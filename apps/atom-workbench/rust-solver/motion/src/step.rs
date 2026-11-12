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
//! One propagation, whatever the generator (`propagate`; spectral.rs). Ĥ(s) = Ĥ₀ + D(s): Ĥ₀ = Ĥ(0), the generator at
//! the step's middle, holds the atom's own frequencies -- detunings, light shifts, decay rates --, which may turn any
//! number of times over the step (an explicit excited state twenty linewidths off its light turns 48 rad in 0.0625 µs,
//! and rings at that for 2/Γ after every jump); D(s) is what the motion along the path and the lights' beats change,
//! slow by the step's own choice. The two are integrated apart:
//! - Ĥ₀ exactly: in its eigenbasis its phases and decays are factors over any width (the interaction picture,
//!   `spectral::Picture`), so the state is known at any time of the step for n exponentials and two products of n²;
//! - D through the step's three nodes alone, its middle and its two ends (two more generators of the model; none where
//!   Ĥ is the same at every time, `Middle::still`: every pair of states holds one wave vector): D(s) = sA + s²B. A pair
//!   of Ĥ₀'s states that turns far between two nodes for what changes -- where its following leaves no more than its
//!   hops would (`spectral::followed`; spectral.rs, # The split) -- follows the change (a polynomial in the time, to
//!   the second order: the generator's own eigenvectors along the path); the slower pairs, with the change's own light
//!   shift, are two hops of the interaction picture a step, the first Magnus term with its integral over the pair's
//!   turn exact, and a quintic between the nodes (spectral.rs). The propagation's work is the same whatever Ĥ₀'s
//!   frequencies times the step are. The gradient's slow change is taken through the same three nodes (`propagate`).
//! The step's time integrals -- the kick, the mean populations -- are read on that state at the nodes of a
//! Gauss–Legendre quadrature at the bandwidth of the forms ⟨x|·|x⟩ on a state Ĥ₀ propagates (`quadrature`: to
//! round-off), piece by piece between the step's jumps (# Jumps), and the norm's root between them. These nodes meet
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
//!   τ sinc(Ω_mτ/2) (`shear`). Of a state that does not change it depends on r_h alone: the conservative kick is a shear
//!   and the step symplectic (propagation.md §2.3);
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
//! are in the Q_j, integrated to round-off, and the model's forms are three. Where Ĥ is still, Ĝ is one matrix and the
//! whole is its one form with ∫ρ = Σ_k w_k ρ_k. (With a decay the mean is a quotient by the norm, whose own ringing --
//! of the size Γ/Δ times the excited share -- puts a part of the quotient at twice the forms' bandwidth, which the
//! quadrature does not resolve: the kick errs by a part of that ringing, measured 2·10⁻⁷ of the dipole kick at twenty
//! linewidths and Δτ = 50, 10⁻⁵ at Δτ = 200 (tests). Without a decay nothing does.)
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
//! κ_h + Δκ/2 (propagation.md §2.3). It depends on the kick the step makes, so it is predicted: κ_m = κ_h + K(ψ₀)/2, K(ψ)
//! the kick of a state that does not change -- its shear at κ_h and the reference's τ∇V_ref(r_h) -- on the step's start
//! state (`predicted`). The generator and the path are both taken at κ_m: one middle, one propagation a step.
//! - The prediction errs by half of what the state's change adds to the kick, τ ΔF/2 (ΔF the force's change over the
//!   step through the state): O(τ²) where the step resolves the internal state. The kick's sensitivity to v_m,
//!   (τ/2)|∂Δκ/∂κ_m|, is O(τ), so the kick errs by O(τ³), the scheme's own local order; the pilot's companion at twice
//!   the step reads it with the rest of the step's error (solve.rs, # The step: each level predicts from its own
//!   steps' starts).
//! - The reverse step predicts from its own start, the step's end: its path's momentum is κ_h + Δκ − K(ψ₁)/2 against the
//!   step's κ_h + K(ψ₀)/2. They differ by (K(ψ₀) + K(ψ₁))/2 − Δκ, the held kick's trapezoid against the kick: O(τ³), so a
//!   step and its reverse miss the start by O(τ⁴), beyond the scheme's order (tests), and by nothing where the kick
//!   does not depend on the path (one internal state: a conservative trap's step is its own inverse to round-off).
//! - Where the step does not resolve the internal state (a jump's bare state relaxing in 1/Γ ≪ τ) ΔF is O(1) and the
//!   path's velocity errs in first order over that step; the companion reads that too.
//!
//! # Jumps (SPEC §1.6)
//! ‖ψ‖² falls monotonically between jumps (the decay is −(i/2)ΣJ†J by the model's construction). A jump is made inside
//! the step, where the step's own propagation reaches the trajectory's threshold θ: at the root u* of ln‖x(u)‖² = ln θ
//! on the step's path, by bracketed inverse interpolation to the norm's rounding (`root`), from the state x(u*) that
//! propagation has there, and **at the path's own point there** -- the position r_h + v_m u*, the momentum κ_m the
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
//!   nodes are laid from its start to the step's end, and where the norm reaches the threshold before that, laid again
//!   up to the root -- states at nodes, no hop;
//! - a jump's kick δκ enters the reference's flow where it is made, exactly: a kick at the time u from the middle ends
//!   the step as the kick δκ cos ωu and the displacement −(ħ/m) δκ sin(ωu)/ω at the middle do, per principal axis (the
//!   identity of `spreading` and of `Segment`);
//! - a jump that lands in another cluster goes on in that cluster's generator at the same middle (`Dynamics::middle`),
//!   on the same path: its picture is made there, its part of the kick is its own forms'.
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
//! Σ Re⟨∇u|⊗|∇u⟩ of the jumps' amplitudes u = Jψ, neither divided by the jumps' rate (model/photon.rs). They are
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
//!   The path's momentum κ_m is predicted without the path's increment (its mean is none): the propagation, its norm
//!   and so the jumps' times do not depend on the deviates.
//!   F_s and 2D are those of the state at the step's middle; what of them goes with an explicit excited state's decay
//!   takes that state's population averaged over the step, by the kick's own quadrature (`propagate`): a jump's bare
//!   state fills its excited state in 1/Γ, which a step need not resolve, and the photons it scatters over the step
//!   are its decay's integral, not its middle's value times τ.
//!
//! # The path between a step's ends
//! A trajectory is read at times that are no step's end (an output, a cut of T's window: adapt.rs, # The reads), and
//! the read is of the scheme's order there too. What a step knows between its ends:
//! - The internal state at any time: its propagation has x(u), u the time from the middle, on the straight path through
//!   the middle (spectral.rs), and ψ_s = e^{iK_s·v_m u} x_s. The times read are stops of the propagation, as its
//!   quadrature's nodes are (`Midway::attempt`): each takes the state as the propagation passes it, whichever jumps
//!   lie before it. That path lies ½au² + ½a_B h² from the atom's (a its
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

use std::sync::OnceLock;

use crate::decide::COMPLEX;
use crate::linalg::{self, inner, norm_sqr, C64, Mat, LANES};
use crate::model::{Cluster, Point, Quadratic, Scattering, STATES};
use crate::random::Stream;
use crate::spectral::Picture;
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

    /// The reference potential's second-order expansion at r (SPEC §2.1): the fixed reference state's trap potential,
    /// `Model::expansion(r, p_ref, &model.prepared())`. The splitting is exact for any reference; this one moves the trap's
    /// harmonic part, and its gradient and curvature at the step's middle, exactly.
    fn reference(&self, r: [f64; 3]) -> Quadratic;

    /// A cluster: its states and its Doppler frame K_s (`Model::clusters()[c]`).
    fn cluster(&self, cluster: usize) -> &Cluster;

    /// The cluster a ground state belongs to (`Model::cluster_of`).
    fn cluster_of(&self, ground: usize) -> usize;

    /// The cluster's generator at the position r and the time t with every light on, for every momentum and along every
    /// straight path through the point (`Middle`): made once a step, at its middle.
    fn middle(&self, cluster: usize, r: [f64; 3], t: f64) -> Self::Middle<'_>;

    /// A jump's photon of the cluster's state ψ (unnormalized) at a point (SPEC §1.6; model/photon.rs
    /// `Model::emission_of`): the model makes the rate W = Σ_a‖u_a‖² of every class the state emits there, in the
    /// classes' order; `draw` picks one of them by its place among those rates, or none; and the model makes the photon
    /// of the class drawn (`Emitted`) -- the one class whose amplitudes and kick a jump reads. None where `draw` picks
    /// none.
    fn emit(&self, cluster: usize, at: &Point, psi: &[C64], draw: &mut dyn FnMut(&[f64]) -> Option<usize>) -> Option<Emitted>;
    /// Where a jump's bare state `f` lands in a cluster at a point: the state the trajectory goes on in, with the norm
    /// it goes on with (a fold's bare state is its slow state and a part that is gone within the excited state's
    /// lifetime; the slow state's norm is the bare state's survival extrapolated back to the jump, below ‖f‖ or above
    /// it; model.rs `Model::landed`). The state itself where the model folds nothing of that.
    fn landed(&self, _cluster: usize, _at: &Point, f: &[C64]) -> Internal {
        f.iter().copied().collect()
    }

    /// The loss energy E_s = (ħ/m)κ²/2 + V_run(r; p(ψ)) of the cluster's state at a point (`Model::energy`, rad/µs).
    fn energy(&self, cluster: usize, at: &Point, psi: &[C64]) -> f64;

    /// The loss test at a point (`Model::lost`): outside the region and E_s above its far value.
    fn lost(&self, cluster: usize, at: &Point, psi: &[C64]) -> bool;

    /// The displayed rows of the cluster's state at a point (`Model::readout`).
    fn readout(&self, cluster: usize, at: &Point, psi: &[C64]) -> [f64; STATES];

    /// The counted work of one middle (`decide::Work`, real multiply-adds), made once per run: the price of a step's
    /// middle beside its own work and its nodes' (`step_work`).
    fn assembly_work(&self) -> f64;

    /// The counted real multiply-adds of one jump's photons (`Dynamics::emit`), made once per run: `jump` adds it to the
    /// trajectory's work.
    fn photon_work(&self) -> f64;
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

    /// Re tr(Ĝ_a(s) ρ) per lab axis a (rad/µs/µm): the gradient of Ĥ's Hermitian part at the time s from the middle along
    /// the path, contracted with a Hermitian ρ on the states (n × n, row by row).
    fn force(&self, path: &Self::Path, s: f64, rho: &[C64]) -> [f64; 3];

    /// The gradient at the fixed point by its tones, G(r_h, t) = Σ_m G_m e^{iΩ_m(t − t_m)}: to `each` every Ω_m (rad/µs)
    /// with Re tr(G_m ρ) per lab axis, for a Hermitian ρ.
    fn tones(&self, path: &Self::Path, rho: &[C64], each: &mut dyn FnMut(f64, [f64; 3]));

    /// What the light a state scatters at the point gives the momentum per unit time beside the Ehrenfest force and the
    /// jumps' kicks (`Model::scattering_on`): the force and the second moment's rate the jumps do not carry, of the
    /// state ψ at the middle's time and the path's momentum, with the states' populations averaged over the step (of
    /// unit sum) for what goes with an explicit state's decay -- its rate is the step's mean, not the middle's. Asked
    /// of the middle, which holds the point: the model makes nothing there a second time.
    fn scattering(&self, path: &Self::Path, psi: &[C64], populations: &[f64]) -> Scattering;

    /// The counted complex multiply-adds of one `generator`, one `force` and one `tones`.
    fn work(&self) -> [f64; 3];
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
    /// The waiting time's threshold θ ∈ (0, 1]: the next jump is where ‖ψ‖² reaches it.
    pub threshold: f64,
    /// The jumps made: the next one draws from `Stream::jump(seed, index, jumps + 1)`.
    pub jumps: u32,
    /// The time the atom was lost (SPEC §1.7: checked at every step's end and after every jump); None while held.
    pub lost: Option<f64>,
    /// The run's seed and the trajectory's number: its jumps' streams.
    pub seed: u64,
    pub index: u64,
    /// The work counted (`decide::Work`, real multiply-adds) of the integrator's own algorithms over the trajectory --
    /// the generator's decomposition, its following and hops, the states at the kick's nodes and their sums, with the
    /// model's generators and forces at the step's nodes (`adapt` adds each step's) --, and its jumps' photons
    /// (`Dynamics::photon_work`).
    pub work: f64,
}

impl Trajectory {
    /// A trajectory at a basis state (a ground state of the model basis) at (r, κ) and time 0: ψ the unit vector of that
    /// state on its cluster, the first threshold from jump 0's stream (SPEC §2.4).
    pub fn new<D: Dynamics>(model: &D, state: usize, r: [f64; 3], kappa: [f64; 3], seed: u64, index: u64) -> Trajectory {
        let cluster = model.cluster_of(state);
        let states = &model.cluster(cluster).states;
        let at = states.iter().position(|&s| s == state).expect("a ground state lies on its cluster");
        let mut psi = Internal::zeros(states.len());
        psi[at] = C64::ONE;
        let threshold = Stream::jump(seed, index, 0).open_uniform();
        Trajectory {
            r,
            kappa,
            psi,
            cluster,
            time: 0.0,
            threshold,
            jumps: 0,
            lost: None,
            seed,
            index,
            work: 0.0,
        }
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

/// The reference's exact flow (A) of a step: the expansion point p (µm), the principal axes (unit rows), per axis the
/// curvature λ (rad/µs/µm²) and the gradient g at p along it (rad/µs/µm), and ħ/m.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Flow {
    pub centre: [f64; 3],
    pub axes: [[f64; 3]; 3],
    pub curvatures: [f64; 3],
    pub gradient: [f64; 3],
    pub hbar_over_mass: f64,
}

impl Flow {
    /// The flow of the expansion `q` at p.
    fn new(q: &Quadratic, centre: [f64; 3], hbar_over_mass: f64) -> Flow {
        let (curvatures, axes) = linalg::symmetric3_eigen(&q.hessian);
        let gradient = std::array::from_fn(|i| linalg::dot3(axes[i], q.gradient));
        Flow { centre, axes, curvatures, gradient, hbar_over_mass }
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
        let mut sizes = [self.centre.iter().fold(0.0, |m: f64, x| m.max(x.abs())), 0.0];
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
        self.curvatures.iter().map(|l| (self.hbar_over_mass * l.abs()).sqrt()).fold(0.0, f64::max)
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
    let apart = |a: [f64; 3], b: [f64; 3]| (0..3).map(|i| (a[i] - b[i]).abs()).fold(0.0, f64::max);
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
        contraction = contraction.max(moved / last);
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
// Quadrature to round-off.

/// A Gauss–Legendre rule of m nodes on [−1, 1] (nodes, weights) and its panel phase x_m: the largest ν w at which its
/// error on a component e^{iνs} over a panel of width w, w (νw)^{2m} (m!)⁴/((2m + 1)((2m)!)³) (Davis & Rabinowitz
/// 2.7.12), is within the unit round-off times w -- the panel a rule integrates an entire function of that bandwidth on to
/// round-off.
struct Rule {
    nodes: Vec<f64>,
    weights: Vec<f64>,
    phase: f64,
}

impl Rule {
    fn new(m: usize) -> Rule {
        let (nodes, weights) = crate::special::gauss_legendre(m).into_iter().unzip();
        let ln_factorial = |n: usize| (1..=n).map(|k| (k as f64).ln()).sum::<f64>();
        let ln = UNIT_ROUNDOFF.ln() + ((2 * m + 1) as f64).ln() + 3.0 * ln_factorial(2 * m) - 4.0 * ln_factorial(m);
        Rule { nodes, weights, phase: (ln / (2 * m) as f64).exp() }
    }
}

/// The rules of 2 to 64 nodes. The least is the smallest rule there is (`FORMS`, `MOMENTS` ask for no fewer). The
/// largest is not derived: a rule's panel phase over its nodes, x_m/m, rises with m towards 8/e (Stirling on the
/// error's factorials), so no rule is the last that pays; 64 is at three quarters of that limit (u^{1/128}). Every
/// rule integrates to round-off: the largest sets the nodes of a stiff step's quadrature, its cost alone.
fn rules() -> &'static [Rule] {
    static RULES: OnceLock<Vec<Rule>> = OnceLock::new();
    RULES.get_or_init(|| (2..=64).map(Rule::new).collect())
}

/// The rule and the panels over a width of an entire integrand of bandwidth ν to round-off: the fewest nodes (the
/// smaller rule of two that take as many), each panel of width w within its rule's phase, νw ≤ x_m, of the rules of
/// `nodes` nodes and more -- a rule of m nodes is exact for the polynomials of degree 2m − 1, which is what is asked
/// of it where the bandwidth vanishes. Its node count is the propagation's states and the kick's sums (`propagate`,
/// `count::impulse`). The rules are looked at from the smallest up as far as the first that takes one panel: every
/// larger rule has more nodes than that one.
fn plan(width: f64, bandwidth: f64, nodes: usize) -> (&'static Rule, usize) {
    let mut least: Option<(&'static Rule, usize)> = None;
    for rule in rules().iter().filter(|rule| rule.nodes.len() >= nodes) {
        let panels = ((bandwidth * width / rule.phase).ceil() as usize).max(1);
        if least.map_or(true, |(best, most)| rule.nodes.len() * panels < best.nodes.len() * most) {
            least = Some((rule, panels));
        }
        if panels == 1 {
            break;
        }
    }
    least.expect("a rule")
}

/// The nodes and weights over [a, b] of an entire integrand of bandwidth ν to round-off (`plan`, of the rules of
/// `nodes` nodes and more), in their order from a to b (the rule's nodes are held from the largest down), panel by
/// panel as they are read: a stiff generator's millions of nodes are never held at once.
fn quadrature(a: f64, b: f64, bandwidth: f64, nodes: usize) -> impl Iterator<Item = (f64, f64)> {
    let (rule, panels) = plan((b - a).abs(), bandwidth, nodes);
    let w = (b - a) / panels as f64;
    (0..panels).flat_map(move |p| {
        let centre = a + (p as f64 + 0.5) * w;
        rule.nodes.iter().zip(&rule.weights).rev().map(move |(x, weight)| (centre + 0.5 * w * x, 0.5 * w * weight))
    })
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
        let sigma = (theta * (1.0 - theta)).max(0.0);
        sigma * (1.0 - 3.0 * sigma)
    }
    /// ∫κ_a dt and ∫κ_a² dt (1/µm·µs, 1/µm²·µs) per lab axis over [from, to] within the path's part of the step:
    /// Gauss–Legendre to round-off on the path, κ² of bandwidth twice the flow's rate (`quadrature`). κ² in the mean
    /// over the Brownian path: the path is the mean given the step's two moments, and the square of what they do not
    /// tell is added, (Λ Λᵀ)_aa |τ| ∫σ(1 − 3σ)dt (`untold`; ∫₀^θ σ(1 − 3σ)dθ = θ²/2 − 4θ³/3 + 3θ⁴/2 − 3θ⁵/5, 1/15 over the
    /// step) -- so that a cut of a step has the diffusion's input up to it, as the whole step has. (The residual is the
    /// free flight's; the reference's rotation changes it by its (ωτ)² share, the third order.)
    pub fn moments(&self, from: f64, to: f64) -> ([f64; 3], [f64; 3]) {
        let (lo, hi) = (from.min(to), from.max(to));
        let sign = if to >= from { 1.0 } else { -1.0 };
        let (mut one, mut two) = ([0.0; 3], [0.0; 3]);
        if !(hi > lo) {
            return (one, two);
        }
        for (t, w) in quadrature(lo, hi, 2.0 * self.flow.rate(), MOMENTS) {
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
    let (n, scale) = (x.len(), 1.0 / norm_sqr(x));
    let width = row(n);
    let (mut re, mut im) = ([0.0; row(STATES)], [0.0; row(STATES)]);
    for (z, (re, im)) in x.iter().zip(re.iter_mut().zip(&mut im)) {
        (*re, *im) = (z.re, z.im);
    }
    for (a, left) in x.iter().enumerate() {
        let (left, from) = (*left * scale, a / LANES * LANES);
        for (at, (re, im)) in re[from..width].chunks_exact(LANES).zip(im[from..width].chunks_exact(LANES)).enumerate() {
            // x_a x_b*/⟨x|x⟩ on a vector of columns b.
            let (mut real, mut imaginary) = ([0.0; LANES], [0.0; LANES]);
            for l in 0..LANES {
                real[l] = left.re.mul_add(re[l], left.im * im[l]);
                imaginary[l] = left.im.mul_add(re[l], -(left.re * im[l]));
            }
            let place = 2 * a * width + from + at * LANES;
            for (sum, w) in sums.chunks_exact_mut(2 * n * width).zip(weights) {
                let (head, tail) = sum[place..place + width + LANES].split_at_mut(width);
                for l in 0..LANES {
                    head[l] = w.mul_add(real[l], head[l]);
                    tail[l] = w.mul_add(imaginary[l], tail[l]);
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

/// A state's density matrix ρ = |x⟩⟨x|/⟨x|x⟩, ρ_ab = x_a x_b*/‖x‖², row by row: a mean ⟨x|G|x⟩/⟨x|x⟩ is tr(Gρ), which the
/// model contracts (`Middle::force`, `Middle::tones`).
fn density(x: &[C64], rho: &mut [C64]) {
    let (n, scale) = (x.len(), 1.0 / norm_sqr(x));
    for (a, left) in x.iter().enumerate() {
        let left = *left * scale;
        for (b, right) in x.iter().enumerate() {
            rho[a * n + b] = left * right.conj();
        }
    }
}

/// The shear (module documentation, # The kick): −∫⟨ψ|G(r_h, t)|ψ⟩dt/‖ψ‖² over the step of a state held, at the fixed r_h
/// each tone at its beat Ω_m alone, ∫e^{iΩ_m(t − t_m)}dt = τ sinc(Ω_mτ/2). `rho` takes the state's density.
fn shear<M: Middle>(along: &Along<M>, rho: &mut [C64], psi: &[C64], tau: f64) -> [f64; 3] {
    density(psi, rho);
    let mut f = [0.0; 3];
    along.middle.tones(&along.path, rho, &mut |omega, form| {
        let w = tau * sinc(0.5 * omega * tau);
        for a in 0..3 {
            f[a] -= w * form[a];
        }
    });
    f
}

/// A trajectory's scratch memory for its steps, made with it and fitted to each step's cluster (`Scratch::fit`): the
/// generator's picture along the step's path (`spectral::Picture`: the generator at the step's nodes, its decomposition,
/// its following, the state's path), one array carved (`Scratch::carve`) into the middle state's density and a
/// Hermitian sum the model takes, and the nodes' weighted sums of the states' densities as `accumulate` holds them,
/// twice: the sums, and the sums as they stood where the piece being propagated began (`Midway::attempt`). Every use
/// writes what it reads: nothing a step left in it is read by another.
#[derive(Default)]
pub struct Scratch {
    picture: Picture,
    rest: Vec<C64>,
    sums: Vec<f64>,
}

struct Carved<'a> {
    picture: &'a mut Picture,
    held: &'a mut [C64],
    sum: &'a mut [C64],
    sums: &'a mut [f64],
    saved: &'a mut [f64],
}

impl Scratch {
    /// The scratch for a cluster of n states.
    fn fit(&mut self, n: usize) {
        self.picture.fit(n);
        self.rest.resize(2 * n * n, C64::ZERO);
        self.sums.resize(2 * SUMS * 2 * n * row(n), 0.0);
    }
    /// The scratch's parts.
    fn carve(&mut self) -> Carved<'_> {
        let n = self.picture.states();
        let (held, sum) = self.rest.split_at_mut(n * n);
        let (sums, saved) = self.sums.split_at_mut(SUMS * 2 * n * row(n));
        Carved { picture: &mut self.picture, held, sum, sums, saved }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// The propagations.

/// The counted work (`decide::Work`, real multiply-adds) of the integrator's operations on n states, one formula each,
/// the model's own counts of its generator, its force and its tones at a node with them (`Middle::work`): what a
/// step adds up and what `step_work` prices a step by.
mod count {
    use super::COMPLEX;
    /// The propagation: the model's generator at the step's nodes -- three, or the one of a path it is still along --
    /// and the picture's own count of its complex multiply-adds (`spectral::Picture::work`: the decomposition, the
    /// slopes, the following, the hops, the states at the nodes).
    pub fn propagation(model: [f64; 3], still: bool, picture: f64) -> f64 {
        COMPLEX * (if still { 1.0 } else { 3.0 } * model[0] + picture)
    }
    /// The kick's part of the state through `nodes` nodes: per node its density's upper triangle into the weighted
    /// sums (one on a still path, three on a beating one); then the model's one form of the sum, or on a beating path
    /// the middle state's density, its tones, and the three nodes' forms.
    pub fn impulse(n: usize, model: [f64; 3], nodes: usize, still: bool) -> f64 {
        let (sums, forms) = if still { (1, model[1]) } else { (3, (n * n) as f64 + model[2] + 3.0 * model[1]) };
        COMPLEX * ((nodes * (1 + sums) * n * (n + 1) / 2) as f64 + forms)
    }
    /// The shear of the start state, the prediction of the path's momentum: its density and its tones.
    pub fn shear(n: usize, model: [f64; 3]) -> f64 {
        COMPLEX * ((n * n) as f64 + model[2])
    }
}

/// The node count of a quadrature over τ at a bandwidth (`plan`).
fn nodes(tau: f64, bandwidth: f64) -> usize {
    let (rule, panels) = plan(tau.abs(), bandwidth, FORMS);
    rule.nodes.len() * panels
}

/// The picture of the generator along a step's path (`spectral::Picture`), prepared: the model's generator at the
/// step's middle and, unless it is still along every path, at the step's two ends.
fn pictured<M: Middle>(along: &Along<M>, picture: &mut Picture, tau: f64) {
    let (h, still) = (0.5 * tau, along.middle.still());
    along.middle.generator(&along.path, 0.0, &mut picture.base);
    if !still {
        along.middle.generator(&along.path, -h, &mut picture.ends[0]);
        along.middle.generator(&along.path, h, &mut picture.ends[1]);
    }
    picture.prepare(h, still);
}

/// A node's state into the kick's sums (`accumulate`): with its weight where Ĥ is still along the path, with its
/// weight times the Lagrange polynomials of the step's three nodes on a beating one, whose sums `weights` takes.
fn gather(still: bool, h: f64, u: f64, weight: f64, x: &[C64], weights: &mut [f64; 3], sums: &mut [f64]) {
    if still {
        accumulate(x, &[weight], sums);
    } else {
        let s = u / h;
        let lagrange = [0.5 * s * (s - 1.0), 1.0 - s * s, 0.5 * s * (s + 1.0)].map(|l| weight * l);
        for (sum, l) in weights.iter_mut().zip(lagrange) {
            *sum += l;
        }
        accumulate(x, &lagrange, sums);
    }
}

/// A root in [lo, hi] of ln D(s) = ln θ for a non-increasing D with D(hi) ≤ θ, D from `norm` and D at the bracket's ends
/// given (lo itself where D(lo) ≤ θ already): by the inverse interpolation of ln D − ln θ through its last three points
/// (their parabola's root; their last two's line where three do not tell), kept inside the bracket (its middle where
/// both step out), until ln D is within its rounding of ln θ or the iteration no longer moves; and where `ROUNDS` of
/// them have not got there, by the bracket's halving until its ends are neighbours -- the root ends by its rounding or
/// by its bracket, never by a count.
///
/// `floor` is the rounding of ln D as it is computed -- twice the state's relative to its norm, and the norm's own sum --,
/// the two logarithms' beside it: within it s is the root as far as the norm tells, and the interpolation's next point,
/// taken without another evaluation, is the root to the norm's actual rounding over its rate. (Beyond it the points
/// wander in that rounding until the bracket closes on neighbouring numbers: some ten evaluations more.) The norm's rate
/// 2 Im⟨x|Ĥ|x⟩ is not asked: between two nodes it is not the slope of the state's own norm to the last digit, and a Newton
/// iteration with it converged at that difference's pace, in twice the evaluations.
fn root(mut norm: impl FnMut(f64) -> f64, (lo, at_lo): (f64, f64), (hi, at_hi): (f64, f64), theta: f64, floor: f64) -> f64 {
    let target = theta.ln();
    let floor = floor + 2.0 * UNIT_ROUNDOFF * target.abs();
    let (a, b) = (at_lo.ln() - target, at_hi.ln() - target);
    if !(a > floor) {
        return lo;
    }
    if !(b < -floor) {
        return hi;
    }
    // The last three points (s, ln D − ln θ), the latest first, and their interpolation's root where it lies inside.
    let interpolated = |[(x1, y1), (x2, y2), (x3, y3)]: [(f64, f64); 3], lo: f64, hi: f64| {
        let parabola = x1 * y2 * y3 / ((y1 - y2) * (y1 - y3)) + x2 * y1 * y3 / ((y2 - y1) * (y2 - y3)) + x3 * y1 * y2 / ((y3 - y1) * (y3 - y2));
        let line = x1 - y1 * (x1 - x2) / (y1 - y2);
        [parabola, line].into_iter().find(|&s| s > lo && s < hi)
    };
    let (mut lo, mut hi) = (lo, hi);
    let mut points = [(hi, b), (lo, a), (f64::NAN, f64::NAN)];
    let mut s = interpolated(points, lo, hi).unwrap_or(0.5 * (lo + hi));
    for _ in 0..ROUNDS {
        let g = norm(s).ln() - target;
        if g > 0.0 {
            lo = s;
        } else {
            hi = s;
        }
        points = [(s, g), points[0], points[1]];
        let refined = interpolated(points, lo, hi);
        if g.abs() <= floor {
            return refined.unwrap_or(s);
        }
        let next = refined.unwrap_or(0.5 * (lo + hi));
        if next == s || !(hi > lo) {
            return s;
        }
        s = next;
    }
    loop {
        let middle = 0.5 * (lo + hi);
        if !(middle > lo && middle < hi) {
            return hi;
        }
        if norm(middle).ln() - target > 0.0 {
            lo = middle;
        } else {
            hi = middle;
        }
    }
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
    /// The trajectory's states made inside the step, in the order of their times: after each of its jumps, and at the
    /// times read.
    pub made: Vec<Made>,
    /// The step's counted work (real multiply-adds: its own and its nodes' generators and forces; its jumps' photons
    /// are counted on the states). The middles it went through are counted where they are made (`Midway`).
    pub work: f64,
}

/// A state of the trajectory made inside a step: after a jump -- with the jump's kick (1/µm) -- or at a time read.
/// The internal state, the cluster, the time and the trajectory's counters are the propagation's there; the position
/// and the momentum are the caller's to set on the step's path (`Segment::after`, `Segment::place`).
#[derive(Clone, Debug)]
pub struct Made {
    pub kick: Option<[f64; 3]>,
    pub state: Trajectory,
}

/// Why a step or a jump could not be made: the reference's fixed point is not a contraction at this step (`middle`); the
/// state left the floating-point range; or the norm reached its threshold with no photon to emit (`jump`: the model's
/// decay and its photons contradict each other). The advance takes the step at its half at the first two (adapt.rs, #
/// The step), and ends at the third.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Refused {
    Reference,
    Overflow,
    Emission,
}

/// The path's momentum of the step of τ through a middle from a state (module documentation, # The path's momentum),
/// predicted: κ_h and half the kick of the start state held there -- its shear at κ_h and the reference's τ∇V_ref(r_h).
fn predicted<M: Middle>(scratch: &mut Scratch, middle: &M, psi: &[C64], kappa_h: [f64; 3], reference: [f64; 3], tau: f64) -> [f64; 3] {
    let held = shear(&Along::new(middle, kappa_h, [0.0; 3]), scratch.carve().held, psi, tau);
    std::array::from_fn(|a| kappa_h[a] + 0.5 * (held[a] + tau * reference[a]))
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

/// The stops of a step's propagation beside its quadrature's nodes: the step's middle, a time read, the step's end.
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
/// scattered light's force and its diffusion's factor at the middle, once it is passed; the counted work; and the
/// states made.
struct March {
    u: f64,
    state: Trajectory,
    read: usize,
    passed: bool,
    weights: [f64; 3],
    impulse: [f64; 3],
    scattered: Option<([f64; 3], [[f64; 3]; 3])>,
    work: f64,
    made: Vec<Made>,
}

impl<'a, D: Dynamics> Midway<'a, D> {
    /// The middle of the step of τ (negative: the reverse step) from a trajectory's position, momentum, cluster and
    /// time: the reference's fixed point (`middle`; refused where it is not one) and the model's generator there.
    pub fn new(model: &'a D, from: &Trajectory, tau: f64) -> Result<Midway<'a, D>, Refused> {
        let half = 0.5 * tau;
        let (flow, reached) = middle(model, from.r, from.kappa, half)?;
        let generator = model.middle(from.cluster, reached.0, from.time + half);
        Ok(Midway { model, tau, hbar_over_mass: model.hbar_over_mass(), flow, reached, reference: flow.gradient_at(reached.0), generator })
    }

    /// The step through this middle of a state at its start -- the middle's own position, momentum, cluster and time,
    /// any internal state of the cluster --, with its jumps (module documentation, # Jumps) and the states at the
    /// times `reads` (increasing, after the step's start and no later than its end): the path's momentum predicted
    /// from the state (`predicted`); the propagation at it and on its path, cluster by cluster as the jumps land
    /// (`stretch`); the kick -- the propagation's impulse, the reference's τ∇V_ref(r_h), the scattered light's force
    /// and its diffusion's part of the Brownian path `brownian` (module documentation, # The absorbed light's
    /// momentum), and each jump's own where it is made -- and the second half flow. `scratch` is the trajectory's. A
    /// jump the model gives no photon is refused (`jump`).
    pub fn attempt(&self, from: &Trajectory, brownian: &Brownian, reads: &[f64], scratch: &mut Scratch) -> Result<Step, Refused> {
        let (tau, hm, n) = (self.tau, self.hbar_over_mass, from.psi.len());
        let h = 0.5 * tau;
        let (flow, (r_h, kappa_h), reference) = (self.flow, self.reached, self.reference);
        scratch.fit(n);
        let kappa_m = predicted(scratch, &self.generator, &from.psi, kappa_h, reference, tau);
        let mut march = March { u: -h, state: from.clone(), read: 0, passed: false, weights: [0.0; 3], impulse: [0.0; 3], scattered: None, work: count::shear(n, self.generator.work()), made: Vec::new() };
        // Cluster by cluster: the start's generator, then that of every cluster a jump lands in, at the same middle.
        let mut landed: Option<D::Middle<'a>> = None;
        while !self.stretch(landed.as_ref().unwrap_or(&self.generator), kappa_m, from.time, reads, &mut march, scratch)? {
            landed = Some(self.model.middle(march.state.cluster, r_h, from.time + h));
        }
        let March { state: mut end, read, impulse, scattered, work, mut made, .. } = march;
        // The scattered light at the middle, on the state there: its force over the step, and its diffusion's factor on
        // the Brownian path's increment (the momentum) and on its first moment about the middle (the position).
        let (force, spread) = scattered.expect("a step passes its middle");
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
            return Err(Refused::Overflow);
        }
        (end.r, end.kappa, end.time) = (r1, kappa1, from.time + tau);
        // A time read at the step's end is the end itself.
        made.extend(reads[read..].iter().map(|&time| Made { kick: None, state: Trajectory { time, ..end.clone() } }));
        let segment = Segment::new(from.time, tau, flow, (from.r, from.kappa), kick, spread);
        Ok(Step { end, segment, made, work })
    }

    /// The propagation of B on the step's path from where the march stands, in the cluster its state is in, as far as
    /// the step's end (true) or a jump that lands in another cluster (false): the cluster's generator `generator` at
    /// the middle with the path through it at κ_m, its picture (`spectral::Picture`), and piece by piece between the
    /// jumps
    /// - the picture launched from the piece's first state (`spectral::Picture::launch`: the state is then known at
    ///   any time from there on);
    /// - the frame's state read at the stops in the path's order: the nodes of a quadrature that resolves the
    ///   generator's own frequencies -- Gauss–Legendre from the piece's start to the step's end at the bandwidth of the
    ///   forms on a state it propagates (`quadrature`) --, each with its weight; the step's middle, the times read
    ///   and the step's end, where the state is kept -- a node within the time's round-off of one of them is it;
    /// - where ‖ψ‖² is at the threshold or below at a stop: the norm's root between the last two stops, the state read
    ///   at the root's times (`spectral::Picture::between`; `root`); the piece's nodes laid again from its start to
    ///   the root, the sums as they stood at the piece's start; and the jump at the root on the path's own point
    ///   (module documentation, # Jumps; `jump`), its state there the next piece's first.
    ///
    /// The stretch's part of the kick is exact in its decomposition (module documentation, # The kick): the force
    /// along the path, −∫Re tr(Ĝ(u)ρ(u))du over the stretch, Ĝ through the step's three nodes u_j = −h, 0, h and their
    /// Lagrange polynomials ℓ_j -- Σ_j Re tr(Ĝ(u_j) S_j), S_j = Σ_k w_k ℓ_j(u_k) ρ_k over the stretch's nodes, three
    /// forms of the model --; and in the stretch the step's middle lies in, about the middle state held in the frame
    /// over the whole step: its shear at r_h, each tone at its beat alone (`shear`), and S_j less W_j ρ_m, W_j = ∫ℓ_j
    /// over the step -- the weights' sum over every piece of the step, and of what is left of it past this stretch.
    /// Of a state that does not change that difference is its rounding, and nothing where there is one state. Where
    /// Ĥ is still along every path Ĝ is one matrix, and the stretch's part its one form with ∫ρ(u)du. The middle's
    /// stretch reads the scattered light there, on the middle state and its own populations' mean.
    fn stretch(&self, generator: &D::Middle<'a>, kappa_m: [f64; 3], start: f64, reads: &[f64], march: &mut March, scratch: &mut Scratch) -> Result<bool, Refused> {
        let (tau, hm) = (self.tau, self.hbar_over_mass);
        let h = 0.5 * tau;
        let velocity = kappa_m.map(|k| hm * k);
        let along = Along::new(generator, kappa_m, velocity);
        let (n, still, costs, cluster) = (along.n, generator.still(), generator.work(), march.state.cluster);
        scratch.fit(n);
        let Carved { picture, held, sum, sums, saved } = scratch.carve();
        // The nodes' sums: one where Ĥ is still, the three nodes' on a beating path.
        let size = (if still { 1 } else { SUMS }) * 2 * n * row(n);
        let (sums, saved) = (&mut sums[..size], &mut saved[..size]);
        pictured(&along, picture, tau);
        let bandwidth = picture.bandwidth;
        sums.fill(0.0);
        let ahead = |u: f64| (u + h) / tau;
        let (begin, mut counted, mut middle) = (march.u, 0, None);
        let mut x = [C64::ZERO; STATES];
        let done = loop {
            // A piece: from the state the march stands at, the sums and the weights as they are.
            let from = march.u;
            let launched = along.turned(&march.state.psi, from, true);
            picture.launch(from, &launched[..n]);
            saved.copy_from_slice(sums);
            let (theta, weights) = (march.state.threshold, march.weights);
            let (mut at, mut norm) = (from, norm_sqr(&march.state.psi));
            let mut nodes = quadrature(from, h, bandwidth, FORMS).peekable();
            let mut fixed = [(!march.passed).then_some((0.0, Stop::Middle)), Some((h, Stop::End))].into_iter().flatten().peekable();
            let crossed = loop {
                // The next stop: a node, a kept stop -- the earlier of the next fixed one and the next time read, a
                // time read at the step's end being the end's own -- or both at once.
                let read = reads.get(march.read).map(|&time| (time - start) - h).filter(|&u| ahead(u) < 1.0);
                let kept = match (fixed.peek().copied(), read) {
                    (Some((stop, _)), Some(u)) if ahead(u) < ahead(stop) => Some((u, Stop::Read)),
                    (stop, _) => stop,
                };
                let (u, weight, keep) = match (nodes.peek().copied(), kept) {
                    (Some((u, w)), Some((stop, kind))) if (u - stop).abs() <= f64::EPSILON * h.abs() => {
                        nodes.next();
                        (stop, Some(w), Some(kind))
                    }
                    (Some((u, w)), Some((stop, _))) if ahead(u) < ahead(stop) => {
                        nodes.next();
                        (u, Some(w), None)
                    }
                    (_, Some((stop, kind))) => (stop, None, Some(kind)),
                    (_, None) => unreachable!("a piece's stops end with the step's end"),
                };
                let bound = picture.advance(u, &mut x[..n]);
                let now = norm_sqr(&x[..n]);
                if now <= theta {
                    let floor = 2.0 * (picture.rounding + bound) + rounding(n + 1);
                    let between = |time: f64| {
                        let mut z = [C64::ZERO; STATES];
                        picture.between(time - h, &mut z[..n]);
                        norm_sqr(&z[..n])
                    };
                    break Some(root(between, (at + h, norm), (u + h, now), theta, floor) - h);
                }
                if let Some(w) = weight {
                    counted += 1;
                    gather(still, h, u, w, &x[..n], &mut march.weights, sums);
                }
                match keep {
                    Some(Stop::Middle) => {
                        (middle, march.passed) = (Some(x), true);
                        fixed.next();
                    }
                    Some(Stop::Read) => {
                        let state = Trajectory { psi: along.turned(&x[..n], u, false)[..n].iter().copied().collect(), time: reads[march.read], ..march.state.clone() };
                        march.made.push(Made { kick: None, state });
                        march.read += 1;
                    }
                    Some(Stop::End) => break None,
                    None => {}
                }
                (at, norm) = (u, now);
            };
            let Some(crossed) = crossed else {
                march.state.psi.copy_from_slice(&along.turned(&x[..n], h, false)[..n]);
                march.u = h;
                break true;
            };
            // The piece again, as far as the root: its nodes, and the state at the root.
            sums.copy_from_slice(saved);
            march.weights = weights;
            picture.again(&launched[..n]);
            if ahead(crossed) > ahead(from) {
                for (u, w) in quadrature(from, crossed, bandwidth, FORMS) {
                    picture.advance(u, &mut x[..n]);
                    counted += 1;
                    gather(still, h, u, w, &x[..n], &mut march.weights, sums);
                }
            }
            picture.advance(crossed, &mut x[..n]);
            // The jump, on the path's own point.
            let state = &mut march.state;
            state.psi.copy_from_slice(&along.turned(&x[..n], crossed, false)[..n]);
            (state.r, state.kappa, state.time) = (std::array::from_fn(|a| self.reached.0[a] + velocity[a] * crossed), kappa_m, start + (crossed + h));
            jump(self.model, state)?;
            let kick = std::array::from_fn(|a| state.kappa[a] - kappa_m[a]);
            march.made.push(Made { kick: Some(kick), state: state.clone() });
            march.u = crossed;
            if !(ahead(crossed) < 1.0) {
                break true;
            }
            if march.state.cluster != cluster {
                break false;
            }
        };
        // The stretch's part of the kick; in the middle's stretch about the middle state, with the scattered light.
        let length = march.u - begin;
        let mut populations = [0.0; STATES];
        let mut mean = |sum: &[C64]| {
            for (s, p) in populations[..n].iter_mut().enumerate() {
                *p += sum[s * n + s].re / length;
            }
        };
        let part = if still {
            whole(sums, 0, n, sum);
            mean(sum);
            along.middle.force(&along.path, 0.0, sum).map(|f| -f)
        } else {
            let mut out = [0.0; 3];
            if let Some(middle) = &middle {
                if !done {
                    for (u, w) in quadrature(march.u, h, 0.0, FORMS) {
                        let s = u / h;
                        for (sum, l) in march.weights.iter_mut().zip([0.5 * s * (s - 1.0), 1.0 - s * s, 0.5 * s * (s + 1.0)]) {
                            *sum += w * l;
                        }
                    }
                }
                out = shear(&along, held, &middle[..n], tau);
            }
            for (j, at) in [-h, 0.0, h].into_iter().enumerate() {
                whole(sums, j, n, sum);
                mean(sum);
                if middle.is_some() {
                    for (q, rho) in sum.iter_mut().zip(held.iter()) {
                        *q -= *rho * march.weights[j];
                    }
                }
                let form = along.middle.force(&along.path, at, sum);
                for a in 0..3 {
                    out[a] -= form[a];
                }
            }
            out
        };
        for a in 0..3 {
            march.impulse[a] += part[a];
        }
        if let Some(middle) = &middle {
            // A stretch of no length (two jumps at the middle's own time) has no mean: its populations are the middle state's.
            if !(length != 0.0) {
                let norm = norm_sqr(&middle[..n]);
                for (p, z) in populations[..n].iter_mut().zip(&middle[..n]) {
                    *p = z.norm_sqr() / norm;
                }
            }
            let scattered = along.middle.scattering(&along.path, &middle[..n], &populations[..n]);
            march.scattered = Some((scattered.force, factor(&scattered.diffusion, scattered.size, scattered.links)));
        }
        march.work += count::propagation(costs, still, picture.work) + count::impulse(n, costs, counted, still);
        Ok(done)
    }
}

/// The Stumpff function c₃(x) = (1 − c₁(x))/x = Σ_j (−x)^j/(3 + 2j)!, entire in x, by its series to round-off: the
/// quotient cancels where x is small, and the series' terms fall from the first while |x| is below its second term's
/// 20 (a step's x = ω²τ² is below π² on a trapping axis, SPEC §2.2; on a hyperbolic one every term is positive).
fn stumpff3(x: f64) -> f64 {
    let (mut term, mut sum, mut j) = (1.0 / 6.0, 0.0, 0.0);
    while sum + term != sum {
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

/// The counted work of one step of τ on the path through a middle at a momentum and a velocity: what `Step::work`
/// counts, priced without taking the step -- the solve's price of a model it has not run, at its pilot's states and
/// steps. It makes no middle: the caller's is a state's own, or one of the cluster made once -- the nodes' counts are
/// the cluster's whatever the point (`Middle::work`), and the point gives the generator's picture alone: its
/// decomposition, its slopes and its following as the step makes them (`pictured`), the quadrature's nodes at its
/// bandwidth, and the launch with a state at each of them, at the middle and at the end (`spectral::Picture::price`);
/// the kick's sums and forms, and the shear. The middle itself is the model's (`Dynamics::assembly_work`, one a
/// step).
pub fn step_work<M: Middle>(middle: &M, kappa: [f64; 3], velocity: [f64; 3], tau: f64) -> f64 {
    let (n, costs, still) = (middle.states(), middle.work(), middle.still());
    let along = Along::new(middle, kappa, velocity);
    let mut scratch = Scratch::default();
    scratch.fit(n);
    let picture = &mut scratch.picture;
    pictured(&along, picture, tau);
    let nodes = nodes(tau, picture.bandwidth);
    let own = picture.work + picture.price(nodes + 2);
    count::propagation(costs, still, own) + count::impulse(n, costs, nodes, still) + count::shear(n, costs)
}

// ---------------------------------------------------------------------------------------------------------------------
// The jump.

/// A factor Λ of a diffusion's second-moment rate C = 2D, Λ Λᵀ = C (positive semidefinite: the second moment's rate less
/// what the jumps carry of it, Cauchy–Schwarz; model/photon.rs): its Cholesky factor's columns, so that Λ on a Brownian
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

/// The jump of a trajectory at its state (SPEC §1.6): from `Stream::jump(seed, index, jumps + 1)` the next threshold, the
/// class ∝ Σ_a‖u_a‖² among the rates the model makes of the state's classes, whose photon alone the model then makes
/// (`Dynamics::emit`), the polarization across the direction by ‖Σ_a ε_a u_a‖², and the direction from the dipole pattern
/// tr G − n·Re(G)·n (at most tr G, as Re G ≥ 0) by exact rejection from the uniform sphere; then ψ ← f/‖f‖ on the landing
/// cluster and κ += −k_φ n + s b, the emitted photon's recoil and the class's conditional mean of the absorbed momentum
/// as far as one photon's goes (`Emitted::kick`; module documentation, # The absorbed light's momentum) -- at most
/// k_φ + k_c, whatever the light does at the point. The point is the trajectory's as it is given: a step makes its
/// jumps on its path's points (module documentation, # Jumps), and the loss after a jump is its caller's to test, on
/// the trajectory's own place (adapt.rs). A norm that reached its threshold with no photon to emit -- every class's
/// weight zero, or not a number -- is refused (`Refused::Emission`) with the trajectory as it was: the model's decay
/// and its photons contradict each other, and no jump is made of that.
///
/// The landing (`Dynamics::landed`): the trajectory goes on in the state the bare state f/‖f‖ lands in, with that state's
/// norm, against the threshold as every state is -- one rule, whatever the norm. A norm above the threshold waits: from
/// one, ‖f‖'s own, as before; from below one, as long as the rest of the threshold's law says; from above one, longer,
/// by the time its decay takes down to one. A norm at or below the threshold already (only a norm below one meets it
/// but for the one threshold in 2⁵³ that is one itself) is one more photon at once: the bare state emits it as any
/// state does -- its class, polarization and direction drawn anew, its recoil and its bounded kick with them --, the
/// next threshold is drawn anew, and it lands again. So the waiting time after a jump is P(T > t) = min(1, n e^{−γt})
/// for the landing's norm n and the state's rate γ (model.rs `Model::landed`: the fourth order's statistics for either
/// sign of 1 − n), and a state that lands as it is draws nothing more.
pub fn jump<D: Dynamics>(model: &D, trajectory: &mut Trajectory) -> Result<(), Refused> {
    let j = trajectory.jumps + 1;
    let mut stream = Stream::jump(trajectory.seed, trajectory.index, j);
    let threshold = stream.open_uniform();
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
        let Some(photon) = model.emit(trajectory.cluster, &at, &trajectory.psi, &mut draw) else {
            // A norm at its threshold with no photon to emit is refused; a bare state with none goes on as it is.
            if first {
                return Err(Refused::Emission);
            }
            break;
        };
        if first {
            trajectory.jumps = j;
            trajectory.threshold = threshold;
        }
        trajectory.work += model.photon_work();
        let u = &photon.amplitudes;
        let gram: [[f64; 3]; 3] = std::array::from_fn(|a| std::array::from_fn(|b| inner(&u[a], &u[b]).re));
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
        let scale = 1.0 / w.sqrt();
        trajectory.psi = f.iter().map(|z| *z * scale).collect();
        trajectory.cluster = photon.landing;
        first = false;
        // The landing: the state the trajectory goes on in, its norm against the threshold; at the threshold already,
        // one more photon from the bare state and the next threshold.
        let at = trajectory.point();
        let landed = model.landed(trajectory.cluster, &at, &trajectory.psi);
        if norm_sqr(&landed) > trajectory.threshold {
            trajectory.psi = landed;
            break;
        }
        trajectory.threshold = stream.open_uniform();
        (class, polarization) = (stream.uniform(), stream.uniform());
    }
    Ok(())
}

/// The loss test at a trajectory's state (SPEC §1.7: at every step's end, after every jump and at every time read),
/// the one definition: its time once the model finds it lost (`Dynamics::lost`).
pub fn test_loss<D: Dynamics>(model: &D, trajectory: &mut Trajectory) {
    if trajectory.lost.is_none() && model.lost(trajectory.cluster, &trajectory.point(), &trajectory.psi) {
        trajectory.lost = Some(trajectory.time);
    }
}

#[cfg(test)]
mod tests;
