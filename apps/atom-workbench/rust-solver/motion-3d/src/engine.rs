//! Trajectories: a classical point (r, ħκ) and the internal state ψ on one cluster (ground.rs),
//! unnormalized between quantum jumps (the waiting-time unraveling of the cluster's generator).
//!
//! # One step (flow–kick–flow)
//! The trap's harmonic part along each of its principal axes, ½ m ω_i² x_i² (`Settings::axes`,
//! `Settings::frequencies`), is moved exactly: (r, κ) turns in phase space for dt/2 to (r_h, κ_h);
//! there the cluster's H_eff at the step's middle time gives U = exp(−i H_eff dt/2) (`HalfStep`); ψ_h = U ψ;
//! κ_h += dt (F − F_harmonic) with F the Ehrenfest force −∇Re⟨H_eff⟩ (the trap's full force and
//! the light's) and F_harmonic its harmonic part; ψ ← U ψ_h; and the harmonic flow for dt/2 again.
//! Only the anharmonic rest and the light are split, so the step is not bound to the trap period.
//! The scalar part of H_herm (the trap's depth) is a phase and left out of U.
//!
//! # Beyond the trap's region
//! A splitting about an integrable reference is accurate while the rest is a small perturbation of it
//! (Wisdom–Holman's mixed-variable splittings); where the system leaves the reference's domain the
//! field's practice is a smooth changeover of what the reference takes (Chambers' hybrid integrator,
//! its changeover function; r-RESPA's switching functions), not a hard switch. The harmonic part about
//! the trap's centre is no model of the trap past its region (U_h > D, `Settings::region`): there it grows as
//! ζ² while the trap's potential saturates at its depth, and the kick cancels its rotation -- an atom
//! barely bound far out on a shallow trap's axis took an energy error (ω dt)² U_h, many times its
//! binding (SHALLOW_PROBE's tweezer alone, light off: 4 of 1024 such atoms lost over 2 ms at 0.25 us,
//! none at any step from 0.5 us down with the weight below), and a retro-reflected tweezer's
//! neighbouring sites, ten times the depth up the central site's parabola, moved their κ by 50-190 %
//! between 0.2 and 0.05 us (2-14 % with it). So each step's flow takes the harmonic part weighted by
//! w = 1 within the region and (2u − 1)/u² beyond it, u = U_h/D, at the step's start (`reference`;
//! continuous with its slope, its potential below 2D), the kick the rest: a step of any w is a
//! second-order splitting of the same Hamiltonian, so the weight moves the step's error, not what it
//! integrates, and an atom within the region (every one of a trap many k_BT deep) takes the step as
//! before, bit for bit.
//!
//! # A standing wave's sites
//! A retro-reflected tweezer's neighbouring sites are as stiff as the central one, and the region's weight leaves
//! their stiffness to the kick: the step resolves ω dt there, while the harmonic part's rotation is exact at the
//! centre. So where the trap is a standing wave (model.rs `Lattice`) the harmonic part is the atom's own site's at the
//! step's start (`reference`): about the nearest antinode z_n on the axis, its stiffness the site's, A(z_n) = 1/(1 +
//! z_n²/z_R²) over the trap's centre's (`Lattice::share`), u = U_h⊥/(A D) + tan²Φ -- Φ the distance from the site in the standing wave's phase,
//! tan²Φ the atom's level above its site's bottom over its depth below the far level, (kx)² near the bottom, the
//! across part as before --, so that w falls to zero at the node between two sites: the reference is continuous
//! through it, with no switch, though its centre changes there. An atom within its site's region takes the same
//! symmetric step about its site as the central site's atoms about the focus.
//!
//! # The exact step (explicit excited states, beats)
//! An explicit excited state rotates at its detuning in the frame, and its saturation, Rabi
//! oscillation and decay set the internal state's pace; a beating generator (two colours on one line,
//! a term whose residual its frame keeps: model.rs `Components`) changes within the step. Holding
//! either at the step's middle ties the motion's step to the internal state's pace, so the step takes
//! the internal state exactly instead -- multirate: the motion's step keeps its own size. The flow for
//! dt/2 to (r_h, κ_h); there the cluster takes its generator's components once
//! (`Segment`), H(t) = H₀ + Σ_j e^{iΩ_j t} H_j -- the tables and the envelopes held at r_h, the beams' phases advancing along the straight path r_h + v(t − t_m), each beat's frequency taking its q·v (model.rs, # The atom's path) --, and ψ is propagated through the step on a grid of spacing δ
//! from its start, to the root of ‖ψ(t)‖² = θ (safeguarded Newton on ln‖ψ‖², whose derivative is
//! −⟨ψ|i(H − H†)|ψ⟩/‖ψ‖²) where it has one. A jump's step is then redone to that root, as a cluster of ground
//! states' is (`Engine::step`): flow, evolution and kick of that length, each on its own path, so that the state and the photons' folded amplitudes hold the beams' phases of the place they reach; the jump at that place, the rest of
//! the step a step of its own from the state the jump left, in the cluster it landed in. Held through the jump, the two
//! states' forces act at the one r_h, against r_h + v (t* − τ)/2 and r_h + v t*/2: an error of the work, per jump
//! (C₁ − C₂) v² t*(τ − t*)/2 with C the states' curvatures of ⟨ψ|H(r)|ψ⟩ at their fixed ψ -- a dressed ground state's a
//! light shift's (2k)², the bare state a jump leaves none -- that heats as v², a friction of its own against the light's. The
//! 1D σ⁺σ⁻ molasses at 3 mW/cm², −6 Γ (5 ms, 2048 atoms) stood at T_x 16.8 ± 0.9 µK at a step of 1.35 us and 32 ± 2 at 2 us against 10.6 ± 0.3 at small steps, where a
//! cluster of ground states' step takes 9.2 ± 0.5 and 11.1 ± 0.8; redone, 10.0 ± 0.5 and 10.8 ± 0.6. No check of the jump-free flow sees it. On a grid:
//! - a static generator through its eigendecomposition H = R Λ R⁻¹ (eigen.rs), exact at any time,
//!   where its rounding -- R's condition times the double precision, grown by the phases over the
//!   step -- is within the exponentials' share (`Settings::truncation`); near an exceptional point by
//!   the propagator exp(−iHδ) (its Taylor series to double precision) and the series on the vector off
//!   the grid;
//! - a beating one by CF4, the fourth-order commutator-free Magnus scheme (`cf4`, exact where H is
//!   static), over each spacing, its exponentials' series on the vector, H(t) by its matrices' entries
//!   (`Entries`: a cluster's drives and a beat's terms are few of its n²). A pair of beats ±Ω whose whole effect on the step is within the
//!   exponentials' share is left out (`Segment::resolve`: Duhamel's formula integrated by parts bounds
//!   it, its boundary terms paid once over the time to the next jump, `Settings::horizon`, and its
//!   integral per step) -- a gray molasses' explicit states keep terms of 10⁻⁹ of its generator at the
//!   hyperfine splitting, a red molasses' of 10⁻⁵; the solve books them in its ledger. A cluster of ground states
//!   with beats that are all left out at the model's reach (`quiet_clusters`: a bias field's Zeeman beats of 10⁻⁵ to 10⁻³⁰ of its
//!   Raman couplings, near the generator's scale and so kept by the rule, the couplings static in its frames) is static within that share and takes the plain step:
//!   6 us per step against 26 on the 1 G molasses, whose T is the same.
//! The impulse is the conditional force's time integral along that evolution, ∫ ⟨ψ|F|ψ⟩/⟨ψ|ψ⟩ dt, by composite Simpson on the grid, applied with the
//! jumps' kicks and less dt F_harmonic(r_h) before the flow for dt/2 again. Over whole panels of h = 2δ Simpson's sum of a component A e^{iωt} of the force is exactly S times its integral,
//! S = (u/3)(2 + cos u)/sin u = 1 + u⁴/180 + …, u = ωh/2 (`simpson_excess`): it errs by S − 1 of an integral of at most min(τ, 2/|ω|) |A| -- a fast component's does not accumulate --, until the grid
//! aliases it at u = π. A static generator's propagation is exact on any grid, so that is the whole of its grid's error, and a grid that reads only its frequencies' bound ν = 2‖H‖₁ takes the
//! fewest whole panels that keep (S − 1) min(1, 2/(ντ)) within `QUADRATURE`, the worst a state can make of it (`static_panels`). Over successive steps those errors cancel as the integrals do, at one S. So a grid through the eigendecomposition, which follows the state
//! from step to step and S with it, takes no such credit: each frequency λ_a* − λ_b weighted by its amplitude in the state, (S − 1) of τ to the first order within the share (`Segment::weighted`). With min(τ, 2/|ω|) there, which lets S − 1 run up to tens near its pole, the 10 G scene's T_x
//! stood 3.6 ± 1.2 µK higher at 3-6 ms and T_y 3.7 ± 1.5 at 6-10 ms (a pair of runs), where the kernel's grid, S − 1 ≤ 0.16, moved nothing beyond its statistics. On a beating grid the same bound, at ν = 2(‖H₀‖₁ + Σ_j ‖H_j‖₁) + max|Ω_j|, is far
//! above what either error takes, the state's fast part being small beside the slow one. So a solve measures the phase of the grid that meets both shares,
//! the populations' over the waiting time and the impulse's `QUADRATURE`, by doubling the grid on its states
//! (`Engine::grid_error`, solve.rs `grid_phase`; `Settings::node_phase`, the bound's by default), CF4's own error
//! of fourth order in the beats' derivatives (validated against a
//! fine CF4, the master equation and the population core: examples/multirate.rs). Neither share sees a small part of the state that is a whole channel between ground states: a far-detuned explicit state
//! driven through a beat is kicked as the beat's phase steps from one spacing to the next, the kicks adding as the spacing nears 2π over its detuning and the beat, and staying well past it. Its population lies far below the
//! populations' largest difference and the impulse barely feels it, while its photons pump the atom out of its cluster, and what they move adds up over the run until the populations relax. (The PGC with a
//! resonant probe with F' = 2 and F' = 1 explicit, 100 us: F = 1 stands 0.023 ± 0.008 above a grid of 0.6 times the spacing, at the phase those two shares read 0.26 and 0.69 of themselves; the numerical share is 0.002.)
//! So a solve also reads what a step lands between the groups of ground states, over the run (`Engine::grid_landings`, solve.rs `landed_excess`). What is left is the
//! motion's own splitting, which the step-doubling check measures. A trajectory whose jump lands it in
//! such a cluster takes the rest of its step so.
//!
//! # Jumps
//! When ‖ψ‖² falls to the trajectory's threshold θ (uniform in (0, 1]), the crossing time t* is
//! found on ln‖ψ‖² (linear on the half step where it lies), the step redone from its start to t*
//! (flow, kick and flow of that length) and the jump made there; the rest of the step is taken from
//! the state the jump left, so the force follows the internal state on either side of the jump --
//! with the jump inside a step the force of the state before it would act on the whole step, an
//! error first order in dt that the jump-free step doubling cannot see. The jump draws the class φ
//! of photons of one frequency -- and the cluster it lands the atom in -- with probability
//! Σ_a ‖u_a‖², u_a = J_{a,φ} ψ: the folded drives' C_a R_c ψ less an explicit state's amplitude
//! C_a ψ_e (the folded amplitude A_c = −R_c with its physical sign, the jump's overall sign the
//! release's, ground.rs); the photon's direction n from the dipole pattern tr G − n·Re(G)·n,
//! G_ab = ⟨u_a|u_b⟩, exactly (rejection from the uniform sphere); its polarization ε across n from
//! ‖Σ_a ε_a u_a‖² on a real orthonormal pair; then ψ ← f = Σ_a ε_a u_a and the emission recoil
//! κ −= k_φ n.
//!
//! # Absorption
//! The light absorbed with the photon moves the atom too, by one beam's ħ k_l (bounded kicks, the beams'
//! own recoil). The photon f = Σ_a ε_a u_a is a sum of parts, each near beam's folded amplitude f_l (model.rs
//! `Photons::absorption`) and the explicit states' v. Beams that do not interfere share it by ‖f_l‖², the incoherent
//! attribution. Interference moves the mean recoil off the shares': a part's projection on the photon,
//! Re⟨f|f_l⟩/⟨f|f⟩, is its true weight in the mean -- the projections sum to one, and Σ_l Re⟨f|f_l⟩ k_l/⟨f|f⟩ is the
//! beams' phases' part of the photon's exact conditional mean b = Im⟨f|∇f⟩/⟨f|f⟩ (the jump term's first Wigner moment;
//! ∇f_l = i k_l f_l for a plane wave, an explicit state's amplitude does not depend on r) -- but a projection can be
//! negative and draws nothing. So the draw takes the shares moved the least (χ²) to that mean, nonnegative
//! (`part_weights`), and what b holds beyond them -- the envelopes, the intensity's gradient -- is a deterministic kick: the
//! jumps carry the exact mean force of the absorbed light, and per jump the beams' incoherent second moment,
//! Σ_l q_l k_l k_lᵀ: the shares' where nothing interferes, k_z² wherever every beam's recoil has the same projection (a 1D
//! molasses, a tetrahedral MOT's axis), a running wave's ħk per photon. Only a mean beyond every mixture of the beams'
//! recoils (a superoscillation by a dark point) leaves more of a deterministic kick, a second moment no mixture can
//! lower, at most the localized one below (`absorbed_recoil`). Adding b to a kick drawn by the shares, less their mean,
//! instead counted the interference's mean on top of the incoherent spread: on a tetrahedral MOT's z it gave 0.142 µK per
//! photon against 0.120 incoherent and 0.083 localized, and T_z 13 % above the explicit model's. The localized second moment,
//! ‖∇f‖²/⟨f|f⟩ (the second-order Wigner value at a point), is not taken: against the tweezer PGC's 1D quantum reference
//! (TW2, x alone) it cools 2–4 % further below it at 4–5 ħω, where the incoherent one is 4–8 % low (# What the kicks leave out).
//! An explicit state's absorbed light acts through the Ehrenfest force -- the stimulated force of its kept drives,
//! −∇Re⟨ψ|V|ψ⟩ --, which carries its mean: its part's recoil has mean zero at the jump, the recoil of a beam drawn by
//! its drives -- each explicit state's weight in the photon shared by the drives into it, Σ |a_l Σ_k c_lk U_k ψ_b|² -- less
//! their mean a, plus the mean's shot noise ±a with equal chance --
//! per photon the beams' second moment, without the mean twice. A running wave's beam axis takes R E_r (1 + ⟨n²⟩) so; with
//! the spread alone it took R E_r ⟨n²⟩. The draw's uniform picks the part, its leftover the explicit part's beam, its
//! lower bits the sign: no further deviate.
//!
//! # The transient after a jump
//! A jump leaves the atom in a bare ground state ψ′, while the fold's photons are those of the steady excited amplitude of that state, v = X ψ′ (X = −Σ e^{−iε̃t} R_c per
//! coherent group of channels, model.rs `Transient`). The amplitude builds up from zero, e(τ) = v − e^{−iKτ} v to first order in the drives, and over the time it rings the atom emits otherwise than the fold's
//! steady rate has it, by
//!
//! ```text
//! ∫ dτ Σ_a C_a (e e† − v v†) C_a† = Σ_a C_a (Z − a v† − v a†) C_a†        Z = ∫ dτ e^{−iKτ} v v† e^{iK†τ}        a = ∫ dτ e^{−iKτ} v = i R2 ψ′
//! ```
//!
//! -- on K's eigenstates, Δ_i their detunings from a channel, v_i v_i′* (1/(i(Δ_i − Δ_i′) + (Γ_i + Γ_i′)/2) − 1/(iΔ_i + Γ_i/2) − 1/(−iΔ_i′ + Γ_i′/2)), on one state (κ_i/Γ_i) |v_i|², κ_i = 1 − Γ_i²/(Δ_i² + Γ_i²/4).
//! Small as its weight is, it matters where the atoms' states leave a slow mode rarely: on the tetrahedral MOT's frozen |2,1⟩, which scatters 0.36 /us and leaves at 0.0123 /us, this photon runs at 0.0034 /us, and the
//! tweezer's tensor shift, 2.3 Γ across F′ = 3, rings its sublevels apart: it lands 66 % back in the mode with F′ = 3 explicit, 96 % if they stay coherent, the whole of the fourth order's error out of the mode. The map's parts:
//! - Z, of trace ‖v‖², is the excited amplitude a jump leaves to decay: the ringing's own photon, with probability p = Σ ‖v‖² over the coherent groups (their beats average their interference out). Its unraveling is the
//!   emission itself: a group and an excited manifold by their weights, at a time τ ~ Γ e^{−Γτ}, Γ the manifold's width, the photon's amplitude u_a = C_a e^{−iHτ} v -- each landing state at its frame at t + τ --, its landing cluster by the weights
//!   there, its kicks a jump's. The mixture over τ is Z exactly, whatever H's off-diagonal part, each pair of its eigenstates at its coherence Γ/(Γ + iδ), the landing states' frames' difference in δ.
//!   (A manifold's states' widths agree to 1e-4 of Γ, their mean, and D1 and D2 are 1e6 widths apart.) Only a photon takes H's eigenstates (`Model::ringing`, as far as a coupling moves its amplitude beyond the numerical share of the rule's budget): one jump in a hundred on that MOT.
//! - The interference, −(a v† + v a†), of trace −w, w = Σ (1 − κ_i) |v_i|² ≥ 0, is what the steady amplitude's photons lack while it builds. Where every pair's terms are equal it is w of those photons themselves: a delay of the next jump,
//!   which at fourth order the slow part's weight holds: A = ‖Πψ′‖² has A + p − 1 = w to the fold's next order, so that after no ringing photon (1 − p) the atom goes on from Πψ′/√A at the threshold θ (1 − p)/A: the slow part's whole weight is kept, and the next jump's first w never comes.
//!   What the delay leaves, the terms' spread, is ς of the photons' weight, ς the spread of Γ_i/z_i over v, of first order in S (`Model::ringing_spread`: 0.06 on the MOT, where its effect is measured at 2e-5, and of the order of one on a line the light brings to resonance): the fold's charge books S² + ς S (solve.rs `fold_charges`).
//! - At second order and with explicit states, where the slow part is the state itself, the atom stays in ψ′ after no photon, without the delay: the rates' error is of the fold's own order there (2.7 % on the MOT with or without it), and the populations move less for it (a near-resonant shallow probe's at the focus 0.0017 against 0.0026
//!   with it); the state's tilt (1 − ½Y)ψ′, Y = Σ X†X, the map's linear form, moves the MOT's |2,1⟩ by 4e-4. So a jump asks for no amplitude unless its draw falls within a bound of p from the columns' norms (`Model::ringing_bound`), about one in ten at p = 1 %; at fourth order p is the metric's form on ψ′, which Π holds.
//!
//! Against the rung with F′ = 3 explicit on the MOT, the frozen master equation's largest |Δp| at x times its near beams' intensities (x = 1, ½, ¼): fourth order 0.0150, 0.0077, 0.0040 with a photon coherent over the tensor split, 0.0004, 0.0001, 0.0001 with this one
//! (S² = 3e-4 at x = 1; H's couplings across F′, of mixing 3 %, are 3e-4 of its first order against H's diagonal alone); second order 0.0134, 0.0070, 0.0036 without a transient, 0.0029, 0.0013, 0.0005 with it.
//! The photon's draws come from the jump's stream before the directions' loops. A photon's own landing takes its slow part, the third photon's chance -- S² -- in its threshold (`slow_start`), as the bare state a solve starts from does at fourth order.
//! On the moving atom (1 ms, the second half's means against no such photon): the MOT's F = 1 8 % up, as the frozen atom's, and its T 2.2 % at second order, 1.0 % at fourth (S = 2.1 %, T's charge √2 S), 0.7 % and 0.4 % of it this photon's kicks,
//! the rest where it lands; the 3D PGC's F = 1 4 % up and its T within 0.3 % (a recorded rung with F′ = 3 explicit: F = 1 3-6 % up, T within 0.4 %). The MOT's photons per atom now hold the core's at fourth order (0.8 % high before), its T the rung's within 0.8 %.
//!
//! # What the kicks leave out
//! The Ehrenfest force averages the dipole force over a superposition of ground states, and a kick
//! drawn per beam is not correlated with the state the jump leaves: at a fixed point the momentum
//! diffusion is a part of the second-order Wigner expansion's (0.23-0.27 of it at a lin-perp-lin
//! focus). The expansion's own trajectory scheme -- a drift tr A(rho), a momentum-internal
//! correlation M and Gaussian noise of rate 2 tr A(M) + B -- is exact on a prescribed path but not
//! for an atom moving freely in the trap, where M shears into a position correlation that acts back
//! through the force's gradient (part of the Sisyphus friction) and the variance it prescribes
//! falls as well as rises: it heats the tweezer PGC far above the GMW reference
//! (claude/motion3d-m-scheme). The trapped T of this engine stays within a few percent of the 1D
//! GMW references in the validated regime (solve.rs).
//!
//! # First order in the Doppler shift
//! The folded responses take the kinetic shift of absorption to first order, R − w R2, in the generator -- through the second order's decay tables at a transfer of
//! their own, what that leaves bounded on T -- as in the jumps' photons (model.rs, # The photons and the Doppler term: the mean force on a state then the adiabatic expansion's to that
//! order), good to (kv/σ)², the largest (w_l/σ_c)² over the near beams, σ_c the nearest excited
//! pole of beam l's carrier (model.rs `Model::doppler_ratio`). A step whose (kv/σ)² exceeds the
//! solve's allowance (`Settings::doppler`) takes the exact response at its jumps (model.rs `Exact`), its hazard still the
//! generator's first order, its error of that (kv/σ)² on the few atoms that hot;
//! the atoms' mean (kv/σ)² is the ledger's. An explicit drive needs neither: its beams' phases advance along the atom's path within the step (model.rs, # The atom's path), so that the state sees them at their Doppler-shifted
//! frequencies at any step: an atom moving through a running wave is an atom at rest at the detuning less k·v (tests/engine.rs). No step check sees what their absence does: the phases' steps between steps stand for the Doppler shift only where the step
//! resolves the internal state's memory, and a jump erases that memory. A weak probe's T_z at 100 us (a tweezer and 0.3 mW/cm² on z, 4096 atoms): 78.4, 82.9, 91.3, 94.1, 94.3 uK at 1.67, 0.71, 0.18, 0.045, 0.011 us held, where the step's doubling read 2.8 %, and 93.2, 93.1,
//! 93.9, 94.3 on the path; detuned by +6 MHz, the lost friction over-heating, 42.3, 38.9, 38.2 against 38.3, 38.2, 38.1.
//!
//! # Loss
//! An atom is lost when it is no longer bound to the trap beam and has left the trap's region: its
//! mechanical energy, ħ²κ²/2m plus its potential (model.rs `Model::trap_potential`: Re⟨ψ|H_eff|ψ⟩ of its
//! current internal state less the same with the trap beam off, so U → 0 far from it), is positive, and
//! the trap's harmonic part there lies above the region's depth (`Settings::region`, a solve's the
//! reference state's depth). An unbound atom inside the region is held until it leaves -- it may be
//! bound again before it does, as an atom that a photon's
//! recoil or a bright state's light shift lifts above the trap's top for a moment is: judged at every
//! step's end alone, those moments were caught or missed as the step fell, and the loss converged with
//! the step only as fast as it resolved them. The energy is of one state at one time: of a step without
//! a jump, its middle -- the point the flow's first half reached, κ there with half the kick, ψ after
//! the first half step --, else its end. (The middle's point with the end's κ errs by ∇U·v dt/2, first
//! order: on a shallow trap's dark atoms it lost 2 % more of them at a 2 μs step.)
//!
//! # Randomness
//! Every jump draws from its own counter-based stream (rng.rs: the run's seed, the trajectory, the
//! jump's number), so runs that differ only in the time step make the same choices: the
//! step-doubling check compares them. A jump draws its class, the next threshold, the polarization,
//! the absorbed beam and the transient's five first and the
//! directions' rejection loops last, so that
//! a trajectory's twin on the same streams (control.rs) keeps its thresholds and choices whatever its
//! loops take.
//!
//! # A held atom
//! A held twin (control.rs) does not move and its generator is static: its engine (`Engine::held`)
//! takes every cluster's generator and jump operators at its point once, steps by the
//! half step's propagators, and draws a jump's direction and polarization without the kicks.
use crate::eigen::{decompose,Eigen};
use crate::linear::zeroed;
use crate::model::{Absorption,Components,Exact,Local,Model,Point,MOST_MIXED,MOST_STATES};
use crate::rng::{Purpose,Rng};
use crate::Complex;

/// The impulse's quadrature share in the exact step (module documentation, # The exact step): composite Simpson's error within it of the integrand's size times the step.
pub const QUADRATURE:f64=crate::solve::NUMERICAL_SHARE*crate::solve::TEMPERATURE_TARGET;
/// A beating grid's phase by default, its spacing times half its force's frequencies' bound (`Settings::node_phase`): (ν·2δ)⁴/2880, `simpson_excess` to its first order, at `QUADRATURE`.
pub(crate) fn node_phase()->f64{(2880.*QUADRATURE).powf(0.25)/4.}
/// S − 1, S the ratio of composite Simpson's sum of a component e^{iωt} over whole panels of h to its integral, u = ωh/2: u⁴/180 = (ωh)⁴/2880 to the first order, all its orders' coefficients positive (so that at a
/// complex ω, a decaying component, |ω| bounds it), its pole at u = π the grid aliasing the component. At u = 0 it reads as no excess: NaN exceeds nothing.
fn simpson_excess(u:f64)->f64{u*(2.+u.cos())/(3.*u.sin())-1.}
/// The fewest whole panels over τ of a static generator's grid, x = ντ/2, ν its frequencies' bound (module documentation, # The exact step): (S − 1) min(1, 2/(ντ)) within `QUADRATURE` at u = νh/2 = x/panels. And no spacing wider than
/// one part of the exponentials' series, x/(2 panels) = ‖H‖₁δ ≤ 1 at ν = 2‖H‖₁ (`series_within`'s widest at a share at the round-off, the kernel's): the series takes its fewest products per unit of time there, and past it, in parts, more
/// than the forces it saves repay (the 10 G scene's kernel per trajectory, against (ν·2δ)⁴/2880 of τ: −49 % at 1.7 times that phase, −55 % here, 2.17 times it, −33 % at 2.7), which keeps u ≤ 2, clear of the pole.
pub(crate) fn static_panels(x:f64)->usize{
    let room=QUADRATURE*x.max(1.);let mut panels=((0.5*x).ceil() as usize).max(1);
    while simpson_excess(x/panels as f64)>room{panels+=1;}
    panels
}

/// A trajectory's state.
#[derive(Clone,Debug)]
pub struct Trajectory{
    pub cluster:usize,
    pub psi:[Complex;MOST_MIXED],
    pub r:[f64;3],
    /// The momentum over ħ (1/um).
    pub kappa:[f64;3],
    /// The time (us).
    pub time:f64,
    pub threshold:f64,
    pub alive:bool,
    pub jumps:u64,
    /// The largest bound per unit time (1/us) of the beats its exact steps left out (`Segment::resolve`):
    /// over the waiting time, its share of the ledger.
    pub beats:f64,
    /// The run's seed and the trajectory's number: its jumps' streams.
    pub seed:u64,
    pub index:u64,
}

/// What the engine needs besides the model.
#[derive(Clone,Debug)]
pub struct Settings{
    pub dt:f64,
    /// The lab axes the atom moves along (the others stay at 0).
    pub free:[bool;3],
    /// The trap's harmonic part, moved exactly: its principal axes (unit rows) and frequencies
    /// ω_i (rad/us), when the atom moves along every axis; along the free lab axes alone, the
    /// frequencies per lab axis.
    pub axes:[[f64;3];3],
    pub frequencies:[f64;3],
    pub lab:[f64;3],
    /// The bound on each exponential's truncated Taylor remainder relative to ‖ψ‖ (`propagate`):
    /// a share of the step's own error budget in a solve, f64::EPSILON for double precision.
    pub truncation:f64,
    /// The motion's sums (T, ⟨v⟩, n̄) averaged over one trap period of each axis centred on
    /// each output time (lib.rs `simulate`, a solve's), or taken at the output time.
    pub average:bool,
    /// The loss by the mechanical energy (a solve's); off where a validation holds the atom.
    pub loss:bool,
    /// The largest (kv/σ_min)² a jump keeps the first order in the Doppler shift to (# First order in the Doppler shift).
    pub doppler:f64,
    /// The share of ‖ψ‖ per step within which the exact step leaves out a generator's beats
    /// (`Segment::resolve`): the exponentials' share in a solve, also in its pilot, whose exponentials
    /// are summed to double precision; `truncation` by default.
    pub beats:f64,
    /// The time (us) over which a trajectory's per-step errors accumulate until a jump resets them: the
    /// solve's waiting time, its pilots' run; a beat left out costs its boundary terms once over it
    /// (`Segment::resolve`). The step by default.
    pub horizon:f64,
    /// The trap's region the loss takes (`Engine::step`): an unbound atom is lost once outside it, where
    /// the trap's harmonic part (`axes`, `frequencies`) lies above this depth (rad/us) -- a solve's is the
    /// reference state's, U_h ≤ D (model.rs `Model::depth`). 0, the default, is none: an atom
    /// is lost as soon as it is unbound.
    pub region:f64,
    /// The phase of a beating segment's grid, its spacing times the bound of half its force's frequencies
    /// (`Segment::build`): by default `node_phase`, the quadrature's bound; a solve's as its errors ask
    /// (solve.rs `grid_phase`).
    pub node_phase:f64,
    /// The trap's standing wave (model.rs `Lattice`): the harmonic part is the atom's own site's (`Engine::reference`).
    pub lattice:Option<crate::model::Lattice>,
    /// The trap's centre (model.rs `Trap`): where the harmonic part and its region stand when every lab axis is free (`about`).
    pub centre:[f64;3],
}
impl Settings{
    /// The settings of a trap (model.rs `Trap`), every axis free, the loss on, the first order in
    /// the Doppler shift everywhere.
    pub fn new(dt:f64,trap:&crate::model::Trap,truncation:f64,average:bool)->Settings{
        Settings{dt,free:[true;3],axes:trap.axes,frequencies:trap.frequencies,lab:trap.lab,truncation,average,loss:true,doppler:f64::INFINITY,beats:truncation,horizon:dt,region:0.,node_phase:node_phase(),lattice:trap.lattice,centre:trap.centre}
    }
    /// Where the harmonic part stands: the trap's centre when every lab axis is free, else the focus, the free axes' origin.
    pub fn about(&self)->[f64;3]{if self.free==[true;3]{self.centre}else{[0.;3]}}
    /// r less its own site's place where the harmonic part is the atom's own site's (`Engine::reference`), else less the trap's centre: the start of
    /// the harmonic control's twin (control.rs), which turns about that site as the atom does.
    pub fn about_site(&self,r:[f64;3])->[f64;3]{
        match self.lattice{
            Some(l)if self.free==[true;3]&&self.region>0.=>{let z=l.site(l.nearest(dot(l.axis,r)).0);std::array::from_fn(|a|r[a]-z*l.axis[a])}
            _=>{let c=self.about();std::array::from_fn(|a|r[a]-c[a])}
        }
    }
}

/// The harmonic flow along one axis for τ: (x, κ) turned by ωτ in phase space (x in um, κ in 1/um,
/// hm = ħ/m); free flight when ω is 0.
fn flow(x:f64,kappa:f64,omega:f64,hm:f64,tau:f64)->(f64,f64){
    if omega==0.{return(x+hm*kappa*tau,kappa);}
    let(s,c)=(omega*tau).sin_cos();(x*c+hm*kappa/omega*s,kappa*c-omega/hm*x*s)
}
fn dot(a:[f64;3],b:[f64;3])->f64{a[0]*b[0]+a[1]*b[1]+a[2]*b[2]}

fn norm2(psi:&[Complex],n:usize)->f64{psi[..n].iter().map(|z|z.norm_sqr()).sum()}

/// ψ ← exp(−i H τ) ψ for an n × n matrix H (row-major in an 8 × 8 array): Taylor series on the vector
/// in sub-steps of ‖H‖τ ≤ `SUB_STEP` (‖H‖ the largest column sum of |re| + |im|, above the 1-norm), each
/// summed through the term whose bound (‖H‖τ)^m/m! falls below its share of `truncation` (relative
/// to ‖ψ‖; the remainder past it is at most half that bound).
pub fn propagate(h:&[Complex;MOST_STATES*MOST_STATES],n:usize,tau:f64,psi:&mut [Complex],truncation:f64){
    macro_rules! sized{($($n:literal)*)=>{match n{$($n=>propagate_sized::<$n>(h,tau,psi,truncation),)*_=>unreachable!("a cluster of ground states has 1 to 8 states")}}}
    sized!(1 2 3 4 5 6 7 8)
}
/// The largest column sum of |re| + |im| of an n × n matrix (row-major): the bound of its 1-norm the
/// series takes.
pub fn column_norm(h:&[Complex],n:usize)->f64{
    let mut norm=0_f64;for b in 0..n{let mut column=0.;for a in 0..n{column+=h[a*n+b].re.abs()+h[a*n+b].im.abs();}norm=norm.max(column);}
    norm
}
/// A sub-step's ‖H‖τ (`propagate`): a series' terms per unit of ‖H‖τ fall with it (at a truncation of 1e-5, 13 at ½, 6.5 at 2,
/// 4 at 6), the largest that still sums to double precision within its 40 terms, its largest term's round-off 65 ε.
const SUB_STEP:f64=6.;
/// The series of exp(−iHτ) for a generator of that norm (`propagate`): its sub-steps of ‖H‖τ ≤ `SUB_STEP` and
/// each one's terms, through the first whose bound y^m/m! is within the sub-step's share of the
/// truncation bound.
pub fn series(norm:f64,tau:f64,truncation:f64)->(usize,usize){
    let x=norm*tau.abs();let parts=((x/SUB_STEP).ceil() as usize).max(1);
    (parts,terms_within(x/parts as f64,truncation/parts as f64,40).0)
}
/// The terms that sum exp(y)'s series within `share`, through the first whose bound y^m/m! is within it, at most `most`, and that bound.
fn terms_within(y:f64,share:f64,most:usize)->(usize,f64){
    let mut terms=1;let mut size=y;while size>share&&terms<most{terms+=1;size*=y/terms as f64;}
    (terms,size)
}
fn propagate_sized<const N:usize>(h:&[Complex;MOST_STATES*MOST_STATES],tau:f64,psi:&mut [Complex],truncation:f64){
    let(parts,terms)=series(column_norm(h,N),tau,truncation);let step=tau/parts as f64;
    let mut v=[Complex::ZERO;N];v.copy_from_slice(&psi[..N]);
    for _ in 0..parts{
        let mut term=v;let mut sum=v;
        for k in 1..=terms{
            let scale=step/k as f64;let mut next=[Complex::ZERO;N];
            for a in 0..N{let mut s=Complex::ZERO;for b in 0..N{s+=h[a*N+b]*term[b];}next[a]=Complex::new(s.im*scale,-s.re*scale);}
            term=next;for a in 0..N{sum[a]+=term[a];}
        }
        v=sum;
    }
    psi[..N].copy_from_slice(&v);
}

/// The fewest n × n products by which scaling and squaring sums exp(−iX), ‖X‖ ≤ x, within `share`: s squarings of the Taylor polynomial of X/2^s, its terms through the first whose
/// bound y^m/m! is within its 2^−s of the share, the part no lower than the double precision's round-off. None where no such part sums it.
fn squarings(x:f64,share:f64)->Option<(i32,usize)>{
    let mut best:Option<(i32,usize)>=None;
    for s in 0..64{
        let scale=0.5_f64.powi(s);let(y,part)=(x*scale,share*scale);if part<=f64::EPSILON{break;}
        let(terms,size)=terms_within(y,part,40);
        if size>part{continue;}
        match best{Some((b,m))if m-1+b as usize<=terms-1+s as usize=>break,_=>best=Some((s,terms))}
    }
    best
}
/// exp(−iHτ) of an n × n generator (row-major in an 8 × 8 array), by `squarings`'s s and terms.
fn scaled_exponential<const N:usize>(h:&[Complex;MOST_STATES*MOST_STATES],tau:f64,(s,terms):(i32,usize))->[Complex;MOST_STATES*MOST_STATES]{
    let times=|a:&[Complex;MOST_STATES*MOST_STATES],b:&[Complex;MOST_STATES*MOST_STATES]|{let mut out=[Complex::ZERO;MOST_STATES*MOST_STATES];
        for i in 0..N{for k in 0..N{let y=a[i*N+k];for j in 0..N{out[i*N+j]+=y*b[k*N+j];}}}out};
    let step=tau*0.5_f64.powi(s);let mut x=[Complex::ZERO;MOST_STATES*MOST_STATES];for k in 0..N*N{x[k]=Complex::new(h[k].im*step,-h[k].re*step);}
    let(mut term,mut sum)=(x,x);for a in 0..N{sum[a*N+a]+=Complex::ONE;}
    for k in 2..=terms{term=times(&term,&x);let scale=1./k as f64;for i in 0..N*N{term[i]=term[i]*scale;sum[i]+=term[i];}}
    for _ in 0..s{sum=times(&sum,&sum);}
    sum
}
/// A step's half steps on one generator (`Engine::advance`), exp(−iHτ) twice: its series on the vector (`propagate`), or, where that takes more than one part and more products
/// (n² each, against n³ of `squarings`, and two by U) -- a generator of ‖H‖τ ≫ 1, a cluster's states' frames apart: THREE's half of its CPU solve, 200 products a half step --, U once.
enum HalfStep{Series,Matrix(Box<[Complex;MOST_STATES*MOST_STATES]>)}
impl HalfStep{
    fn new(h:&[Complex;MOST_STATES*MOST_STATES],n:usize,tau:f64,truncation:f64)->HalfStep{
        let norm=column_norm(h,n);let(parts,terms)=series(norm,tau,truncation);if parts==1{return HalfStep::Series;}
        let Some((s,m))=squarings(norm*tau.abs(),truncation)else{return HalfStep::Series};
        if (m-1+s as usize)*n+2>=2*parts*terms{return HalfStep::Series;}
        macro_rules! sized{($($n:literal)*)=>{match n{$($n=>Box::new(scaled_exponential::<$n>(h,tau,(s,m))),)*_=>unreachable!("a cluster of ground states has 1 to 8 states")}}}
        HalfStep::Matrix(sized!(1 2 3 4 5 6 7 8))
    }
    fn apply(&self,h:&[Complex;MOST_STATES*MOST_STATES],n:usize,tau:f64,psi:&mut [Complex],truncation:f64){
        match self{HalfStep::Series=>propagate(h,n,tau,psi,truncation),HalfStep::Matrix(u)=>multiply(&u[..],n,psi)}
    }
}

/// ‖H‖∞ bounded above: the largest row sum (with `column_norm` above the spectral norm, √(‖H‖₁‖H‖∞)).
fn row_norm(h:&[Complex],n:usize)->f64{(0..n).map(|a|(0..n).map(|b|h[a*n+b].re.abs()+h[a*n+b].im.abs()).sum::<f64>()).fold(0.,f64::max)}
/// ψ ← exp(−i H τ) ψ on any number of states, to double precision: `propagate`'s series.
pub fn propagate_dyn(h:&[Complex],n:usize,tau:f64,psi:&mut [Complex]){
    let(parts,terms)=series(column_norm(h,n),tau,f64::EPSILON);let step=tau/parts as f64;
    let(mut term,mut next)=([Complex::ZERO;MOST_MIXED],[Complex::ZERO;MOST_MIXED]);
    for _ in 0..parts{
        term[..n].copy_from_slice(&psi[..n]);
        for k in 1..=terms{
            let scale=step/k as f64;
            for a in 0..n{let mut s=Complex::ZERO;for b in 0..n{s+=h[a*n+b]*term[b];}next[a]=Complex::new(s.im*scale,-s.re*scale);}
            term[..n].copy_from_slice(&next[..n]);for a in 0..n{psi[a]+=term[a];}
        }
    }
}
/// exp(−i H τ) of an n × n matrix with ‖H‖₁τ ≤ 1 (row-major): its Taylor series to double precision.
fn exponential(h:&[Complex],n:usize,tau:f64)->Vec<Complex>{
    let x:Vec<Complex>=h.iter().map(|z|Complex::new(z.im*tau,-z.re*tau)).collect();
    let mut sum=vec![Complex::ZERO;n*n];for a in 0..n{sum[a*n+a]=Complex::ONE;}let mut term=sum.clone();let mut next=vec![Complex::ZERO;n*n];
    let size=column_norm(h,n)*tau.abs();let mut bound=1.;
    for k in 1..40{
        for a in 0..n{for b in 0..n{let mut s=Complex::ZERO;for c in 0..n{s+=term[a*n+c]*x[c*n+b];}next[a*n+b]=s*(1./k as f64);}}
        std::mem::swap(&mut term,&mut next);for(s,t)in sum.iter_mut().zip(&term){*s+=*t;}
        bound*=size/k as f64;if bound<=f64::EPSILON{break;}
    }
    sum
}
fn multiply(u:&[Complex],n:usize,psi:&mut [Complex]){
    let mut v=[Complex::ZERO;MOST_MIXED];v[..n].copy_from_slice(&psi[..n]);
    for a in 0..n{let mut s=Complex::ZERO;for b in 0..n{s+=u[a*n+b]*v[b];}psi[a]=s;}
}

/// CF4, the fourth-order commutator-free Magnus scheme of two exponentials (Blanes & Moan): over h from t,
/// ψ ← exp(−ih(w₂H₁ + w₁H₂)) exp(−ih(w₁H₁ + w₂H₂)) ψ, H_i = H(t + c_i h) at the Gauss nodes c = ½ ∓ √3/6,
/// w = ¼ ± √3/6 -- exact where H is static (w₁ + w₂ = ½).
fn cf4()->([f64;2],[f64;2]){let r=3f64.sqrt()/6.;([0.5-r,0.5+r],[0.25+r,0.25-r])}
/// e^{−izs}.
fn rotation(z:Complex,s:f64)->Complex{Complex::cis(-z.re*s)*(z.im*s).exp()}
/// out = a b of n × n matrices (row-major), a's zero entries skipped (a beat's matrix is sparse).
fn product(a:&[Complex],b:&[Complex],n:usize,out:&mut Vec<Complex>){
    zeroed(out,n*n);for i in 0..n{for k in 0..n{let x=a[i*n+k];if x==Complex::ZERO{continue;}for j in 0..n{out[i*n+j]+=x*b[k*n+j];}}}
}
/// out = a b with b's zero entries skipped.
fn product_right(a:&[Complex],b:&[Complex],n:usize,out:&mut Vec<Complex>){
    zeroed(out,n*n);for k in 0..n{for j in 0..n{let y=b[k*n+j];if y==Complex::ZERO{continue;}for i in 0..n{out[i*n+j]+=a[i*n+k]*y;}}}
}

/// A segment's state between its grid's nodes: ψ, or a spectral segment's coordinates R⁻¹ψ.
type State=[Complex;MOST_MIXED];

/// One cluster's internal evolution in an exact step (module documentation): its generator's components
/// on the atom's path, the step's start (the beats' phases are absolute), the grid's spacing δ -- its nodes
/// at start + iδ --, how it propagates, and the bound of the beats it left out.
struct Segment{components:Components,start:f64,spacing:f64,how:How,dropped:f64}
/// How a segment propagates over its grid.
enum How{
    /// A static generator through its eigendecomposition.
    Spectral(Spectral),
    /// A static generator whose eigenvectors are too ill conditioned (near an exceptional point): the
    /// propagator over a spacing and the decay operator i(H − H†).
    Taylor{propagator:Vec<Complex>,decay:Vec<Complex>},
    /// A beating generator: CF4 over each spacing, on its matrices' entries.
    Beating(Entries),
}
thread_local!{
    /// A thread's last segment's buffers, for its next (as model.rs `LOCAL_BUFFERS`, `COMPONENT_BUFFERS`): the exact step takes a segment per step and allocates nothing for it. (On a beating
    /// cluster of 30 pairs of beats the system heap took a third of a CPU solve.) A second segment alive at once (`Engine::grid_error`) starts empty, and the last one back stays.
    static SPECTRAL:std::cell::Cell<Spectral>=std::cell::Cell::default();
    static ENTRIES:std::cell::Cell<Entries>=std::cell::Cell::default();
    static BOUNDS:std::cell::Cell<Bounds>=std::cell::Cell::default();
}
impl Drop for Segment{
    fn drop(&mut self){
        match &mut self.how{
            How::Spectral(s)=>{let s=std::mem::take(s);let _=SPECTRAL.try_with(|c|c.set(s));}
            How::Beating(e)=>{let e=std::mem::take(e);let _=ENTRIES.try_with(|c|c.set(e));}
            How::Taylor{..}=>{}
        }
    }
}
/// A static generator through its eigendecomposition H = R Λ R⁻¹ (eigen.rs), in the coordinates
/// d = R⁻¹ψ: e^{−iλδ} per spacing; the norm's, the force's and the norm's rate's forms in them,
/// R†R, R†G_aR and i(λ_a* − λ_b)(R†R)_ab; per force form and the norm's, |X_ab| and |X_ab| |λ_a* − λ_b|⁴
/// (`Segment::weighted`). The identity and a product, its forms' working room.
#[derive(Default)]
struct Spectral{eigen:Eigen,gram:Vec<Complex>,forces:[Vec<Complex>;3],rate:Vec<Complex>,moments:[[Vec<f64>;2];4],identity:Vec<Complex>,product:Vec<Complex>}
/// What `Segment::bounds` finds per pair of beats -- its index, its costs on the state and on the force's integral --, those `Segment::resolve` leaves out, and the commutators' working room.
#[derive(Default)]
struct Bounds{pairs:Vec<(usize,f64,f64)>,dropped:Vec<bool>,left:Vec<Complex>,right:Vec<Complex>,first:Vec<Complex>,second:Vec<Complex>}
/// out = h x − x h (`product_right`, `product`: x's zero entries skipped), `left` and `right` its working room.
fn commutator(h:&[Complex],x:&[Complex],n:usize,left:&mut Vec<Complex>,right:&mut Vec<Complex>,out:&mut Vec<Complex>){
    product_right(h,x,n,left);product(x,h,n,right);out.clear();out.extend(left.iter().zip(right.iter()).map(|(p,q)|*p-*q));
}
/// A beating generator by its matrices' entries (`Entries::new`), flat. Per part -- 0 the static one, 1 + j the kept beat j -- its
/// entries (row, column, value; round-off left out; `parts` where each begins) and its size, above ‖·‖₁. The force's forms: the static gradients' upper
/// triangle (row, column, per axis the entry, doubled off the diagonal: Re⟨x|G|x⟩ of a Hermitian G) and each positive beat's entries (`beats` where each begins, the last where they end).
/// The share of ‖ψ‖ each exponential's series is summed to; whether the processor has AVX2 and FMA; the static part of an exponential's matrix, (w₁ + w₂) H₀ as `Working` holds it; the working matrix (`Working`).
#[derive(Default)]
struct Entries{n:usize,entries:Vec<(usize,usize,Complex)>,parts:Vec<usize>,sizes:Vec<f64>,forms:Vec<(usize,usize,[Complex;3])>,beats:Vec<usize>,share:f64,wide:bool,fixed:[Vec<f64>;2],working:std::cell::RefCell<Working>}
/// An exponential's matrix w₁H(t₁) + w₂H(t₂), column-major, its real and imaginary parts apart (the product by a vector goes
/// column by column, in the processor's vectors), and each pair of beats' phases at t₁ and t₂.
#[derive(Default)]
struct Working{re:Vec<f64>,im:Vec<f64>,phases:Vec<[Complex;2]>}
/// The series of exp(−i B τ) on a vector, x its real and imaginary parts: `parts` × `terms` products by B (`Working`).
#[inline(always)]
fn series_body(re:&[f64],im:&[f64],n:usize,parts:usize,terms:usize,tau:f64,xr:&mut [f64;MOST_MIXED],xi:&mut [f64;MOST_MIXED]){
    let step=tau/parts as f64;
    for _ in 0..parts{
        let(mut tr,mut ti)=(*xr,*xi);
        for k in 1..=terms{
            let scale=step/k as f64;let(mut yr,mut yi)=([0.;MOST_MIXED],[0.;MOST_MIXED]);
            for((hr,hi),(&cr,&ci))in re.chunks_exact(n).zip(im.chunks_exact(n)).zip(tr[..n].iter().zip(&ti[..n])){
                for(((ar,ai),&r),&i)in yr[..n].iter_mut().zip(&mut yi[..n]).zip(hr).zip(hi){*ar+=r*cr-i*ci;*ai+=r*ci+i*cr;}
            }
            for(((((t,u),x),z),&a),&b)in tr[..n].iter_mut().zip(&mut ti[..n]).zip(&mut xr[..n]).zip(&mut xi[..n]).zip(&yr[..n]).zip(&yi[..n]){*t=b*scale;*u=-a*scale;*x+=*t;*z+=*u;}
        }
    }
}
#[target_feature(enable="avx2,fma")]
unsafe fn series_wide(re:&[f64],im:&[f64],n:usize,parts:usize,terms:usize,tau:f64,xr:&mut [f64;MOST_MIXED],xi:&mut [f64;MOST_MIXED]){series_body(re,im,n,parts,terms,tau,xr,xi)}
impl Entries{
    /// A beating generator's entries, each exponential's series within `share` of ‖ψ‖ (double precision at least).
    fn new(c:&Components,share:f64)->Entries{
        let n=c.n();let half=c.omegas().len()/2;let mut e=ENTRIES.take();
        e.n=n;e.share=share.max(f64::EPSILON);e.wide=std::arch::is_x86_feature_detected!("avx2")&&std::arch::is_x86_feature_detected!("fma");
        let Entries{entries,parts,sizes,forms,beats,fixed,working,..}=&mut e;
        sizes.clear();sizes.push(column_norm(c.hamiltonian(),n));sizes.extend(c.beats().map(|b|column_norm(b.hamiltonian,n)));
        // Entries within the generator's round-off are none (a cluster keeps drives of 1e-30, static in its frames).
        let floor=(f64::EPSILON*sizes.iter().sum::<f64>()).powi(2);
        entries.clear();parts.clear();parts.push(0);
        for h in std::iter::once(c.hamiltonian()).chain(c.beats().map(|b|b.hamiltonian)){entries.extend(h.iter().enumerate().filter(|(_,z)|z.norm_sqr()>floor).map(|(k,&z)|(k/n,k%n,z)));parts.push(entries.len());}
        let scale={let(_,w)=cf4();w[0]+w[1]};for x in fixed.iter_mut(){zeroed(x,n*n);}for &(a,b,z)in &entries[..parts[1]]{let k=b*n+a;fixed[0][k]=z.re*scale;fixed[1][k]=z.im*scale;}
        let statics=[c.gradient(0),c.gradient(1),c.gradient(2)];
        let gradients=std::iter::once(statics).chain((half..c.omegas().len()).map(|j|c.beat(j).gradient));
        let top=gradients.clone().flat_map(|g|g.into_iter().flatten()).fold(0_f64,|m,z|m.max(z.norm_sqr()))*f64::EPSILON*f64::EPSILON;
        let form=|g:&[&[Complex];3],a:usize,b:usize,scale:f64|->Option<(usize,usize,[Complex;3])>{let k=a*n+b;g.iter().any(|x|x[k].norm_sqr()>top).then(||(a,b,[g[0][k]*scale,g[1][k]*scale,g[2][k]*scale]))};
        forms.clear();beats.clear();
        forms.extend((0..n).flat_map(|a|(a..n).map(move|b|(a,b))).filter_map(|(a,b)|form(&statics,a,b,if a==b{1.}else{2.})));beats.push(forms.len());
        for g in gradients.skip(1){forms.extend((0..n*n).filter_map(|k|form(&g,k/n,k%n,1.)));beats.push(forms.len());}
        let w=working.get_mut();zeroed(&mut w.re,n*n);zeroed(&mut w.im,n*n);zeroed(&mut w.phases,half);
        e
    }
    /// Part q's entries.
    fn part(&self,q:usize)->&[(usize,usize,Complex)]{&self.entries[self.parts[q]..self.parts[q+1]]}
    /// The static gradients' forms.
    fn statics(&self)->&[(usize,usize,[Complex;3])]{&self.forms[..self.beats[0]]}
    /// The positive beat i's forms.
    fn beat(&self,i:usize)->&[(usize,usize,[Complex;3])]{&self.forms[self.beats[i]..self.beats[i+1]]}
    /// The pairs of beats kept.
    fn pairs(&self)->usize{self.beats.len()-1}
    /// x ← exp(−iτ(w₂H₁ + w₁H₂)) exp(−iτ(w₁H₁ + w₂H₂)) x, H_i = H(t_i) (`cf4`): `omegas` the positive beats, each one's adjoint beat
    /// taking its phases' conjugates.
    fn cf4(&self,omegas:&[f64],times:[f64;2],w:[f64;2],tau:f64,x:&mut State){
        let mut working=self.working.borrow_mut();let Working{re,im,phases}=&mut *working;let(n,half)=(self.n,phases.len());
        for(p,omega)in phases.iter_mut().zip(omegas){*p=[Complex::cis(omega*times[0]),Complex::cis(omega*times[1])];}
        let(mut xr,mut xi)=([0.;MOST_MIXED],[0.;MOST_MIXED]);for a in 0..n{xr[a]=x[a].re;xi[a]=x[a].im;}
        for weights in [w,[w[1],w[0]]]{
            re.copy_from_slice(&self.fixed[0]);im.copy_from_slice(&self.fixed[1]);let mut size=(weights[0]+weights[1])*self.sizes[0];
            for(i,p)in phases.iter().enumerate(){
                let z=p[0]*weights[0]+p[1]*weights[1];let(down,up)=(half-i,half+1+i);
                for &(a,b,y)in self.part(down){let k=b*n+a;re[k]+=z.re*y.re+z.im*y.im;im[k]+=z.re*y.im-z.im*y.re;}
                for &(a,b,y)in self.part(up){let k=b*n+a;re[k]+=z.re*y.re-z.im*y.im;im[k]+=z.re*y.im+z.im*y.re;}
                size+=z.abs()*(self.sizes[down]+self.sizes[up]);
            }
            let(parts,terms)=series_within(size*tau.abs(),self.share);
            if self.wide{unsafe{series_wide(re,im,n,parts,terms,tau,&mut xr,&mut xi);}}else{series_body(re,im,n,parts,terms,tau,&mut xr,&mut xi);}
        }
        for a in 0..n{x[a]=Complex::new(xr[a],xi[a]);}
    }
    /// ⟨x|H_q|x⟩ of a part.
    fn form(&self,q:usize,x:&State)->Complex{self.part(q).iter().fold(Complex::ZERO,|s,&(a,b,z)|s+x[a].conj()*z*x[b])}
    /// What a node costs on this generator at the spacing δ: its complex multiply-adds (a form's real part per axis half of one) and
    /// its phases. CF4's two exponentials, their series by the n × n matrix and the parts' combinations, and the force's forms.
    fn node_work(&self,delta:f64)->[f64;2]{
        let half=self.pairs() as f64;let(parts,terms)=series_within(0.5*self.sizes.iter().sum::<f64>()*delta,self.share);
        let combined=self.entries.len() as f64;
        let force=2.5*self.statics().len() as f64+4.*(self.forms.len()-self.statics().len()) as f64;
        [2.*((parts*terms*self.n*self.n) as f64+combined)+force,3.*half]
    }
}
/// The fewest products of H that sum exp(−iX) on a vector, ‖X‖ ≤ x, within `share` of ‖ψ‖: Taylor's series in equal parts, each
/// through the first term whose bound y^m/m! is within its part's share. The products per unit of x fall with y, from 16 at y = ½
/// to 5 at y = 4 (at a share of 1e-6), while the series' round-off, about e^y of the double precision, stays within the share. A share too near the round-off for any part leaves y = 1, whose
/// parts round the least, (x/y) e^y of the double precision least there.
fn series_within(x:f64,share:f64)->(usize,usize){
    let widest=(share/f64::EPSILON).ln().max(1.);let mut parts=((x/widest).ceil() as usize).max(1);let mut best=(parts,usize::MAX);
    loop{
        let part=share/parts as f64;let(terms,size)=terms_within(x/parts as f64,part,64);
        if size>part{parts+=1;continue;}
        if parts*terms>=best.0.saturating_mul(best.1){return best;}
        best=(parts,terms);parts+=1;
    }
}

impl Segment{
    /// The segment of a cluster's components over a step of τ from `start`, its exponentials within
    /// `truncation`, the beats it leaves out within `beats` over `horizon`, a beating grid at `phase` (`Settings`).
    fn new(mut components:Components,start:f64,tau:f64,truncation:f64,beats:f64,horizon:f64,phase:f64)->Segment{
        let dropped=Segment::resolve(&mut components,tau,beats,horizon);
        let mut segment=Segment::build(components,start,tau,truncation,phase);segment.dropped=dropped;segment
    }
    fn build(components:Components,start:f64,tau:f64,truncation:f64,phase:f64)->Segment{
        let n=components.n();
        if components.omegas().is_empty(){
            // The eigendecomposition where its rounding -- R's condition times the double precision, grown
            // by the phases over the step -- is within the exponentials' share.
            let mut spectral=SPECTRAL.take();
            if decompose(&mut spectral.eigen,components.hamiltonian(),n){
                let e=&spectral.eigen;let spread=e.values.iter().flat_map(|a|e.values.iter().map(move|b|(a.conj()-*b).abs())).fold(0.,f64::max);
                if e.condition*f64::EPSILON*(1.+spread*tau)<=truncation{return Segment::spectral(components,start,tau,spectral,spread);}
            }
            SPECTRAL.set(spectral);
            let h=components.hamiltonian();
            let decay:Vec<Complex>=(0..n*n).map(|k|{let(a,b)=(k/n,k%n);(h[a*n+b]-h[b*n+a].conj())*Complex::I}).collect();
            let spacing=tau/(2*static_panels(column_norm(h,n)*tau)) as f64;
            let propagator=exponential(h,n,spacing);
            return Segment{components,start,spacing,how:How::Taylor{propagator,decay},dropped:0.};
        }
        // The force's frequencies are within 2‖H(t)‖ + max|Ω|, ‖H(t)‖ ≤ ‖H₀‖ + Σ‖H_j‖; whole panels of two spacings,
        // each spacing's two exponentials within their part of the exponentials' share.
        let norm=column_norm(components.hamiltonian(),n)+components.beats().map(|b|column_norm(b.hamiltonian,n)).sum::<f64>()
            +0.5*components.omegas().iter().map(|w|w.abs()).fold(0.,f64::max);
        let steps=2*((0.5*tau*norm/phase).ceil() as usize).max(1);
        let how=How::Beating(Entries::new(&components,truncation/(2*steps) as f64));
        Segment{components,start,spacing:tau/steps as f64,how,dropped:0.}
    }
    /// A beating segment at half the spacing, of the same components (`Engine::grid_error`), over its step of τ.
    fn refined(&self,tau:f64,truncation:f64)->Segment{
        let steps=2*(tau/self.spacing).round() as usize;
        Segment{components:self.components.clone(),start:self.start,spacing:tau/steps as f64,how:How::Beating(Entries::new(&self.components,truncation/(2*steps) as f64)),dropped:self.dropped}
    }
    /// The beats a step resolves: the pairs ±Ω whose cost on the step (`bounds`) is within the share are
    /// left out, the cheapest first while their costs' sum stays within it -- on ‖ψ‖ and on the
    /// force's integral (‖G₀‖ + Σ_j ‖G_j‖)τ: a stimulated force stands in the beats its beams' phases move it to. Their cost.
    fn resolve(c:&mut Components,tau:f64,truncation:f64,horizon:f64)->f64{
        let half=c.omegas().len()/2;let mut work=BOUNDS.take();let scale=Segment::bounds(c,tau,truncation,horizon,&mut work);let Bounds{pairs,dropped,..}=&mut work;
        // The cheapest first, in their order where they cost the same.
        pairs.sort_unstable_by(|a,b|a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        let(mut moved,mut pushed)=(0.,0.);zeroed(dropped,half);
        for &(i,x,f)in pairs.iter(){if moved+x<=truncation&&pushed+f<=truncation*scale{moved+=x;pushed+=f;dropped[i]=true;}else{break;}}
        c.retain(|j|!dropped[if j<half{half-1-j}else{j-half}]);
        BOUNDS.set(work);
        moved
    }
    /// Per pair of beats ±Ω (`resolve`), into `work`: its index and what leaving it out costs a step, on the state and
    /// on the force's integral; and the force's integral (‖G₀‖ + Σ_j ‖G_j‖)τ, returned. Dropping B e^{iΩt} from H moves the
    /// propagator by ‖∫ U_rest(τ,s) B U(s) e^{iΩs} ds‖ (Duhamel's formula); integrated by parts it splits
    /// into boundary terms, which successive steps' cancel -- a trajectory pays them once over its horizon H,
    /// until a jump --, and an integral that accumulates: the cost is boundary·τ/H + integral, the smaller
    /// of two bounds. Once: boundary 2b/|Ω|, integral 2hbτ/|Ω|; twice: boundary 2b/|Ω| + 2x/Ω² and integral
    /// τy/Ω², x = ‖[H₀,B]‖ + 3b b′, y = ‖[H₀,[H₀,B]]‖ + 3‖[H₀,B]‖ b′ + 6h b b′ + 3b b′² + 2b h″ + |Ω| b b_± --
    /// b = ‖B‖, h the generator's bound, b′ all the beats' sizes, h″ the other beats' Σ|Ω_k|‖H_k‖, b_± the pair's own: only a beat with its own negative stands still, every other's product is the slower beat's in h″ (norms
    /// above the spectral one). The force's integral alike, its integrand's derivatives ⟨ψ|G|ψ⟩/⟨ψ|ψ⟩
    /// within 4hg and (24h² + 4h″)g (h″ every beat's). Either bound holds the first boundary term: where that alone exceeds
    /// `share`, no pair of it is left out, and it stands for the bound without the commutators.
    fn bounds(c:&Components,tau:f64,share:f64,horizon:f64,work:&mut Bounds)->f64{
        let n=c.n();let size=|h:&[Complex]|column_norm(h,n).max(row_norm(h,n));let half=c.omegas().len()/2;let h0=c.hamiltonian();
        let all=c.beats().map(|b|size(b.hamiltonian)).sum::<f64>();let bound=size(h0)+all;
        let pace=c.beats().map(|b|b.omega.abs()*size(b.hamiltonian)).sum::<f64>();
        let scale=(0..3).map(|a|size(c.gradient(a))+c.beats().map(|b|size(b.gradient[a])).sum::<f64>()).fold(0.,f64::max)*tau;let once=(tau/horizon).min(1.);
        let Bounds{pairs,left,right,first,second,..}=work;
        let mut state=|b:&[Complex],w:f64,others:f64,own:f64|->f64{
            let m=size(b);if 2.*m/w*once>share{return 2.*m/w*once;}
            commutator(h0,b,n,left,right,first);commutator(h0,first,n,left,right,second);let(x1,x2)=(size(first),size(second));
            let x=x1+3.*m*all;let y=x2+3.*x1*all+6.*bound*m*all+3.*m*all*all+2.*m*others+w*m*own;
            (2.*m/w*once+2.*bound*m*tau/w).min((2.*m/w+2.*x/(w*w))*once+tau*y/(w*w))};
        let force=|g:f64,w:f64|(2.*g/w*once+4.*bound*g*tau/w).min((2.*g/w+8.*bound*g/(w*w))*once+tau*(24.*bound*bound+4.*pace)*g/(w*w));
        pairs.clear();
        for i in 0..half{
            let(down,up)=(c.beat(half-1-i),c.beat(half+i));let w=up.omega.abs();
            if w==0.{pairs.push((i,f64::INFINITY,f64::INFINITY));continue;}
            let own=size(down.hamiltonian)+size(up.hamiltonian);let others=pace-w*own;
            let moved=state(down.hamiltonian,w,others,own)+state(up.hamiltonian,w,others,own);
            pairs.push((i,moved,(0..3).map(|a|force(size(down.gradient[a]),w)+force(size(up.gradient[a]),w)).fold(0.,f64::max)));
        }
        scale
    }
    /// A static segment through its eigendecomposition: Simpson's frequencies are the spectrum's
    /// differences λ_a* − λ_b, so the grid's spacing has (spread·2δ)⁴/2880 within `QUADRATURE`.
    fn spectral(components:Components,start:f64,tau:f64,mut spectral:Spectral,spread:f64)->Segment{
        let Spectral{eigen:e,gram,forces,rate,moments,identity,product:xr}=&mut spectral;let n=e.n;let r=&e.vectors;
        let steps=((spread*tau/(2.*node_phase())).ceil() as usize).max(1);let spacing=tau/steps as f64;
        // R† X R.
        let mut sandwich=|x:&[Complex],out:&mut Vec<Complex>|{product(x,r,n,xr);zeroed(out,n*n);
            for i in 0..n{for k in 0..n{let y=r[k*n+i].conj();for j in 0..n{out[i*n+j]+=y*xr[k*n+j];}}}};
        zeroed(identity,n*n);for a in 0..n{identity[a*n+a]=Complex::ONE;}
        sandwich(identity,gram);for a in 0..3{sandwich(components.gradient(a),&mut forces[a]);}
        rate.clear();rate.extend((0..n*n).map(|k|{let(a,b)=(k/n,k%n);Complex::I*(e.values[a].conj()-e.values[b])*gram[k]}));
        for(f,[size,fourth])in moments.iter_mut().enumerate(){
            let m=if f<3{&forces[f]}else{&*gram};size.clear();size.extend(m.iter().map(|z|z.abs()));
            fourth.clear();fourth.extend((0..n*n).map(|k|size[k]*(e.values[k/n].conj()-e.values[k%n]).abs().powi(4)));
        }
        Segment{components,start,spacing,how:How::Spectral(spectral),dropped:0.}
    }
    fn n(&self)->usize{self.components.n()}
    /// A spectral segment's grid spacing for the evolution of the state x over `remaining` (`threshold`
    /// the norm it may stop at), each frequency weighted by its amplitude: in the eigenbasis every
    /// component of the force's forms and of the norm's, c = (a, b), is A_c e^{iω_c s}, ω_c = λ_a* − λ_b, with
    /// |A_c(s)| ≤ |x_a||x_b||X_ab| (the coordinates only decay), and composite Simpson errs on it by at most
    /// |A_c| τ (|ω_c|·2δ)⁴/2880: the spacing whose fourth moment Σ_c |A_c||ω_c|⁴ (2δ)⁴/2880 keeps every form
    /// within `QUADRATURE` of its total |A|, the integrand's scale at the start, the norm's fall to its
    /// least over the evolution taken in (‖ψ‖² ≥ max(threshold, its end)); the spacings over `remaining`
    /// whole. No credit for a fast component's integral, as this grid moves with the state (module documentation, # The exact step).
    fn weighted(&self,x:&State,remaining:f64,threshold:f64)->Option<f64>{
        let How::Spectral(spectral)=&self.how else{return None};let n=self.n();
        let start=self.norm(x);let mut end=*x;self.advance(&mut end,0.,remaining,None);let least=self.norm(&end).max(threshold).min(start);
        let allowed=QUADRATURE*least/start;let mut size=[0.;MOST_MIXED];for a in 0..n{size[a]=x[a].abs();}
        // The largest fourth moment relative to its total over the forms.
        let ratio=spectral.moments.iter().map(|[total,fourth]|{let(mut t,mut f)=(0.,0.);
            for a in 0..n{let(mut ta,mut fa)=(0.,0.);for b in 0..n{ta+=total[a*n+b]*size[b];fa+=fourth[a*n+b]*size[b];}t+=size[a]*ta;f+=size[a]*fa;}
            if t>0.{f/t}else{0.}}).fold(0.,f64::max);
        if !(ratio>0.){return Some(remaining);}
        let widest=0.5*(2880.*allowed/ratio).powf(0.25);let steps=(remaining/widest).ceil().max(1.);
        Some(remaining/steps)
    }
    /// The gauge's phases over a time from the path's middle: ψ_s = e^{iK_s·v(t − t_m)} x_s (model.rs `Components`).
    fn gauged(&self,x:&mut [Complex],elapsed:f64){let c=&self.components;if c.gauged{for a in 0..self.n(){x[a]=x[a]*Complex::cis(c.gauge[a]*elapsed);}}}
    /// The state of ψ at the time `at`.
    fn enter(&self,psi:&[Complex],at:f64)->State{
        let n=self.n();let mut y=[Complex::ZERO;MOST_MIXED];y[..n].copy_from_slice(&psi[..n]);self.gauged(&mut y,self.components.middle-at);
        match &self.how{How::Spectral(s)=>{let inverse=&s.eigen.inverse;let mut x=[Complex::ZERO;MOST_MIXED];for a in 0..n{x[a]=(0..n).fold(Complex::ZERO,|s,b|s+inverse[a*n+b]*y[b]);}x},_=>y}
    }
    /// ψ of a state at the time `t`.
    fn leave(&self,x:&State,psi:&mut [Complex],t:f64){
        let n=self.n();
        match &self.how{How::Spectral(s)=>{let vectors=&s.eigen.vectors;for a in 0..n{psi[a]=(0..n).fold(Complex::ZERO,|s,b|s+vectors[a*n+b]*x[b]);}},_=>psi[..n].copy_from_slice(&x[..n])}
        self.gauged(psi,t-self.components.middle);
    }
    /// A state propagated over `len` from the time t; `node`: the grid's node i where it is the whole
    /// spacing from it.
    fn advance(&self,x:&mut State,t:f64,len:f64,node:Option<usize>){
        let n=self.n();
        match(&self.how,node){
            (How::Spectral(s),_)=>{for a in 0..n{x[a]=x[a]*rotation(s.eigen.values[a],len);}}
            (How::Taylor{propagator,..},Some(_))=>multiply(propagator,n,x),
            (How::Taylor{..},None)=>propagate_dyn(self.components.hamiltonian(),n,len,x),
            (How::Beating(e),_)=>{let(c,w)=cf4();e.cf4(&self.components.omegas()[e.pairs()..],[t+c[0]*len,t+c[1]*len],w,len,x);}
        }
    }
    /// ‖ψ‖² of a state.
    fn norm(&self,x:&State)->f64{match &self.how{How::Spectral(s)=>crate::model::hermitian_form(&s.gram,x,self.n()),_=>norm2(x,self.n())}}
    /// The conditional force ⟨ψ|F|ψ⟩/⟨ψ|ψ⟩ at the time t and ‖ψ‖²; `node`: the grid's node there.
    fn force(&self,x:&State,t:f64,node:Option<usize>)->([f64;3],f64){
        let n=self.n();let norm=self.norm(x);
        let force=match(&self.how,node){
            // G(t) = G₀ + Σ_j e^{iΩ_j t} G_j, the beats in pairs G_{−Ω} = G_Ω†: Re⟨G₀⟩ + 2 Σ_{Ω>0} Re e^{iΩt}⟨G_Ω⟩.
            (How::Beating(e),_)=>{
                let mut f=[0.;3];
                for &(a,b,g)in e.statics(){let w=x[a].conj()*x[b];for k in 0..3{f[k]+=g[k].re*w.re-g[k].im*w.im;}}
                for(i,omega)in self.components.omegas()[e.pairs()..].iter().enumerate(){
                    let phase=Complex::cis(omega*t);let mut s=[Complex::ZERO;3];
                    for &(a,b,g)in e.beat(i){let w=x[a].conj()*x[b];for k in 0..3{s[k]+=g[k]*w;}}
                    for k in 0..3{f[k]+=2.*(phase*s[k]).re;}
                }
                f.map(|v|-v/norm)
            }
            (How::Spectral(s),_)=>std::array::from_fn(|a|-crate::model::hermitian_form(&s.forces[a],x,n)/norm),
            (How::Taylor{..},_)=>std::array::from_fn(|a|-crate::model::hermitian_form(self.components.gradient(a),x,n)/norm),
        };
        (force,norm)
    }
    /// −d‖ψ‖²/dt of a state at the time t: ⟨ψ|i(H − H†)|ψ⟩ = −2 Im⟨ψ|H|ψ⟩.
    fn rate(&self,x:&State,t:f64)->f64{
        let n=self.n();
        match &self.how{
            How::Spectral(s)=>-crate::model::hermitian_form(&s.rate,x,n),
            How::Taylor{decay,..}=>crate::model::hermitian_form(decay,x,n),
            How::Beating(e)=>{
                let mut z=e.form(0,x);
                for(j,omega)in self.components.omegas().iter().enumerate(){z+=Complex::cis(omega*t)*e.form(1+j,x);}
                -2.*z.im
            }
        }
    }
    /// From ψ (unnormalized) at the time `at`, the evolution for at most `remaining` until ‖ψ‖² reaches
    /// `threshold`: the time taken, whether it reached it (ψ then at the root), and the impulse ∫ ⟨F⟩ dt --
    /// composite Simpson on panels of two spacings from a node, a panel to the next node first where the
    /// evolution starts between nodes (after a jump), and what is left last; a spectral segment's grid
    /// from `at`, weighted by the state's amplitudes.
    fn evolve(&self,psi:&mut [Complex],at:f64,remaining:f64,threshold:f64)->(f64,bool,[f64;3]){
        let(n,end)=(self.n(),at+remaining);let mut impulse=[0.;3];
        let mut x=self.enter(psi,at);let mut t=at;
        // The grid: a spectral segment's from `at`, its spacing weighted by the state's amplitudes (`weighted`)
        // and its phases over it; the others' from the segment's start.
        let weighted=self.weighted(&x,remaining,threshold);
        let(origin,delta)=match weighted{Some(d)=>(at,d),None=>(self.start,self.spacing)};
        let phases:Option<State>=match(&self.how,weighted){(How::Spectral(s),Some(d))=>{let mut p=[Complex::ZERO;MOST_MIXED];for a in 0..n{p[a]=rotation(s.eigen.values[a],d);}Some(p)},_=>None};
        let step=|x:&mut State,t:f64,len:f64,node:Option<usize>|match(&phases,node){(Some(p),Some(_))=>{for a in 0..n{x[a]=x[a]*p[a];}}_=>self.advance(x,t,len,node)};
        // The first node at or after `at`.
        let mut i=((at-origin)/delta).ceil().max(0.) as usize;
        let(mut fa,mut na)=self.force(&x,t,(origin+i as f64*delta==t).then_some(i));
        while t<end{
            let node=origin+i as f64*delta;
            // The panel [t, b]: two spacings from node i, else to the next node, at most to the end.
            let(b,whole)=if node>t{(node.min(end),false)}else{let b=origin+(i+2) as f64*delta;if b<=end{(b,true)}else{(end,false)}};
            let(width,half)=(b-t,0.5*(b-t));
            let mut middle=x;step(&mut middle,t,half,whole.then_some(i));
            let mut last=middle;step(&mut last,t+half,half,whole.then_some(i+1));
            let(fm,nm)=self.force(&middle,t+half,whole.then_some(i+1));
            let(fb,nb)=self.force(&last,b,if whole{Some(i+2)}else if b==node{Some(i)}else{None});
            if nb<=threshold{
                let s=self.root(&x,&middle,t,[na,nm,nb],width,threshold);
                let mut h=x;self.advance(&mut h,t,0.5*s,None);let mut y=h;self.advance(&mut y,t+0.5*s,0.5*s,None);
                let(fh,_)=self.force(&h,t+0.5*s,None);let(fs,_)=self.force(&y,t+s,None);
                for a in 0..3{impulse[a]+=s/6.*(fa[a]+4.*fh[a]+fs[a]);}
                self.leave(&y,psi,t+s);return(t+s-at,true,impulse);
            }
            for a in 0..3{impulse[a]+=width/6.*(fa[a]+4.*fm[a]+fb[a]);}
            x=last;fa=fb;na=nb;t=b;if whole{i+=2;}
        }
        self.leave(&x,psi,end);
        (remaining,false,impulse)
    }
    /// The time s in (0, width] at which ‖ψ(s)‖² = threshold from the panel's start x (its middle at half
    /// the width): Newton on ln‖ψ(s)‖² with its derivative −rate/‖ψ‖², from linear interpolation of the
    /// logarithm on the panel's half where it lies, kept inside the bracket (bisection when it steps out).
    fn root(&self,x:&State,middle:&State,t:f64,norms:[f64;3],width:f64,threshold:f64)->f64{
        let[na,nm,nb]=norms;let(target,half)=(threshold.ln(),0.5*width);
        let(mut lo,mut hi)=if nm<=threshold{(0.,half)}else{(half,width)};
        let(la,lb)=if nm<=threshold{(na.ln(),nm.ln())}else{(nm.ln(),nb.ln())};
        let mut s=if lb<la{(lo+(target-la)/(lb-la)*(hi-lo)).clamp(lo,hi)}else{hi};
        for _ in 0..64{
            let(mut y,from)=if s<=half{(*x,0.)}else{(*middle,half)};self.advance(&mut y,t+from,s-from,None);
            let norm=self.norm(&y);let g=norm.ln()-target;
            if g>0.{lo=s;}else{hi=s;}
            let rate=self.rate(&y,t+s)/norm;
            let next=if rate>0.{s+g/rate}else{0.5*(lo+hi)};
            let next=if next>lo&&next<hi{next}else{0.5*(lo+hi)};
            if (next-s).abs()<=f64::EPSILON*width.max(s){return next;}
            s=next;
        }
        s
    }
}

/// The fourth order's start from a bare ground state at t = 0, or a ringing photon's landing (module documentation, # The transient; into a cluster with
/// its tables; model.rs `Model::slow_part`): ψ ← Π f/√A, its slow part, and the threshold θ ← θ/A,
/// A = ‖Π f‖² -- the survival after the dressing's transient is A times the slow part's. A < 1 (far
/// from resonance): with probability 1 − A the transient emits at once (θ/A ≥ 1, a jump at the step's
/// start); A > 1 (near it): the dressing's rebuild delays the next jump.
pub fn slow_start(model:&Model,t:&mut Trajectory,point:&Point,time:f64){
    let Some(pi)=model.slow_part(t.cluster,point,time)else{return};let n=model.dimension(t.cluster);
    let mut y=[Complex::ZERO;MOST_MIXED];for a in 0..n{for b in 0..n{y[a]+=pi[a*n+b]*t.psi[b];}}
    let(w,before)=(norm2(&y,n),norm2(&t.psi,n));let a=w/before;
    for z in y[..n].iter_mut(){*z=*z*(before/w).sqrt();}t.psi=y;t.threshold/=a;
}

/// What the exact step does with a cluster at a point (`Engine::probe`, examples/multirate.rs): its
/// segment's path ("spectral", "taylor", "beating"), the grid's spacing and nodes, per pair of
/// beats ±Ω its frequency, its bound and whether it was left out (`Segment::resolve`), the share,
/// the seconds its components and its segment take to build, and on a beating grid a node's work (`Entries::node_work`).
pub struct Probe{pub path:&'static str,pub spacing:f64,pub nodes:usize,pub beats:Vec<(f64,f64,bool)>,pub share:f64,pub seconds:[f64;2],pub weighted:Option<usize>,pub node:[f64;2]}

/// What one step from a trajectory's state, normalized, lands between groups of ground states (`Engine::grid_landings`, solve.rs `landed_excess`): the group it starts in, its largest; per group the master
/// equation's source Σ_x ‖P_g J_x ψ(s)‖² summed over the step's reads, ψ the no-jump state unnormalized; the ground states' population at the step's end, and the norm there.
pub struct Landings{pub from:usize,pub sources:Vec<f64>,pub stay:Vec<f64>,pub left:f64}

/// A real orthonormal pair across the unit vector n.
fn across(n:[f64;3])->[[f64;3];2]{
    let a=if n[0].abs()<=n[1].abs()&&n[0].abs()<=n[2].abs(){[1.,0.,0.]}else if n[1].abs()<=n[2].abs(){[0.,1.,0.]}else{[0.,0.,1.]};
    let cross=|u:[f64;3],v:[f64;3]|[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]];
    let e1=cross(n,a);let l=(e1[0]*e1[0]+e1[1]*e1[1]+e1[2]*e1[2]).sqrt();let e1=e1.map(|x|x/l);[e1,cross(n,e1)]
}

/// A photon's direction n from the dipole pattern of its amplitudes' Gram matrix G
/// (G_ab = ⟨u_a|u_b⟩): density tr G − n·Re(G)·n, by rejection from the uniform sphere (the density
/// is at most tr G and at least a third of its mean).
pub fn dipole_direction(rng:&mut Rng,gram:&[[Complex;3];3])->[f64;3]{
    let trace=gram[0][0].re+gram[1][1].re+gram[2][2].re;
    loop{
        let d=rng.direction();let mut q=0.;for a in 0..3{for b in 0..3{q+=d[a]*gram[a][b].re*d[b];}}
        if rng.uniform()*trace<=trace-q{return d;}
    }
}

/// A photon of amplitudes u_a on nt states: its direction from the dipole pattern, its polarization
/// across it (by `polarization`, uniform), the state it leaves f = Σ ε_a u_a and ‖f‖².
fn emit(rng:&mut Rng,u:&[[Complex;MOST_STATES];3],nt:usize,polarization:f64)->([f64;3],[f64;3],[Complex;MOST_STATES],f64){
    let gram:[[Complex;3];3]=std::array::from_fn(|a|std::array::from_fn(|b|(0..nt).fold(Complex::ZERO,|acc,i|acc+u[a][i].conj()*u[b][i])));
    let direction=dipole_direction(rng,&gram);
    let pair=across(direction);
    let combine=|e:&[f64;3]|->[Complex;MOST_STATES]{let mut out=[Complex::ZERO;MOST_STATES];for i in 0..nt{for a in 0..3{out[i]+=u[a][i]*e[a];}}out};
    let amplitudes=pair.map(|e|combine(&e));let weights=amplitudes.map(|v|norm2(&v,nt));
    let chosen=if polarization*(weights[0]+weights[1])<weights[0]{0}else{1};
    (direction,pair[chosen],amplitudes[chosen],weights[chosen])
}

/// A part's share below the largest by single precision's round-off squared is the GPU's amplitudes' round-off: C's eigenvalues below this of its trace, 2⁻⁴⁶, 64 of double
/// precision's, are no direction on either device (`part_weights`).
const SHARE_ROUNDING:f64=f32::EPSILON as f64*f32::EPSILON as f64;
/// The least spread of a photon's parts' recoils along a direction, of the largest recoil, for which their weights move along it (`part_weights`). Within it the
/// weights could move the drawn mean by that spread at most and the deterministic kick carries it instead, the mean exact: the photon's second moment moves by twice the spread, of k², at most, within
/// T's numerical share (`QUADRATURE`). Beams along one axis on carriers apart, 1e-7 of a recoil at 36 MHz, are one direction so -- the kernel's single-precision recoils do not even tell that apart --, and a step
/// along their difference, which no weight could complete, would only drop one part or another.
pub(crate) const RECOIL_SPREAD:f64=0.5*QUADRATURE;

/// The weights of a photon's parts (# Absorption): their shares s_c moved the least (χ²) to the mean t of their recoils' means
/// μ_c, nonnegative: q_c ← q_c + α a_c β·(μ_c − m), a_c the parts still on (first the shares), m their mean, β = C⁺ (t − Σ q μ)
/// on their covariance C's range, the directions above its round-off (`SHARE_ROUNDING`) along which the parts on spread by
/// more than `RECOIL_SPREAD`, α ≤ 1 up to the first weight that reaches zero -- it
/// leaves, and the others move on, at most once per part. A t beyond the recoils' hull leaves them on its face, the rest
/// to the caller. All zero without a share.
pub fn part_weights(s:&[f64],mu:&[[f64;3]],target:[f64;3])->Vec<f64>{
    let n=s.len();let total:f64=s.iter().sum();if !(total>0.){return vec![0.;n];}
    let mut q:Vec<f64>=s.iter().map(|x|x/total).collect();let mut on=q.clone();
    let spread=RECOIL_SPREAD*mu.iter().map(|m|(m[0]*m[0]+m[1]*m[1]+m[2]*m[2]).sqrt()).fold(0.,f64::max);
    for _ in 0..n{
        let sum:f64=on.iter().sum();if !(sum>0.){break;}
        let centre:[f64;3]=std::array::from_fn(|a|(0..n).map(|c|on[c]*mu[c][a]).sum::<f64>()/sum);
        let g:[f64;3]=std::array::from_fn(|a|target[a]-(0..n).map(|c|q[c]*mu[c][a]).sum::<f64>());
        let mut h=[[0.;3];3];for c in 0..n{for i in 0..3{for j in 0..3{h[i][j]+=on[c]*(mu[c][i]-centre[i])*(mu[c][j]-centre[j]);}}}
        let(values,vectors)=crate::linear::symmetric_eigen_pivoted(h);let floor=SHARE_ROUNDING*values.iter().map(|v|v.abs()).sum::<f64>();
        let mut beta=[0.;3];
        for i in 0..3{
            let spreads=(0..n).any(|c|on[c]>0.&&(0..3).map(|a|vectors[i][a]*(mu[c][a]-centre[a])).sum::<f64>().abs()>spread);
            if values[i]>floor&&spreads{let p=(0..3).map(|a|vectors[i][a]*g[a]).sum::<f64>()/values[i];for a in 0..3{beta[a]+=p*vectors[i][a];}}
        }
        let d:Vec<f64>=(0..n).map(|c|on[c]*(0..3).map(|a|beta[a]*(mu[c][a]-centre[a])).sum::<f64>()).collect();
        let alpha=(0..n).filter(|&c|on[c]>0.&&d[c]<0.).map(|c|-q[c]/d[c]).fold(1.,f64::min);
        for c in 0..n{q[c]+=alpha*d[c];if on[c]>0.&&d[c]<0.&&q[c]<=-4.*f64::EPSILON*alpha*d[c]{q[c]=0.;on[c]=0.;}}
        if !(alpha<1.){break;}
    }
    let sum:f64=q.iter().sum();if sum>0.{for x in q.iter_mut(){*x/=sum;}}
    q
}

/// The explicit part's sign (`absorbed_recoil`) takes the lower half of the bits of the draw's uniform: the upper half
/// picks the part and the beam, so it is independent of them, and it does not depend on their weights' round-off.
pub const SIGN_BITS:f64=(1u64<<26) as f64;

/// A photon's absorbed recoil (# Absorption) from the uniform u: its parts -- per near beam its wave vector, share and
/// projection on the photon; the explicit states' share and their beams' wave vectors and drive weights --, and its exact
/// conditional mean. u picks the part by its weight (`part_weights`, the explicit part's recoil mean zero), its leftover the
/// explicit part's beam, and its lower bits (`SIGN_BITS`) the sign of its mean's shot noise; the kick is the picked beam's
/// recoil (the explicit part's less its drives' mean, ± that mean) and the rest of the exact mean.
pub fn absorbed_recoil(beams:&[([f64;3],f64,f64)],explicit:Option<(f64,&[([f64;3],f64)])>,mean:[f64;3],u:f64)->[f64;3]{
    let mut s:Vec<f64>=beams.iter().map(|x|x.1).collect();let mut mu:Vec<[f64;3]>=beams.iter().map(|x|x.0).collect();
    if let Some((v,_))=explicit{s.push(v);mu.push([0.;3]);}
    let target:[f64;3]=std::array::from_fn(|a|beams.iter().map(|x|x.2*x.0[a]).sum());
    let q=part_weights(&s,&mu,target);
    let Some(last)=(0..q.len()).rev().find(|&c|q[c]>0.)else{return mean};
    let mut x=u;let mut pick=last;
    for(c,&v)in q.iter().enumerate(){if x<v{pick=c;break;}x-=v;}
    let left=(x/q[pick]).clamp(0.,1.);
    let mut kick=if pick<beams.len(){beams[pick].0}else{
        let drives=explicit.map_or(&[][..],|x|x.1);let total:f64=drives.iter().map(|d|d.1).sum();
        if !(total>0.){[0.;3]}else{
            let average:[f64;3]=std::array::from_fn(|a|drives.iter().map(|d|d.1*d.0[a]).sum::<f64>()/total);
            let mut y=left*total;let mut l=(0..drives.len()).rev().find(|&i|drives[i].1>0.).unwrap();
            for(i,d)in drives.iter().enumerate(){if y<d.1{l=i;break;}y-=d.1;}
            let sign=if (u*SIGN_BITS).fract()<0.5{1.}else{-1.};
            std::array::from_fn(|a|drives[l].0[a]-average[a]+sign*average[a])
        }
    };
    for a in 0..3{kick[a]+=mean[a]-(0..q.len()).map(|c|q[c]*mu[c][a]).sum::<f64>();}
    kick
}

/// The stream of a trajectory's jump number `jump` (the initial threshold is jump 0's).
pub fn jump_stream(seed:u64,index:u64,jump:u64)->Rng{Rng::new(seed,Purpose::Jump,index,jump)}

/// An atom held at a point (control.rs `Twin`), its generator static there: every cluster's H_eff,
/// per class and photon axis its jump operator L (target × cluster, row-major), and its post-jump
/// transient, computed once.
pub struct Held{pub point:Point,pub locals:Vec<Local>,pub jumps:Vec<Vec<[Vec<Complex>;3]>>}

/// The engine: a model and its settings; for a held atom its generator and jumps there, and each
/// cluster's propagator exp(−i H dt/2) over half the settings' step.
/// Per cluster of ground states with beats the largest bound of those beats (`Segment::resolve`, its step's) where the exact step leaves them all out at the model's reach -- wherever the atom goes:
/// a generator static within the exponentials' share, the plain step's, at a tenth of the work (# The exact step). None for the others.
pub(crate) fn quiet_clusters(model:&Model,s:&Settings)->Vec<Option<f64>>{
    (0..model.clusters()).map(|k|{
        if model.mixed(k)||!model.beating(k){return None;}
        let mut most=0_f64;
        for &r in model.reach_points(){
            let mut c=model.components(k,&model.point(r,[0.;3]));most=most.max(Segment::resolve(&mut c,s.dt,s.beats,s.horizon));
            if !c.omegas().is_empty(){return None;}
        }
        Some(most)
    }).collect()
}

pub struct Engine<'a>{pub model:&'a Model,pub settings:Settings,held:Option<(&'a Held,Vec<[Complex;MOST_STATES*MOST_STATES]>)>,quiet:Vec<Option<f64>>}

impl<'a> Engine<'a>{
    pub fn new(model:&'a Model,settings:Settings)->Engine<'a>{let quiet=quiet_clusters(model,&settings);Engine{model,settings,held:None,quiet}}
    /// The engine of an atom held at a point: no axis free, no loss, the held generator and jumps;
    /// the half step's propagators by the exponentials' own series on the basis states. Its clusters
    /// are of ground states (a twin exists for those alone, control.rs).
    pub fn held(model:&'a Model,settings:&Settings,held:&'a Held)->Engine<'a>{
        let mut s=settings.clone();s.free=[false;3];s.loss=false;
        let half=held.locals.iter().enumerate().map(|(k,local)|{let n=model.dimension(k);let mut u=[Complex::ZERO;MOST_STATES*MOST_STATES];
            for j in 0..n{let mut e=[Complex::ZERO;MOST_STATES];e[j]=Complex::ONE;propagate(&local.hamiltonian,n,0.5*s.dt,&mut e,s.truncation);for i in 0..n{u[i*n+j]=e[i];}}u}).collect();
        Engine{model,settings:s,held:Some((held,half)),quiet:vec![]}
    }
    /// The trap's harmonic flow for τ about c, its part weighted by w (`reference`): along its principal axes
    /// when every lab axis is free, else along the free lab axes, at the frequencies ω √w.
    fn flow(&self,t:&mut Trajectory,tau:f64,(c,w):([f64;3],f64)){
        let(s,hm)=(&self.settings,self.model.hbar_over_mass);let scale=if w==1.{1.}else{w.sqrt()};
        if s.free==[true;3]{
            let(mut r,mut kappa)=([0.;3],[0.;3]);
            for i in 0..3{let(x,q)=flow(dot(s.axes[i],t.r)-dot(s.axes[i],c),dot(s.axes[i],t.kappa),s.frequencies[i]*scale,hm,tau);for a in 0..3{r[a]+=x*s.axes[i][a];kappa[a]+=q*s.axes[i][a];}}
            (t.r,t.kappa)=(std::array::from_fn(|a|r[a]+c[a]),kappa);
        }else{for a in 0..3{if s.free[a]{(t.r[a],t.kappa[a])=flow(t.r[a],t.kappa[a],s.lab[a]*scale,hm,tau);}}}
    }
    /// The harmonic part's centre and weight at a step's start r (module documentation, # Beyond the trap's region,
    /// # A standing wave's sites): one within the region, U_h ≤ D (`Settings::region`), and (2u − 1)/u² beyond it, u =
    /// U_h/D -- continuous with its slope at the region's edge, the weighted part's potential below 2D
    /// everywhere --, about the trap's centre or the atom's own site; one without a region.
    fn reference(&self,r:[f64;3])->([f64;3],f64){
        let s=&self.settings;let(depth,about)=(s.region,s.about());if !(depth>0.){return(about,1.);}
        let changeover=|u:f64|if u<=1.{1.}else{(2.*u-1.)/(u*u)};
        match s.lattice.filter(|_|s.free==[true;3]){
            None=>(about,changeover(self.harmonic_potential(std::array::from_fn(|a|r[a]-about[a]))/depth)),
            Some(l)=>{
                let(n,phase)=l.nearest(dot(l.axis,r));let z=l.site(n);let c=l.axis.map(|x|x*z);let share=l.share(z);
                let along=dot(l.axis,r)-z;let across:[f64;3]=std::array::from_fn(|a|r[a]-c[a]-along*l.axis[a]);
                let tangent=phase.tan();(c,share*changeover(self.harmonic_potential(across)/(depth*share)+tangent*tangent))
            }
        }
    }
    /// The harmonic part's potential at r (rad/us): along the trap's principal axes, or the free lab axes.
    fn harmonic_potential(&self,r:[f64;3])->f64{
        let(s,hm)=(&self.settings,self.model.hbar_over_mass);
        if s.free==[true;3]{(0..3).map(|i|{let x=dot(s.axes[i],r);s.frequencies[i]*s.frequencies[i]*x*x}).sum::<f64>()/(2.*hm)}
        else{(0..3).filter(|&a|s.free[a]).map(|a|s.lab[a]*s.lab[a]*r[a]*r[a]).sum::<f64>()/(2.*hm)}
    }
    /// The harmonic part's force on κ at r (1/um per us).
    fn harmonic(&self,r:[f64;3])->[f64;3]{
        let(s,hm)=(&self.settings,self.model.hbar_over_mass);let mut f=[0.;3];
        if s.free==[true;3]{for i in 0..3{let x=dot(s.axes[i],r)*s.frequencies[i]*s.frequencies[i]/hm;for a in 0..3{f[a]-=x*s.axes[i][a];}}}
        else{for a in 0..3{f[a]=-s.lab[a]*s.lab[a]/hm*r[a];}}
        f
    }

    /// One step's flow, kick and internal evolution of length dt without jumps: the norms of ψ at
    /// the start, the middle and the end, the place and momentum of the middle, the kick's momentum (the light's
    /// and the trap's anharmonic part) and ψ at the middle. A cluster of ground states (`exact` takes
    /// the others). Where ‖ψ‖² ends at or below `threshold` the step is to be redone to its crossing (`step`): the norms, the place and the middle's ψ alone, the rest of `t` for its caller to restore.
    fn advance(&self,t:&mut Trajectory,dt:f64,threshold:f64)->(f64,f64,f64,([f64;3],[f64;3]),[f64;3],[Complex;MOST_MIXED]){
        let(m,s)=(self.model,&self.settings);
        if let Some((h,half))=&self.held{
            // Held: the internal state alone, by the generator there -- a whole step by the half
            // step's propagator, a step's part by the series.
            let n=m.dimension(t.cluster);let local=&h.locals[t.cluster];let n0=norm2(&t.psi,n);let mut psi=t.psi;
            let half_step=|psi:&mut [Complex;MOST_MIXED]|if dt==s.dt{let u=&half[t.cluster];let v=*psi;for i in 0..n{let mut z=Complex::ZERO;for j in 0..n{z+=u[i*n+j]*v[j];}psi[i]=z;}}
                else{propagate(&local.hamiltonian,n,0.5*dt,psi,s.truncation);};
            half_step(&mut psi);let nh=norm2(&psi,n);let middle=psi;half_step(&mut psi);let n1=norm2(&psi,n);
            t.psi=psi;t.time+=dt;return(n0,nh,n1,(h.point.r,h.point.kappa),[0.;3],middle);
        }
        let reference=self.reference(t.r);let(c,w)=reference;
        self.flow(t,0.5*dt,reference);
        let point=m.point(t.r,t.kappa);
        let n=m.dimension(t.cluster);let local=m.local(t.cluster,&point,t.time+0.5*dt);
        let n0=norm2(&t.psi,n);let half=HalfStep::new(&local.hamiltonian,n,0.5*dt,s.truncation);
        let mut psi=t.psi;half.apply(&local.hamiltonian,n,0.5*dt,&mut psi,s.truncation);let nh=norm2(&psi,n);let middle=psi;
        half.apply(&local.hamiltonian,n,0.5*dt,&mut psi,s.truncation);let n1=norm2(&psi,n);
        let place=(point.r,point.kappa);let mut kick=[0.;3];
        if !(n1>threshold){return(n0,nh,n1,place,kick,middle);}
        let force=m.force(t.cluster,&local,&point,&middle);let harmonic=self.harmonic(std::array::from_fn(|a|t.r[a]-c[a]));
        for a in 0..3{if s.free[a]{kick[a]=dt*(force[a]-w*harmonic[a]);t.kappa[a]+=kick[a];}}
        t.psi=psi;
        self.flow(t,0.5*dt,reference);
        t.time+=dt;
        (n0,nh,n1,place,kick,middle)
    }

    /// A cluster's components over a step of τ from `start` through a point on the atom's path (model.rs `Model::on_path`, # The atom's path), its momentum the step's middle's: the point's and half the kick the step takes, of its reference (c, w)
    /// less its harmonic part, from the force at the point on the state at the step's start.
    fn on_path(&self,cluster:usize,point:&Point,psi:&[Complex],start:f64,tau:f64,(c,w):([f64;3],f64))->Components{
        let(m,s)=(self.model,&self.settings);let mut components=m.components(cluster,point);
        if m.moves(cluster){
            let middle=start+0.5*tau;let force=components.force_at(middle,psi);let harmonic=self.harmonic(std::array::from_fn(|a|point.r[a]-c[a]));
            let kappa=std::array::from_fn(|a|point.kappa[a]+if s.free[a]{0.5*tau*(force[a]-w*harmonic[a])}else{0.});
            m.on_path(cluster,&mut components,kappa,middle);
        }
        components
    }
    /// The exact step of τ (module documentation): the flow for τ/2; at the point there the cluster's
    /// generator at the middle time once, the internal state propagated exactly (`jumps` false: with no
    /// jump, the step-doubling check's), the impulse along it; the kick; the flow for τ/2. The kick's
    /// momentum (the light's and the trap's anharmonic part). Where ‖ψ‖² reaches the threshold in it, the
    /// step is redone to that crossing, the jump made at the place it reaches and the rest of the step
    /// taken from there (the kick then none of the caller's: only the check, without jumps, reads it). A part of a step
    /// leaves out beats within its part of the share (a short part took the step's whole share, and what it left out booked 12 % on the 1 G molasses).
    fn exact(&self,t:&mut Trajectory,tau:f64,exact:bool,jumps:bool)->[f64;3]{
        let(m,s)=(self.model,&self.settings);
        let before=jumps.then(||t.clone());
        let reference=self.reference(t.r);let(c,w)=reference;
        self.flow(t,0.5*tau,reference);
        let point=m.point(t.r,t.kappa);let start=t.time;
        let segment=Segment::new(self.on_path(t.cluster,&point,&t.psi,start,tau,reference),start,tau,s.truncation,s.beats*(tau/s.dt).min(1.),s.horizon,s.node_phase);t.beats=t.beats.max(segment.dropped/tau);
        let(taken,jumped,impulse)=segment.evolve(&mut t.psi,start,tau,if jumps{t.threshold}else{0.});drop(segment);
        if let(true,Some(from))=(jumped,before){
            let beats=t.beats;*t=from;t.beats=beats;
            self.exact(t,taken,exact,false);self.jump(t,t.r,t.time,exact);
            if taken<tau{self.exact(t,tau-taken,exact,true);}
            t.time=start+tau;
            return [0.;3];
        }
        let harmonic=self.harmonic(std::array::from_fn(|a|t.r[a]-c[a]));let mut kick=[0.;3];
        for a in 0..3{if s.free[a]{kick[a]=impulse[a]-tau*w*harmonic[a];t.kappa[a]+=kick[a];}}
        self.flow(t,0.5*tau,reference);
        t.time=start+tau;
        kick
    }

    /// The exact step's segment of a cluster at a point over τ from `start` (`Probe`).
    pub fn probe(&self,cluster:usize,point:&Point,start:f64,tau:f64,psi:&[Complex])->Probe{
        let clock=std::time::Instant::now();let c=self.on_path(cluster,point,psi,start,tau,self.reference(point.r));let built=clock.elapsed().as_secs_f64();
        let mut work=Bounds::default();Segment::bounds(&c,tau,self.settings.beats,self.settings.horizon,&mut work);let half=c.omegas().len()/2;let omegas=c.omegas()[half..].to_vec();
        let clock=std::time::Instant::now();let s=Segment::new(c,start,tau,self.settings.truncation,self.settings.beats,self.settings.horizon,self.settings.node_phase);let segment=clock.elapsed().as_secs_f64();
        let kept=s.components.omegas();
        let(path,node)=match &s.how{How::Spectral(_)=>("spectral",[0.;2]),How::Taylor{..}=>("taylor",[0.;2]),How::Beating(e)=>("beating",e.node_work(s.spacing))};
        let weighted=s.weighted(&s.enter(psi,start),tau,0.).map(|d|(tau/d).round() as usize);
        Probe{path,spacing:s.spacing,nodes:(tau/s.spacing).round() as usize,beats:work.pairs.iter().map(|&(i,x,_)|(omegas[i],x,!kept.contains(&omegas[i]))).collect(),share:self.settings.beats,seconds:[built,segment],weighted,node}
    }
    /// The largest phase whose grid differs from the next coarser's at a trajectory's state: its step in one panel of two spacings. None where its cluster takes no beating segment (solve.rs `grid_phase`).
    pub fn coarsest_phase(&self,t:&Trajectory)->Option<f64>{
        if !self.takes_exact(t.cluster){return None;}
        let(m,s)=(self.model,&self.settings);
        let segment=Segment::new(self.on_path(t.cluster,&m.point(t.r,t.kappa),&t.psi,t.time,s.dt,self.reference(t.r)),t.time,s.dt,s.truncation,s.beats,s.horizon,s.node_phase);
        matches!(segment.how,How::Beating(_)).then(||s.node_phase*s.dt/(2.*segment.spacing))
    }
    /// A beating segment's grid at `phase` against one of half that spacing, one jump-free step from a trajectory's state at its
    /// point: the largest difference of its populations, of the impulse, and the impulse's size; None where its cluster
    /// takes no beating segment (solve.rs `grid_phase`).
    pub fn grid_error(&self,t:&Trajectory,phase:f64)->Option<[f64;3]>{
        if !self.takes_exact(t.cluster){return None;}
        let(m,s)=(self.model,&self.settings);let n=m.dimension(t.cluster);let c=self.on_path(t.cluster,&m.point(t.r,t.kappa),&t.psi,t.time,s.dt,self.reference(t.r));
        let coarse=Segment::new(c,t.time,s.dt,s.truncation,s.beats,s.horizon,phase);if !matches!(coarse.how,How::Beating(_)){return None;}
        let fine=coarse.refined(s.dt,s.truncation);let(mut a,mut b)=(t.psi,t.psi);
        let(_,_,one)=coarse.evolve(&mut a,t.time,s.dt,0.);let(_,_,two)=fine.evolve(&mut b,t.time,s.dt,0.);
        let(x,y)=(norm2(&a,n),norm2(&b,n));
        Some([(0..n).map(|i|(a[i].norm_sqr()/x-b[i].norm_sqr()/y).abs()).fold(0.,f64::max),(0..3).map(|k|(one[k]-two[k]).powi(2)).sum::<f64>().sqrt(),two.iter().map(|v|v*v).sum::<f64>().sqrt()])
    }
    /// The state a jump leaves a trajectory in, on a copy (`jump`, its own stream): a bare state, whose amplitudes ring at the drives' detunings where the trajectory's own have relaxed (solve.rs `grid_phase`).
    pub fn landed(&self,t:&Trajectory)->Trajectory{let mut x=t.clone();let(r,time)=(x.r,x.time);self.jump(&mut x,r,time,false);x}
    /// One step's landings from a trajectory's state (`Landings`) on a grid of half the spacing at `phase` and on the grid at it, none where its cluster takes no beating grid (the first is then its own
    /// propagation); `draws` the reads' places, each in its equal share of the step, in [0, 1), the same on both (solve.rs `landed_excess`).
    pub fn grid_landings(&self,t:&Trajectory,phase:f64,groups:&[usize],count:usize,draws:&[f64])->(Landings,Option<Landings>){
        let(m,s)=(self.model,&self.settings);let point=m.point(t.r,t.kappa);let c=self.on_path(t.cluster,&point,&t.psi,t.time,s.dt,self.reference(t.r));
        let at=|c:Components,phase:f64|Segment::new(c,t.time,s.dt,s.truncation,s.beats,s.horizon,phase);
        let coarse=m.beating(t.cluster).then(||at(c.clone(),phase)).filter(|x|matches!(x.how,How::Beating(_)));
        let fine=match &coarse{Some(x)=>x.refined(s.dt,s.truncation),None=>at(c,0.5*phase)};
        (self.landings(t,&point,&fine,groups,count,draws),coarse.map(|x|self.landings(t,&point,&x,groups,count,draws)))
    }
    /// A step's landings through a segment: at each read the photons of the state the evolution holds there, which a jump would take -- off a copy from the last whole panel before it, as a jump's
    /// (`Segment::evolve`), the grid's own path going on untouched: reads that broke its panels refined the grid they read (F = 1 on a PGC's explicit rung at phase 9.9, 0.06 at 4 reads, 0.0003 at 32).
    fn landings(&self,t:&Trajectory,point:&Point,segment:&Segment,groups:&[usize],count:usize,draws:&[f64])->Landings{
        let(m,s)=(self.model,&self.settings);let n=m.dimension(t.cluster);let g=&m.ground;let cl=&g.clusters[t.cluster];
        let size=norm2(&t.psi,n).sqrt();let mut x=t.psi;for z in x[..n].iter_mut(){*z=*z*(1./size);}
        let of=|x:&[Complex]|{let mut p=vec![0.;count];for(a,&state)in cl.states.iter().enumerate(){p[groups[state]]+=x[a].norm_sqr();}p};
        let start=of(&x);let from=(0..count).max_by(|&a,&b|start[a].total_cmp(&start[b])).unwrap();
        let(mut at,panel,mut sources)=(t.time,2.*segment.spacing,vec![0.;count]);
        for(k,place)in draws.iter().enumerate(){
            let to=t.time+s.dt*(k as f64+place)/draws.len() as f64;let node=t.time+((to-t.time)/panel).floor()*panel;
            if node>at{segment.evolve(&mut x,at,node-at,0.);at=node;}
            let mut y=x;if to>at{segment.evolve(&mut y,at,to-at,0.);}
            // The photons at the place the path reaches at the read: the amplitude y holds the beams' phases there.
            let there=m.point(std::array::from_fn(|a|point.r[a]+m.hbar_over_mass*point.kappa[a]*(to-t.time-0.5*s.dt)),point.kappa);
            let photons=m.photons(t.cluster,&there,&y,to,None);
            for(class,u)in cl.classes.iter().zip(&photons.amplitude){for(row,&state)in g.clusters[class.target].states.iter().enumerate(){sources[groups[state]]+=(0..3).map(|a|u[a][row].norm_sqr()).sum::<f64>();}}
        }
        segment.evolve(&mut x,at,t.time+s.dt-at,0.);
        Landings{from,sources,stay:of(&x),left:norm2(&x,n)}
    }
    /// Whether a cluster takes the exact step: one with explicit excited states or a beating generator
    /// (a held atom's clusters are of ground states and static, control.rs).
    pub(crate) fn takes_exact(&self,cluster:usize)->bool{self.held.is_none()&&(self.model.mixed(cluster)||(self.model.beating(cluster)&&self.quiet[cluster].is_none()))}
    /// The mechanical energy of a trajectory's state (rad/us): ħ²κ²/2m and its potential, the
    /// Hermitian part of its generator's expectation (the trap's and the light's, the force's).
    pub fn energy(&self,t:&Trajectory)->f64{
        let m=self.model;let g=m.generator(t.cluster,&m.point(t.r,t.kappa),t.time);let n=g.n;let norm=norm2(&t.psi,n);
        let mut e=Complex::ZERO;for a in 0..n{for b in 0..n{e+=t.psi[a].conj()*g.hamiltonian[a*n+b]*t.psi[b];}}
        0.5*m.hbar_over_mass*t.kappa.iter().map(|k|k*k).sum::<f64>()+g.scalar+e.re/norm
    }
    /// One jump-free step of dt from a trajectory's state, by the step its cluster takes: the norms
    /// at its start and end and its kick.
    fn free_step(&self,t:&mut Trajectory,dt:f64)->(f64,f64,[f64;3]){
        if self.takes_exact(t.cluster){
            let n=self.model.dimension(t.cluster);let n0=norm2(&t.psi,n);let kick=self.exact(t,dt,false,false);(n0,norm2(&t.psi,n),kick)
        }else{let(n0,_,n1,_,kick,_)=self.advance(t,dt,0.);(n0,n1,kick)}
    }
    /// A trajectory's state after `count` jump-free steps of dt (the deterministic flow the step check
    /// compares): flow, kick and the internal state's no-jump evolution, no jump, no loss.
    pub fn coast(&self,t:&Trajectory,dt:f64,count:usize)->Trajectory{let mut x=t.clone();for _ in 0..count{self.free_step(&mut x,dt);}x}
    /// The step-doubling error of one step of dt from a trajectory's state, without jumps: one step
    /// against two of dt/2, as (|Δκ|, |kick|) -- the momentum's difference and the step's
    /// non-harmonic impulse (1/um) -- and (max |Δp|, |Δ ln‖ψ‖²|, |ln‖ψ‖²|) -- the largest difference of
    /// the cluster's state populations, of the norm's decay and the decay itself; and the one step's end (r, κ).
    pub fn doubling(&self,t:&Trajectory,dt:f64)->([f64;2],[f64;3],([f64;3],[f64;3])){
        let n=self.model.dimension(t.cluster);
        let mut one=t.clone();let(n0,n1,kick)=self.free_step(&mut one,dt);
        let mut two=t.clone();self.free_step(&mut two,0.5*dt);let(_,n2,_)=self.free_step(&mut two,0.5*dt);
        let dk=(0..3).map(|a|(one.kappa[a]-two.kappa[a]).powi(2)).sum::<f64>().sqrt();let impulse=kick.iter().map(|x|x*x).sum::<f64>().sqrt();
        let(a,b)=(norm2(&one.psi,n),norm2(&two.psi,n));
        let dp=(0..n).map(|i|(one.psi[i].norm_sqr()/a-two.psi[i].norm_sqr()/b).abs()).fold(0.,f64::max);
        ([dk,impulse],[dp,((n1/n0).ln()-(n2/n0).ln()).abs(),(n1/n0).ln().abs()],(one.r,one.kappa))
    }

    /// One step of dt: flow, kick and flow (`advance`), or the exact step in a cluster with explicit
    /// excited states. Where ‖ψ‖² crosses the threshold, the step is redone to the crossing, the jump
    /// made there and the rest of the step taken from it in the state the jump left -- exactly when
    /// it lands in a cluster with explicit states: the force and the motion follow the internal state
    /// on either side of the jump. Then the loss.
    pub fn step(&self,t:&mut Trajectory){
        let(m,s)=(self.model,&self.settings);
        // The first order in the Doppler shift where it holds, by the step's start; the exact response beyond: read at its jumps and by an exact step alone.
        let kappa=t.kappa;let exact=||m.doppler_ratio(kappa)>s.doppler;
        // The state the loss is judged at: the step's middle where it went without a jump -- the point
        // the flow's first half reached, κ there with half the kick, ψ after the first half step, the
        // middle's time --, else its end.
        let(place,middle)=if self.takes_exact(t.cluster){self.exact(t,s.dt,exact(),true);((t.r,t.kappa),None)}else{
            let mut left=s.dt;
            if let Some(Some(dropped))=self.quiet.get(t.cluster){t.beats=t.beats.max(dropped/s.dt);}
            loop{
                let start=t.clone();
                let threshold=t.threshold;let(n0,nh,n1,place,kick,half)=self.advance(t,left,threshold);
                if n1>t.threshold{let kappa:[f64;3]=std::array::from_fn(|a|place.1[a]+0.5*kick[a]);break(place,Some((kappa,half,t.time-0.5*left)));}
                // The crossing on ln‖ψ‖², linear on the half where it lies; the step redone to it.
                let(ta,tb,la,lb)=if nh<=t.threshold{(0.,0.5*left,n0.ln(),nh.ln())}else{(0.5*left,left,nh.ln(),n1.ln())};
                let at=if lb<la{(ta+(t.threshold.ln()-la)/(lb-la)*(tb-ta)).clamp(ta,tb)}else{tb};
                *t=start;if at>0.{self.advance(t,at,0.);}
                self.jump(t,t.r,t.time,exact());
                left-=at;
                if !(left>0.){break((t.r,t.kappa),None);}
                if self.takes_exact(t.cluster){self.exact(t,left,exact(),true);break((t.r,t.kappa),None);}
            }
        };
        // The loss: an atom unbound -- its mechanical energy above the trap's far level -- outside the trap's
        // region (`Settings::region`), of that one state.
        if s.loss&&self.harmonic_potential({let c=s.about();std::array::from_fn(|a|place.0[a]-c[a])})>s.region{
            let point=m.point(place.0,place.1);
            let(kappa,psi,time)=match &middle{Some((kappa,psi,time))=>(*kappa,psi,*time),None=>(t.kappa,&t.psi,t.time)};
            let kinetic=0.5*m.hbar_over_mass*kappa.iter().map(|k|k*k).sum::<f64>();
            if kinetic+m.trap_bound(t.cluster,&point,psi)>0.&&kinetic+m.trap_potential(t.cluster,&point,psi,time)>0.{t.alive=false;}
        }
    }

    /// A quantum jump at r and time `time` with the trajectory's (unnormalized) state, with the exact
    /// response when `exact`; then the post-jump transient.
    pub fn jump(&self,t:&mut Trajectory,r:[f64;3],time:f64,exact:bool){
        let m=self.model;t.jumps+=1;let mut rng=jump_stream(t.seed,t.index,t.jumps);
        let n=m.dimension(t.cluster);let norm=norm2(&t.psi,n).sqrt();
        let mut psi=t.psi;for z in psi[..n].iter_mut(){*z=*z*(1./norm);}
        let classes=&m.ground.clusters[t.cluster].classes;
        // Each class's amplitudes u_x = J_{φ,x} ψ: from the photons at the point, or a held atom's
        // jump operators; the photons kept for the kicks of a moving one.
        let point=self.held.as_ref().map_or_else(||m.point(r,t.kappa),|(h,_)|h.point);
        let photons=self.held.is_none().then(||{let responses=if exact{Exact::new(m,r,t.kappa).ok()}else{None};m.photons(t.cluster,&point,&psi,time,responses)});
        let held:Vec<[[Complex;MOST_STATES];3]>=self.held.as_ref().map_or(vec![],|(h,_)|h.jumps[t.cluster].iter().zip(classes).map(|(l,c)|{let nt=m.ground_dimension(c.target);
            std::array::from_fn(|x|{let mut u=[Complex::ZERO;MOST_STATES];for i in 0..nt{for j in 0..n{u[i]+=l[x][i*n+j]*psi[j];}}u})}).collect());
        let amplitude=photons.as_ref().map_or(&held,|p|&p.amplitude);
        // The class, and the cluster it lands the atom in.
        let weights:Vec<f64>=classes.iter().zip(amplitude).map(|(c,u)|{let nt=m.ground_dimension(c.target);(0..3).map(|a|norm2(&u[a],nt)).sum()}).collect();
        let total:f64=weights.iter().sum();
        if !(total>0.){t.threshold=rng.open_uniform();return;}
        // The draws the state and the kicks depend on first -- the class, the next threshold, the
        // polarization, the absorbed beam, the transient's --, the directions' rejection loops after
        // them: a twin trajectory on the same stream (control.rs) keeps its thresholds and choices
        // whatever its loops take.
        let(mut x,threshold,polarization,absorbed)=(rng.uniform()*total,rng.open_uniform(),rng.uniform(),rng.uniform());
        let mut pick=weights.len()-1;
        for(i,&w)in weights.iter().enumerate(){if x<w{pick=i;break;}x-=w;}
        let class=&classes[pick];let nt=m.ground_dimension(class.target);
        let transient:Option<[f64;5]>=self.held.is_none().then(||std::array::from_fn(|_|rng.uniform()));
        let(direction,e,f,size)=emit(&mut rng,&amplitude[pick],nt,polarization);
        // The recoils of a moving atom: the photon's, the absorbed beam's and the exact conditional
        // mean's correction.
        if let Some(photons)=&photons{
            let(gradient,parts)=photons.absorption(pick,e,&f);
            self.kick(t,direction,class.wave_number,&f,size,&gradient,&parts,absorbed);
        }
        let mut next=[Complex::ZERO;MOST_MIXED];for i in 0..nt{next[i]=f[i]*(1./size.sqrt());}
        t.cluster=class.target;t.psi=next;t.threshold=threshold;
        if let Some(draws)=transient{self.transient(t,&point,time,draws,&mut rng);}
    }

    /// A moving atom's recoils from a photon (direction n, wave number k, the state it leaves f and
    /// ‖f‖², the gradient of its amplitudes along the polarization ε per row and axis, and its absorbed
    /// light's parts): −k n, and the absorbed light's with the exact conditional mean Im⟨f|∇f⟩/‖f‖²
    /// (`absorbed_recoil`, by the uniform `absorbed`).
    fn kick(&self,t:&mut Trajectory,direction:[f64;3],k:f64,f:&[Complex;MOST_STATES],size:f64,du:&[[Complex;3];MOST_STATES],parts:&Absorption,absorbed:f64){
        let m=self.model;let mut kick=direction.map(|x|-k*x);
        let mean:[f64;3]=std::array::from_fn(|x|{let mut s=0.;for i in 0..MOST_STATES{s+=(f[i].conj()*du[i][x]).im;}s/size});
        let beams:Vec<([f64;3],f64,f64)>=parts.beams.iter().map(|&(l,s,w)|(m.geometry[l].wave_vector(),s,w)).collect();
        let drives=parts.explicit.as_ref().map(|(_,d)|d.iter().map(|&(l,w)|(m.geometry[l].wave_vector(),w)).collect::<Vec<_>>());
        let explicit=parts.explicit.as_ref().zip(drives.as_deref()).map(|((w,_),d)|(*w,d));
        let recoil=absorbed_recoil(&beams,explicit,mean,absorbed);
        for a in 0..3{kick[a]+=recoil[a];}
        for a in 0..3{if self.settings.free[a]{t.kappa[a]+=kick[a];}}
    }

    /// The post-jump transient (module documentation) from the normalized ground state ψ′ the jump left: with probability p = ‖Xψ′‖²
    /// the ringing's photon (`ringing_photon`), and the slow part of the state it leaves; else at fourth order its slow part Πψ′/√A with the threshold θ (1 − p)/A (`slow_start`'s, A = ‖Πψ′‖²).
    /// `draws`: whether, the photon's time, its landing cluster, its polarization, its beam.
    fn transient(&self,t:&mut Trajectory,point:&Point,time:f64,draws:[f64;5],rng:&mut Rng){
        let m=self.model;let k=t.cluster;let psi=t.psi;let n0=m.ground_dimension(k);
        let p=match m.slow_and_metric(k,point,time){
            Some((pi,x))=>{
                let form=(0..n0).map(|a|psi[a].conj()*(0..n0).fold(Complex::ZERO,|s,b|s+x[a*n0+b]*psi[b])).fold(Complex::ZERO,|s,z|s+z).re;
                if draws[0]>=form{
                    let mut y=[Complex::ZERO;MOST_MIXED];for a in 0..n0{for b in 0..n0{y[a]+=pi[a*n0+b]*psi[b];}}
                    let a=norm2(&y,n0);for z in y[..n0].iter_mut(){*z=*z*(1./a.sqrt());}t.psi=y;t.threshold*=(1.-form)/a;return;
                }
                form
            }
            None=>{if draws[0]>=m.ringing_bound(k,point,&psi){return;}0.}
        };
        let v=m.ringing_amplitudes(k,&m.responses(k,point),&psi,time);
        let total:f64=v.iter().map(|v|v.iter().map(|z|z.norm_sqr()).sum::<f64>()).sum();
        // At fourth order the draw is within the metric's form, X on the fourth order's carriers; without it within the bound, and the amplitudes' own weight decides.
        let within=if p>0.{draws[0]*total/p}else{draws[0]};
        if within<total{self.ringing_photon(t,point,time,&v,within,draws,rng);slow_start(m,t,point,time);}
    }
    /// The ringing's photon (module documentation) from ψ′, v its amplitudes per coherent group: the group and the manifold by their weights (`x` within their sum, the
    /// whether's uniform), the time τ ~ Γ e^{−Γτ} at the manifold's width, the amplitude C_a e^{−iHτ} v, its landing cluster by the weights there, its kicks a jump's.
    fn ringing_photon(&self,t:&mut Trajectory,point:&Point,time:f64,v:&[Vec<Complex>],x:f64,draws:[f64;5],rng:&mut Rng){
        let m=self.model;let g=&m.ground;let k=t.cluster;let psi=t.psi;
        let manifolds=&g.excited_groups;let weight=|c:usize,e:usize|manifolds[e].iter().map(|&i|v[c][i].norm_sqr()).sum::<f64>();
        let(mut x,mut pick)=(x,(0,0));
        'pick:for c in 0..v.len(){for e in 0..manifolds.len(){let w=weight(c,e);if w>0.{pick=(c,e);if x<w{break 'pick;}x-=w;}}}
        let(c,e)=pick;let levels=&manifolds[e];
        let ring=m.ringing(k,point.r,e);let tau=-(1.-draws[1]).ln()/ring.width;let ops=m.ringing_photon(&ring,tau,time);
        // Its amplitude on each cluster's ground states; the cluster by the weights.
        let amplitude=|target:usize|->[[Complex;MOST_STATES];3]{std::array::from_fn(|a|{let mut u=[Complex::ZERO;MOST_STATES];
            for(row,&to)in g.clusters[target].states.iter().enumerate(){for &i in levels{u[row]+=ops[a][(to,i)]*v[c][i];}}u})};
        let photons:Vec<[[Complex;MOST_STATES];3]>=(0..g.clusters.len()).map(amplitude).collect();
        let weights:Vec<f64>=photons.iter().enumerate().map(|(target,u)|(0..3).map(|a|norm2(&u[a],m.ground_dimension(target))).sum()).collect();
        let total:f64=weights.iter().sum();if !(total>0.){return;}
        let mut y=draws[2]*total;let mut target=weights.len()-1;for(i,&w)in weights.iter().enumerate(){if y<w{target=i;break;}y-=w;}
        let nt=m.ground_dimension(target);
        let(direction,polar,f,size)=emit(rng,&photons[target],nt,draws[3]);
        let(_,du,beams)=m.transient_photon(k,point,time,c,levels,target,&ops,&psi);
        // The beams' parts and the photon's gradient along the polarization (`kick`), of the photon's own sign:
        // transient_photon's amplitude is Σ e^{−iε̃t} R ψ′, the photon its negative (X = −Σ e^{−iε̃t} R, model.rs `transient`).
        let parts=Absorption{beams:beams.iter().map(|(l,u)|{let along=|i:usize|(0..3).fold(Complex::ZERO,|s,a|s+u[a][i]*polar[a]);
            (*l,(0..nt).map(|i|along(i).norm_sqr()).sum(),-(0..nt).map(|i|(f[i].conj()*along(i)).re).sum::<f64>()/size)}).collect(),explicit:None};
        let along:[[Complex;3];MOST_STATES]=std::array::from_fn(|i|std::array::from_fn(|x|-(0..3).fold(Complex::ZERO,|s,a|s+du[a][x][i]*polar[a])));
        self.kick(t,direction,m.carriers[g.clusters[k].coherent[c][0].0.carrier].wave_number,&f,size,&along,&parts,draws[4]);
        let mut next=[Complex::ZERO;MOST_MIXED];for i in 0..nt{next[i]=f[i]*(1./size.sqrt());}
        t.cluster=target;t.psi=next;
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    use crate::linear::uniform;
    use std::f64::consts::TAU;

    /// A kick's mean and second moment per axis over the draw's uniform (a low-discrepancy sequence).
    fn over_the_draw(kick:impl Fn(f64)->[f64;3])->([f64;3],[f64;3]){
        let n=1usize<<16;let(mut m,mut s)=([0.;3],[0.;3]);
        for i in 0..n{let k=kick(((i as f64+0.5)*0.6180339887498949).fract());for a in 0..3{m[a]+=k[a]/n as f64;s[a]+=k[a]*k[a]/n as f64;}}
        (m,s)
    }
    fn close(a:[f64;3],b:[f64;3],within:f64)->bool{(0..3).all(|x|(a[x]-b[x]).abs()<=within)}
    /// A tetrahedral MOT's three beam pairs, all of one |k_z|.
    fn pairs()->Vec<[f64;3]>{let k=8.05;
        [[0.577,0.,0.816],[-0.577,0.,-0.816],[-0.289,0.5,0.816],[0.289,-0.5,-0.816],[-0.289,-0.5,0.816],[0.289,0.5,-0.816]].iter().map(|n:&[f64;3]|n.map(|x|k*x)).collect()}

    /// A running wave's photon takes its recoil in either model: folded ħk each, its mean the recoil; explicit ±ħk
    /// about the Ehrenfest force's mean, zero at the jump -- the same second moment, the recoil formula's.
    #[test]
    fn a_running_wave_s_photon_takes_its_recoil_in_either_model(){
        let k=[3.,0.,7.];
        for u in [0.01,0.4,0.99]{assert!(close(absorbed_recoil(&[(k,0.3,1.)],None,k,u),k,1e-12));}
        let(mean,second)=over_the_draw(|u|absorbed_recoil(&[],Some((1.,&[(k,0.7)])),[0.;3],u));
        assert!(close(mean,[0.;3],1e-3),"{mean:?}");assert!(close(second,[9.,0.,49.],1e-9),"{second:?}");
    }
    /// The draw over a photon's beams: without interference (the projections the shares) the shares, the second moment
    /// theirs; with a negative projection the mean is reached by nonnegative weights, the kicks the beams' own and k_z² on an
    /// axis of one |k_z|; beyond the recoils' hull the rest of the mean is added, and the envelopes' part always -- the
    /// mean exact in each.
    #[test]
    fn the_absorbed_recoil_keeps_the_exact_mean_and_the_beams_recoils(){
        let k=pairs();let of=|s:&[f64],w:&[f64]|->Vec<([f64;3],f64,f64)>{(0..6).map(|c|(k[c],s[c],w[c])).collect()};
        let mixture=|w:&[f64]|->[f64;3]{std::array::from_fn(|a|(0..6).map(|c|w[c]*k[c][a]).sum::<f64>())};
        let s=[0.31,0.05,0.22,0.11,0.2,0.11];let b=of(&s,&s);let m=mixture(&s);let q=part_weights(&s,&k,m);
        for c in 0..6{assert!((q[c]-s[c]).abs()<=1e-12,"{q:?}");}
        let(mean,second)=over_the_draw(|u|absorbed_recoil(&b,None,m,u));assert!(close(mean,m,2e-3),"{mean:?}");
        for a in 0..3{let incoherent=(0..6).map(|c|s[c]*k[c][a]*k[c][a]).sum::<f64>();assert!((second[a]-incoherent).abs()<=2e-2,"{second:?}");}
        // A part interferes destructively: the mean within the hull is reached by the beams' own kicks.
        let w=[0.45,-0.02,0.2,0.05,0.2,0.12];let b=of(&s,&w);let m=mixture(&w);let q=part_weights(&s,&k,m);
        assert!(q.iter().all(|&x|x>=0.)&&(q.iter().sum::<f64>()-1.).abs()<=1e-12,"{q:?}");
        assert!(close(std::array::from_fn(|a|(0..6).map(|c|q[c]*k[c][a]).sum::<f64>()),m,1e-9),"{q:?}");
        let(mean,second)=over_the_draw(|u|{let kick=absorbed_recoil(&b,None,m,u);assert!(k.iter().any(|x|close(*x,kick,1e-9)),"{kick:?}");kick});
        assert!(close(mean,m,2e-3),"{mean:?}");assert!((second[2]-k[0][2]*k[0][2]).abs()<=1e-9,"{second:?}");
        // A beam of negligible share takes a negligible part of the move.
        let tiny=[0.5,1e-14,0.2,0.1,0.1,0.1];let q=part_weights(&tiny,&k,mixture(&[0.5,0.,0.25,0.05,0.1,0.1]));assert!(q[1]<=1e-9,"{q:?}");
        // Two counter-propagating beams by a dark point (10, −9): 19 ħk, beyond them: the beam toward it and the rest.
        let two=[([0.,0.,8.],0.55,10.),([0.,0.,-8.],0.45,-9.)];
        let target=[0.,0.,152.];assert!(part_weights(&[0.55,0.45],&[two[0].0,two[1].0],target)[1]==0.);
        for u in [0.1,0.9]{assert!(close(absorbed_recoil(&two,None,target,u),target,1e-9));}
        // What the exact mean holds beyond the beams' phases is a deterministic kick.
        let extra=[0.3,-0.2,0.1];let(mean,_)=over_the_draw(|u|absorbed_recoil(&b,None,std::array::from_fn(|a|m[a]+extra[a]),u));
        assert!(close(mean,std::array::from_fn(|a|m[a]+extra[a]),2e-3),"{mean:?}");
    }
    /// Two beams along one axis on carriers apart, by a dark point (the probed PGC's jump that once parted the devices: shares of 1e-9, the mean beyond the recoils' hull): whatever their split
    /// below `RECOIL_SPREAD` and however the recoils lie, the weights end on the hull's edge from them to the beam across, at the foot of the mean on it, and neither beam is lost: the direction between the two takes no step.
    #[test]
    fn beams_along_one_axis_are_one_direction(){
        let k=8.05282506255877;let s=[9.173492148945024e-10,9.90160465550982e-10,0.2665137905681378,0.1858455805747167,0.18667338805763312,0.36096723889200283,0.0];
        let target=[5.1052716918023044,-1.2922150921409203,5.606014164812307];let axis=[1./14_f64.sqrt(),2./14_f64.sqrt(),3./14_f64.sqrt()];
        for angle in [0_f64,0.3,1.1,2.9]{
            // Rodrigues' rotation: the frames in which C's zero eigenvalues are its round-off.
            let turn=|v:[f64;3]|->[f64;3]{let(sin,cos)=angle.sin_cos();let dot=(0..3).map(|a|axis[a]*v[a]).sum::<f64>();
                let cross=[axis[1]*v[2]-axis[2]*v[1],axis[2]*v[0]-axis[0]*v[2],axis[0]*v[1]-axis[1]*v[0]];std::array::from_fn(|a|v[a]*cos+cross[a]*sin+axis[a]*dot*(1.-cos))};
            for split in [9.47e-8,3e-7,1e-6,1e-5,1e-4]{
                let mu=[[k*(1.+split),0.,0.],[k,0.,0.],[-k,0.,0.],[0.,k,0.],[0.,-k,0.],[0.,0.,k],[0.,0.,0.]].map(turn);
                let q=part_weights(&s,&mu,turn(target));let foot=(target[0]-target[2]+k)/(2.*k);
                assert!(q[0]>0.&&q[1]>0.,"{angle} {split}: {q:?}");assert!(q[2]==0.&&q[3]==0.&&q[4]==0.&&q[6]==0.,"{angle} {split}: {q:?}");
                assert!((q[0]+q[1]-foot).abs()<=1e-4&&(q[5]-(1.-foot)).abs()<=1e-4,"{angle} {split}: {q:?} against {foot}");
            }
        }
    }
    /// A photon of folded beams and explicit states: the mean is the folded part's whatever the explicit part weighs (its
    /// kicks have mean zero, the Ehrenfest force carrying its mean); the explicit part's second moment is its drives'.
    #[test]
    fn the_explicit_part_s_kicks_add_no_mean(){
        let k=pairs();let drives:Vec<([f64;3],f64)>=k.iter().copied().zip([0.4,0.1,0.2,0.1,0.15,0.05]).collect();
        let beams=[(k[0],0.02,0.02),(k[1],0.01,-0.01)];let mean:[f64;3]=std::array::from_fn(|a|0.02*k[0][a]-0.01*k[1][a]);
        let(got,_)=over_the_draw(|u|absorbed_recoil(&beams,Some((0.97,&drives)),mean,u));assert!(close(got,mean,3e-3),"{got:?}");
        let(got,second)=over_the_draw(|u|absorbed_recoil(&[],Some((1.,&drives)),[0.;3],u));assert!(close(got,[0.;3],3e-3),"{got:?}");
        let total:f64=drives.iter().map(|d|d.1).sum();
        for a in 0..3{let incoherent=drives.iter().map(|d|d.1*d.0[a]*d.0[a]).sum::<f64>()/total;assert!((second[a]-incoherent).abs()<=2e-2,"{second:?}");}
    }
    /// A random Hermitian n × n matrix of entries up to `size`.
    fn hermitian(next:&mut impl FnMut()->f64,n:usize,size:f64)->Vec<Complex>{
        let mut h=vec![Complex::ZERO;n*n];
        for i in 0..n{for j in 0..=i{let z=Complex::new(next()*size,next()*size);if i==j{h[i*n+i]=Complex::new(z.re,0.);}else{h[i*n+j]=z;h[j*n+i]=z.conj();}}}
        h
    }
    /// A random generator of n states (an H_eff's shape: Hermitian, its energies spread, less (i/2) a
    /// decay on the diagonal) with Hermitian beats at ±Ω for each of `omegas` (H_{−Ω} = H_Ω†, so the norm
    /// decays monotonically), and its gradients, the beats' paired (G_{−Ω} = G_Ω†).
    fn random(n:usize,omegas:&[f64],seed:u64)->Components{
        let mut next=uniform(seed);
        let mut h=hermitian(&mut next,n,60.);for i in 0..n{h[i*n+i]+=Complex::new(80.*next(),-12.*(next()+0.5));}
        let gradient=std::array::from_fn(|_|hermitian(&mut next,n,30.));
        let raw:Vec<(Vec<Complex>,[Vec<Complex>;3])>=omegas.iter().map(|_|{let b:Vec<Complex>=(0..n*n).map(|_|Complex::new(next()*20.,next()*20.)).collect();
            (b,std::array::from_fn(|_|(0..n*n).map(|_|Complex::new(next()*10.,next()*10.)).collect()))}).collect();
        let adjoint=|x:&[Complex]|->Vec<Complex>{(0..n*n).map(|k|x[(k%n)*n+k/n].conj()).collect()};
        let(mut hs,mut gs)=(vec![h],vec![gradient]);
        for(b,g)in raw.iter().rev(){hs.push(adjoint(b));gs.push(std::array::from_fn(|a|adjoint(&g[a])));}
        for(b,g)in raw{hs.push(b);gs.push(g);}
        let matrices:Vec<Complex>=hs.into_iter().flatten().chain(gs.into_iter().flat_map(|g|g.into_iter().flatten())).collect();
        let beats:Vec<f64>=omegas.iter().rev().map(|&w|-w).chain(omegas.iter().copied()).collect();
        Components{n,scalar:0.,parts:1+beats.len(),places:(1..=beats.len()).collect(),omegas:beats,matrices,gauge:[0.;MOST_MIXED],middle:0.,gauged:false}
    }
    /// H_q of a test's components, before any beat is left out.
    fn h_of(c:&mut Components,q:usize)->&mut [Complex]{let nn=c.n*c.n;&mut c.matrices[q*nn..(q+1)*nn]}
    /// G_q along an axis, likewise.
    fn g_of(c:&mut Components,q:usize,a:usize)->&mut [Complex]{let nn=c.n*c.n;let at=(c.parts+3*q+a)*nn;&mut c.matrices[at..at+nn]}
    fn state(n:usize,seed:u64)->[Complex;MOST_MIXED]{
        let mut next=uniform(seed);let mut psi=[Complex::ZERO;MOST_MIXED];for z in psi[..n].iter_mut(){*z=Complex::new(next(),next());}
        let norm=norm2(&psi,n).sqrt();for z in psi[..n].iter_mut(){*z=*z*(1./norm);}psi
    }
    /// ψ(t₀ + τ) for ψ' = −iH(t)ψ by RK4 in `steps` steps, and ∫ ⟨ψ|F|ψ⟩/⟨ψ|ψ⟩ dt by Simpson on them.
    fn reference(c:&Components,psi:&[Complex;MOST_MIXED],t0:f64,tau:f64,steps:usize)->([Complex;MOST_MIXED],[f64;3]){
        let n=c.n();let h=tau/steps as f64;
        let derivative=|t:f64,y:&[Complex;MOST_MIXED]|->[Complex;MOST_MIXED]{let g=c.at(t);let mut d=[Complex::ZERO;MOST_MIXED];
            for a in 0..n{let s=(0..n).fold(Complex::ZERO,|s,b|s+g.hamiltonian[a*n+b]*y[b]);d[a]=Complex::new(s.im,-s.re);}d};
        let force=|t:f64,y:&[Complex;MOST_MIXED]|c.at(t).force(y);
        let mut y=*psi;let mut impulse=[0.;3];let mut fa=force(t0,&y);
        let add=|y:&[Complex;MOST_MIXED],k:&[Complex;MOST_MIXED],s:f64|->[Complex;MOST_MIXED]{let mut o=*y;for a in 0..n{o[a]+=k[a]*s;}o};
        let rk4=|t:f64,y:&[Complex;MOST_MIXED],h:f64|{let k1=derivative(t,y);let k2=derivative(t+0.5*h,&add(y,&k1,0.5*h));let k3=derivative(t+0.5*h,&add(y,&k2,0.5*h));let k4=derivative(t+h,&add(y,&k3,h));
            let mut o=*y;for a in 0..n{o[a]+=(k1[a]+k2[a]*2.+k3[a]*2.+k4[a])*(h/6.);}o};
        for i in 0..steps{let t=t0+i as f64*h;let ym=rk4(t,&y,0.5*h);let yb=rk4(t+0.5*h,&ym,0.5*h);let(fm,fb)=(force(t+0.5*h,&ym),force(t+h,&yb));
            for a in 0..3{impulse[a]+=h/6.*(fa[a]+4.*fm[a]+fb[a]);}y=yb;fa=fb;}
        (y,impulse)
    }
    fn distance(a:&[Complex],b:&[Complex],n:usize)->f64{(0..n).map(|i|(a[i]-b[i]).norm_sqr()).sum::<f64>().sqrt()/norm2(b,n).sqrt()}

    /// A segment of a beating generator at a given spacing, its series to double precision.
    fn beating(c:Components,start:f64,spacing:f64)->Segment{let how=How::Beating(Entries::new(&c,f64::EPSILON));Segment{components:c,start,spacing,how,dropped:0.}}
    /// CF4 is exact for a static generator and of fourth order for a beating one: its error falls 16-fold
    /// with the spacing halved, against RK4 at a far finer step; and at the node rule's spacing it is far
    /// below the impulse's quadrature share.
    #[test]
    fn cf4_is_exact_for_a_static_generator_and_of_fourth_order_with_beats(){
        let n=5;let tau=0.1;let psi=state(n,3);
        let still=random(n,&[],7);
        let exact=Segment::new(still.clone(),0.,tau,0.,0.,tau,node_phase());let mut a=psi;exact.evolve(&mut a,0.,tau,0.);
        let cf4=beating(still,0.,tau/8.);let mut b=psi;cf4.evolve(&mut b,0.,tau,0.);
        assert!(distance(&b,&a,n)<=1e-12,"static: {:e}",distance(&b,&a,n));
        let c=random(n,&[150.,300.],11);let(target,_)=reference(&c,&psi,0.,tau,20000);
        let error=|steps:usize|{let s=beating(c.clone(),0.,tau/steps as f64);let mut x=psi;s.evolve(&mut x,0.,tau,0.);distance(&x,&target,n)};
        let e:Vec<f64>=[32,64,128].iter().map(|&s|error(s)).collect();
        assert!(e[1]/e[2]>12.&&e[1]/e[2]<20.,"errors {e:?}: ratio {}",e[1]/e[2]);
        let rule=Segment::new(c.clone(),0.,tau,0.,0.,tau,node_phase());let mut x=psi;rule.evolve(&mut x,0.,tau,0.);
        assert!(distance(&x,&target,n)<=1e-6,"at the node rule's spacing {}: {:e}",rule.spacing,distance(&x,&target,n));
    }

    /// A beating grid on its entries gives the dense evolution (RK4 at a far finer step): the state and the impulse, from the
    /// step's start and from a time between nodes (after a jump), its series' share within the state's error there.
    #[test]
    fn a_beating_grid_on_its_entries_gives_the_dense_evolution(){
        let n=5;let(start,tau)=(0.37,0.1);let mut c=random(n,&[170.,410.],43);
        // A cluster's pattern: most entries zero, and some of the round-off's size.
        for q in 1..c.parts{for k in 0..n*n{if (k/n+k%n)%2==0{h_of(&mut c,q)[k]=Complex::ZERO;for a in 0..3{g_of(&mut c,q,a)[k]=Complex::ZERO;}}}}
        for a in 0..n{for b in 0..n{if a!=b&&(a+2*b)%3==0{let z=Complex::new(1e-30,1e-31);let h=h_of(&mut c,0);h[a*n+b]=z;h[b*n+a]=z.conj();}}}
        let s=Segment::new(c.clone(),start,tau,1e-9,0.,tau,node_phase()/8.);
        let How::Beating(e)=&s.how else{panic!("the beats are kept")};assert!(e.part(0).len()<n*n&&e.part(1).len()<n*n,"{} and {} of {}",e.part(0).len(),e.part(1).len(),n*n);
        for offset in [0.,0.0123]{
            let psi=state(n,9);let(target,reference)=reference(&c,&psi,start+offset,tau-offset,20000);
            let mut a=psi;let(_,_,mine)=s.evolve(&mut a,start+offset,tau-offset,0.);
            assert!(distance(&a,&target,n)<=1e-8,"offset {offset}: {:e}",distance(&a,&target,n));
            let size=reference.iter().map(|x|x.abs()).fold(0.,f64::max);
            for k in 0..3{assert!((mine[k]-reference[k]).abs()<=1e-6*size,"offset {offset}, axis {k}: {} against {}",mine[k],reference[k]);}
        }
    }

    /// A half step by scaling and squaring where its series is long is the series, within the truncation: a stiff generator of frames apart, ‖H‖τ of 100, its products by far the fewer; a mild one stays on the series.
    #[test]
    fn a_stiff_half_step_by_squaring_is_its_series(){
        let n=4;let mut c=random(n,&[],47);for a in 0..n{h_of(&mut c,0)[a*n+a]+=Complex::new(150.*a as f64,0.);}
        let mut h=[Complex::ZERO;MOST_STATES*MOST_STATES];h[..n*n].copy_from_slice(c.hamiltonian());
        let(tau,truncation)=(0.2,1e-7);let psi=state(n,53);
        let half=HalfStep::new(&h,n,tau,truncation);assert!(matches!(half,HalfStep::Matrix(_)),"a stiff generator takes the matrix");
        let(mut a,mut b)=(psi,psi);half.apply(&h,n,tau,&mut a,truncation);propagate(&h,n,tau,&mut b,f64::EPSILON);
        assert!(distance(&a,&b,n)<=truncation,"{:e}",distance(&a,&b,n));
        for z in h.iter_mut(){*z=*z*1e-3;}assert!(matches!(HalfStep::new(&h,n,tau,truncation),HalfStep::Series),"a mild one its series");
        assert!(squarings(100.,f64::EPSILON).is_none());
    }

    /// The series on the vector takes the fewest products its share allows, within the share; at a share at the
    /// series' round-off its parts are those that round the least (x = 1: 1 × 18 products; parts of ½ took 2 × 15, and rounded by more).
    #[test]
    fn the_series_takes_the_fewest_products_within_its_share(){
        for(x,share)in [(0.3,1e-6),(1.3,8e-6),(4.,1e-6),(30.,1e-4),(1.,f64::EPSILON),(200.,1e-5)]{
            let(parts,terms)=series_within(x,share);let y=x/parts as f64;
            let bound=(1..=terms).fold(1.,|s,k|s*y/k as f64);
            assert!(bound*parts as f64<=share,"x {x} share {share}: {parts} × {terms}, bound {bound:e}");
            assert!(y<=(share/f64::EPSILON).ln().max(1.)+1e-12,"x {x} share {share}: y {y}");
            let(sub,each)=series(x,1.,share);assert!(parts*terms<=sub*each,"x {x} share {share}: {parts} × {terms} against {sub} × {each}");
        }
    }

    /// A static generator through its eigendecomposition and through its Taylor propagator: the same
    /// root of the norm and state there, and impulses within the quadrature's bound of a fine grid's.
    #[test]
    fn the_spectral_and_the_taylor_segments_agree(){
        let n=6;let tau=0.1;let c=random(n,&[],13);let psi=state(n,17);
        let spectral=Segment::new(c.clone(),0.,tau,1e-9,1e-9,tau,node_phase());let taylor=Segment::new(c.clone(),0.,tau,0.,0.,tau,node_phase());
        assert!(matches!(spectral.how,How::Spectral{..})&&matches!(taylor.how,How::Taylor{..}));
        // The norm at the end, and a threshold between it and 1.
        let mut end=psi;let(_,_,whole)=taylor.evolve(&mut end,0.,tau,0.);let threshold=0.5*(1.+norm2(&end,n));
        let(mut a,mut b)=(psi,psi);
        let(sa,ja,ia)=spectral.evolve(&mut a,0.,tau,threshold);let(sb,jb,ib)=taylor.evolve(&mut b,0.,tau,threshold);
        assert!(ja&&jb,"the root lies in the step");
        assert!((sa-sb).abs()<=1e-12*tau,"roots {sa} and {sb}");
        assert!(distance(&a,&b,n)<=1e-11,"states at the root: {:e}",distance(&a,&b,n));
        assert!((norm2(&a,n)-threshold).abs()<=1e-12,"the norm at the root {} against {threshold}",norm2(&a,n));
        // The impulses against the whole step's at a 64-fold finer grid's spacing (Simpson's bound).
        let finer=Segment{components:taylor.components.clone(),start:0.,spacing:taylor.spacing/64.,how:How::Taylor{propagator:exponential(taylor.components.hamiltonian(),n,taylor.spacing/64.),decay:vec![]},dropped:0.};
        let mut x=psi;let(_,_,reference)=finer.evolve(&mut x,0.,tau,0.);let mut y=psi;let(_,_,mine)=spectral.evolve(&mut y,0.,tau,0.);
        let size=reference.iter().map(|v|v.abs()).fold(0.,f64::max);
        for k in 0..3{assert!((mine[k]-reference[k]).abs()<=QUADRATURE*size&&(whole[k]-reference[k]).abs()<=QUADRATURE*size,"axis {k}: {} and {} against {}",mine[k],whole[k],reference[k]);}
        // To the root, whose impulse the redone step does not read: within the share of the force's size times the step, the net impulse there being small beside it.
        let size=(0..3).map(|k|column_norm(c.gradient(k),n)).fold(0.,f64::max)*tau;
        for k in 0..3{assert!((ia[k]-ib[k]).abs()<=QUADRATURE*size,"to the root, axis {k}: {} against {} of {size}",ia[k],ib[k]);}
    }

    /// A fast weak beat within the exponentials' share is left out (the step then static, through its
    /// eigendecomposition), and the state differs from the beat's resolved evolution within its bound; a
    /// share too small for it keeps it.
    #[test]
    fn a_beat_within_the_share_is_left_out(){
        let n=4;let tau=0.05;let w=4e4;let mut c=random(n,&[w],29);
        for q in 1..c.parts{for z in h_of(&mut c,q).iter_mut(){*z=*z*1e-5;}for a in 0..3{for z in g_of(&mut c,q,a).iter_mut(){*z=*z*1e-5;}}}
        let size=|h:&[Complex]|column_norm(h,n).max(row_norm(h,n));
        let bound=2.*c.beats().map(|b|size(b.hamiltonian)).sum::<f64>()*(1.+(size(c.hamiltonian())+c.beats().map(|b|size(b.hamiltonian)).sum::<f64>())*tau)/w;
        let left=Segment::new(c.clone(),0.,tau,1e-6,1e-6,tau,node_phase());assert!(matches!(left.how,How::Spectral{..}),"the beat left out");
        assert!(matches!(Segment::new(c.clone(),0.,tau,1e-12,1e-12,tau,node_phase()).how,How::Beating(_)),"a smaller share keeps it");
        let resolved=beating(c,0.,TAU/w/16.);
        let psi=state(n,31);let(mut a,mut b)=(psi,psi);left.evolve(&mut a,0.,tau,0.);resolved.evolve(&mut b,0.,tau,0.);
        assert!(distance(&a,&b,n)<=bound,"{:e} against the bound {bound:e}",distance(&a,&b,n));
    }

    /// Composite Simpson's sum of a component over whole panels is S times its integral, at any frequency and decay short of the aliasing; S's first order is (ωh)⁴/2880, and the
    /// fewest panels meet their bound where one less would not.
    #[test]
    fn simpson_s_sum_of_a_component_is_s_times_its_integral(){
        for(omega,decay,tau,panels)in [(100.,0.,0.1,7),(400.,0.,0.2,50),(1000.,0.,0.3,60),(3000.,0.,0.07,40),(300.,19.,0.2,13)]{
            let w:Complex=Complex::new(omega,decay);let h=tau/panels as f64;let at=|t:f64|(Complex::I*w*t).exp();
            let mut sum=Complex::ZERO;for p in 0..panels{let t=p as f64*h;sum+=(at(t)+at(t+0.5*h)*4.+at(t+h))*(h/6.);}
            let integral=(at(tau)-Complex::ONE)/(Complex::I*w);
            // S at the complex phase: (u/3)(2 + cos u)/sin u, cos and sin of u = ωh/2 by e^{iu}.
            let u=w*(0.5*h);let(e,back)=((Complex::I*u).exp(),(Complex::I*u*(-1.)).exp());let(c,s)=((e+back)*0.5,(e-back)*Complex::new(0.,-0.5));
            let ratio=u*(c+Complex::new(2.,0.))*(1./3.)/s;
            assert!((sum-ratio*integral).abs()<=1e-12*integral.abs().max(tau*1e-3),"ω {omega}: {sum:?} against {:?}",ratio*integral);
            if decay==0.{assert!((ratio.re-1.-simpson_excess(u.re)).abs()<=1e-12&&ratio.im.abs()<=1e-12,"{ratio:?}");}
            assert!((ratio-Complex::ONE).abs()<=simpson_excess(u.abs())+1e-12,"a complex phase's excess within its modulus': {ratio:?}");
        }
        for u in [1e-2,0.1,0.5,1.]{let leading=u*u*u*u/180.;assert!((simpson_excess(u)-leading).abs()<=0.2*u*u*leading+1e-14,"u {u}: {} against {leading}",simpson_excess(u));}
        for x in [0_f64,0.3,1.,2.5,7.,20.,39.,41.,300.,1e4]{
            let panels=static_panels(x);let room=QUADRATURE*x.max(1.);let cap=((0.5*x).ceil() as usize).max(1);
            assert!(x==0.||simpson_excess(x/panels as f64)<=room,"x {x}: {panels} panels");
            assert!(panels==cap||simpson_excess(x/(panels-1) as f64)>room,"x {x}: {panels} panels, one less meets it");
        }
    }

    /// The rule's grid on a state holding every frequency, from a phase over the step below one to hundreds: the impulse within the quadrature's share of the force's size times the step.
    #[test]
    fn a_static_grid_holds_the_impulse_within_its_share(){
        let n=4;
        for(shift,tau)in [(0.,0.01),(0.,0.05),(400.,0.1),(2000.,0.1),(2000.,0.4)]{
            let mut c=random(n,&[],53);for a in 2..n{h_of(&mut c,0)[a*n+a]+=Complex::new(shift,0.);}
            let rule=Segment::new(c.clone(),0.,tau,0.,0.,tau,node_phase());assert!(matches!(rule.how,How::Taylor{..}));
            let fine=Segment{components:c.clone(),start:0.,spacing:rule.spacing/16.,how:How::Taylor{propagator:exponential(c.hamiltonian(),n,rule.spacing/16.),decay:vec![]},dropped:0.};
            let psi=state(n,59);let(mut a,mut b)=(psi,psi);let(_,_,mine)=rule.evolve(&mut a,0.,tau,0.);let(_,_,reference)=fine.evolve(&mut b,0.,tau,0.);
            let size=(0..3).map(|k|column_norm(c.gradient(k),n)).fold(0.,f64::max)*tau;
            for k in 0..3{assert!((mine[k]-reference[k]).abs()<=QUADRATURE*size,"shift {shift}, τ {tau}, axis {k}: {} against {} of {size}",mine[k],reference[k]);}
        }
    }

    /// A static generator whose fast part the state hardly holds: its spectral grid weighted by the
    /// state's amplitudes is far coarser than the rule's, and the impulse stays within the quadrature's
    /// share of a fine grid's.
    #[test]
    fn a_spectral_grid_follows_the_state_s_amplitudes(){
        let n=6;let tau=0.2;let mut c=random(n,&[],37);
        // Two blocks: three slow states near zero, three at −400 rad/us, coupled weakly; the state in the first.
        for a in 0..n{for b in 0..n{let scale=if a<3&&b<3{0.1}else if (a<3)!=(b<3){0.05}else{1.};let h=h_of(&mut c,0);h[a*n+b]=h[a*n+b]*scale;}}
        for a in 3..n{h_of(&mut c,0)[a*n+a]+=Complex::new(-400.,0.);}
        let mut psi=state(n,41);for z in psi[3..n].iter_mut(){*z=*z*0.01;}let norm=norm2(&psi,n).sqrt();for z in psi[..n].iter_mut(){*z=*z*(1./norm);}
        let s=Segment::new(c.clone(),0.,tau,1e-9,1e-9,tau,node_phase());assert!(matches!(s.how,How::Spectral{..}));
        let weighted=s.weighted(&s.enter(&psi,0.),tau,0.).unwrap();
        // The cross components' weight is about the fast part's amplitude, 1%: (ω·2δ)⁴ may grow about a hundredfold.
        assert!(weighted>=2.*s.spacing,"weighted {weighted:e} against the rule's {:e}",s.spacing);
        let fine=Segment{components:c,start:0.,spacing:s.spacing/16.,how:How::Taylor{propagator:exponential(s.components.hamiltonian(),n,s.spacing/16.),decay:vec![]},dropped:0.};
        let(mut a,mut b)=(psi,psi);let(_,_,mine)=s.evolve(&mut a,0.,tau,0.);let(_,_,reference)=fine.evolve(&mut b,0.,tau,0.);
        assert!(distance(&a,&b,n)<=1e-11,"states {:e}",distance(&a,&b,n));
        let size=reference.iter().map(|v|v.abs()).fold(0.,f64::max);
        for k in 0..3{assert!((mine[k]-reference[k]).abs()<=QUADRATURE*size,"axis {k}: {} against {}",mine[k],reference[k]);}
    }

    /// A beating segment's root of the norm and the impulse up to it: CF4 at the node rule's spacing
    /// against an eight-fold finer one.
    #[test]
    fn a_beating_segment_finds_the_norm_s_root(){
        let n=5;let tau=0.12;let c=random(n,&[170.,410.],19);let psi=state(n,23);
        let rule=Segment::new(c.clone(),0.,tau,0.,0.,tau,node_phase());let fine=beating(c,0.,rule.spacing/8.);
        let mut end=psi;rule.evolve(&mut end,0.,tau,0.);let threshold=0.5*(1.+norm2(&end,n));
        let(mut a,mut b)=(psi,psi);let(sa,ja,ia)=rule.evolve(&mut a,0.,tau,threshold);let(sb,jb,ib)=fine.evolve(&mut b,0.,tau,threshold);
        assert!(ja&&jb);
        assert!((sa-sb).abs()<=1e-7*tau,"roots {sa} and {sb}");
        assert!(distance(&a,&b,n)<=1e-6,"states at the root: {:e}",distance(&a,&b,n));
        let size=ib.iter().map(|v|v.abs()).fold(0.,f64::max);
        for k in 0..3{assert!((ia[k]-ib[k]).abs()<=QUADRATURE*size,"axis {k}: {} against {}",ia[k],ib[k]);}
    }
}
