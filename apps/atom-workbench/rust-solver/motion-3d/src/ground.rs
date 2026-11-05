//! The atom's internal state on clusters of states: the ground manifold and the excited states
//! whose couplings must be kept, every other excited state folded at second order; its generator
//! at any point r of the trap, to first order in the Doppler and recoil shifts of the folded part.
//!
//! # Far components
//! Every beam drives a co-rotating component, V_l e^{−iδ_l t} with δ_l = 2π(ν_l − ν_frame), and a
//! counter-rotating one, R_l e^{+i2π(ν_l + ν_frame)t}, each with its amplitude a_l(r). The beams of
//! one frequency drive one field, Σ_l a_l(r) V_l (and Σ_l a_l(r)* R_l): its components are one
//! component each (`Component`, its size the beams' sum, every beam at its peak in phase),
//! classified by one rule against the 3D solve's budget (ledger.rs, `classify`): a component folds
//! when it is far, |ν| ≥ FOLD_DISTANCE ‖L0‖ with its mixing m = size/(|ν| − ‖L0‖) within
//! m² ≤ budget, its error booked -- whatever the frequencies of the others: a van Vleck fold's
//! denominators are exact, and it needs no gap between what it keeps and what it folds (the
//! population core's lattice does, for its own time scales). A folded component enters as
//! its second-order (van Vleck) shift on the ground and the excited states, with the exact
//! bare-energy denominators -- quadratic in its field, so that its beams interfere: a standing wave,
//! a lattice of crossed beams --; the others are near carriers. Each beam's static shift through the
//! states outside the model O_l (in H0 at the focus, the beams of one frequency summed there) is
//! quadratic in its field too. The beams of one frequency whose fields are parallel (`Unit`: a
//! retro-reflected tweezer, a beam alone) shift the states by their summed field's intensity
//! s_u(r) = |Σ_l a_l(r) γ_l|²/N_u alone, exactly:
//!
//! ```text
//! H(r) = H0 + Σ_u (s_u(r) − s_u(0)) O_u + Σ_u s_u(r) F_u      (`Ground::hamiltonian_at`)
//! ```
//!
//! O_u and F_u the unit's outside shift and fold at s_u = 1 (a beam alone: s = |a_l|², O_l, F_l). The
//! beams of a frequency are one unit, their dominant polarization's, or a unit each, whichever leaves
//! out the smaller interference (`coherence`): the remainder of polarizations a little off a common
//! one, or the interference of polarizations that differ -- a polarization lattice, which the beams'
//! intensities alone do not fix --, its bound in the ledger (`Ground::interference`, solve.rs).
//!
//! # Where the trap is
//! A beam focused at the origin need not hold its atoms there: a reflection's standing wave has its antinodes a mirror phase φ's φ/2k along the axis from the focus, whatever the beam's
//! frequency, and φ past 90° makes the focus a slope's top. The trap of a potential is the nearest bottom of a well to the focus, whatever the field: its descent from the focus
//! (`descent`, `Ground::well`), and everything about the trap reads that place -- its frequencies, depth and region, the atoms' reach. Where two wells are as near (a top: φ = 180°), the descent's
//! first step falls into one, the same one every time, and they are mirror images. At a focus that is a well's bottom the descent stands still, the focus's very bits.
//!
//! # One rule over the ground and the excited states
//! The near components of one frequency (equal `Beam::frequency_index`) form a carrier c at δ_c;
//! carrier c drives V_c(r) = Σ_{l∈c} a_l(r) V_l e^{−iδ_c t} + h.c., each beam with its own coupling
//! V_l at the focus and its own amplitude a_l(r). Every coupling of the generator is judged by one
//! rule against the budget: folded at second order when its mixing fits, kept explicit otherwise.
//! - A drive, carrier c from ground state b into excited state e (`Drive`): its size g = Σ_l |V_l[e,b]|
//!   (every beam at its peak), its mismatch ΔE = E_e − E_b − δ_c − iΓ_e/2, complex, and its distance
//!   d the smallest |ΔE| over the atom's reach (H(r)'s energies at every corner of the units'
//!   intensities there: they are affine in them; `Ground::reach`), its saturation
//!   s = SATURATION (g / max(d − dressing, 1/T))², T the solve's duration (below which a mismatch is a
//!   resonance over the run), the dressing zero at first (the closure, below). The second-order fold
//!   of a ground state's drives errs by their total saturation S_b = Σ_e s: its rates by S_b relative
//!   (a two-level transition's scattering rate is the folded one over 1 + s, its light shift's error
//!   s/2 relative), its populations by at most S_b -- S²/(2(1 + S)) for one transition in its steady
//!   state, at most S/e for a pumping step (R t e^{−Rt}); measured against the 32-state core at the
//!   focus, 0.01–0.79 S_b (SAT, PUMP, THREE, Λ, the 3D PGC). Each ground state keeps its drives from
//!   the most saturated down until the rest's total fits the budget; those fold.
//! - H's static coupling between two excited states (the tweezer's tensor shift between hyperfine
//!   levels and sublevels): it folds when (g/d)² is within the budget, g its largest size and d their
//!   smallest separation over the whole trap -- each Gaussian beam's intensity from 0 to its peak, the
//!   tables' range: a pair whose separation closes anywhere there (sublevels whose Zeeman and tensor
//!   shifts cross) would make an explicit state's first-order dressing singular there.
//! - The ground states' couplings (below), judged with the folded drives alone; and the two photons
//!   of an explicit state's jump landing in two ground states, their coherence |Σ_a C_a C_a*|.
//! An excited state is explicit when a drive into it is kept, or a kept coupling joins it to an
//! explicit one. Closure: every folded drive from a cluster's ground state, every folded coupling
//! of an explicit state, and every folded pair of channels between two of its ground states, is judged
//! again with the cluster's dressing -- the spectral norm of the bounds of what stays in its static block, its states' light
//! shifts and its kept couplings (the block's spread: a pair not far beyond it would beat between the block's eigenstates at their nearness) -- subtracted from its
//! distance; one that fails is kept (promoted), a drive static in the frames between two states of one
//! cluster is kept (it costs nothing), and the rule runs again until nothing changes. Each pass takes the estimates at its own
//! dressing and keeps first, per ground state, the most saturated drive of a folded total beyond the allowance: a folded near drive's
//! light is most of the dressing until it is kept (PGCP_10's probe, a dressing above its distances of 21 to 33 MHz), and what is judged
//! under it stays kept for nothing -- a cooling drive's saturation read at a quarter of its distance, the pairs of channels within that spread.
//! What folds stays out of it: it beats away at its own frequency, and its bound says nothing of the block. A near tweezer's cross terms with the cooling beams, bounds of 2.9 GHz at a beat of 6 THz (m² 2e-7), gave a
//! D1 gray molasses at 808 nm a dressing of 7.6 GHz past its 6.8 GHz hyperfine splitting, against 47 MHz from what stays: pairs of m² ≤ 1e-9 kept at that beat, a grid of 30 000 nodes per us, 256 trajectories of 100 us in
//! 2930 CPU-s against 0.8, T the same within 0.5 of its standard errors at every output on shared streams.
//! The couplings between explicit states and the pairs are judged once the drives stand. The rungs then keep more as the allowance
//! falls. When every drive folds, the clusters are the ground states' alone.
//!
//! A folded drive into an explicit state and a kept drive into it are a pair of channels too, and their coupling through that state,
//! −V′[e,a]* G V[e,b], lies in neither part: X takes V^f, zero on the kept entry, and the state's amplitude does not feel a folded drive. So each
//! such pair is judged as the ground states' are, by `foldable` at its worst separation over the trap, within one cluster with its dressing taken from
//! that, its size g g′ (1/d + 1/d′ + Γ/(d d′)), and the folded drive is kept where the pair may not fold (`Ground::keep_partners`), a residual beat in the
//! cluster's frames or none. Read off the frames alone, a kept pair's partner fell out wherever a probe's and a cooling drive's one state split the
//! ground states' frames by the 36 MHz between them: on PGCP_10 the rungs of 7 explicit states lost Raman couplings between sublevels of 0.3-1.1 rad/us beside
//! the largest's 4-5, and their T stood 2.9 ± 0.9 % and 10.1 ± 0.9 % high at a fold's charge of 0.28 % and 0.07 %, against 15 explicit states (8192 trajectories at 100 us); on
//! those rungs now 1.3 ± 0.8 % and 0.9 ± 0.9 %. The pairs at the beat of two carriers fold beyond the dressing's reach: on the first rung, whose cooling drives into the probe's states
//! lie within it, a frozen atom's p(2,2) went from 0.515 to 0.541 and its photons from 1.281 to 1.312 /us, against 0.539 and 1.309 in the master equation of F = 2 and all the
//! D2 states without any fold, and T from −2.0 ± 0.9 % to +0.4 ± 0.9 %.
//!
//! # Elimination of the folded drives
//! Each beam's folded coupling V_l^f is V_l with every kept drive's entry zeroed (`Ground::folded`).
//! With the ground energies E_b, the diagonal of H(r)'s ground block, and the excited states'
//! non-Hermitian Hamiltonian K = H(r)_ee − (i/2) D, D = Σ_a C_a†C_a|_ee, every excited state's
//! amplitude driven by the folded couplings follows the ground states adiabatically, beam by beam and
//! column by column (`Elimination`):
//!
//! ```text
//! R_l[:,b] = (K − E_b − δ_c)⁻¹ V_l^f[:,b]        R2_l[:,b] = (K − E_b − δ_c)⁻² V_l^f[:,b]        R_c(r) = Σ_{l∈c} a_l(r) R_l
//! ```
//!
//! and the folded excited amplitude is A_c = −R_c: its physical sign, with which it interferes with
//! an explicit state's amplitude of the same frequency in a photon. A jump operator's overall sign is
//! no physics; the jumps are taken as −(C_a ψ_e + C_a A_c ψ) = C_a R_c ψ − C_a ψ_e, so that a cluster
//! of ground states keeps the release's photons sign for sign (and the GPU's kernel, gpu/mod.rs).
//!
//! # Couplings between ground states
//! In the frame where every ground state b rotates at its own energy, the folded drives couple the
//! ground states pairwise, each coupling rotating at its pair's separation:
//! - absorbing carrier c from b and emitting into c′, landing in a (the channels (c, b) and
//!   (c′, a)): X_ab = [−½(V_c′^f†R_c + R_c′†V_c^f) − (i/2) R_c′†D R_c]_ab at
//!   Ω = (E_a + δ_c′) − (E_b + δ_c) -- for a = b and c = c′ the light shift and scattering of b,
//!   between the sublevels of a hyperfine level (c = c′) the light's own couplings at their Zeeman
//!   separation, between F = 1 and F = 2 the Raman terms at the Raman detuning, on one state
//!   (a = b, c ≠ c′) the beat of two carriers;
//! - H's static couplings between ground states, at their separation;
//! - the two photons of a jump from b landing in a₁ and a₂, interfering at the difference of their
//!   frequencies.
//!
//! Each is judged by the rule of the far components (`classify`): its frequency the separation
//! (with H(r)'s Zeeman and state-dependent shifts, the worst over the trap; the light's own shifts
//! and rates bounded in the eliminated generator's ‖L0‖), its size the bound of its superoperator
//! over the trap (every beam at most its peak amplitude). A folded coupling enters at second order:
//! - a pair of channels: through the near light's dressing (below), the far channel (c′, a) an
//!   intermediate of the excited states' self-energy in the response from (c, b);
//! - a static coupling: its van Vleck shift with the exact bare-energy denominators;
//! - a pair of photons: they are distinguishable, and the jump's channels split by frequency.
//!
//! # The near light's dressing
//! The response from a channel (c, b) sees every other near light at all orders in its
//! denominator, as a far beam's fold is in H(r) (`Ground::dress_light`): b's energy shifted by every
//! other carrier d's light shift, z = E_b + δ_c + Σ_{d≠c} ΔE_b^d, ΔE_b^d = −Re(V_d^f†(K − E_b − δ_d)⁻¹V_d^f)_bb,
//! and the excited states' self-energy through the folded pairs' far channels (d, a) -- the virtual
//! states the cluster's own dynamics does not resolve; a kept pair's are its cluster's --, each
//! intermediate's energy dressed by every near carrier, Ẽ_a = E_a + Σ_d ΔE_a^d:
//!
//! ```text
//! Σ = Σ_(d,a) V_d^f[:,a] V_d^f[:,a]†/(z − Ẽ_a − δ_d)        R_c[:,b] = (K + Σ − z)⁻¹ V_c^f[:,b]
//! ```
//!
//! -- for a ground state alone; within a cluster its static block's eigenstates take the place of its
//! energy E_b (# The static ground block).
//!
//! -- a strong beam a few GHz off a probe's line moves it by its light shift, a near tweezer's
//! shift moves every other light's line. The folded pairs' van Vleck paths of the release,
//! W̃ D U = V_ĉ†G Σ G V_c, were this response's first order, wrong where Σ is not small against the
//! pole and, added outside the response, a decay without its jumps. Every coupling, decay and photon
//! below takes the dressed response: −½(V†R + R†V), −(i/2) R†DR and C R, a Lindblad generator by
//! construction (the Hermitian part, the decay its jumps' Σ L†L). Each near carrier's field is its
//! units' (the dominant polarization's, a beam alone its own, at their intensities: the tables'
//! axes). What that leaves out -- the interference of a carrier's units taken apart, the dressing's
//! next order, ΔE^d by the admixture under d, and an intermediate's energy within its cluster's static
//! block, the block's spread about it (its row's other entries, Gershgorin's radius) -- changes a
//! response at first order by G times it; its bound (`Ground::dressing_bounds`, through the dressed resolvent applied to the
//! intermediates' fields and projected on the response), weighted by the channel's share of the
//! generator, is the ledger's.
//!
//! # The static ground block
//! A cluster's ground states need not be the eigenstates of what moves them statically: a tweezer's
//! vector light shift across the field mixes the Zeeman sublevels by more than their splitting (the
//! validation's F6, 6.7 rad/us against 0.7), as does another near carrier's light shift. A ground
//! superposition evolving under its static block responds at the block's eigenenergies, not at the
//! states' diagonal ones: the response solves (K + Σ − δ_c) R − R h_c = V_c^f on the cluster, h_c its
//! static block as carrier c's responses see it -- H(r)'s ground block on the cluster (its static
//! couplings: the far light's folds, the field's Zeeman terms) with every other near carrier's light
//! shift between its states, ΔE^d_{bb′} = −½(v_b†g_b′ + (v_b′†g_b)*) --, by its eigenstates, each with each
//! ground state's own self-energy at the eigenstate's energy (its far channels, above):
//!
//! ```text
//! h_c = Σ_j λ_j u_j u_j†        R_c = Σ_j Σ_b (K + Σ_b(z_j) − z_j)⁻¹ V_c^f[:,b] u_j[b] u_j†,   z_j = λ_j + δ_c
//! ```
//!
//! (`Block`, `Mode`). Summed over a degenerate eigenspace, u_j[b] u_j† is its projector's row b, P_E[b,:]: R
//! does not depend on the eigenbasis within it -- eigenvalues equal to their round-off are one energy, taken
//! through its projector --, and it is continuous in the mixing; for a state nothing mixes, u_j = e_b, it is
//! the channel's own dressed response above. Its cost: per eigen-energy one factorization for the states
//! without far channels, one per state with them; a state whose weight leaves its term below double
//! precision's round-off of the generator is left out with its bound (‖(K + Σ − z)⁻¹‖ ≤ 2/γ, γ the decay's
//! floor, `Ground::floors`). An intermediate in the cluster itself is taken at the state's own energy
//! z_b = h_bb + δ_c, the difference in its error: at z_j an eigenstate mostly a would return into itself
//! through b's share, c's own light shift on a for the pair's separation -- the cluster's dynamics taken
//! again as a virtual state. What the block leaves out -- each entry's left-out part, the units'
//! interference in the shifts and their next order -- moves its eigenvalues and eigenvectors, and the
//! response at first order by Σ_ij M_i[z_i, z_j] (u_i†E u_j) u_j† (M_i(z) eigenstate i's response at energy z,
//! its divided differences); its bound is the dressing's (`Ground::dressing_bounds`). A block whose responses change
//! by less than the numerical share (`MIXING_SHARE`) relative to themselves when its off-diagonal part is left out,
//! measured against its eigenstates' at the eliminations that judge the trap, is taken unmixed (`Ground::unmixed`):
//! each state its own eigenstate at its own energy h_bb + δ_c, one factorization per state as for a lone state, the
//! largest change booked (`Ground::unmixed_error`). Its eigenstates would cost a factorization per eigen-energy and
//! state for that change alone: a molasses' F = 2, whose sublevels the other carrier's shifts mix by 6e-4 to 0.2
//! rad/us (its eigenstates turned 45°, the responses 1e-5 apart), a Zeeman ladder coupled at 6e-7 rad/us. A Λ
//! scheme, a tweezer's vector shift across the field or a far beam's tensor shift mix the responses by 2e-3 to 6e-2:
//! those blocks keep their eigenstates. Left out: the couplings that beat (a pair of carriers' Raman terms, the time
//! step's) and the static couplings to other clusters (their fold, the rule's ledger).
//!
//! # Clusters and the mixed generator
//! The states joined by kept couplings form a cluster (`Cluster`), the trajectory's state space
//! between jumps: near B = 0 a whole hyperfine level, at large fields single sublevels (the
//! rate-equation limit), F = 1 and F = 2 together only through a Raman-resonant pair (a Λ scheme,
//! gray molasses, EIT); with its explicit excited states, a saturated transition, a near-resonant
//! probe, EIT's dark state in full. A cluster's basis is its ground states, then its explicit excited
//! states. Its frame (`Cluster::frames`, `Cluster::excited_frames`, θ_e = θ_b + δ_c along a kept drive)
//! makes the strongest kept couplings static; the others keep their residual beat e^{iΩ̃t}, which the
//! time step resolves. The clusters are fixed for a run, judged over the trap. A jump projects onto
//! the cluster it lands in, its channels its classes of photons of one frequency (`Class`):
//!
//! ```text
//! H_eff = Σ_s (E_s − θ_s)|s⟩⟨s| + Σ_kept e^{iΩ̃t} X_ab |a⟩⟨b|
//!       + Σ_kept drives e^{iΩ̃t} V_c[e,b] |e⟩⟨b| + h.c. + Σ_{e,e′∈P} (H_ee′ kept, or its van Vleck shift)
//!       + Σ_{e,e′∈P} ½ Σ_b V_c^f[e,b] V_c^f[e′,b]* (Re 1/(K_ee − δ_c − E_b) + Re 1/(K_e′e′ − δ_c − E_b)) |e⟩⟨e′|
//!       − (i/2) Σ_φ J_φ†J_φ  (on the explicit states; the folded part is X's)
//! J_{a,φ} = Σ_{(c,b→b′)∈φ} e^{iω̃t} [C_a R_c]_{b′b} |b′⟩⟨b| − Σ_{(e→b′)∈φ} e^{iω̃t} [C_a]_{b′e} |b′⟩⟨e|
//! ```
//!
//! ω̃ = θ_b′ − θ_e for an explicit state's photon, θ_b′ − θ_b − δ_c for a folded one. An explicit state's amplitude holds the frequency of every kept drive
//! that reaches it, through H's kept couplings too, whatever its frame: its photon shares a class with the folded photons of those frequencies and with the explicit states that hold
//! one of them, each at its residual. Classes by the frame alone dropped the interference of a probe-driven state's photon with the probe's folded photons through F′ = 2 (on PGCP_10 a rate between
//! sublevels 11 % low, against the second order of every excited state) and what two frames' explicit states land coherently (a frozen atom's p(2,2) 0.46, against 0.54). The decay between
//! ground states is X's on the kept pairs of channels; a class that joins two folded photons landing in one
//! state from channels without a kept coupling -- through an explicit state's photon of their frequency --
//! adds their cross term alone (`FoldedCross`): the decay is the jumps' product on every entry.
//!
//! Readout: the bare ground density is ρ − ½{Y, ρ}, Y = Σ R′†R over the co-resonant channel
//! pairs, and an excited state's population tr(E_i ρ), E_i = Σ R′†|i⟩⟨i|R, plus ρ_ee when explicit.
//!
//! # Fourth order
//! A cluster of ground states folds its drives at fourth order where that costs less (solve.rs): its
//! error is S² instead of S (`Ground::fourth` dense, model.rs `FourthLayout` its tables). On the slow
//! invariant subspace Ψ = (x, W x) of the no-jump generator, i Ẇ = V + K W − W (H_g + V†W), W = −R + W3
//! with, per triple of carriers (c1, c2, c4) at ω = δ_c1 − δ_c2 + δ_c4,
//!
//! ```text
//! W3[:,b] = G_b(ω) Σ_b′ R_c1[:,b′] (V_c2†R_c4)[b′,b]        G_b(ω) = (K − E_b − ω)⁻¹
//! ```
//!
//! (every ground state b′ between), over the carriers the fourth order takes (`fourth_carriers`: those beyond the
//! budget's square, and every carrier whose photons share a class with theirs). Its frequencies are their carriers'
//! integer combinations, evaluated once (`Ground::frequency`): a photon's triple is at its carrier's δ_c exactly.
//! The second order's R is dressed by the near light (# The near light's dressing), and the fourth order is its
//! expansion around that dressing:
//! - at a carrier c's frequency (a photon's triples, and Π's YW below) G(ω) is c's block's, as R's (# The static ground
//!   block): from a source at place b, Σ_E (K + Σ_b(z_E) − z_E)⁻¹ P_E[b, ·] over the block's eigen-energies
//!   (`fourth_column`; Y's rows by the projectors' transpose): a near tweezer's shift moves the pole W3 sees as it
//!   moves R's, and a mixed static block's eigenstates respond at their own energies;
//! - W3 leaves out what R already holds (`Ground::excluded`): the triples (d, d, c) through b′ = a for (d, a) a far
//!   channel of (c, b) -- Σ's first order: in W3 its Rayleigh–Schrödinger form, the Brillouin–Wigner one less the
//!   intermediate's admixture, which the dressing takes whole --, and in the triples (c, d, d), d ≠ c, d's light
//!   shift between the cluster's states, the Hermitian part of (V_d†R_d)[b′, b], c's block's (an unmixed block's at
//!   b′ = b alone: the off-diagonal part it leaves out W3 takes) -- its anti-Hermitian part, b's scattering by d,
//!   stays. Counted twice, they
//!   tripled the 808 nm tweezer's probe's rates and took F10's populations 0.4 off the core.
//!
//! The state is y = M^{1/2} x, M = 1 + X with X = R†R on the
//! cluster's coherent channel pairs (the readout's Y; the other pairs beat and average out), so that
//! ‖y‖² is the survival probability, and
//! - H̃'s Hermitian part is H_eff's and that of H^(4) = V†W3 − ½[X, V†R], on the entries the second
//!   order would keep a coupling at their beat Ω = θ_a − θ_b + ν (ν the monomial's carriers'): static, below
//!   the cut, a kept coupling's residual on the entry, or one the rule does not let fold (`fourth_keeps`).
//!   Another entry beats faster than the time step follows (between F = 1 and F = 2 at the hyperfine
//!   splitting) and is left out like a folded coupling, its mixing (b/(|Ω| − l0))² booked, b = S l0 the
//!   size of H^(4);
//! - its anti-Hermitian part is −(i/2) Σ L̃†L̃, the jumps' own -- the norm's loss is their rate by
//!   construction --, L̃ = C_a (R_c (1 − ½X) − W3^c), W3^c the triples at δ_c (a photon of carrier c), for the
//!   fourth order's carriers (the others' photons stay the second order's); two photons' products over the landing
//!   states they share a class in, the folded crosses' (`FoldedCross`) with the coherent pairs';
//! - after a jump the bare state f has the slow part Π f, Π = 1 + ½X + YW,
//!   (YW)_ab = −Σ_{c1,c2} V_c1[:,a]† G_a(δ_c1) R_c2[:,b] at its beat: a map at the jump's instant, every
//!   entry kept (a fast entry pairs with the photon's own into a slow product); with the ringing's photon of probability p
//!   its weight A = ‖Π f‖² sets the waiting time's threshold, θ (1 − p)/A (engine.rs, # The transient), A + p − 1 the excited states' 1 − κ, κ = 1 − Γ²/(Δ² + Γ²/4), Δ from the channel's dressed energy (`state_factor`): the 808 nm
//!   tweezer's probe is on its shifted line, a delay (κ = −3), not 32 MHz off;
//! - the engine's readout (metric 1 + X) and the force stay the second order's: the readout errs
//!   within S², T by the force's fold S.
//!
//! # First order in the Doppler and recoil shifts
//! Absorbing from beam l shifts the excited state by the kinetic energy w_l = (ħ/m)(k_l·κ + k_l²/2);
//! to first order R_l → R_l − w_l R2_l, in the generator and the jumps alike (model.rs, # The photons and the
//! Doppler term), and `Ground::response` gives the shifted response to all
//! orders. An explicit drive needs neither: its beams' phases e^{ik_l·r(t)} advance along the atom's path within a step (model.rs, # The atom's path).
//!
//! # Approximations and their estimates (the solve's ledger)
//! - the far components' fold: Σ m²;
//! - the couplings' fold: Σ m² of the folded pairs of channels and static couplings;
//! - the drives' fold: the largest folded total saturation S_b of a ground state;
//! - the zeroth-order elimination: η², the folded excited population along the trajectories (lib.rs);
//! - first order in the Doppler shift: (kv/σ_min)² (engine.rs), what its transfer leaves on T (model.rs `Model::first_error`);
//! - an emitted photon has its carrier's wave number: ≤ 1e-7 on the D2 line.
use crate::ledger::BUDGET;
use crate::linear::{hermitian_eigen,hermitian_eigenvalues,Dense,Lu};
use crate::{Beam,Complex,Request,STATES};
use atom_population_core::lattice::classification::FOLD_DISTANCE;
use rayon::prelude::*;
use std::f64::consts::TAU;

/// A drive's saturation parameter is 2(g/|ΔE|)²: the relative error of its linear response, the
/// estimate of its fold.
pub const SATURATION:f64=2.;
/// The numerical share an unmixed block may change its responses by, relative to themselves (`Ground::unmixed_blocks`):
/// the numerical tolerances' (solve.rs, # Shares; `NUMERICAL_SHARE` of ε = `POPULATION_TARGET`), the one switch between a
/// block's eigenstates and its states on their own.
pub const MIXING_SHARE:f64=crate::solve::NUMERICAL_SHARE*crate::solve::POPULATION_TARGET;

/// The co-rotating component of the beams of one frequency (`counter` false: V e^{−iνt},
/// ν = δ = 2π(ν_beam − ν_frame)) or their counter-rotating one (R e^{−iνt}, ν = −2π(ν_beam + ν_frame)),
/// each with its Hermitian conjugate; its size Σ_l 2‖V_l‖_F bounds its commutator, every beam at its
/// peak in phase, and `folded` is the rule's verdict.
#[derive(Clone,Debug)]
pub struct Component{pub beams:Vec<usize>,pub counter:bool,pub nu:f64,pub size:f64,pub folded:bool}
/// A unit of the light's static shifts (module documentation, # Far components): its beams, of one
/// frequency, and each one's weight γ_l relative to the first's field -- V_l = γ_l V_first,
/// R_l = γ_l* R_first, O_l = |γ_l|² O_first --, N = (Σ_l |γ_l|)² and its intensity at the focus s(0)
/// = |Σ_l γ_l|²/N; and its field's coupling as a combination of its beams', W = Σ_l u_l V_l (`coherence`).
/// A beam alone: γ = 1, N = 1, s(0) = 1, u = 1.
#[derive(Clone,Debug)]
pub struct Unit{pub beams:Vec<usize>,pub weights:Vec<Complex>,pub norm:f64,pub focus:f64,pub combination:Vec<Complex>}
impl Unit{
    /// Its intensity s = |Σ_l a_l γ_l|²/N at the beams' amplitudes a, in [0, 1].
    pub fn intensity(&self,a:impl Fn(usize)->Complex)->f64{self.field(a).norm_sqr()/self.norm}
    /// Σ_l a_l γ_l at the beams' amplitudes a.
    pub fn field(&self,a:impl Fn(usize)->Complex)->Complex{self.beams.iter().zip(&self.weights).fold(Complex::ZERO,|s,(&l,&w)|s+a(l)*w)}
    /// Whether its intensity varies over the trap: a Gaussian beam's, or interfering beams'.
    pub fn varies(&self,beams:&[Beam])->bool{self.beams.len()>1||beams[self.beams[0]].waist_um.is_some()}
}
/// The near components of one frequency.
#[derive(Clone,Debug)]
pub struct Carrier{
    /// Its beams (indices into `Ground::beams`).
    pub beams:Vec<usize>,
    /// δ_c = 2π(ν_c − ν_frame) (rad/us).
    pub detuning:f64,
    /// Its photons' wave number |k| (rad/um).
    pub wave_number:f64,
}
/// An absorption channel: carrier `carrier` absorbed from ground state `state` (a position in
/// `Ground::ground_states`).
#[derive(Clone,Copy,Debug,PartialEq,Eq,PartialOrd,Ord)]
pub struct Channel{pub carrier:usize,pub state:usize}
/// A jump's photon of a folded drive: carrier `carrier` absorbed from ground state `from`, the atom
/// landing in `to`.
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub struct Photon{pub carrier:usize,pub from:usize,pub to:usize}
/// A drive: carrier `carrier` from ground state `ground` into excited state `excited` (positions in
/// `Ground::ground_states` and `Ground::excited_states`).
#[derive(Clone,Copy,Debug,PartialEq,Eq,PartialOrd,Ord)]
pub struct Drive{pub carrier:usize,pub excited:usize,pub ground:usize}
/// What a coupling between ground states is.
#[derive(Clone,Debug,PartialEq)]
pub enum Kind{
    /// H's static coupling between ground states `a` and `b`.
    Static{a:usize,b:usize},
    /// The coupling of two channels: absorbing one, emitting into the other's carrier and landing
    /// in its state, and back.
    Raman{first:Channel,second:Channel},
    /// Two photons of a jump from one state into two.
    Photons{first:Photon,second:Photon},
    /// Two photons of an explicit excited state's jump landing in two ground states.
    Direct{excited:usize,first:usize,second:usize},
}
/// A coupling between ground states, its frequency (the worst separation over the trap, rad/us),
/// its size (the bound of its superoperator) and the rule's verdict.
#[derive(Clone,Debug)]
pub struct Item{pub kind:Kind,pub frequency:f64,pub size:f64,pub folded:bool}
/// What a coupling of an excited state is: a drive, or H's static coupling between excited states
/// `a` < `b` (positions in `Ground::excited_states`).
#[derive(Clone,Copy,Debug,PartialEq)]
pub enum ExcitedKind{Drive(Drive),Static{a:usize,b:usize}}
/// A coupling of an excited state judged by the rule: its size and distance over the trap (rad/us),
/// its power (a drive's Σ_l |V_l[e,b]|², the beams' incoherent sum; a static coupling's size²), the
/// rule's estimates (with its cluster's dressing when the closure judged it again) -- `estimate` with
/// every beam at its peak in phase, a bound over the positions, and `mean` over the light's pattern,
/// the beams' interference averaged out, what a moving atom sees --, the verdict, and whether the
/// closure kept it.
#[derive(Clone,Debug)]
pub struct ExcitedItem{pub kind:ExcitedKind,pub size:f64,pub power:f64,pub distance:f64,pub estimate:f64,pub mean:f64,pub folded:bool,pub promoted:bool}

/// The kept couplings of a cluster from one carrier to another: absorbing `absorbed`, emitting
/// into `emitted`, each entry (row a, column b, in the cluster's order) with its residual beat
/// Ω̃ = θ_a − θ_b + δ_emitted − δ_absorbed (rad/us). Between two explicit states of a group of one
/// carrier (row and column past the cluster's ground states), the entry is their folded drives' van
/// Vleck shift.
#[derive(Clone,Debug)]
pub struct Group{pub absorbed:usize,pub emitted:usize,pub entries:Vec<(usize,usize,f64)>}
/// A jump channel from a cluster: the photons that share a frequency landing in cluster `target` -- the
/// folded drives' (`photons`) and the explicit states' (`direct`: the explicit state's place in the
/// cluster, the ground state it lands in, a position in `Ground::ground_states`) -- each with its
/// residual ω̃ − ω̃₀ (rad/us), and their wave number.
#[derive(Clone,Debug)]
pub struct Class{pub target:usize,pub photons:Vec<(Photon,f64)>,pub direct:Vec<(usize,usize,f64)>,pub wave_number:f64}
/// Two folded photons of a class landing in one state from channels without a kept coupling between them (a
/// class joins them through an explicit state's photon of their frequency): their jump's cross term is the
/// decay's, Σ_{to,a} (C_a R_d)[to,a]* (C_a R_c)[to,b] over the landing states they share, on the entry (row:
/// the state of the channel `emitted` d, column: the one of `absorbed` c) at their residual beat w_c − w_d in
/// the class -- the folded pair's Hermitian part stays its van Vleck shift.
#[derive(Clone,Debug)]
pub struct FoldedCross{pub row:usize,pub col:usize,pub absorbed:usize,pub emitted:usize,pub residual:f64,pub tos:Vec<usize>}
/// A cluster: its ground states (positions in `Ground::ground_states`) and explicit excited states
/// (positions in `Ground::excited_states`), its basis in that order, and each one's frame θ (rad/us);
/// its kept couplings between ground states (per carrier pair, and H's static ones: row, column),
/// the van Vleck paths of its folded static couplings (row, column and the far ground state, between
/// states of one frame); its kept drives (row, column, carrier,
/// residual θ_e − θ_b − δ_c), H's kept couplings between its explicit states (row, column) and the
/// van Vleck paths of their folded ones (row, column, the far excited state); its jumps' classes and
/// its co-resonant channels (each with its residual, the readout's coherent sums). Each frame is its
/// cluster's first state's energy plus an integer combination of the carriers' detunings
/// (`combinations`, per place: the ground states', then the explicit ones'), and every residual the
/// integer combination of its terms, each evaluated once (`Ground::beat`): equal in exact arithmetic,
/// equal in floating point.
#[derive(Clone,Debug)]
pub struct Cluster{
    pub states:Vec<usize>,
    pub excited:Vec<usize>,
    pub frames:Vec<f64>,
    pub excited_frames:Vec<f64>,
    pub combinations:Vec<Vec<i32>>,
    pub groups:Vec<Group>,
    pub statics:Vec<(usize,usize)>,
    pub static_paths:Vec<(usize,usize,usize)>,
    pub drives:Vec<(usize,usize,usize,f64)>,
    pub excited_statics:Vec<(usize,usize)>,
    pub excited_paths:Vec<(usize,usize,usize)>,
    pub classes:Vec<Class>,
    pub coherent:Vec<Vec<(Channel,f64)>>,
    pub folded_crosses:Vec<FoldedCross>,
}
impl Cluster{
    /// A ground state's place in the cluster.
    pub fn position(&self,state:usize)->Option<usize>{self.states.iter().position(|&s|s==state)}
    /// An explicit excited state's place in the cluster (after its ground states).
    pub fn excited_position(&self,state:usize)->Option<usize>{self.excited.iter().position(|&s|s==state).map(|p|self.states.len()+p)}
    /// The cluster's dimension: its ground and explicit excited states.
    pub fn dimension(&self)->usize{self.states.len()+self.excited.len()}
    /// The frame of the cluster's state at place x.
    pub fn frame(&self,x:usize)->f64{if x<self.states.len(){self.frames[x]}else{self.excited_frames[x-self.states.len()]}}
}

/// The elimination for one H(r): the ground block of H (its diagonal the energies E_b), the excited
/// block of H and K, and every near beam's excited response to its folded coupling.
#[derive(Clone,Debug)]
pub struct Elimination{
    pub energies:Vec<f64>,
    pub ground:Dense,
    /// H's excited block (rad/us, Hermitian) and its diagonal, the excited states' energies.
    pub excited:Dense,
    pub excited_energies:Vec<f64>,
    /// K, the excited states' non-Hermitian Hamiltonian (rad/us).
    pub excited_hamiltonian:Dense,
    /// Per beam, R_l and R2_l, excited ← ground (dimensionless and us).
    pub responses:Vec<Dense>,
    pub derivatives:Vec<Dense>,
    /// Per carrier and cluster the block its responses see (`Ground::dress_light`; empty before the rule's
    /// clusters, None where the carrier reaches none of the cluster's states).
    pub blocks:Vec<Vec<Option<Block>>>,
}
impl Elimination{
    /// What a folded drive at δ from ground state b shifts an explicit state e by, per |V[e,b]|²: Re 1/(K_ee − E_b − δ), the state's amplitude
    /// living at its complex energy; 1/Γ_e at most. The rule folds a drive by its complex mismatch, so that it may cross the state's
    /// line within the trap: its bare 1/(E_e − E_b − δ) has a pole there of no width (the validation's F2, case 109: 1 rad/us at a table's node
    /// 1e-3 of the intensities from it, against 1e-3 rad/us here), which read 22 % at any grid and kept the rung that holds the near drive from the choice.
    pub fn shift_denominator(&self,e:usize,b:usize,delta:f64)->f64{(Complex::ONE/(self.excited_hamiltonian[(e,e)]-Complex::new(self.energies[b]+delta,0.))).re}
}
/// A carrier's block on a cluster (module documentation, # The static ground block): the cluster's
/// static ground block as the carrier's responses see it -- H's, with every other near carrier's light
/// shift --, its eigenstates (the columns of `vectors`, over the cluster's states; an unmixed block's the
/// states themselves, `Ground::unmixed`), the bound of each entry's left-out part (row-major over the
/// cluster's states, rad/us), and its eigen-energies' terms.
#[derive(Clone,Debug)]
pub struct Block{pub vectors:Dense,pub error:Vec<f64>,pub modes:Vec<Mode>}
/// An eigen-energy's terms of the response R = Σ_E Σ_b G_b(z_E) V^f[:,b] P_E[b,:] (`Ground::dress_light`): its
/// eigenstates (`members`, equal to the eigenvalues' round-off; their projector P_E) and z_E = λ_E + δ_c (rad/us);
/// the places of the cluster's states it takes -- those without far channels through one resolvent
/// (K − z_E)⁻¹ (`plain`, its factorization `lu` at the elimination), each with far channels through its own
/// (`dressed`) --, and those left out (`skipped`): each place with its resolvent's bound times its source,
/// (2/γ) ‖V^f[:,b]‖ (`Ground::floors`), the term's bound that times |P_E[b,:]|, below double precision's
/// round-off of the generator.
#[derive(Clone,Debug)]
pub struct Mode{pub members:Vec<usize>,pub energy:f64,pub plain:Vec<usize>,pub lu:Option<Lu>,pub dressed:Vec<Dressing>,pub skipped:Vec<(usize,f64)>}
impl Block{
    /// The projector of eigen-energy m's eigenstates between the cluster's places p and x, Σ_j u_j[p] u_j[x]*.
    pub fn projector(&self,m:usize,p:usize,x:usize)->Complex{self.modes[m].members.iter().fold(Complex::ZERO,|s,&j|s+self.vectors[(p,j)]*self.vectors[(x,j)].conj())}
}
/// A term's dressing (`Ground::dress_light`): its state's place in the cluster, the excited states'
/// self-energy Σ_b(z_j) of the state's far channels (`Ground::vias`), each one with its D_a and D_a² times
/// its energy's error; and its factorization at the elimination.
#[derive(Clone,Debug)]
pub struct Dressing{pub place:usize,pub sigma:Dense,pub vias:Vec<(Channel,f64,f64)>,pub lu:Option<Lu>}
/// A near carrier's unit for the dressing (`Ground::dress_light`): its carrier, its folded coupling (the
/// dominant polarization's W^f = Σ_l u_l V_l^f, a beam alone its own V_l^f) and its beams' remainders
/// off it, V_l^f − γ_l W^f.
#[derive(Clone,Debug)]
pub struct NearUnit{pub carrier:usize,pub coupling:Dense,pub remainders:Vec<Dense>}

/// A cluster of ground states' fourth-order fold at a point and a time (`Ground::fourth`, the dense
/// reference): H̃ (its anti-Hermitian part its jumps' −(i/2) Σ L̃†L̃), every class's jumps L̃ per axis
/// (target cluster's ground states × cluster), the cluster's post-jump map Π (as a jump's target:
/// the slow part M^{1/2}(1 + YW) of a bare ground state, its weight A = ‖Π f‖²), and the readout to
/// fourth order: M^{−1/2} (the bare ground density M^{−1/2} ρ M^{−1/2}) and per excited state the
/// operator M^{−1/2} Σ_c W_c[i,:]† W_c[i,:] M^{−1/2} of its population; and per excited state i the
/// transient's population after a jump into the cluster, Q_i = Σ_g (κ_i/Γ_i) X_g[i,:]†X_g[i,:] of the
/// bare state f (per coherent group of channels, κ = 1 − Γ²/(Δ² + Γ²/4): the
/// dressing rings above its mean far from resonance, and rebuilds below it at resonance).
#[derive(Clone,Debug)]
pub struct Fourth{pub effective:Dense,pub jumps:Vec<[Dense;3]>,pub slow:Dense,pub root:Dense,pub excited:Vec<Dense>,pub surplus:Vec<(usize,Dense)>}

/// A cluster's generator at a point and a time (the dense reference of model.rs): H_eff, every class's
/// jumps per axis (target cluster's ground states × cluster), the readout's Y and E_i (per excited
/// state).
#[derive(Clone,Debug)]
pub struct Local{pub effective:Dense,pub jumps:Vec<[Dense;3]>,pub admixture:Dense,pub excited_readout:Vec<Dense>}

/// The internal state's generator of a request: the operators act on the ground states in the order
/// of `ground_states`, the excited ones in the order of `excited_states`; rates in rad/us, jumps in
/// 1/sqrt(us).
#[derive(Clone,Debug)]
pub struct Ground{
    /// The request's ground and excited states (indices into `Request::states`).
    pub ground_states:Vec<usize>,
    pub excited_states:Vec<usize>,
    /// Each ground state's F.
    pub levels:Vec<u8>,
    /// D = Σ_a C_a†C_a on the excited states, and C_a (ground ← excited) on the request's axes.
    pub decay:Dense,
    pub emission:[Dense;3],
    /// The request's beams, and each one's co-rotating V_l (excited ← ground at the focus, rad/us)
    /// and its folded part V_l^f (every kept drive's entry zeroed).
    pub beams:Vec<Beam>,
    pub couplings:Vec<Dense>,
    pub folded:Vec<Dense>,
    /// H0 at the focus (32 × 32, rad/us); the units of the light's static shifts, and per unit its
    /// shift through the states outside the model O_u and the fold F_u of its folded components (both
    /// 32 × 32, at s_u = 1); per frequency of several beams the interference its units leave out
    /// (`coherence`).
    pub hamiltonian:Dense,
    pub units:Vec<Unit>,
    pub outside:Vec<Dense>,
    pub folds:Vec<Dense>,
    pub interference:Vec<Interference>,
    /// Every frequency's two components, classified, and the bare ‖L0‖ they were judged against.
    pub components:Vec<Component>,
    pub l0:f64,
    pub carriers:Vec<Carrier>,
    /// The solve's duration T (us; 0 without one): a mismatch below 1/T is a resonance over the run.
    pub duration:f64,
    /// Per unit, its least intensity within the atom's reach (`Ground::reach`): 1 for a plane wave.
    pub reach:Vec<f64>,
    /// The bottom of the reference state's well in H(r) alone (`Ground::well`): the atoms' reach is about it.
    pub centre:[f64;3],
    /// The couplings of the excited states, classified, with the closure's verdicts.
    pub excited_items:Vec<ExcitedItem>,
    /// The fold's charge: the largest total saturation over the light's pattern of a ground state's
    /// folded drives (with its cluster's dressing), its populations' error bound.
    pub charge:f64,
    /// The ground states' couplings, classified, the rule's cut between near and far frequencies
    /// and the eliminated generator's ‖L0‖ they were judged against (rad/us).
    pub items:Vec<Item>,
    pub cut:f64,
    pub generator_scale:f64,
    /// The light shift's bound over the trap (rad/us), part of that scale.
    pub light:f64,
    /// The nearest excited pole any coupled ground state sees over the trap, min σ_min(K − E_b − δ_c),
    /// overall and per carrier (its folded drives').
    pub pole:f64,
    pub poles:Vec<f64>,
    /// The clusters, and each ground state's cluster.
    pub clusters:Vec<Cluster>,
    pub cluster:Vec<usize>,
    /// Per ground state, its cluster's explicit excited states (flags over `excited_states`): the
    /// folded amplitudes it drives do not pass through them (`Ground::eliminate`).
    pub explicit_of:Vec<Vec<bool>>,
    /// The excited states by manifold (a line's, D1 or D2): a folded amplitude's ringing after a jump keeps to its manifold (engine.rs, # The transient).
    pub excited_groups:Vec<Vec<usize>>,
    /// The near light's dressing (`Ground::dress_light`): per carrier and ground state the far channels of
    /// its folded pairs, per unit of a near carrier its fields, and whether the eliminations are dressed --
    /// after the rule, whose judgements are the bare light's.
    pub vias:Vec<Vec<Vec<Channel>>>,
    pub near_units:Vec<Option<NearUnit>>,
    pub dressing:bool,
    /// Per cluster the floor of its responses' resolvents, γ/2 ≤ σ_min(K + Σ − z) at any real z: γ the smallest
    /// eigenvalue of the decay D decoupled from its explicit states (K's anti-Hermitian part is −(i/2) D there, Σ
    /// is Hermitian), |x†(K + Σ − z)x| ≥ ½ x†D x.
    pub floors:Vec<f64>,
    /// Per carrier and cluster whether its block is taken unmixed: each state at its own energy h_bb + δ_c, the block's
    /// off-diagonal part left out (`Ground::unmixed_blocks`, # The static ground block); and the largest relative change
    /// of their responses it makes, booked.
    pub unmixed:Vec<Vec<bool>>,
    pub unmixed_error:f64,
    /// Per carrier and ground state its dressed response's nearest pole over the trap (rad/us).
    pub channel_poles:Vec<Vec<f64>>,
    /// Per excited state, the excited states H's folded couplings join it to with a separation the
    /// run resolves (its distance over the trap beyond 1/T): its dressing's and van Vleck shift's
    /// partners. A folded coupling of a smaller separation is one of size below 1/T (the rule's), a
    /// resonance of nothing over the run.
    pub dressed:Vec<Vec<usize>>,
}

/// The manifold and F of a state label `manifold-F-m`.
fn level(label:&str)->Result<(&str,u8),String>{
    let mut parts=label.splitn(3,'-');
    match(parts.next(),parts.next().and_then(|f|f.parse().ok()),parts.next()){
        (Some(manifold),Some(f),Some(_))=>Ok((manifold,f)),_=>Err(format!("The state label {label} is not manifold-F-m")),
    }
}
/// The LU factors of K + shift, the denominator of an excited response.
pub(crate) fn denominator(k:&Dense,shift:f64)->Result<Lu,String>{
    let mut a=k.clone();for i in 0..a.rows{a[(i,i)]+=Complex::new(shift,0.);}
    Lu::new(&a).ok_or_else(||"The excited states' resolvent is singular".to_string())
}
/// Σ_l weight_l m_l over the given beams.
fn combine(beams:&[usize],weights:&[Complex],matrices:&[Dense])->Dense{
    let mut out=Dense::zeros(matrices[0].rows,matrices[0].cols);for &l in beams{out.add_scaled(weights[l],&matrices[l]);}out
}
/// The rows × cols block of a square matrix.
fn sub(m:&Dense,rows:&[usize],cols:&[usize])->Dense{Dense::from_fn(rows.len(),cols.len(),|i,j|m[(rows[i],cols[j])])}
fn frobenius(m:&Dense)->f64{m.entries.iter().map(|z|z.norm_sqr()).sum::<f64>().sqrt()}
/// The second-order (van Vleck) shift of a coupling V e^{−iνt} + h.c. (V excited ← ground) with the
/// exact bare-energy denominators: on the ground states ½ Σ_e V_ea* V_ea′ [1/(E_a + ν − E_e) + 1/(E_a′ + ν − E_e)],
/// on the excited ones ½ Σ_a V_ea V_e′a* [1/(E_e − ν − E_a) + 1/(E_e′ − ν − E_a)].
fn van_vleck(energies:&[f64],v:&Dense,nu:f64,ground:&[usize],excited:&[usize])->Dense{
    let mut out=Dense::zeros(STATES,STATES);
    for(i,&a)in ground.iter().enumerate(){for(j,&b)in ground.iter().enumerate(){let mut z=Complex::ZERO;
        for(e,&x)in excited.iter().enumerate(){z+=v[(e,i)].conj()*v[(e,j)]*(0.5*(1./(energies[a]+nu-energies[x])+1./(energies[b]+nu-energies[x])));}
        out[(a,b)]=z;}}
    for(e,&x)in excited.iter().enumerate(){for(f,&y)in excited.iter().enumerate(){let mut z=Complex::ZERO;
        for(i,&a)in ground.iter().enumerate(){z+=v[(e,i)]*v[(f,i)].conj()*(0.5*(1./(energies[x]-nu-energies[a])+1./(energies[y]-nu-energies[a])));}
        out[(x,y)]=z;}}
    out
}

/// The interference of the beams of one frequency their units leave out (`coherence`): the beams,
/// the bound of its shift over the trap (rad/us), whether they are one unit (their dominant
/// polarization's) or a unit each, whether they are a near carrier (their co-rotating field summed
/// in its tables, exactly: the units hold only their counter-rotating fold and outside shift), and
/// the largest of their units' shifts O_u + F_u (rad/us).
#[derive(Clone,Debug)]
pub struct Interference{pub beams:Vec<usize>,pub bound:f64,pub coherent:bool,pub near:bool,pub shift:f64}
impl Interference{
    /// Its estimate in the ledger: its bound relative to the nearest pole (the elimination's) and to
    /// the generator's scale -- the eliminated generator's, or the units' shifts where larger (the far
    /// light's energies: a trap of far light alone) --, as a profile's.
    pub fn estimate(&self,g:&Ground)->f64{self.bound*(1./g.pole.max(f64::MIN_POSITIVE)).max(1./g.generator_scale.max(self.shift).max(f64::MIN_POSITIVE))}
}

/// A matrix of the request with its entries at the round-off of its largest set to zero: what a selection rule forbids and the polarizations' rotation leaves at 1e-16 of the allowed is no coupling,
/// whatever its frequency (`Ground::couples`, the rule's items).
fn rounded(mut m:Dense)->Dense{
    let floor=64.*f64::EPSILON*m.entries.iter().map(|z|z.abs()).fold(0.,f64::max);
    for z in m.entries.iter_mut(){if z.abs()<=floor{*z=Complex::ZERO;}}
    m
}
/// The units of the beams of one frequency (module documentation, # Far components) and the
/// interference they leave out, bounded over the trap (every beam at most its peak), whichever of two
/// leaves out less:
/// - one unit, their dominant polarization: W = Σ_l u_l V_l, u the leading eigenvector of their
///   couplings' Gram matrix ⟨V_l, V_k⟩ (the best rank-one approximation V_l ≈ γ_l W, γ_l = ⟨W, V_l⟩/‖W‖²),
///   the counter-rotating R_W = Σ_l u_l* R_l and the outside shift O_W = Σ_l |γ_l|² O_l/Σ_l |γ_l|⁴, exact
///   for parallel fields (V_l = γ_l W, R_l = γ_l* R_W, O_l = |γ_l|² O_W). It leaves out the remainders
///   Δ_l = V_l − γ_l W (Δ′_l = R_l − γ_l* R_W): per folded component 2Γ Σ_l ‖C(W, Δ_l)‖ + Σ_lk ‖C(Δ_l, Δ_k)‖,
///   Γ = Σ_l |γ_l| and C the van Vleck shift's sesquilinear part (VV(A) = C(A, A)), ‖C(A, B)‖ ≤ ½(‖X‖ + ‖Y‖)
///   with X = VV(A + B) − VV(A) − VV(B) and Y = VV(A + iB) − VV(A) − VV(B); and of the outside shifts the
///   beams' own mismatch Σ_l ‖O_l − |γ_l|² O_W‖ and their cross terms' Σ_{l≠k} ‖O_W‖(|γ_l| ρ_k + ρ_l |γ_k| + ρ_l ρ_k),
///   ρ_l = ‖Δ_l‖/‖W‖ the remainder's field relative to W's (the couplings of the D lines' full multiplets
///   are an isometry of the field; the request gives no outside cross terms, and the estimate is exact
///   for a scalar polarizability). Both vanish with the remainders: beams of one polarization to
///   round-off are their unit to round-off, nearly parallel ones (a reflection's polarization a little
///   off) their unit with a bound of the same smallness;
/// - a unit per beam, which leaves out their interference: per pair of beams 2‖C(V_l, V_k)‖ of the
///   folded components and 2 sqrt(‖O_l‖ ‖O_k‖) of the outside shifts (exact were they of one sign) --
///   for polarizations that differ (crossed beams, a polarization lattice) the smaller.
fn coherence(beams:&[usize],couplings:&[Dense],counters:&[Dense],outside:&[Dense],folded:[Option<f64>;2],fold:&dyn Fn(&Dense,f64)->Dense)->(Vec<(Unit,Dense,Dense)>,Option<Interference>){
    let folds_of=|v:&Dense,r:&Dense|{let mut f=Dense::zeros(STATES,STATES);
        if let Some(nu)=folded[0]{f.add_scaled(Complex::ONE,&fold(v,nu));}if let Some(nu)=folded[1]{f.add_scaled(Complex::ONE,&fold(r,nu));}f};
    let single=|l:usize|(Unit{beams:vec![l],weights:vec![Complex::ONE],norm:1.,focus:1.,combination:vec![Complex::ONE]},outside[l].clone(),folds_of(&couplings[l],&counters[l]));
    if beams.len()==1{return (vec![single(beams[0])],None);}
    let inner=|a:&Dense,b:&Dense|a.entries.iter().zip(&b.entries).fold(Complex::ZERO,|s,(x,y)|s+x.conj()**y);
    let combination=|m:&[Dense],w:&dyn Fn(usize)->Complex|{let mut out=Dense::zeros(m[0].rows,m[0].cols);for(i,&l)in beams.iter().enumerate(){out.add_scaled(w(i),&m[l]);}out};
    // ‖C(A, B)‖'s bound by the polarization identity.
    let cross=|a:&Dense,b:&Dense,nu:f64|{let mut base=fold(a,nu);base.add_scaled(Complex::ONE,&fold(b,nu));
        let(mut sum,mut turned)=(a.clone(),a.clone());sum.add_scaled(Complex::ONE,b);turned.add_scaled(Complex::new(0.,1.),b);
        let(mut x,mut y)=(fold(&sum,nu),fold(&turned,nu));x.add_scaled(-Complex::ONE,&base);y.add_scaled(-Complex::ONE,&base);0.5*(x.spectral_norm()+y.spectral_norm())};
    let n=beams.len();
    // A unit per beam: their interference.
    let mut apart=0.;
    for i in 0..n{for k in i+1..n{let(l,m)=(beams[i],beams[k]);
        for(nu,v)in [(folded[0],couplings),(folded[1],counters)]{if let Some(nu)=nu{apart+=2.*cross(&v[l],&v[m],nu);}}
        apart+=2.*(outside[l].spectral_norm()*outside[m].spectral_norm()).sqrt();
    }}
    // The dominant polarization's unit and its remainders.
    let gram:Vec<Complex>=(0..n*n).map(|x|inner(&couplings[beams[x/n]],&couplings[beams[x%n]])).collect();
    let together=crate::eigen::eigen(&gram,n).and_then(|e|{
        let j=(0..n).max_by(|&a,&b|e.values[a].re.total_cmp(&e.values[b].re))?;let u:Vec<Complex>=(0..n).map(|i|e.vectors[i*n+j]).collect();
        let w=combination(couplings,&|i|u[i]);let ww=inner(&w,&w).re;if !(ww>0.){return None;}
        let rw=combination(counters,&|i|u[i].conj());
        let gamma:Vec<Complex>=beams.iter().map(|&l|inner(&w,&couplings[l])*(1./ww)).collect();
        let quartic:f64=gamma.iter().map(|g|g.norm_sqr().powi(2)).sum();
        let ow=combination(outside,&|i|Complex::new(gamma[i].norm_sqr()/quartic,0.));
        let remainder=|m:&[Dense],base:&Dense,g:&dyn Fn(Complex)->Complex|->Vec<Dense>{beams.iter().zip(&gamma).map(|(&l,&c)|{let mut d=m[l].clone();d.add_scaled(-g(c),base);d}).collect()};
        let(dv,dr)=(remainder(couplings,&w,&|c|c),remainder(counters,&rw,&|c|c.conj()));
        let total:f64=gamma.iter().map(|g|g.abs()).sum();
        let mut bound=0.;
        for(nu,base,d)in [(folded[0],&w,&dv),(folded[1],&rw,&dr)]{
            let Some(nu)=nu else{continue};
            for i in 0..n{bound+=2.*total*cross(base,&d[i],nu)+fold(&d[i],nu).spectral_norm();for k in i+1..n{bound+=2.*cross(&d[i],&d[k],nu);}}
        }
        let rho:Vec<f64>=dv.iter().map(|d|(inner(d,d).re/ww).sqrt()).collect();let size=ow.spectral_norm();
        for(i,&l)in beams.iter().enumerate(){let mut d=outside[l].clone();d.add_scaled(Complex::new(-gamma[i].norm_sqr(),0.),&ow);bound+=d.spectral_norm();
            for k in 0..n{if k!=i{bound+=size*(gamma[i].abs()*rho[k]+rho[i]*gamma[k].abs()+rho[i]*rho[k]);}}}
        let norm=total*total;let focus=gamma.iter().fold(Complex::ZERO,|s,&g|s+g).norm_sqr()/norm;
        Some(((Unit{beams:beams.to_vec(),weights:gamma,norm,focus,combination:u},ow.scaled(Complex::new(norm,0.)),folds_of(&w,&rw).scaled(Complex::new(norm,0.))),bound))
    });
    let(units,bound,coherent)=match together{Some((unit,bound))if bound<=apart=>(vec![unit],bound,true),_=>(beams.iter().map(|&l|single(l)).collect(),apart,false)};
    let shift=units.iter().map(|(_,o,f)|{let mut x=o.clone();x.add_scaled(Complex::ONE,f);x.spectral_norm()}).fold(0.,f64::max);
    (units,Some(Interference{beams:beams.to_vec(),bound,coherent,near:folded[0].is_none(),shift}))
}

/// m = size/(|ν| − l0), a coupling's mixing.
pub fn mixing(nu:f64,size:f64,l0:f64)->f64{size/(nu.abs()-l0).max(f64::MIN_POSITIVE)}
/// An item of frequency ν and size g may fold, at second order, when it is beyond the generator's scale l0,
/// |ν| ≥ FOLD_DISTANCE l0 (the population core's distance: ‖(L0 − iν)⁻¹‖ ≤ 1/(|ν| − l0) with room), with
/// its mixing m² = (g/(|ν| − l0))² within the budget, the error it leaves (booked: `Ground::fold_estimates`).
pub fn foldable(nu:f64,size:f64,l0:f64)->bool{nu.abs()>=FOLD_DISTANCE*l0&&mixing(nu,size,l0).powi(2)<=BUDGET}
/// The smallest ratio of the frequency gap the classes' photons interfere below, far over near: the population
/// core's for its lattice (`classify`'s cut).
pub fn fold_gap()->f64{1./BUDGET.sqrt()}
/// The rule over items of frequency ν and size g: every item that may fold folds (`foldable`; one of size
/// zero trivially) -- a van Vleck fold's denominators are exact, and it needs no gap between the
/// frequencies it keeps and those it folds (the population core's lattice cuts its frequencies at a gap
/// for its own time scales, which kept foldable items below the cut, and refused scenes for them where they
/// could not stay: the validation's F2). The verdicts (true: folded) and the cut the classes' photons
/// interfere below, kept the population core's: their largest ratio gap of at least `fold_gap` above every
/// frequency that may not fold (FOLD_DISTANCE l0 or below where every one may), infinite without one --
/// photons of frequencies closer than it share a class, their interference kept with its beat.
pub fn classify(items:&[(f64,f64)],l0:f64)->(Vec<bool>,f64){
    let mut frequencies:Vec<(f64,bool)>=items.iter().filter(|i|i.1>0.).map(|&(nu,size)|(nu.abs(),foldable(nu,size,l0))).collect();
    frequencies.sort_by(|a,b|a.0.total_cmp(&b.0));
    frequencies.dedup_by(|a,b|{let same=a.0==b.0;if same{b.1&=a.1;}same});
    let first=frequencies.iter().rposition(|f|!f.1).map_or(0,|i|i+1);
    let mut best:Option<(f64,usize)>=None;
    for i in first..frequencies.len(){
        let ratio=if i==0{f64::INFINITY}else{frequencies[i].0/frequencies[i-1].0};
        if i>0&&ratio<fold_gap(){continue;}
        if best.is_none_or(|(r,_)|ratio>r){best=Some((ratio,i));}
    }
    let cut=match best{None=>f64::INFINITY,Some((_,0))=>(FOLD_DISTANCE*l0).min(frequencies[0].0),Some((_,i))=>(frequencies[i-1].0*frequencies[i].0).sqrt()};
    (items.iter().map(|&(nu,size)|size==0.||foldable(nu,size,l0)).collect(),cut)
}
/// The rule's estimates for a coupling of an excited state: k (g / max(d − dressing, 1/T))² and over
/// the light's pattern k P/max(d − dressing, 1/T)², P its power, k = SATURATION for a drive and 1 for
/// a static coupling; infinite when the distance left is zero.
fn excited_estimate(item:&ExcitedItem,dressing:f64,floor:f64)->(f64,f64){
    let k=match item.kind{ExcitedKind::Drive(_)=>SATURATION,ExcitedKind::Static{..}=>1.};let d=(item.distance-dressing).max(floor);
    if d>0.{(k*(item.size/d).powi(2),k*item.power/(d*d))}else{(f64::INFINITY,f64::INFINITY)}
}

/// A union of sets over 0..n: each element's representative.
struct Union{parent:Vec<usize>}
impl Union{
    fn new(n:usize)->Union{Union{parent:(0..n).collect()}}
    fn find(&mut self,x:usize)->usize{let mut x=x;while self.parent[x]!=x{self.parent[x]=self.parent[self.parent[x]];x=self.parent[x];}x}
    fn join(&mut self,a:usize,b:usize){let(a,b)=(self.find(a),self.find(b));if a!=b{self.parent[a.max(b)]=a.min(b);}}
    /// The sets, each in increasing order, by their smallest element.
    fn sets(&mut self)->Vec<Vec<usize>>{
        let n=self.parent.len();let mut out:Vec<Vec<usize>>=vec![];let mut at=vec![usize::MAX;n];
        for x in 0..n{let r=self.find(x);if at[r]==usize::MAX{at[r]=out.len();out.push(vec![]);}out[at[r]].push(x);}
        out
    }
}

/// The bounds over the trap the couplings between ground states are judged by, from the
/// eliminations (of the folded drives) at intensities spanning every Gaussian beam's range: per pair
/// of channels the largest |X_ab| and |X_ba| (each beam at most its peak), per photon its largest
/// amplitude per axis, per pair of states the largest static coupling, the worst separations, the
/// eliminated generator's scale and the pole.
struct Bounds{
    /// [c][c′][a][b]: the bound of |X_ab| absorbing c, emitting c′.
    raman:Vec<Vec<Vec<Vec<f64>>>>,
    /// [c][b][a][axis]: the bound of |[C_axis R_c]_ab|.
    photons:Vec<Vec<Vec<[f64;3]>>>,
    /// [a][b]: the largest |H_ab|.
    statics:Vec<Vec<f64>>,
    /// [a][b]: the largest E_a − E_b over the intensities (−high[b][a] the smallest).
    high:Vec<Vec<f64>>,
    /// The eliminated generator's light shift and scattering (rad/us): bounds over the trap.
    light:f64,scattering:f64,
    poles:Vec<f64>,
}
impl Bounds{
    /// The worst separation of a pair of states over the trap with a constant beat added: E_a − E_c + beat over its range, zero where that holds zero.
    fn separation(&self,a:usize,c:usize,beat:f64)->f64{let(lo,hi)=(beat-self.high[c][a],beat+self.high[a][c]);if lo<=0.&&hi>=0.{0.}else{lo.abs().min(hi.abs())}}
}

/// The corners of the varying units' intensities (`Unit::varies`) and the levels between them,
/// {0, ¼, ½, ¾, 1} each (the others at their own): H depends on the point through them, affinely.
fn intensity_points(units:&[Unit],beams:&[Beam])->Vec<Vec<f64>>{
    let varying:Vec<usize>=(0..units.len()).filter(|&u|units[u].varies(beams)).collect();
    let levels=[0.,0.25,0.5,0.75,1.];let points=levels.len().pow(varying.len() as u32);
    (0..points).map(|point|{let mut intensities:Vec<f64>=units.iter().map(|u|u.focus).collect();
        for(f,&u)in varying.iter().enumerate(){intensities[u]=levels[(point/levels.len().pow(f as u32))%levels.len()];}intensities}).collect()
}

/// The points an atom's reach is judged at: the thermal ellipsoid about the trap's centre (`Ground::well`), its ends along its principal axes
/// (`axes`' rows, half-widths `reach`), and with `inside` a grid of nine per axis within it too, for
/// a pattern on the wavelength's scale (interfering beams).
pub(crate) fn reach_points(centre:[f64;3],axes:&[[f64;3];3],reach:[f64;3],inside:bool)->Vec<[f64;3]>{
    let at=|q:[f64;3]|->[f64;3]{std::array::from_fn(|i|centre[i]+(0..3).map(|a|q[a]*axes[a][i]).sum::<f64>())};
    let mut out:Vec<[f64;3]>=(0..3).flat_map(|a|[-1.,1.].map(move|s|(a,s))).map(|(a,s)|{let mut q=[0.;3];q[a]=s*reach[a];at(q)}).collect();
    if inside{for i in -4_i32..=4{for j in -4_i32..=4{for k in -4_i32..=4{
        if i*i+j*j+k*k<=16{out.push(at([i as f64*reach[0]/4.,j as f64*reach[1]/4.,k as f64*reach[2]/4.]));}
    }}}}
    out
}

/// The central differences a trap's derivatives are taken over: a thousandth of its beam's waist.
pub(crate) fn difference(waist:f64)->f64{1e-3*waist}
/// A potential u at r, its gradient and its Hessian, by central differences over h.
pub(crate) fn slopes(u:&dyn Fn([f64;3])->f64,r:[f64;3],h:f64)->(f64,[f64;3],[[f64;3];3]){
    let e=|a:usize|{let mut v=[0.;3];v[a]=h;v};let add=|x:[f64;3],y:[f64;3]|->[f64;3]{std::array::from_fn(|i|x[i]+y[i])};let neg=|x:[f64;3]|x.map(|v|-v);let at=|x:[f64;3]|u(add(r,x));
    let u0=u(r);
    let gradient=std::array::from_fn(|a|(at(e(a))-at(neg(e(a))))/(2.*h));
    let hessian=std::array::from_fn(|a|std::array::from_fn(|b|if a==b{(at(e(a))+at(neg(e(a)))-2.*u0)/(h*h)}
        else{(at(add(e(a),e(b)))-at(add(e(a),neg(e(b))))-at(add(neg(e(a)),e(b)))+at(add(neg(e(a)),neg(e(b)))))/(4.*h*h)}));
    (u0,gradient,hessian)
}
/// The bottom of the well a potential u's descent from `start` reaches (module documentation, # Where the trap is), or `start` where it
/// reaches none: a step of Newton's along each eigenvector of the Hessian curved upward, downhill along the others -- at a top, where a tie has no slope to follow, along the eigenvector as
/// it is --, within a radius from h up, doubled after a step it cut short and halved where u did not fall. It ends where a step's predicted gain is within u's rounding, its point a bowl's bottom or none; or where it
/// leaves `inside`. A bowl at the start is the answer as it stands: the start's very bits.
pub(crate) fn descent(u:&dyn Fn([f64;3])->f64,start:[f64;3],h:f64,inside:&dyn Fn([f64;3])->bool)->[f64;3]{
    let(mut r,mut radius)=(start,h);let(mut u0,mut g,mut c)=slopes(u,r,h);
    loop{
        let(values,vectors)=crate::linear::symmetric_eigen(c);let round=64.*f64::EPSILON*u0.abs();
        let along=vectors.map(|v|v[0]*g[0]+v[1]*g[1]+v[2]*g[2]);
        let bowl=values.iter().all(|&l|l*h*h>round);
        let steps:[f64;3]=std::array::from_fn(|i|if values[i]*h*h>round{-along[i]/values[i]}else if along[i].abs()*h>round{-along[i].signum()*radius}else if values[i]*h*h< -round{radius}else{0.});
        let length=steps.iter().map(|x|x*x).sum::<f64>().sqrt();
        loop{
            let scale=if length>radius{radius/length}else{1.};
            let gain=-(0..3).map(|i|{let x=steps[i]*scale;along[i]*x+0.5*values[i]*x*x}).sum::<f64>();
            if !(gain>round){return if bowl{r}else{start};}
            let next:[f64;3]=std::array::from_fn(|a|r[a]+(0..3).map(|i|steps[i]*scale*vectors[i][a]).sum::<f64>());
            if !inside(next){return start;}
            if u(next)<u0{r=next;(u0,g,c)=slopes(u,r,h);if scale<1.{radius*=2.;}break;}
            radius=radius.min(length)*0.5;
        }
    }
}

/// A block's eigen-energy m's terms for one beam (`Mode`), each its response, with `second` its second, and its
/// weights over the cluster's places: the plain ones per eigenstate j, Σ_b V^f[:,b] u_j[b] through `plain` with
/// u_j[:]*; each dressed one, V^f[:,b] through its own resolvent (`dressed`) with P_E[b,:]. `sources` per ground state.
fn mode_terms(cl:&Cluster,block:&Block,m:usize,sources:&[Vec<Complex>],plain:Option<&Lu>,dressed:&[&Lu],second:bool)->Vec<(Vec<Complex>,Option<Vec<Complex>>,Vec<Complex>)>{
    let(u,mode,n0)=(&block.vectors,&block.modes[m],cl.states.len());let mut columns:Vec<(Vec<Complex>,&Lu,Vec<Complex>)>=vec![];
    if let Some(lu)=plain{for &j in &mode.members{
        let mut column=vec![Complex::ZERO;sources[cl.states[mode.plain[0]]].len()];
        for &i in &mode.plain{let w=u[(i,j)];for(x,y)in column.iter_mut().zip(&sources[cl.states[i]]){*x+=*y*w;}}
        columns.push((column,lu,(0..n0).map(|x|u[(x,j)].conj()).collect()));
    }}
    for(d,&lu)in mode.dressed.iter().zip(dressed){columns.push((sources[cl.states[d.place]].clone(),lu,(0..n0).map(|x|block.projector(m,d.place,x)).collect()));}
    columns.into_iter().map(|(mut column,lu,weights)|{lu.solve(&mut column);let next=second.then(||{let mut x=column.clone();lu.solve(&mut x);x});(column,next,weights)}).collect()
}
/// A term's response spread over the cluster's columns: out[:,b] += r w[b].
fn spread(cl:&Cluster,r:&[Complex],weights:&[Complex],out:&mut Dense){
    for(&b,&w)in cl.states.iter().zip(weights){if w==Complex::ZERO{continue;}for(x,&y)in r.iter().enumerate(){out[(x,b)]+=y*w;}}
}

/// The atoms' reach along an axis of a harmonic trap at ω (rad/us): the end of the thermal ellipsoid, the region within ln(1/`BUDGET`) k_B T of its bottom, which holds all but the budget of a Boltzmann ensemble (T in μK, hm = ħ/m). None where the axis does not confine.
pub(crate) fn thermal_reach(w:f64,temperature_uk:f64,hm:f64)->Option<f64>{
    (w>0.).then(||(2.*BUDGET.recip().ln()*crate::stats::KT_PER_UK*temperature_uk*hm).sqrt()/w)
}

/// Two frames are one: equal to the rounding of the sums of detunings they are made of.
pub(crate) fn same_frame(a:f64,b:f64)->bool{(a-b).abs()<=64.*f64::EPSILON*(a.abs()+b.abs())}

/// The partition the rule leaves for one set of verdicts on the excited couplings: the ground
/// couplings classified, the cut, the clusters (sets of nodes: the ground states, then the excited
/// states at ng + e), each node's cluster, and the frames (NaN off the clusters).
struct Partition{items:Vec<Item>,cut:f64,sets:Vec<Vec<usize>>,set:Vec<Option<usize>>,frames:Vec<f64>,combinations:Vec<Vec<i32>>}

impl Ground{
    /// The generator of a request with every drive folded (`Ground::with` at an infinite allowance).
    pub fn new(request:&Request)->Result<Ground,String>{Ground::with(request,f64::INFINITY)}
    /// The generator of a request, its drives kept until each ground state's folded rest is within
    /// `allowance` (its total saturation over the light's pattern, the fold's charge), or why it has
    /// none: a counter-rotating component that may not fold, or a static coupling between ground
    /// states that may not.
    pub fn with(request:&Request,allowance:f64)->Result<Ground,String>{Ground::bare(request)?.clustered(allowance)}
    /// The rule at an allowance on a generator before it (`bare`), which a solve's candidates share.
    pub(crate) fn clustered(&self,allowance:f64)->Result<Ground,String>{let mut ground=self.clone();ground.cluster_states(allowance)?;Ok(ground)}
    /// A request's generator before the rule: its states, components, carriers and reach.
    pub(crate) fn bare(request:&Request)->Result<Ground,String>{
        let labels=request.states.iter().map(|s|level(s)).collect::<Result<Vec<_>,_>>()?;
        let ground_states:Vec<usize>=(0..STATES).filter(|&i|labels[i].0=="ground").collect();
        let excited_states:Vec<usize>=(0..STATES).filter(|&i|labels[i].0!="ground").collect();
        let ne=excited_states.len();
        let driven=|k:usize|labels[k/STATES].0!="ground"&&labels[k%STATES].0=="ground";
        for beam in &request.beams{for m in [&beam.coupling,&beam.counter_coupling]{
            if (0..STATES*STATES).any(|k|!driven(k)&&(m.real[k]!=0.||m.imag[k]!=0.)){return Err(format!("Beam {}'s couplings must be excited ← ground",beam.id));}
        }}
        let all:Vec<usize>=(0..STATES).collect();
        let mut decay=Dense::zeros(ne,ne);
        for c in &request.emission{let c=rounded(Dense::block(c,&all,&excited_states));decay.add_scaled(Complex::ONE,&c.adjoint_product(&c));}
        let emission:[Dense;3]=std::array::from_fn(|a|rounded(Dense::block(&request.emission[a],&ground_states,&excited_states)));
        let hamiltonian=Dense::block(&request.hamiltonian,&all,&all);
        if (0..STATES).any(|i|(0..STATES).any(|j|i!=j&&hamiltonian[(i,j)]!=Complex::ZERO)){return Err("H0 must be diagonal in the model basis".into());}
        let energies:Vec<f64>=(0..STATES).map(|i|hamiltonian[(i,i)].re).collect();
        // Every component, classified by the population core's rule. ‖L0‖ with H0's diagonal centred:
        // its spread, the decay and the emission.
        let frame=request.frame_frequencies_mhz[excited_states[0]];
        let spread=energies.iter().copied().fold(f64::NEG_INFINITY,f64::max)-energies.iter().copied().fold(f64::INFINITY,f64::min);
        let l0=spread+frobenius(&decay)+emission.iter().map(|c|frobenius(c).powi(2)).sum::<f64>();
        let couplings:Vec<Dense>=request.beams.iter().map(|b|rounded(Dense::block(&b.coupling,&excited_states,&ground_states))).collect();
        let counters:Vec<Dense>=request.beams.iter().map(|b|rounded(Dense::block(&b.counter_coupling,&excited_states,&ground_states))).collect();
        let outside:Vec<Dense>=request.beams.iter().map(|b|rounded(Dense::block(&b.outside_shift,&all,&all))).collect();
        let detuning=|beam:&Beam|beam.frequency_index.iter().zip(&request.frequencies_mhz).map(|(&n,&f)|n as f64*f).sum::<f64>();
        // The beams of each frequency, in the order of their first.
        let mut frequencies:Vec<Vec<usize>>=vec![];
        for(l,beam)in request.beams.iter().enumerate(){
            match frequencies.iter_mut().find(|g|request.beams[g[0]].frequency_index==beam.frequency_index){Some(g)=>g.push(l),None=>frequencies.push(vec![l])}
        }
        let mut components:Vec<Component>=frequencies.iter().flat_map(|beams|{let delta=detuning(&request.beams[beams[0]]);
            [Component{beams:beams.clone(),counter:false,nu:TAU*delta,size:beams.iter().map(|&l|2.*frobenius(&couplings[l])).sum(),folded:false},
             Component{beams:beams.clone(),counter:true,nu:-TAU*(delta+2.*frame),size:beams.iter().map(|&l|2.*frobenius(&counters[l])).sum(),folded:false}]}).collect();
        let(verdicts,_)=classify(&components.iter().map(|c|(c.nu,c.size)).collect::<Vec<_>>(),l0);
        for(c,v)in components.iter_mut().zip(verdicts){c.folded=v;}
        let names=|beams:&[usize]|beams.iter().map(|&l|request.beams[l].name.as_str()).collect::<Vec<_>>().join(" + ");
        // A counter-rotating component that may not fold has no other place: the near carriers co-rotate.
        if let Some(c)=components.iter().find(|c|c.counter&&!c.folded&&c.size>0.){
            return Err(format!("「{}」的反旋耦合不能按 van Vleck 折叠：频率 {:.1} THz（生成元尺度的 {:.1} 倍，须至少 {FOLD_DISTANCE} 倍），混合 m² = {:.1e}（预算 {BUDGET:.0e}）；3D 运动的近共振载波只取同旋分量，无法保留它",
                names(&c.beams),c.nu.abs()/TAU/1e6,c.nu.abs()/l0,mixing(c.nu,c.size,l0).powi(2)));}
        // The units (`Unit`) and their outside shifts and folds, and per frequency of several beams the
        // interference they leave out (`coherence`).
        let fold=|v:&Dense,nu:f64|van_vleck(&energies,v,nu,&ground_states,&excited_states);
        let(mut parts,mut interference):(Vec<(Unit,Dense,Dense)>,Vec<Interference>)=(vec![],vec![]);
        for beams in &frequencies{
            let(co,counter)=(components.iter().find(|c|!c.counter&&c.beams==*beams).unwrap(),components.iter().find(|c|c.counter&&c.beams==*beams).unwrap());
            let folded=[co.folded.then_some(co.nu),counter.folded.then_some(counter.nu)];
            let(units,item)=coherence(beams,&couplings,&counters,&outside,folded,&fold);
            parts.extend(units);interference.extend(item);
        }
        // In the order of their first beams: a scene of single beams has a unit per beam, in its order.
        parts.sort_by_key(|p|p.0.beams[0]);
        let(mut units,mut unit_outside,mut unit_folds)=(vec![],vec![],vec![]);
        for(u,o,f)in parts{units.push(u);unit_outside.push(o);unit_folds.push(f);}
        // The near carriers: the near co-rotating components, each one's beams that drive.
        let carriers:Vec<Carrier>=components.iter().filter(|c|!c.counter&&!c.folded&&c.size>0.).map(|c|{
            let beams:Vec<usize>=c.beams.iter().copied().filter(|&l|frobenius(&couplings[l])>0.).collect();
            let first=&request.beams[beams[0]];Carrier{beams,detuning:TAU*detuning(first),wave_number:first.wave_number_per_um}
        }).collect();
        let levels=ground_states.iter().map(|&i|labels[i].1).collect();
        let mut excited_groups:Vec<Vec<usize>>=vec![];let mut manifolds:Vec<&str>=vec![];
        for(i,&s)in excited_states.iter().enumerate(){match manifolds.iter().position(|&m|m==labels[s].0){Some(m)=>excited_groups[m].push(i),None=>{manifolds.push(labels[s].0);excited_groups.push(vec![i]);}}}
        let duration=request.times_us.last().copied().unwrap_or(0.).max(0.);
        let mut ground=Ground{ground_states,excited_states,levels,decay,emission,beams:request.beams.clone(),folded:couplings.clone(),couplings,hamiltonian,units,outside:unit_outside,folds:unit_folds,interference,
            components,l0,carriers,duration,reach:vec![],excited_items:vec![],charge:0.,items:vec![],cut:f64::INFINITY,generator_scale:0.,light:0.,pole:f64::INFINITY,poles:vec![],clusters:vec![],cluster:vec![],
            explicit_of:vec![],excited_groups,dressed:vec![],vias:vec![],near_units:vec![],dressing:false,floors:vec![],unmixed:vec![],unmixed_error:0.,channel_poles:vec![],centre:[0.;3]};
        (ground.reach,ground.centre)=ground.reach(request);
        Ok(ground)
    }

    /// Every unit's intensity at a point r (um from the focus).
    pub fn intensities_at(&self,r:[f64;3])->Vec<f64>{let a=self.amplitudes_at(r);self.units.iter().map(|u|u.intensity(|l|a[l])).collect()}
    /// H at a point r (um from the focus): H0 with each unit's outside shift and fold at its
    /// intensity there.
    pub fn hamiltonian_at(&self,r:[f64;3])->Dense{self.hamiltonian_with(&self.intensities_at(r))}
    /// H with every unit at the given intensity s_u (module documentation, # Far components).
    pub fn hamiltonian_with(&self,intensities:&[f64])->Dense{
        let mut h=self.hamiltonian.clone();
        for((&s,unit),(outside,fold))in intensities.iter().zip(&self.units).zip(self.outside.iter().zip(&self.folds)){
            if s!=unit.focus{h.add_scaled(Complex::new(s-unit.focus,0.),outside);}
            h.add_scaled(Complex::new(s,0.),fold);
        }
        h
    }
    /// Every unit at its intensity at the focus.
    pub fn focus_intensities(&self)->Vec<f64>{self.units.iter().map(|u|u.focus).collect()}
    /// The unit a beam is in.
    pub fn unit_of(&self,beam:usize)->usize{self.units.iter().position(|u|u.beams.contains(&beam)).unwrap()}

    /// K decoupled from a cluster's explicit excited states P: their off-diagonal entries zeroed (each
    /// keeps its own K_ee). A folded amplitude driven from the cluster's ground states follows the
    /// folded excited states F through K_FF; it does not pass through P, whose amplitude is the
    /// trajectory's own, and whose coupling to F is its dressing (`Ground::dressing`).
    pub fn decoupled(&self,k:&Dense,explicit:&[bool])->Dense{
        let mut out=k.clone();
        for i in 0..k.rows{for j in 0..k.cols{if i!=j&&(explicit[i]||explicit[j]){out[(i,j)]=Complex::ZERO;}}}
        out
    }
    /// An explicit excited state's dressing by H's folded couplings to the other excited states, at
    /// first order: x_f = H_fe/(z_e − K_ff), z_e = K_ee -- each one's amplitude slaved to its own (the
    /// van Vleck fold of those couplings at first order in its operators: its photon is C(|e⟩ + x),
    /// its drive gains x†V), over its partners in `dressed`. The rule kept every coupling whose
    /// separation over the trap does not fit, and `dressed` holds the folded ones the run resolves, so
    /// no denominator is small; zero on the cluster's explicit states.
    pub fn dressing(&self,e:&Elimination,state:usize,explicit:&[bool])->Result<Vec<Complex>,String>{Ok(self.dress(&e.excited,&e.excited_hamiltonian,state,explicit))}
    fn dress(&self,excited:&Dense,k:&Dense,state:usize,explicit:&[bool])->Vec<Complex>{
        let ne=self.excited_states.len();let z=k[(state,state)];
        let mut x=vec![Complex::ZERO;ne];
        if let Some(partners)=self.dressed.get(state){for &f in partners{if !explicit[f]{x[f]=excited[(f,state)]/(z-k[(f,f)]);}}}
        x
    }
    /// The source of a folded response from ground state b through beam l: its folded coupling, less the
    /// kept drives' share the dressing carries into the folded excited states, V^f[:,b] − Σ_{e∈P} x_e V^kept[e,b]
    /// (the dressed far state f̃ = f − Σ_e x_f^(e)* e is driven by ⟨f̃|V|b⟩: at first order in x).
    fn source(&self,l:usize,b:usize,dressings:&[(usize,Vec<Complex>)],column:&mut [Complex]){
        let ne=self.excited_states.len();
        for i in 0..ne{column[i]=self.folded[l][(i,b)];}
        for(e,x)in dressings{let kept=self.couplings[l][(*e,b)]-self.folded[l][(*e,b)];if kept==Complex::ZERO{continue;}for i in 0..ne{column[i]-=x[i]*kept;}}
    }
    /// Each ground state's cluster's explicit states' dressings (state, x) for an excited block and K.
    fn dressings(&self,excited:&Dense,k:&Dense)->Vec<Vec<(usize,Vec<Complex>)>>{
        self.explicit_of.iter().map(|p|(0..p.len()).filter(|&e|p[e]).map(|e|(e,self.dress(excited,k,e,p))).collect()).collect()
    }
    /// Every near beam's sources per ground state its carrier reaches (`source`), at an excited block and
    /// K (indexed by beam, then ground state; empty elsewhere).
    fn sources(&self,excited:&Dense,k:&Dense)->Vec<Vec<Vec<Complex>>>{
        let(ng,ne)=(self.ground_states.len(),self.excited_states.len());let dressings=self.dressings(excited,k);
        let mut out=vec![vec![vec![];ng];self.beams.len()];
        for(c,carrier)in self.carriers.iter().enumerate(){for b in (0..ng).filter(|&b|self.couples(c,b)){for &l in &carrier.beams{
            let mut s=vec![Complex::ZERO;ne];self.source(l,b,dressings.get(b).map_or(&[][..],|d|d),&mut s);out[l][b]=s;
        }}}
        out
    }
    /// K decoupled from a cluster's explicit states (`decoupled`), or K.
    fn cluster_hamiltonian(&self,k:&Dense,cl:&Cluster)->Dense{
        match cl.states.first().and_then(|&b|self.explicit_of.get(b)).filter(|p|p.iter().any(|&x|x)){Some(p)=>self.decoupled(k,p),None=>k.clone()}
    }
    /// A dressed term's resolvent's factorization: K + Σ decoupled from the cluster's explicit states, shifted.
    fn term_denominator(&self,k:&Dense,cl:&Cluster,sigma:&Dense,shift:f64)->Result<Lu,String>{
        let mut dressed=k.clone();dressed.add_scaled(Complex::ONE,sigma);denominator(&self.cluster_hamiltonian(&dressed,cl),shift)
    }

    /// The elimination at the units' intensities (`Ground::hamiltonian_with`): H's ground block and its
    /// energies, its excited block, K, and every near beam's R_l and R2_l of its folded coupling -- each
    /// column through K decoupled from its cluster's explicit states (`decoupled`); before the rule's
    /// clusters at its state's own energy, from then on by each cluster's static block's eigenstates at their
    /// energies (an unmixed block's states at their own, `Ground::unmixed`), dressed by the other near light
    /// (`Ground::dress_light`, `Block`).
    pub fn eliminate(&self,intensities:&[f64])->Result<Elimination,String>{
        let hamiltonian=self.hamiltonian_with(intensities);
        let(ng,ne)=(self.ground_states.len(),self.excited_states.len());
        let ground=sub(&hamiltonian,&self.ground_states,&self.ground_states);
        let energies:Vec<f64>=(0..ng).map(|b|ground[(b,b)].re).collect();
        let excited=sub(&hamiltonian,&self.excited_states,&self.excited_states);
        let excited_energies:Vec<f64>=(0..ne).map(|i|excited[(i,i)].re).collect();
        let mut excited_hamiltonian=excited.clone();excited_hamiltonian.add_scaled(Complex::new(0.,-0.5),&self.decay);
        let(mut responses,mut derivatives)=(vec![Dense::zeros(ne,ng);self.beams.len()],vec![Dense::zeros(ne,ng);self.beams.len()]);
        let decoupled:Vec<Option<Dense>>=self.explicit_of.iter().map(|p|p.iter().any(|&x|x).then(||self.decoupled(&excited_hamiltonian,p))).collect();
        let sources=self.sources(&excited,&excited_hamiltonian);
        if !self.dressing{
            let mut column=vec![Complex::ZERO;ne];
            for(c,carrier)in self.carriers.iter().enumerate(){for b in (0..ng).filter(|&b|self.couples(c,b)){
                let lu=denominator(decoupled.get(b).and_then(|x|x.as_ref()).unwrap_or(&excited_hamiltonian),-energies[b]-carrier.detuning)?;
                for &l in &carrier.beams{
                    column.copy_from_slice(&sources[l][b]);lu.solve(&mut column);for i in 0..ne{responses[l][(i,b)]=column[i];}
                    lu.solve(&mut column);for i in 0..ne{derivatives[l][(i,b)]=column[i];}
                }
            }}
            return Ok(Elimination{energies,ground,excited,excited_energies,excited_hamiltonian,responses,derivatives,blocks:vec![]});
        }
        let mut blocks=self.dress_light(intensities,&ground,&energies,&excited_hamiltonian,&decoupled)?;
        for(c,carrier)in self.carriers.iter().enumerate(){for(k,cl)in self.clusters.iter().enumerate(){
            let Some(block)=blocks[c][k].as_mut() else{continue};
            let kc=self.cluster_hamiltonian(&excited_hamiltonian,cl);
            // Each eigenstate's terms, their factorizations kept for the dressing's bounds at this elimination
            // (`Ground::dressing_bounds`).
            for m in 0..block.modes.len(){
                let mode=&block.modes[m];let z=mode.energy;
                let plain=if mode.plain.is_empty(){None}else{Some(denominator(&kc,-z)?)};
                let dressed=mode.dressed.iter().map(|d|self.term_denominator(&excited_hamiltonian,cl,&d.sigma,-z)).collect::<Result<Vec<Lu>,String>>()?;
                let refs:Vec<&Lu>=dressed.iter().collect();
                for &l in &carrier.beams{for(first,second,weights)in mode_terms(cl,block,m,&sources[l],plain.as_ref(),&refs,true){
                    spread(cl,&first,&weights,&mut responses[l]);spread(cl,&second.unwrap(),&weights,&mut derivatives[l]);
                }}
                let mode=&mut block.modes[m];mode.lu=plain;for(d,lu)in mode.dressed.iter_mut().zip(dressed){d.lu=Some(lu);}
            }
        }}
        Ok(Elimination{energies,ground,excited,excited_energies,excited_hamiltonian,responses,derivatives,blocks})
    }

    /// The near light's dressing at the units' intensities (module documentation, # The near light's
    /// dressing, # The static ground block): per carrier c and cluster it reaches, the block
    /// h_c = H_gg + Σ_{d≠c} ΔE^d on the cluster's states -- d's light shift between b and b′,
    /// −½ Σ_u a_u² (v_ub†g_ub′ + (v_ub′†g_ub)*), g_ub = (K − E_b − δ_d)⁻¹ v_ub (K decoupled as d's response), of each of
    /// d's units' fields v_u (the dominant polarization's W^f at s_u N_u, a beam alone its own V^f at s_l, the
    /// beams' amplitudes bounded by their units' intensities) -- with each entry's left-out part bounded (the
    /// units' interference: their pairs' products exactly, a unit's remainders off its polarization by their
    /// norms; the dressing's next order, the shift times the states' admixture under d); its eigenstates u_j
    /// at z_j = λ_j + δ_c (an unmixed block's: the states themselves at h_bb + δ_c, `Ground::unmixed`); and per
    /// eigenstate the cluster's states it takes, each with far channels with its
    /// self-energy Σ_b(z_j) = Σ_{(d,a)∈vias(c,b)} V_d^f[:,a] V_d^f[:,a]†/(z_j − Ẽ_a − δ_d) (an intermediate in the cluster
    /// itself at the state's own energy z_b = h_bb + δ_c, the difference in its error) -- each intermediate's
    /// energy dressed by every near carrier, Ẽ_a = E_a + Σ_d ΔE_a^d, its error the shifts' left-out parts and
    /// the spread of a's cluster's static block about it (its row's other entries, Gershgorin's radius). A
    /// state whose term's bound is below double precision's round-off of the generator is left out (`Mode`).
    /// The shifts are computed where they enter a block (another carrier reaches the state's cluster) or an
    /// intermediate's energy or spread (its cluster holds one).
    fn dress_light(&self,intensities:&[f64],ground:&Dense,energies:&[f64],k:&Dense,decoupled:&[Option<Dense>])->Result<Vec<Vec<Option<Block>>>,String>{
        let(ng,ne,nc)=(self.ground_states.len(),self.excited_states.len(),self.carriers.len());
        let column=|m:&Dense,b:usize|->Vec<Complex>{(0..ne).map(|i|m[(i,b)]).collect()};
        let dot=|x:&[Complex],y:&[Complex]|x.iter().zip(y).fold(Complex::ZERO,|s,(a,b)|s+a.conj()**b);
        let norm=|x:&[Complex]|x.iter().map(|z|z.norm_sqr()).sum::<f64>().sqrt();
        // Per carrier its units at this node: the amplitude sqrt(s_u N_u), W^f and the remainders.
        let units:Vec<Vec<(f64,&NearUnit)>>=(0..nc).map(|d|self.near_units.iter().enumerate().filter_map(|(u,x)|x.as_ref().filter(|x|x.carrier==d).map(|x|((intensities[u]*self.units[u].norm).max(0.).sqrt(),x))).collect()).collect();
        // Per carrier d and ground state b: its units' fields there, their images under (K − E_b − δ_d)⁻¹ and
        // b's admixture under d.
        type Fields=Vec<(f64,Vec<Complex>,Vec<Vec<Complex>>)>;type Images=Vec<(Vec<Complex>,Vec<Vec<Complex>>)>;
        let mut parts:Vec<Vec<Option<(Fields,Images,f64)>>>=vec![vec![None;ng];nc];
        let holds:Vec<bool>=(0..self.clusters.len()).map(|k|self.vias.iter().flatten().flatten().any(|v|self.cluster[v.state]==k)).collect();
        for d in 0..nc{for b in (0..ng).filter(|&b|self.couples(d,b)){
            let cl=&self.clusters[self.cluster[b]];
            if !holds[self.cluster[b]]&&!(0..nc).any(|c|c!=d&&cl.states.iter().any(|&b2|self.couples(c,b2))){continue;}
            let lu=denominator(decoupled.get(b).and_then(|x|x.as_ref()).unwrap_or(k),-energies[b]-self.carriers[d].detuning)?;
            let solved=|v:&[Complex]|{let mut g=v.to_vec();lu.solve(&mut g);g};
            let fields:Fields=units[d].iter().map(|(a,x)|(*a,column(&x.coupling,b),x.remainders.iter().map(|r|column(r,b)).collect())).collect();
            let images:Images=fields.iter().map(|(_,v,rest)|(solved(v),rest.iter().map(|r|solved(r)).collect())).collect();
            let admixture=fields.iter().zip(&images).map(|((a,_,_),(g,_))|a*a*norm(g).powi(2)).sum();
            parts[d][b]=Some((fields,images,admixture));
        }}
        // d's light shift between ground states b and b2 (b = b2: its shift of b, ΔE_b^d = −Re(V_d^f†(K − E_b − δ_d)⁻¹V_d^f)_bb)
        // and the bound of what it leaves out.
        let between=|d:usize,b:usize,b2:usize|->(Complex,f64){
            let(Some((f1,i1,m1)),Some((f2,i2,m2)))=(&parts[d][b],&parts[d][b2])else{return(Complex::ZERO,0.)};
            let(mut value,mut left)=(Complex::ZERO,0_f64);
            for(u,((a,v,rest),(g,_)))in f1.iter().zip(i1).enumerate(){
                let((_,v2,_),(g2,grest2))=(&f2[u],&i2[u]);
                value-=(dot(v,g2)+dot(v2,g).conj())*(0.5*a*a);
                for(w,((aw,_,_),(gw,_)))in f2.iter().zip(i2).enumerate(){if w!=u{left+=a*aw*dot(v,gw).abs();}}
                for(r,gr)in rest.iter().zip(grest2){left+=a*(dot(v,gr).abs()+dot(r,g2).abs());for r2 in rest{left+=dot(r2,gr).abs();}}
            }
            left+=0.5*(m1+m2)*value.abs();(value,left)
        };
        // Each intermediate's energy dressed by every near carrier, and its error.
        let intermediate:Vec<bool>=(0..ng).map(|a|self.vias.iter().flatten().flatten().any(|v|v.state==a)).collect();
        let dressed:Vec<f64>=(0..ng).map(|a|if intermediate[a]{energies[a]+(0..nc).map(|d|between(d,a,a).0.re).sum::<f64>()}else{energies[a]}).collect();
        let doubt:Vec<f64>=(0..ng).map(|a|if !intermediate[a]{0.}else{(0..nc).map(|d|between(d,a,a).1).sum::<f64>()
            +self.clusters[self.cluster[a]].states.iter().filter(|&&a2|a2!=a).map(|&a2|{let(mut x,mut l)=(ground[(a,a2)],0.);
                for d in 0..nc{let(v,e)=between(d,a,a2);x+=v;l+=e;}x.abs()+l}).sum::<f64>()}).collect();
        let mut out=vec![vec![None;self.clusters.len()];nc];
        for c in 0..nc{for(k,cl)in self.clusters.iter().enumerate(){
            if !cl.states.iter().any(|&b|self.couples(c,b)){continue;}
            let n0=cl.states.len();let mut h=Dense::zeros(n0,n0);let mut left=vec![0.;n0*n0];
            for(i,&b)in cl.states.iter().enumerate(){for(j,&b2)in cl.states.iter().enumerate(){
                h[(i,j)]=ground[(b,b2)];
                for d in (0..nc).filter(|&d|d!=c){let(x,l)=between(d,b,b2);h[(i,j)]+=x;left[i*n0+j]+=l;}
            }}
            let symmetric=Dense::from_fn(n0,n0,|i,j|(h[(i,j)]+h[(j,i)].conj())*0.5);
            let error:Vec<f64>=(0..n0*n0).map(|x|0.5*(left[x]+left[(x%n0)*n0+x/n0])).collect();
            // An unmixed block (`Ground::unmixed`): each state its own eigenstate at its own energy h_bb, the off-diagonal part
            // left out.
            let(values,vectors,groups)=if self.unmixed.get(c).is_some_and(|x|x[k]){
                let mut vectors=Dense::zeros(n0,n0);for i in 0..n0{vectors[(i,i)]=Complex::ONE;}
                ((0..n0).map(|i|symmetric[(i,i)].re).collect::<Vec<f64>>(),vectors,(0..n0).map(|i|vec![i]).collect::<Vec<Vec<usize>>>())
            }else{
                let(values,vectors)=hermitian_eigen(&symmetric);
                // Eigenvalues equal to their round-off are one energy, its eigenstates' basis a choice: the decomposition's
                // residual ‖h − U Λ U†‖_F bounds how far each lies from one of h's (Weyl), and two within twice it of the
                // energy's first are not told apart -- one factorization, the response through its projector.
                let residual={let mut r=symmetric.clone();for i in 0..n0{for j in 0..n0{for x in 0..n0{r[(i,j)]-=vectors[(i,x)]*vectors[(j,x)].conj()*values[x];}}}frobenius(&r)};
                let mut order:Vec<usize>=(0..n0).collect();order.sort_by(|&a,&b|values[a].total_cmp(&values[b]));
                let mut groups:Vec<Vec<usize>>=vec![];
                for &j in &order{match groups.last_mut(){Some(g)if values[j]-values[g[0]]<=2.*residual=>g.push(j),_=>groups.push(vec![j])}}
                (values,vectors,groups)
            };
            // A term's bound, (2/γ) ‖V^f[:,b]‖ |P_E[b,:]| (‖(K + Σ − z)⁻¹‖ ≤ 2/γ, `Ground::floors`; |P_E[b,x]| ≤ sqrt(P_E[b,b])), moves
            // the generator by at most 3 ‖V^f‖ times it: left out where that is below double precision's round-off of its scale.
            let strength:Vec<f64>=cl.states.iter().map(|&b|self.carriers[c].beams.iter().map(|&l|norm(&column(&self.folded[l],b))).sum()).collect();
            let ceiling=f64::EPSILON*self.generator_scale/(3.*strength.iter().copied().fold(0.,f64::max)).max(f64::MIN_POSITIVE);
            let modes=groups.into_iter().map(|members|{
                let z=members.iter().map(|&j|values[j]).sum::<f64>()/members.len() as f64+self.carriers[c].detuning;
                let mut mode=Mode{members,energy:z,plain:vec![],lu:None,dressed:vec![],skipped:vec![]};
                for(i,&b)in cl.states.iter().enumerate(){
                    let weight=mode.members.iter().map(|&j|vectors[(i,j)].norm_sqr()).sum::<f64>().sqrt();
                    if weight==0.||!self.couples(c,b){continue;}
                    if strength[i]*weight/self.floors[k]<=ceiling{mode.skipped.push((i,strength[i]/self.floors[k]));continue;}
                    if self.vias[c][b].is_empty(){mode.plain.push(i);continue;}
                    // An intermediate in the cluster itself at the state's own energy z_b = h_bb + δ_c: at z_E, an eigenstate
                    // mostly a would return into itself through b's share -- c's own light shift on a for the pair's
                    // separation, the cluster's dynamics taken again as a virtual state --; the difference, |z_E − z_b|,
                    // in its energy's error.
                    let(own,mut sigma,mut vias)=(symmetric[(i,i)].re+self.carriers[c].detuning,Dense::zeros(ne,ne),vec![]);
                    for &via in &self.vias[c][b]{
                        let(d,a)=(via.carrier,via.state);let inside=self.cluster[a]==k;
                        let denominator=1./(if inside{own}else{z}-dressed[a]-self.carriers[d].detuning);
                        for(amp,x)in &units[d]{let w=column(&x.coupling,a);let scale=amp*amp*denominator;for p in 0..ne{for q in 0..ne{sigma[(p,q)]+=w[p]*w[q].conj()*scale;}}}
                        vias.push((via,denominator,denominator*denominator*(doubt[a]+if inside{(z-own).abs()}else{0.})));
                    }
                    mode.dressed.push(Dressing{place:i,sigma,vias,lu:None});
                }
                mode
            }).collect();
            out[c][k]=Some(Block{vectors,error,modes});
        }}
        Ok(out)
    }

    /// The dressing's bounds on the generator at an elimination (module documentation, # The near light's
    /// dressing, # The static ground block): per carrier, cluster and column of the response its first-order
    /// change by what the block and the dressings leave out -- the block's entries' left-out parts E,
    /// Σ_ij M_i[z_i, z_j] (u_i†E u_j) u_j† with M_i(z) = Σ_b G_b(z) V^f[:,b] u_i[b] eigenstate i's response at energy z
    /// (its divided differences taken as its slope at z_i), |u_i†E u_j| ≤ Σ_pq |u_i[p]| |E_pq| |u_j[q]|; per dressed
    /// term G_b times its self-energy's left-out parts -- each intermediate D_a² times its energy's error and
    /// D_a Σ_uv a_u a_v* G_b w_u (w_v† R_b), G_b applied to its units' fields and their projections on the term's
    /// response (the units taken apart: their pairs u ≠ v, a unit's remainders off its polarization) --; the
    /// terms left out, their bounds (`Mode`) --, the beams at their peak; the generator's change at most
    /// (‖V‖ + ‖D R‖) ≤ 3‖V‖ times it, ‖V‖ the channel's folded coupling, relative to the generator's scale. A
    /// column the carrier does not couple (a state the block mixes in) is not the generator's: all of it,
    /// through the cluster's widest coupling. The largest.
    pub fn dressing_bounds(&self,e:&Elimination,intensities:&[f64])->Result<f64,String>{
        let ne=self.excited_states.len();let scale=self.generator_scale.max(f64::MIN_POSITIVE);
        let column=|m:&Dense,b:usize|->Vec<Complex>{(0..ne).map(|i|m[(i,b)]).collect()};
        let dot=|x:&[Complex],y:&[Complex]|x.iter().zip(y).fold(Complex::ZERO,|s,(a,b)|s+a.conj()**b).abs();
        let norm=|x:&[Complex]|x.iter().map(|z|z.norm_sqr()).sum::<f64>().sqrt();
        let sources=self.sources(&e.excited,&e.excited_hamiltonian);
        let mut error=0_f64;
        for(c,carrier)in self.carriers.iter().enumerate(){for(k,cl)in self.clusters.iter().enumerate(){
            let Some(block)=e.blocks.get(c).and_then(|x|x[k].as_ref()) else{continue};
            let(n0,nm)=(cl.states.len(),block.modes.len());
            let share:Vec<f64>=cl.states.iter().map(|&b|3.*carrier.beams.iter().map(|&l|norm(&column(&self.folded[l],b))).sum::<f64>()).collect();
            let projector:Vec<Vec<f64>>=(0..nm).map(|m|(0..n0*n0).map(|x|block.projector(m,x/n0,x%n0).abs()).collect()).collect();
            // Per place the bound of its column's change; per eigen-energy its response's slope; per eigen-energy and source
            // place whether its term was solved and what its far channels leave out (β); per place its far channels' largest |D_a|.
            let(mut change,mut slope)=(vec![0.;n0],vec![0.;nm]);
            let(mut solved_term,mut beta,mut reach)=(vec![vec![false;n0];nm],vec![vec![0.;n0];nm],vec![0_f64;n0]);
            for(m,mode)in block.modes.iter().enumerate(){
                for &(p,bound)in &mode.skipped{for x in 0..n0{change[x]+=bound*projector[m][p*n0+x];}}
                if mode.plain.is_empty()&&mode.dressed.is_empty(){continue;}
                for &p in &mode.plain{solved_term[m][p]=true;}
                let lus:Vec<&Lu>=mode.dressed.iter().map(|d|d.lu.as_ref().expect("factorized at the elimination")).collect();
                let terms:Vec<Vec<(Vec<Complex>,Option<Vec<Complex>>,Vec<Complex>)>>=carrier.beams.iter().map(|&l|mode_terms(cl,block,m,&sources[l],mode.lu.as_ref(),&lus,true)).collect();
                // The slope: per term its second response, times its largest weight on a column.
                slope[m]=terms.iter().map(|t|t.iter().map(|(_,second,w)|norm(second.as_ref().unwrap())*w.iter().map(|x|x.abs()).fold(0.,f64::max)).sum::<f64>()).sum();
                let first=if mode.lu.is_some(){mode.members.len()}else{0};
                for(t,d)in mode.dressed.iter().enumerate(){
                    let(lu,mut left)=(lus[t],0.);let solved=|v:&[Complex]|{let mut g=v.to_vec();lu.solve(&mut g);norm(&g)};
                    let responses:Vec<&Vec<Complex>>=terms.iter().map(|x|&x[first+t].0).collect();
                    let projection=|w:&[Complex]|responses.iter().map(|r|dot(w,r)).sum::<f64>();
                    let weight=projector[m][d.place*n0+d.place].sqrt();
                    for &(via,denominator,doubt)in &d.vias{
                        let fields:Vec<(f64,Vec<Complex>,Vec<Vec<Complex>>)>=self.near_units.iter().enumerate().filter_map(|(u,x)|x.as_ref().filter(|x|x.carrier==via.carrier)
                            .map(|x|((intensities[u]*self.units[u].norm).max(0.).sqrt(),column(&x.coupling,via.state),x.remainders.iter().map(|r|column(r,via.state)).collect()))).collect();
                        for(i,(amp,w,rest))in fields.iter().enumerate(){
                            let(g,p)=(solved(w),projection(w));
                            // Σ_b's own slope, −Σ D_a² w w†: the term's slope beyond G_b² V^f.
                            slope[m]+=weight*denominator*denominator*amp*amp*g*p;left+=doubt*amp*amp*g*p;
                            for(i2,(amp2,w2,_))in fields.iter().enumerate(){if i2!=i{left+=denominator.abs()*amp*amp2*g*projection(w2);}}
                            let rests:f64=rest.iter().map(|r|projection(r)).sum();
                            for r in rest{let(gr,pr)=(solved(r),projection(r));left+=denominator.abs()*(amp*(g*pr+gr*p)+gr*rests);}
                        }
                    }
                    solved_term[m][d.place]=true;beta[m][d.place]=left;
                    reach[d.place]=d.vias.iter().map(|v|v.1.abs()).fold(reach[d.place],f64::max);
                }
            }
            // A source's far channels' left-out parts over the columns, [δF_p(h)]_{p,x} = Σ_E δF_p(z_E) P_E[p,x]: on its own
            // column at most Σ_E P_E[p,p] β_Ep; on another, the projectors summing to one, Σ_E (δF_p(z_E) − δF_p(z_E0)) P_E[p,x] with
            // its dominant eigen-energy E0 (and δF_p(z_E0) times the left-out terms' weights), each difference at most β_Ep + β_E0p
            // and |z_E − z_E0| times δF_p's slope, (2‖G‖ + max |D_a|) ≤ (2/floor + max |D_a|) times its largest.
            let floor=self.floors[k];
            for p in 0..n0{
                let top=(0..nm).map(|m|beta[m][p]).fold(0.,f64::max);if top==0.{continue;}
                let m0=(0..nm).max_by(|&a,&b|projector[a][p*n0+p].total_cmp(&projector[b][p*n0+p])).unwrap();let rate=2./floor+reach[p];
                for x in 0..n0{
                    change[x]+=if x==p{(0..nm).filter(|&m|solved_term[m][p]).map(|m|projector[m][p*n0+p]*beta[m][p]).sum::<f64>()}
                    else{(0..nm).map(|m|{let apart=(block.modes[m].energy-block.modes[m0].energy).abs();
                        projector[m][p*n0+x]*if !solved_term[m][p]{beta[m0][p]}else if apart==0.{0.}else{(beta[m][p]+beta[m0][p]).min(apart*rate*top)}}).sum::<f64>()};
                }
            }
            // The block's error: its entries' left-out parts E moving its eigenvalues and eigenvectors, the response at first
            // order by Σ_EF M_E[z_E, z_F] P_E E P_F, its divided differences at most its slope; column x of P_E E P_F at most
            // Σ_pq sqrt(P_E[p,p]) |E_pq| |P_F[q,x]|.
            for m in (0..nm).filter(|&m|slope[m]>0.){for f in 0..nm{
                let mut ef=vec![0.;n0];
                for p in 0..n0{let a=projector[m][p*n0+p].sqrt();if a==0.{continue;}for q in 0..n0{let l=block.error[p*n0+q];if l==0.{continue;}for x in 0..n0{ef[x]+=a*l*projector[f][q*n0+x];}}}
                for x in 0..n0{change[x]+=slope[m]*ef[x];}
            }}
            let widest=share.iter().copied().fold(0.,f64::max);
            for(i,&b)in cl.states.iter().enumerate(){
                if self.couples(c,b){error=error.max(share[i]*change[i]/scale);continue;}
                error=error.max(widest*carrier.beams.iter().map(|&l|norm(&column(&e.responses[l],b))).sum::<f64>()/scale);
            }
        }}
        Ok(error)
    }

    /// A frame or a beat from its integer combination m of the carriers' detunings (`Cluster::combinations`):
    /// root + Σ_c m_c δ_c, one sum in the carriers' order -- equal combinations give equal values, and a
    /// combination of zeros the root exactly.
    pub fn beat(&self,root:f64,m:&[i32])->f64{m.iter().zip(&self.carriers).fold(root,|s,(&n,c)|if n==0{s}else{s+f64::from(n)*c.detuning})}
    /// A sum of carriers' detunings, Σ plus − Σ minus, from its integer combination evaluated once (`Ground::beat`):
    /// exact where its terms cancel (δ_c − δ_d + δ_d is δ_c to the last bit, however large δ_d), so that a fourth-order
    /// triple's frequency is its photon carrier's.
    pub fn frequency(&self,plus:&[usize],minus:&[usize])->f64{let mut m=vec![0;self.carriers.len()];for &c in plus{m[c]+=1;}for &c in minus{m[c]-=1;}self.beat(0.,&m)}
    /// Whether carrier c's folded drives reach ground state b at all (some beam's folded coupling from b).
    pub fn couples(&self,c:usize,b:usize)->bool{self.carriers[c].beams.iter().any(|&l|(0..self.folded[l].rows).any(|e|self.folded[l][(e,b)]!=Complex::ZERO))}

    /// The bounds over the trap (`Bounds`), from the eliminations at intensities {0, ¼, ½, ¾, 1} of
    /// every varying unit (the others at their own): H depends on the point through them, the near
    /// beams' amplitudes through |a_l| ≤ 1.
    fn bounds(&self)->Result<Bounds,String>{
        let(ng,nc)=(self.ground_states.len(),self.carriers.len());
        let mut out=Bounds{raman:vec![vec![vec![vec![0.;ng];ng];nc];nc],photons:vec![vec![vec![[0.;3];ng];ng];nc],statics:vec![vec![0.;ng];ng],
            high:vec![vec![f64::NEG_INFINITY;ng];ng],light:0.,scattering:0.,poles:vec![f64::INFINITY;nc]};
        let mut identity=Dense::zeros(self.excited_states.len(),self.excited_states.len());for i in 0..identity.rows{identity[(i,i)]=Complex::ONE;}
        // A rung's closure takes this several times over: per intensity point, beam pair and carrier on the threads, each sum in its order.
        let points:Vec<Result<(Elimination,Vec<[Dense;3]>,Vec<Result<(Vec<[Dense;3]>,f64),String>>),String>>=intensity_points(&self.units,&self.beams).par_iter().map(|intensities|{
            let e=self.eliminate(intensities)?;
            let pairs:Vec<[Dense;3]>=self.carriers.iter().flat_map(|carrier|self.carriers.iter().flat_map(move|emitted|carrier.beams.iter().flat_map(move|&l|emitted.beams.iter().map(move|&k|(l,k))))).collect::<Vec<_>>()
                .into_par_iter().map(|(l,k)|[self.folded[k].adjoint_product(&e.responses[l]),e.responses[k].adjoint_product(&self.folded[l]),e.responses[k].adjoint_product(&self.decay.product(&e.responses[l]))]).collect();
            let carriers=self.carriers.par_iter().enumerate().map(|(c,carrier)|{
                let jumps:Vec<[Dense;3]>=carrier.beams.iter().map(|&l|std::array::from_fn(|x|self.emission[x].product(&e.responses[l]))).collect();
                let mut pole=f64::INFINITY;
                for b in 0..ng{
                    if !self.couples(c,b){continue;}
                    let lu=denominator(&e.excited_hamiltonian,-e.energies[b]-carrier.detuning)?;
                    pole=pole.min(1./lu.solve_columns(&identity).spectral_norm());
                }
                Ok((jumps,pole))
            }).collect();
            Ok((e,pairs,carriers))
        }).collect();
        for point in points{
            let(e,pairs,carriers)=point?;let mut pairs=pairs.iter();
            for a in 0..ng{for b in 0..ng{
                out.high[a][b]=out.high[a][b].max(e.energies[a]-e.energies[b]);
                if a!=b{out.statics[a][b]=out.statics[a][b].max(e.ground[(a,b)].abs());}
            }}
            // Per pair of beams the three products; per carrier pair their sums of moduli.
            let mut raman=vec![vec![vec![vec![0.;ng];ng];nc];nc];let(mut light,mut scattering)=(vec![0.;ng],vec![0.;ng]);
            for(c,carrier)in self.carriers.iter().enumerate(){for(d,emitted)in self.carriers.iter().enumerate(){
                for _ in &carrier.beams{for _ in &emitted.beams{
                    let [v,w,q]=pairs.next().unwrap();
                    for a in 0..ng{for b in 0..ng{
                        raman[c][d][a][b]+=0.5*(v[(a,b)].abs()+w[(a,b)].abs()+q[(a,b)].abs());
                        if c==d{light[a]+=((v[(a,b)]+w[(a,b)])*0.5).abs();scattering[a]+=q[(a,b)].abs();}
                    }}
                }}
            }}
            for c in 0..nc{for d in 0..nc{for a in 0..ng{for b in 0..ng{out.raman[c][d][a][b]=out.raman[c][d][a][b].max(raman[c][d][a][b]);}}}}
            out.light=out.light.max(light.iter().copied().fold(0.,f64::max));out.scattering=out.scattering.max(scattering.iter().copied().fold(0.,f64::max));
            for(c,carrier)in carriers.into_iter().enumerate(){
                let(jumps,pole)=carrier?;
                for b in 0..ng{for a in 0..ng{for x in 0..3{
                    let n:f64=jumps.iter().map(|j|j[x][(a,b)].abs()).sum();out.photons[c][b][a][x]=out.photons[c][b][a][x].max(n);
                }}}
                out.poles[c]=out.poles[c].min(pole);
            }
        }
        Ok(out)
    }

    /// Each unit's least intensity within the atom's reach: the region of the trap within
    /// ln(1/budget) k_B T of its bottom, the harmonic trap's thermal ellipsoid holding all but the
    /// budget of a Boltzmann ensemble at the request's temperature (model.rs `choose_profiles` judges
    /// the units' profiles over it too), the trap the most populated initial ground state's in
    /// its well in H(r) alone (`Ground::well`); at the ellipsoid's ends along its principal axes, and inside it for
    /// interfering beams (`reach_points`). 1 for a plane wave, 0 for every varying unit when the trap
    /// does not confine. Beyond the reach, what an atom meets is measured along its trajectory (the
    /// folded excited population, the ledger's η²). And that well's place.
    /// The reference state: the request's largest initial population, a ground state (its position in `ground_states`).
    pub fn reference_state(&self,request:&Request)->Result<usize,String>{
        let rho=&request.initial_density;
        self.ground_states.iter().enumerate().max_by(|a,b|rho[a.1*STATES+a.1].re.total_cmp(&rho[b.1*STATES+b.1].re)).map(|x|x.0).ok_or("No ground state".into())
    }
    fn reach(&self,request:&Request)->(Vec<f64>,[f64;3]){
        let unconfined=|centre:[f64;3]|(self.units.iter().map(|u|if u.varies(&self.beams){0.}else{1.}).collect(),centre);
        let Some(waist)=request.beams.get(request.trap_beam).and_then(|b|b.waist_um)else{return unconfined([0.;3])};
        let reference=self.ground_states[self.reference_state(request).unwrap()];
        // The reference state's well and its Hessian there.
        let u=|r:[f64;3]|self.hamiltonian_at(r)[(reference,reference)].re;
        let centre=self.well(request.trap_beam,&u);let(values,axes)=crate::linear::symmetric_eigen(slopes(&u,centre,difference(waist)).2);
        let hm=crate::model::HBAR/request.mass_kg*1e6;
        let reach:[f64;3]=std::array::from_fn(|a|thermal_reach((values[a]*hm).max(0.).sqrt(),request.temperature_uk,hm).unwrap_or(f64::INFINITY));
        if reach.iter().any(|x|!x.is_finite()){return unconfined(centre);}
        (self.units.iter().map(|u|if !u.varies(&self.beams){1.}else{
            reach_points(centre,&axes,reach,u.beams.len()>1).into_iter().map(|r|{let a=self.amplitudes_at(r);u.intensity(|l|a[l])}).fold(1_f64,f64::min)}).collect(),centre)
    }
    /// Where a potential u of the trap beam's light has its trap: the bottom of the well its descent from the focus reaches (`descent`, module documentation, # Where the trap is), the focus where that
    /// leaves the beam's focal volume, its harmonic coordinate 2ρ²/w² + ζ²/z_R² past one, or finds no well.
    pub(crate) fn well(&self,trap_beam:usize,u:&dyn Fn([f64;3])->f64)->[f64;3]{
        let beam=&self.beams[trap_beam];let Some(w)=beam.waist_um else{return [0.;3]};let(n,rayleigh)=(beam.direction,beam.wave_number_per_um*w*w/2.);
        descent(u,[0.;3],difference(w),&|r|{let zeta=r[0]*n[0]+r[1]*n[1]+r[2]*n[2];2.*(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]-zeta*zeta)/(w*w)+zeta*zeta/(rayleigh*rayleigh)<=1.})
    }

    /// The couplings of the excited states, unjudged: every drive with its size and distance over the
    /// atom's reach (`Ground::reach`), and H's static coupling of every pair of excited states with its
    /// size and smallest separation over the whole trap (H's energies are affine in the units'
    /// intensities: their range is reached at the corners of the intensities).
    fn excited_couplings(&self)->Vec<ExcitedItem>{
        let(ng,ne)=(self.ground_states.len(),self.excited_states.len());
        let(mut low,mut high)=(vec![vec![f64::INFINITY;ng];ne],vec![vec![f64::NEG_INFINITY;ng];ne]);
        let(mut elow,mut ehigh,mut esize)=(vec![vec![f64::INFINITY;ne];ne],vec![vec![f64::NEG_INFINITY;ne];ne],vec![vec![0_f64;ne];ne]);
        let varying:Vec<usize>=(0..self.units.len()).filter(|&u|self.units[u].varies(&self.beams)).collect();
        // The drives over the reach (each varying unit at its peak or its least intensity there); H's
        // static couplings over the whole trap (each one from 0 to its peak), the range of the tables.
        let corners=|low:&dyn Fn(usize)->f64|->Vec<Vec<f64>>{(0..1usize<<varying.len()).map(|c|{let mut i=self.focus_intensities();for(f,&u)in varying.iter().enumerate(){i[u]=if c>>f&1==1{low(u)}else{1.};}i}).collect()};
        for intensities in corners(&|l|self.reach[l]){
            let h=self.hamiltonian_with(&intensities);
            for(e,&x)in self.excited_states.iter().enumerate(){for(b,&y)in self.ground_states.iter().enumerate(){let d=h[(x,x)].re-h[(y,y)].re;low[e][b]=low[e][b].min(d);high[e][b]=high[e][b].max(d);}}
        }
        for intensities in corners(&|_|0.){
            let h=self.hamiltonian_with(&intensities);
            for(e,&x)in self.excited_states.iter().enumerate(){for(f,&y)in self.excited_states.iter().enumerate(){
                let d=h[(x,x)].re-h[(y,y)].re;elow[e][f]=elow[e][f].min(d);ehigh[e][f]=ehigh[e][f].max(d);
                if e!=f{esize[e][f]=esize[e][f].max(h[(x,y)].abs());}
            }}
        }
        // |re − i width| at the nearest point of the range [lo, hi] shifted.
        let distance=|lo:f64,hi:f64,shift:f64,width:f64|{let(lo,hi)=(lo-shift,hi-shift);let re=if lo<=0.&&hi>=0.{0.}else{lo.abs().min(hi.abs())};re.hypot(width)};
        let mut out=vec![];
        for(c,carrier)in self.carriers.iter().enumerate(){for e in 0..ne{for b in 0..ng{
            let size:f64=carrier.beams.iter().map(|&l|self.couplings[l][(e,b)].abs()).sum();
            let power:f64=carrier.beams.iter().map(|&l|self.couplings[l][(e,b)].norm_sqr()).sum();
            if size>0.{out.push(ExcitedItem{kind:ExcitedKind::Drive(Drive{carrier:c,excited:e,ground:b}),size,power,
                distance:distance(low[e][b],high[e][b],carrier.detuning,0.5*self.decay[(e,e)].re),estimate:0.,mean:0.,folded:true,promoted:false});}
        }}}
        for a in 0..ne{for b in a+1..ne{let size=esize[a][b].max(esize[b][a]);
            if size>0.{out.push(ExcitedItem{kind:ExcitedKind::Static{a,b},size,power:size*size,distance:distance(elow[a][b],ehigh[a][b],0.,0.),estimate:0.,mean:0.,folded:true,promoted:false});}}}
        out
    }

    /// The drives judged per ground state: kept from the most saturated down (over the light's
    /// pattern) until the total saturation of the rest (their means' sum) fits the allowance, the rest
    /// folded; a drive kept already stays kept. Each one newly kept is passed to `kept` (whose answer
    /// says whether it counts as a change); whether any did. With `one`, at most one per ground state: keeping a drive lowers the
    /// dressing the others' estimates take (the closure).
    fn judge_drives(excited:&mut [ExcitedItem],allowance:f64,one:bool,kept:&mut dyn FnMut(&mut ExcitedItem)->bool)->bool{
        let grounds:Vec<usize>={let mut g:Vec<usize>=excited.iter().filter_map(|x|match x.kind{ExcitedKind::Drive(d)=>Some(d.ground),_=>None}).collect();g.sort();g.dedup();g};
        let mut changed=false;
        for b in grounds{
            let mut drives:Vec<usize>=(0..excited.len()).filter(|&i|matches!(excited[i].kind,ExcitedKind::Drive(d)if d.ground==b)).collect();
            drives.sort_by(|&i,&j|excited[j].mean.total_cmp(&excited[i].mean));
            let mut total:f64=drives.iter().filter(|&&i|excited[i].folded).map(|&i|excited[i].mean).sum();
            for i in drives{
                if !excited[i].folded{continue;}
                if total<=allowance{break;}
                total-=excited[i].mean;excited[i].folded=false;changed|=kept(&mut excited[i]);if one{break;}
            }
        }
        changed
    }
    /// The fold's charge of a set of verdicts: the largest total saturation over the light's pattern
    /// of a ground state's folded drives (the sum of their means).
    fn charge_of(excited:&[ExcitedItem],ng:usize)->f64{
        let mut totals=vec![0.;ng];
        for x in excited.iter().filter(|x|x.folded){if let ExcitedKind::Drive(d)=x.kind{totals[d.ground]+=x.mean;}}
        totals.into_iter().fold(0.,f64::max)
    }
    /// The explicit excited states of a set of verdicts: those a kept drive reaches, and those a
    /// kept static coupling joins to them.
    fn explicit_states(excited:&[ExcitedItem],ne:usize)->Vec<bool>{
        let mut explicit=vec![false;ne];
        for x in excited{if let(ExcitedKind::Drive(d),false)=(x.kind,x.folded){explicit[d.excited]=true;}}
        loop{
            let mut grew=false;
            for x in excited{if let(ExcitedKind::Static{a,b},false)=(x.kind,x.folded){if explicit[a]!=explicit[b]{explicit[a]=true;explicit[b]=true;grew=true;}}}
            if !grew{return explicit;}
        }
    }

    /// The excited couplings judged before the closure: their estimates without a dressing, H's static
    /// couplings folded within the budget, the drives within the allowance.
    fn judged(&self,allowance:f64)->Vec<ExcitedItem>{
        let floor=if self.duration>0.{1./self.duration}else{0.};
        let mut excited=self.excited_couplings();
        for x in excited.iter_mut(){(x.estimate,x.mean)=excited_estimate(x,0.,floor);}
        for x in excited.iter_mut(){if let ExcitedKind::Static{..}=x.kind{x.folded=x.estimate<=BUDGET;}}
        Ground::judge_drives(&mut excited,allowance,false,&mut |_|false);
        excited
    }
    /// The allowances at which the rule's explicit excited states change (`Ground::with`), from all
    /// folded (infinite) down to every drive kept, each the largest that gives its set of explicit
    /// states before the closure: the totals of a ground state's folded drives as it keeps them from
    /// the most saturated down, its total with none kept among them -- its first kept drive lies below that, with
    /// other ground states' still folded (PGCP_10: the near probe's drives alone, its cooling beams folded).
    pub fn allowances(request:&Request)->Result<Vec<f64>,String>{Ok(Ground::bare(request)?.rungs())}
    /// The allowances of a generator before the rule (`allowances`, `bare`).
    pub(crate) fn rungs(&self)->Vec<f64>{self.rungs_where(false)}
    /// The rungs a solve climbs (solve.rs, # The fold's choice): `rungs` less those that add explicit states no drive takes past the rule's per-item threshold at any position (`BUDGET`, the drive's
    /// `estimate`, a bound over the trap). Such a rung keeps what no run's statistics could read, and it can cost without limit: a tensor scene's D1 light on the D2 states, 7 THz away at 1e-12 of saturation, merges
    /// the hyperfine levels into one cluster of 23 states on a grid of that beat, and its first read of the grid outlasted five minutes where the rungs before it took a second in all. (A drive that nears
    /// resonance anywhere in the trap reads there, however low its mean: the retro-reflected tweezer's chosen rung stands at a mean of 2e-3.)
    pub(crate) fn ladder(&self)->Vec<f64>{self.rungs_where(true)}
    fn rungs_where(&self,worth:bool)->Vec<f64>{
        let g=self;let ne=g.excited_states.len();
        let excited=g.judged(f64::INFINITY);
        let mut totals:Vec<f64>=vec![];
        for b in 0..g.ground_states.len(){
            let mut means:Vec<f64>=excited.iter().filter_map(|x|match x.kind{ExcitedKind::Drive(d)if d.ground==b=>Some(x.mean),_=>None}).collect();
            means.sort_by(|x,y|y.total_cmp(x));let mut rest:f64=means.iter().sum();totals.push(rest);
            for m in means{rest-=m;totals.push(rest.max(0.));}
        }
        totals.sort_by(|x,y|y.total_cmp(x));totals.dedup();
        let mut out=vec![f64::INFINITY];let mut last=Ground::explicit_states(&excited,ne);
        for t in totals{
            let explicit=Ground::explicit_states(&g.judged(t),ne);
            if explicit!=last{
                let reads=excited.iter().any(|x|matches!(x.kind,ExcitedKind::Drive(d)if explicit[d.excited]&&!last[d.excited])&&x.estimate>BUDGET);
                if !worth||reads{out.push(t);}
                last=explicit;
            }
        }
        out
    }

    /// The rule over every coupling (module documentation): the excited couplings judged, the
    /// folded drives eliminated, the couplings between ground states judged with them, the clusters
    /// and frames; the closure judging the folded couplings of each cluster again with its dressing,
    /// until nothing changes; then each cluster's machinery.
    fn cluster_states(&mut self,allowance:f64)->Result<(),String>{
        let(ng,ne)=(self.ground_states.len(),self.excited_states.len());
        let floor=if self.duration>0.{1./self.duration}else{0.};
        let mut excited=self.judged(allowance);
        // The pairs of channels the closure keeps (`partition` keeps them whatever `classify` says).
        let mut kept:Vec<Kind>=vec![];
        loop{
            let explicit=Ground::explicit_states(&excited,ne);
            // The folded couplings: every kept drive's entry zeroed in each beam of its carrier.
            self.folded=self.couplings.clone();
            for x in &excited{if let(ExcitedKind::Drive(d),false)=(x.kind,x.folded){for &l in &self.carriers[d.carrier].beams{self.folded[l][(d.excited,d.ground)]=Complex::ZERO;}}}
            let b=self.bounds()?;
            let partition=self.partition(&b,&excited,&explicit,&kept)?;
            // Each ground state's cluster's explicit states, for the eliminations from here on.
            let explicit_of:Vec<Vec<bool>>=(0..ng).map(|x|{let k=partition.set[x];(0..ne).map(|e|explicit[e]&&partition.set[ng+e]==k).collect()}).collect();
            let moved=explicit_of!=self.explicit_of;self.explicit_of=explicit_of;
            // The closure: each cluster's dressing, the spectral norm of the bounds of its static block: its states' light shifts, its kept pairs of channels, static couplings and excited couplings.
            let dressing:Vec<f64>=partition.sets.iter().map(|set|{
                let n=set.len();let mut m=Dense::zeros(n,n);
                for(i,&x)in set.iter().enumerate(){if x<ng{m[(i,i)]=Complex::new((0..b.raman.len()).map(|c|b.raman[c][c][x][x]).sum(),0.);}}
                for item in partition.items.iter().filter(|i|!i.folded){
                    let(x,y)=match item.kind{Kind::Raman{first,second}=>(first.state,second.state),Kind::Static{a,b}=>(a,b),_=>continue};
                    if let(Some(i),Some(j))=(set.iter().position(|&s|s==x),set.iter().position(|&s|s==y)){m[(i,j)]+=Complex::new(0.5*item.size,0.);if i!=j{m[(j,i)]+=Complex::new(0.5*item.size,0.);}}
                }
                for item in excited.iter().filter(|x|!x.folded){
                    let(x,y)=match item.kind{ExcitedKind::Drive(d)=>(ng+d.excited,d.ground),ExcitedKind::Static{a,b}=>(ng+a,ng+b)};
                    if let(Some(i),Some(j))=(set.iter().position(|&s|s==x),set.iter().position(|&s|s==y)){m[(i,j)]+=Complex::new(item.size,0.);m[(j,i)]+=Complex::new(item.size,0.);}
                }
                hermitian_eigenvalues(&m).into_iter().fold(0_f64,|a,v|a.max(v.abs()))}).collect();
            // The folded couplings' estimates at this pass's dressing: what the kept couplings leave of it, the last
            // pass's the estimates the fold's charge books.
            let mut changed=false;let mut inside=vec![false;excited.len()];
            for(i,x)in excited.iter_mut().enumerate().filter(|(_,x)|x.folded){
                let reach=match x.kind{ExcitedKind::Drive(d)=>partition.set[d.ground],
                    ExcitedKind::Static{a,b}=>partition.set[ng+a].or(partition.set[ng+b])};
                let Some(k)=reach else{continue};
                (x.estimate,x.mean)=excited_estimate(x,dressing[k],floor);
                // A coupling static in the frames between two states of one cluster costs nothing kept.
                let m=&partition.combinations;
                inside[i]=match x.kind{
                    ExcitedKind::Drive(d)=>partition.set[ng+d.excited]==Some(k)&&m[ng+d.excited].iter().zip(&m[d.ground]).enumerate().all(|(c,(e,g))|*e==*g+i32::from(c==d.carrier)),
                    ExcitedKind::Static{a,b}=>partition.set[ng+a]==Some(k)&&partition.set[ng+b]==Some(k)&&m[ng+a]==m[ng+b]};
            }
            // The drives first, the most saturated of each ground state's folded total beyond the allowance: a folded near drive's
            // light is most of the dressing before it is kept, and estimates taken under it would keep what its keeping lets fold.
            changed|=Ground::judge_drives(&mut excited,allowance,true,&mut |x|{x.promoted=true;true});
            if !changed{
                for(i,x)in excited.iter_mut().enumerate().filter(|(_,x)|x.folded){
                    if inside[i]||(matches!(x.kind,ExcitedKind::Static{..})&&x.estimate>BUDGET){x.folded=false;x.promoted=true;changed=true;}
                }
            }
            // A folded drive into an explicit state against the drives kept into it.
            if !changed{changed=self.keep_partners(&mut excited,&explicit,&b,&partition.set,&dressing);}
            // A folded pair of channels between two states of one cluster, judged again with its dressing taken from its
            // separation: the cluster's static block mixes its states by up to that much, and a pair that is not far
            // beyond it would beat between the block's eigenstates at their nearness -- kept.
            if !changed{for item in partition.items.iter().filter(|i|i.folded){
                let Kind::Raman{first,second}=item.kind else{continue};
                let(Some(k),Some(k2))=(partition.set[first.state],partition.set[second.state])else{continue};
                if k!=k2||kept.contains(&item.kind){continue;}
                if !foldable((item.frequency-dressing[k]).max(0.),item.size,self.generator_scale){kept.push(item.kind.clone());changed=true;}
            }}
            if !changed&&!moved{self.charge=Ground::charge_of(&excited,ng);return self.assemble(&b,partition,excited);}
        }
    }

    /// The pairs of channels through an explicit state (module documentation, # One rule over the ground and the excited states): each folded drive
    /// into one, against every drive kept into it -- those whose admixture g/d lies above double precision's round-off --, by the ground states' pairs' rule, within
    /// one cluster with its dressing taken from their separation: kept where one may not fold. Whether any was.
    fn keep_partners(&self,excited:&mut [ExcitedItem],explicit:&[bool],b:&Bounds,set:&[Option<usize>],dressing:&[f64])->bool{
        let(ng,l0)=(self.ground_states.len(),self.generator_scale);
        let mixes=|x:&ExcitedItem|x.size/x.distance>f64::EPSILON;
        let kept:Vec<(Drive,f64,f64)>=excited.iter().filter(|x|!x.folded&&mixes(x)).filter_map(|x|match x.kind{ExcitedKind::Drive(d)=>Some((d,x.size,x.distance)),_=>None}).collect();
        let mut changed=false;
        for x in excited.iter_mut().filter(|x|x.folded&&mixes(x)){
            let ExcitedKind::Drive(d)=x.kind else{continue};if !explicit[d.excited]{continue;}
            let gamma=self.decay[(d.excited,d.excited)].re;
            let within=if set[d.ground]==set[ng+d.excited]{set[d.ground].map_or(0.,|k|dressing[k])}else{0.};
            if kept.iter().filter(|p|p.0.excited==d.excited).any(|&(p,size,distance)|{
                let frequency=b.separation(p.ground,d.ground,self.carriers[p.carrier].detuning-self.carriers[d.carrier].detuning);
                !foldable((frequency-within).max(0.),x.size*size*(1./x.distance+1./distance+gamma/(x.distance*distance)),l0)
            }){x.folded=false;x.promoted=true;changed=true;}
        }
        changed
    }

    /// The couplings between ground states classified (`classify`) with the folded drives, and the
    /// clusters they make with the kept excited couplings, and the frames.
    fn partition(&mut self,b:&Bounds,excited:&[ExcitedItem],explicit:&[bool],kept:&[Kind])->Result<Partition,String>{
        let(ng,ne,nc)=(self.ground_states.len(),self.excited_states.len(),self.carriers.len());
        self.pole=b.poles.iter().copied().fold(f64::INFINITY,f64::min);self.poles=b.poles.clone();
        // ‖L0‖ of the eliminated generator without the states' energies: its Hermitian part on both
        // sides, its decay and its jumps.
        let l0=2.*b.light+2.*b.scattering;self.generator_scale=l0;self.light=b.light;
        let separation=|a:usize,c:usize,beat:f64|b.separation(a,c,beat);
        // A coupling at the generator's round-off is no term, whatever its frequency: neither kept, for a beat within the generator's scale, nor a far channel of the responses' dressing (`Model::prune` for the tables).
        let term=|size:f64|size>f64::EPSILON*l0;
        let mut items:Vec<Item>=vec![];
        for a in 0..ng{for c in a+1..ng{if term(2.*b.statics[a][c].max(b.statics[c][a])){
            items.push(Item{kind:Kind::Static{a,b:c},frequency:separation(a,c,0.),size:2.*b.statics[a][c].max(b.statics[c][a]),folded:false});}}}
        let channels:Vec<Channel>=(0..nc).flat_map(|c|(0..ng).map(move|s|Channel{carrier:c,state:s})).filter(|ch|self.couples(ch.carrier,ch.state)).collect();
        for(i,&x)in channels.iter().enumerate(){for &y in &channels[i+1..]{
            // Absorbing x, emitting into y's carrier and landing in y's state: Ω = (E_y + δ_y) − (E_x + δ_x).
            let beat=self.carriers[y.carrier].detuning-self.carriers[x.carrier].detuning;
            let size=2.*b.raman[x.carrier][y.carrier][y.state][x.state].max(b.raman[y.carrier][x.carrier][x.state][y.state]);
            if term(size){items.push(Item{kind:Kind::Raman{first:x,second:y},frequency:separation(y.state,x.state,beat),size,folded:false});}
        }}
        for from in 0..ng{
            let photons:Vec<Photon>=(0..nc).flat_map(|c|(0..ng).map(move|to|Photon{carrier:c,from,to})).filter(|p|b.photons[p.carrier][from][p.to].iter().any(|&n|n>0.)).collect();
            for(i,&p)in photons.iter().enumerate(){for &q in &photons[i+1..]{
                if p.to==q.to{continue;}
                // Their frequencies (δ_p + E_from − E_p.to) − (δ_q + E_from − E_q.to).
                let beat=self.carriers[p.carrier].detuning-self.carriers[q.carrier].detuning;
                let size:f64=(0..3).map(|x|b.photons[p.carrier][from][p.to][x]*b.photons[q.carrier][from][q.to][x]).sum();
                if term(size){items.push(Item{kind:Kind::Photons{first:p,second:q},frequency:separation(q.to,p.to,beat),size,folded:false});}
            }}
        }
        // An explicit state's two photons landing in two ground states: the coherence its jumps make
        // between them, |Σ_a C_a C_a*|, at their separation.
        for e in (0..ne).filter(|&e|explicit[e]){for first in 0..ng{for second in first+1..ng{
            let z=(0..3).fold(Complex::ZERO,|s,a|s+self.emission[a][(first,e)]*self.emission[a][(second,e)].conj());
            if term(z.abs()){items.push(Item{kind:Kind::Direct{excited:e,first,second},frequency:separation(first,second,0.),size:z.abs(),folded:false});}
        }}}
        let(verdicts,cut)=classify(&items.iter().map(|i|(i.frequency,i.size)).collect::<Vec<_>>(),l0);
        for(item,v)in items.iter_mut().zip(verdicts){item.folded=v&&!kept.contains(&item.kind);}
        // A static coupling between F = 1 and F = 2 that may not fold would join the hyperfine levels in one frame,
        // the splitting on its diagonal: the kept static couplings carry no beat.
        if let Some(item)=items.iter().find(|i|!i.folded&&matches!(i.kind,Kind::Static{..})&&{let Kind::Static{a,b}=i.kind else{unreachable!()};self.levels[a]!=self.levels[b]}){
            return Err(format!("光束在 F=1 与 F=2 之间的静态耦合不能按 van Vleck 折叠：间隔 {:.3} GHz（生成元尺度的 {:.1} 倍，须至少 {FOLD_DISTANCE} 倍），混合 m² = {:.1e}（预算 {BUDGET:.0e}）；保留它须在一个子空间里传播超精细分裂，3D 运动暂不支持",
                item.frequency/TAU/1e3,item.frequency/l0,mixing(item.frequency,item.size,l0).powi(2)));}
        // The clusters: the components of the graph of kept couplings over the ground states (nodes
        // 0..ng) and the excited ones (ng + e).
        let mut union=Union::new(ng+ne);
        // An edge (a, b) with θ_a − θ_b its integer combination of the carriers' detunings, and its size.
        let unit=|plus:&[usize],minus:&[usize]|->Vec<i32>{let mut m=vec![0;nc];for &c in plus{m[c]+=1;}for &c in minus{m[c]-=1;}m};
        let mut edges:Vec<(f64,usize,usize,Vec<i32>)>=vec![];
        for item in items.iter().filter(|i|!i.folded){match item.kind{
            Kind::Static{a,b}=>edges.push((item.size,a,b,unit(&[],&[]))),
            Kind::Raman{first,second}=>edges.push((item.size,second.state,first.state,unit(&[first.carrier],&[second.carrier]))),
            Kind::Photons{first,second}=>edges.push((item.size,first.to,second.to,unit(&[first.carrier],&[second.carrier]))),
            Kind::Direct{first,second,..}=>edges.push((item.size,first,second,unit(&[],&[]))),
        }}
        for item in excited.iter().filter(|x|!x.folded){match item.kind{
            ExcitedKind::Drive(d)=>edges.push((item.size,ng+d.excited,d.ground,unit(&[d.carrier],&[]))),
            ExcitedKind::Static{a,b}=>edges.push((item.size,ng+a,ng+b,unit(&[],&[]))),
        }}
        for(_,a,b,_)in &edges{union.join(*a,*b);}
        // Every set holds a ground state (an excited state is in one through its kept drive); its
        // smallest element is one.
        let sets:Vec<Vec<usize>>=union.sets().into_iter().filter(|s|s[0]<ng&&(s.len()>1||s[0]<ng)).collect();
        let mut set=vec![None;ng+ne];for(k,s)in sets.iter().enumerate(){for &x in s{set[x]=Some(k);}}
        // Each cluster's frames: from its first state along a spanning tree of its kept couplings,
        // the strongest first, each nulling its coupling's beat -- the first state's energy plus an
        // integer combination of the carriers' detunings, each frame evaluated once from it
        // (`Ground::beat`): the detunings' round-off along the tree's paths does not enter, and frames
        // equal in exact arithmetic are equal (two sublevels joined through D1 states of a D2-framed
        // excited manifold: θ + δ − δ was θ only to the detuning's round-off, ~1e-8 rad/us).
        let focus=self.eliminate(&self.focus_intensities())?;
        let mut combinations:Vec<Option<Vec<i32>>>=vec![None;ng+ne];
        edges.sort_by(|x,y|y.0.total_cmp(&x.0));
        let mut frames=vec![f64::NAN;ng+ne];
        for s in &sets{
            combinations[s[0]]=Some(vec![0;nc]);
            while let Some((_,a,b,offset))=edges.iter().find(|(_,a,b,_)|combinations[*a].is_none()!=combinations[*b].is_none()&&set[*a]==set[s[0]]){
                let(a,b)=(*a,*b);
                if combinations[a].is_none(){combinations[a]=Some(combinations[b].as_ref().unwrap().iter().zip(offset).map(|(x,y)|x+y).collect());}
                else{combinations[b]=Some(combinations[a].as_ref().unwrap().iter().zip(offset).map(|(x,y)|x-y).collect());}
            }
            for &x in s{frames[x]=self.beat(focus.energies[s[0]],combinations[x].as_ref().unwrap());}
        }
        Ok(Partition{items,cut,sets,set,frames,combinations:combinations.into_iter().map(|m|m.unwrap_or_default()).collect()})
    }

    /// Each cluster's machinery from the rule's partition: its states and frames, its kept couplings
    /// and drives, its folded static couplings' van Vleck paths, its jump classes and co-resonant
    /// channels; and the near light's dressing (its far channels, units and poles).
    fn assemble(&mut self,b:&Bounds,partition:Partition,excited:Vec<ExcitedItem>)->Result<(),String>{
        let(ng,ne,nc)=(self.ground_states.len(),self.excited_states.len(),self.carriers.len());
        let Partition{items,cut,sets,set,frames,combinations}=partition;
        let theta=|node:usize|frames[node];
        // A residual as the integer combination of its terms -- frames (nodes, signed) and carriers
        // (signed) --, evaluated once (`Ground::beat`): zero exactly where its terms cancel.
        let combination=|nodes:&[(usize,i32)],carriers:&[(usize,i32)]|->Vec<i32>{let mut m=vec![0;nc];
            for &(x,s)in nodes{for(a,b)in m.iter_mut().zip(&combinations[x]){*a+=s*b;}}for &(c,s)in carriers{m[c]+=s;}m};
        let residual=|nodes:&[(usize,i32)],carriers:&[(usize,i32)]|->f64{self.beat(0.,&combination(nodes,carriers))};
        let cluster:Vec<usize>=(0..ng).map(|x|set[x].unwrap()).collect();
        let folded_raman=|x:Channel,y:Channel|->Option<bool>{items.iter().find_map(|i|match i.kind{
            Kind::Raman{first,second}if(first==x&&second==y)||(first==y&&second==x)=>Some(i.folded),_=>None})};
        // Two channels are co-resonant: the same, or a kept pair.
        let coresonant=|x:Channel,y:Channel|x==y||folded_raman(x,y)==Some(false);
        let channels:Vec<Channel>=(0..nc).flat_map(|c|(0..ng).map(move|s|Channel{carrier:c,state:s})).filter(|ch|self.couples(ch.carrier,ch.state)).collect();
        let kept_drive=|d:Drive|excited.iter().any(|x|x.kind==ExcitedKind::Drive(d)&&!x.folded);
        let floor=if self.duration>0.{1./self.duration}else{0.};
        let dressed:Vec<Vec<usize>>=(0..ne).map(|e|(0..ne).filter(|&f|excited.iter().any(|x|x.folded&&x.distance>floor&&x.kind==ExcitedKind::Static{a:e.min(f),b:e.max(f)})).collect()).collect();
        let folded_static=|a:usize,b:usize|dressed[a].contains(&b);
        let mut clusters:Vec<Cluster>=sets.iter().map(|nodes|{
            let states:Vec<usize>=nodes.iter().copied().filter(|&x|x<ng).collect();let excited:Vec<usize>=nodes.iter().filter(|&&x|x>=ng).map(|&x|x-ng).collect();
            Cluster{frames:states.iter().map(|&s|theta(s)).collect(),excited_frames:excited.iter().map(|&e|theta(ng+e)).collect(),
                combinations:states.iter().map(|&s|combinations[s].clone()).chain(excited.iter().map(|&e|combinations[ng+e].clone())).collect(),states,excited,
                groups:vec![],statics:vec![],static_paths:vec![],drives:vec![],excited_statics:vec![],excited_paths:vec![],classes:vec![],coherent:vec![],folded_crosses:vec![]}}).collect();
        for(k,cl)in clusters.iter_mut().enumerate(){
            let inside:Vec<Channel>=channels.iter().copied().filter(|ch|cluster[ch.state]==k).collect();
            let n0=cl.states.len();
            // The kept couplings per carrier pair: absorbing x = (c, b), emitting y = (c′, a); between
            // two explicit states of one frame, a carrier's folded drives' van Vleck shift.
            for c in 0..nc{for d in 0..nc{
                let mut entries=vec![];
                for(j,&s)in cl.states.iter().enumerate(){for(i,&t)in cl.states.iter().enumerate(){
                    let(x,y)=(Channel{carrier:c,state:s},Channel{carrier:d,state:t});
                    if !(self.couples(c,s)&&self.couples(d,t)){continue;}
                    if x==y||folded_raman(x,y)==Some(false){entries.push((i,j,residual(&[(t,1),(s,-1)],&[(d,1),(c,-1)])));}
                }}
                if c==d{for(p,&e)in cl.excited.iter().enumerate(){for(q,&f)in cl.excited.iter().enumerate(){
                    let shared=(0..ng).any(|b|self.carriers[c].beams.iter().any(|&l|self.folded[l][(e,b)]!=Complex::ZERO)&&self.carriers[c].beams.iter().any(|&l|self.folded[l][(f,b)]!=Complex::ZERO));
                    if shared&&combinations[ng+e]==combinations[ng+f]{entries.push((n0+p,n0+q,0.));}
                }}}
                if !entries.is_empty(){cl.groups.push(Group{absorbed:c,emitted:d,entries});}
            }}
            let folded_ground_static=|a:usize,b:usize|items.iter().any(|i|i.folded&&i.kind==Kind::Static{a:a.min(b),b:a.max(b)});
            for item in items.iter().filter(|i|!i.folded){if let Kind::Static{a,b}=item.kind{
                if cluster[a]==k{let(i,j)=(cl.position(a).unwrap(),cl.position(b).unwrap());cl.statics.push((i,j));cl.statics.push((j,i));}
            }}
            for i in 0..cl.states.len(){for j in 0..cl.states.len(){if combinations[cl.states[i]]==combinations[cl.states[j]]{
                for f in 0..ng{if folded_ground_static(cl.states[i],f)&&folded_ground_static(f,cl.states[j]){cl.static_paths.push((i,j,f));}}
            }}}
            // The kept drives, H's kept couplings between explicit states, and the van Vleck paths of
            // their folded ones through excited states outside the cluster.
            for(p,&e)in cl.excited.iter().enumerate(){
                for(x,&s)in cl.states.iter().enumerate(){for c in 0..nc{
                    if kept_drive(Drive{carrier:c,excited:e,ground:s}){cl.drives.push((n0+p,x,c,residual(&[(ng+e,1),(s,-1)],&[(c,-1)])));}
                }}
                for(q,&f)in cl.excited.iter().enumerate(){
                    if e!=f&&excited.iter().any(|x|!x.folded&&x.kind==ExcitedKind::Static{a:e.min(f),b:e.max(f)}){cl.excited_statics.push((n0+p,n0+q));}
                    if combinations[ng+e]==combinations[ng+f]{for g in (0..ne).filter(|&g|!cl.excited.contains(&g)){
                        if folded_static(e,g)&&folded_static(g,f){cl.excited_paths.push((n0+p,n0+q,g));}
                    }}
                }
            }
            // The co-resonant channels (the readout's coherent sums), each with its residual.
            let mut union=Union::new(inside.len());
            for i in 0..inside.len(){for j in i+1..inside.len(){if coresonant(inside[i],inside[j]){union.join(i,j);}}}
            cl.coherent=union.sets().into_iter().map(|set|{let x0=inside[set[0]];
                set.iter().map(|&i|{let x=inside[i];(x,residual(&[(x.state,1),(x0.state,-1)],&[(x.carrier,1),(x0.carrier,-1)]))}).collect()}).collect();
        }
        // An explicit state's photons' wave number: a carrier's that drives it (the largest drive).
        let wave_number=|e:usize|->f64{excited.iter().filter_map(|x|match x.kind{ExcitedKind::Drive(d)if d.excited==e=>Some((x.size,d.carrier)),_=>None})
            .max_by(|a,b|a.0.total_cmp(&b.0)).map_or(self.carriers.first().map_or(0.,|c|c.wave_number),|(_,c)|self.carriers[c].wave_number)};
        // The jump classes: the photons from a cluster, joined when they land in one cluster with
        // coherent amplitudes -- in one state from co-resonant sources (channels: the same or a kept
        // pair), in two states as a kept pair of photons of one source, or from two sources whose
        // frequencies meet within the cut: a folded photon's its carrier's, an explicit state's every one its amplitude holds.
        #[derive(Clone,Copy)]
        enum Source{Folded(Photon),Direct(usize,usize)}
        // The frequencies an explicit state's amplitude holds: its frame's and each kept drive's reaching it, through H's kept couplings too.
        let mut holds:Vec<Vec<Vec<i32>>>=(0..ne).map(|e|vec![combination(&[(ng+e,1)],&[])]).collect();
        for x in excited.iter().filter(|x|!x.folded&&x.size/x.distance>f64::EPSILON){if let ExcitedKind::Drive(d)=x.kind{
            let f=combination(&[(d.ground,1)],&[(d.carrier,1)]);if !holds[d.excited].contains(&f){holds[d.excited].push(f);}
        }}
        loop{
            let mut grew=false;
            for x in excited.iter().filter(|x|!x.folded){if let ExcitedKind::Static{a,b}=x.kind{for(u,v)in [(a,b),(b,a)]{for f in holds[u].clone(){if !holds[v].contains(&f){holds[v].push(f);grew=true;}}}}}
            if !grew{break;}
        }
        for k in 0..clusters.len(){
            let mut photons:Vec<Source>=(0..nc).flat_map(|c|clusters[k].states.iter().flat_map(move|&from|(0..ng).map(move|to|Photon{carrier:c,from,to})))
                .filter(|p|b.photons[p.carrier][p.from][p.to].iter().any(|&n|n>0.)).map(Source::Folded).collect();
            // An explicit state's photons: those of its own and of the folded excited states its
            // dressing reaches (H's folded couplings, `Ground::dressing`).
            for &e in &clusters[k].excited{
                let reached:Vec<usize>=(0..ne).filter(|&f|f==e||(!clusters[k].excited.contains(&f)&&folded_static(e,f))).collect();
                for to in 0..ng{if reached.iter().any(|&f|(0..3).any(|a|self.emission[a][(to,f)]!=Complex::ZERO)){photons.push(Source::Direct(e,to));}}
            }
            let to=|p:&Source|match *p{Source::Folded(p)=>p.to,Source::Direct(_,t)=>t};
            // A photon's rate (its target's frame less its source's frame and carrier) as a combination; between two photons from this
            // cluster landing in one cluster their difference (the clusters' first energies cancel), evaluated once.
            let rate=|p:&Source|->Vec<i32>{match *p{Source::Folded(p)=>combination(&[(p.to,1),(p.from,-1)],&[(p.carrier,-1)]),Source::Direct(e,t)=>combination(&[(t,1),(ng+e,-1)],&[])}};
            let apart=|x:Vec<i32>,y:Vec<i32>|->f64{self.beat(0.,&x.iter().zip(&y).map(|(a,b)|a-b).collect::<Vec<i32>>())};
            // A photon's frequencies: the one rate of a folded photon, an explicit state's per frequency its amplitude holds.
            let frequencies=|p:&Source|->Vec<Vec<i32>>{match *p{Source::Folded(_)=>vec![rate(p)],
                Source::Direct(e,t)=>{let to=combination(&[(t,1)],&[]);holds[e].iter().map(|f|to.iter().zip(f).map(|(a,b)|a-b).collect()).collect()}}};
            let meet=|p:&Source,q:&Source|frequencies(p).into_iter().any(|x|frequencies(q).into_iter().any(|y|apart(x.clone(),y).abs()<cut));
            let mut union=Union::new(photons.len());
            for i in 0..photons.len(){for j in i+1..photons.len(){
                let(p,q)=(photons[i],photons[j]);
                if cluster[to(&p)]!=cluster[to(&q)]{continue;}
                let joined=match(p,q){
                    (Source::Folded(p),Source::Folded(q))=>if p.to==q.to{coresonant(Channel{carrier:p.carrier,state:p.from},Channel{carrier:q.carrier,state:q.from})}
                        else if p.from==q.from{items.iter().any(|it|!it.folded&&matches!(it.kind,Kind::Photons{first,second}if(first==p&&second==q)||(first==q&&second==p)))}
                        else{apart(rate(&Source::Folded(p)),rate(&Source::Folded(q))).abs()<cut},
                    (Source::Direct(e,a),Source::Direct(f,b))if e==f=>items.iter().any(|it|!it.folded&&it.kind==Kind::Direct{excited:e,first:a.min(b),second:a.max(b)}),
                    _=>meet(&p,&q),
                };
                if joined{union.join(i,j);}
            }}
            let classes=union.sets().into_iter().map(|set|{let p0=photons[set[0]];let r0=rate(&p0);
                let residual=|p:&Source|apart(rate(p),r0.clone());
                let wave=match p0{Source::Folded(p)=>self.carriers[p.carrier].wave_number,Source::Direct(e,_)=>wave_number(e)};
                let(mut folded,mut direct)=(vec![],vec![]);
                for &i in &set{match photons[i]{Source::Folded(p)=>folded.push((p,residual(&photons[i]))),
                    Source::Direct(e,t)=>direct.push((clusters[k].excited_position(e).unwrap(),t,residual(&photons[i])))}}
                Class{target:cluster[to(&p0)],photons:folded,direct,wave_number:wave}}).collect::<Vec<Class>>();
            // The folded photons a class joins without a kept coupling between their channels: the decay's
            // cross terms the jumps make (`FoldedCross`).
            let mut crosses:Vec<FoldedCross>=vec![];
            for class in &classes{for &(p,wp)in &class.photons{for &(q,wq)in &class.photons{
                if p.to!=q.to||coresonant(Channel{carrier:p.carrier,state:p.from},Channel{carrier:q.carrier,state:q.from}){continue;}
                let(row,col)=(clusters[k].position(p.from).unwrap(),clusters[k].position(q.from).unwrap());
                match crosses.iter_mut().find(|x|x.row==row&&x.col==col&&x.absorbed==q.carrier&&x.emitted==p.carrier&&x.residual==wq-wp){
                    Some(x)=>x.tos.push(p.to),None=>crosses.push(FoldedCross{row,col,absorbed:q.carrier,emitted:p.carrier,residual:wq-wp,tos:vec![p.to]})}
            }}}
            clusters[k].classes=classes;clusters[k].folded_crosses=crosses;
        }
        // The near light's dressing from here on (`Ground::dress_light`): every channel's far channels, every
        // near unit's fields, each cluster's resolvents' floor and the carriers' poles dressed.
        self.vias=(0..nc).map(|c|(0..ng).map(|b|{let x=Channel{carrier:c,state:b};
            channels.iter().copied().filter(|&y|self.couples(c,b)&&folded_raman(x,y)==Some(true)).collect()}).collect()).collect();
        self.near_units=self.units.iter().map(|u|{let c=self.carriers.iter().position(|c|c.beams.contains(&u.beams[0]))?;
            let mut coupling=Dense::zeros(ne,ng);for(&l,&x)in u.beams.iter().zip(&u.combination){coupling.add_scaled(x,&self.folded[l]);}
            let remainders=if u.beams.len()>1{u.beams.iter().zip(&u.weights).map(|(&l,&g)|{let mut r=self.folded[l].clone();r.add_scaled(-g,&coupling);r}).collect()}else{vec![]};
            Some(NearUnit{carrier:c,coupling,remainders})}).collect();
        self.items=items;self.cut=cut;self.clusters=clusters;self.cluster=cluster;self.excited_items=excited;self.dressed=dressed;
        self.floors=self.clusters.iter().map(|cl|0.5*hermitian_eigenvalues(&self.cluster_hamiltonian(&self.decay,cl)).into_iter().fold(f64::INFINITY,f64::min).max(0.)).collect();
        self.dressing=true;self.unmixed_blocks()?;self.channel_poles=self.dressed_poles()?;
        self.poles=self.channel_poles.iter().map(|p|p.iter().copied().fold(f64::INFINITY,f64::min)).collect();self.pole=self.poles.iter().copied().fold(f64::INFINITY,f64::min);
        Ok(())
    }

    /// Every beam's amplitude at a point r (um from the focus).
    pub fn amplitudes_at(&self,r:[f64;3])->Vec<Complex>{self.beams.iter().map(|b|b.amplitude_at(r)).collect()}

    /// The nearest excited pole every channel sees over the trap, its block's dressed responses':
    /// min σ_min of their resolvents (`Mode`) over the eliminations at the varying units' corners and levels.
    fn dressed_poles(&self)->Result<Vec<Vec<f64>>,String>{
        let(ng,nc)=(self.ground_states.len(),self.carriers.len());let mut poles=vec![vec![f64::INFINITY;ng];nc];
        let mut identity=Dense::zeros(self.excited_states.len(),self.excited_states.len());for i in 0..identity.rows{identity[(i,i)]=Complex::ONE;}
        for intensities in intensity_points(&self.units,&self.beams){
            let e=self.eliminate(&intensities)?;
            let blocks:Vec<(usize,usize)>=(0..nc).flat_map(|c|(0..self.clusters.len()).map(move|k|(c,k))).filter(|&(c,k)|e.blocks[c][k].is_some()).collect();
            let found:Vec<f64>=blocks.par_iter().map(|&(c,k)|{let mut pole=f64::INFINITY;
                for mode in &e.blocks[c][k].as_ref().unwrap().modes{for lu in mode.lu.iter().chain(mode.dressed.iter().filter_map(|d|d.lu.as_ref())){pole=pole.min(1./lu.solve_columns(&identity).spectral_norm());}}
                pole}).collect();
            for(&(c,k),&pole)in blocks.iter().zip(&found){for &b in self.clusters[k].states.iter().filter(|&&b|self.couples(c,b)){poles[c][b]=poles[c][b].min(pole);}}
        }
        Ok(poles)
    }

    /// Which blocks are taken unmixed (`Ground::unmixed`, # The static ground block): those whose responses change by less
    /// than the numerical share relative to themselves (`MIXING_SHARE`), measured against the block's
    /// eigenstates' at every elimination `dressed_poles` judges the trap by, the varying units' corners and levels: per column,
    /// every beam's ‖ΔR‖ + σ ‖ΔR2‖ over its ‖R‖ (a column the carrier does not couple, that only mixing gives one, over the
    /// cluster's largest), R2 entering times a Doppler shift below the nearest pole σ the cluster's responses see, ‖R‖/‖R2‖ at
    /// least (the first order's range; beyond its allowance the exact response, engine.rs); the largest change they keep,
    /// booked (`Ground::unmixed_error`).
    fn unmixed_blocks(&mut self)->Result<(),String>{
        let(nc,nk,ne)=(self.carriers.len(),self.clusters.len(),self.excited_states.len());let mut change=vec![vec![0_f64;nk];nc];
        let norm=|m:&Dense,n:Option<&Dense>,b:usize|(0..ne).map(|i|(m[(i,b)]-n.map_or(Complex::ZERO,|n|n[(i,b)])).norm_sqr()).sum::<f64>().sqrt();
        for intensities in intensity_points(&self.units,&self.beams){
            self.unmixed=vec![];let mixed=self.eliminate(&intensities)?;
            self.unmixed=vec![vec![true;nk];nc];let unmixed=self.eliminate(&intensities)?;
            for(c,carrier)in self.carriers.iter().enumerate(){for(k,cl)in self.clusters.iter().enumerate(){
                let size=|m:&[Dense],n:Option<&[Dense]>,b:usize|carrier.beams.iter().map(|&l|norm(&m[l],n.map(|n|&n[l]),b)).sum::<f64>();
                let coupled:Vec<usize>=cl.states.iter().copied().filter(|&b|self.couples(c,b)).collect();if coupled.is_empty(){continue;}
                let largest=coupled.iter().map(|&b|size(&mixed.responses,None,b)).fold(0.,f64::max);
                let pole=coupled.iter().map(|&b|size(&mixed.responses,None,b)/size(&mixed.derivatives,None,b)).fold(f64::INFINITY,f64::min);
                for &b in &cl.states{
                    let apart=size(&mixed.responses,Some(&unmixed.responses[..]),b)+pole*size(&mixed.derivatives,Some(&unmixed.derivatives[..]),b);if apart==0.{continue;}
                    change[c][k]=change[c][k].max(apart/if self.couples(c,b){size(&mixed.responses,None,b)}else{largest});
                }
            }}
        }
        self.unmixed=change.iter().map(|x|x.iter().map(|&y|y<=MIXING_SHARE).collect()).collect();
        self.unmixed_error=change.iter().flatten().copied().filter(|&y|y<=MIXING_SHARE).fold(0.,f64::max);
        Ok(())
    }

    /// A cluster's generator at a point r and a time t (the dense reference of model.rs, with the
    /// elimination of H(r)): H_eff, the jumps of every class, the readout.
    pub fn local_at(&self,e:&Elimination,k:usize,r:[f64;3],t:f64)->Result<Local,String>{
        let cl=&self.clusters[k];let(n0,n,ne)=(cl.states.len(),cl.dimension(),self.excited_states.len());let a=self.amplitudes_at(r);
        let v:Vec<Dense>=self.carriers.iter().map(|c|combine(&c.beams,&a,&self.folded)).collect();
        let full:Vec<Dense>=self.carriers.iter().map(|c|combine(&c.beams,&a,&self.couplings)).collect();
        let rc:Vec<Dense>=self.carriers.iter().map(|c|combine(&c.beams,&a,&e.responses)).collect();
        let mut h=Dense::zeros(n,n);
        for(i,&s)in cl.states.iter().enumerate(){h[(i,i)]=Complex::new(e.energies[s]-cl.frames[i],0.);}
        for(p,&s)in cl.excited.iter().enumerate(){h[(n0+p,n0+p)]=Complex::new(e.excited_energies[s]-cl.excited_frames[p],0.);}
        for &(i,j)in &cl.statics{h[(i,j)]+=Complex::cis((cl.frames[i]-cl.frames[j])*t)*e.ground[(cl.states[i],cl.states[j])];}
        // The folded static couplings' van Vleck shift, with the exact bare-energy denominators.
        for &(i,j,f)in &cl.static_paths{let(s,u)=(cl.states[i],cl.states[j]);
            h[(i,j)]+=e.ground[(s,f)]*e.ground[(f,u)]*(0.5*(1./(e.energies[s]-e.energies[f])+1./(e.energies[u]-e.energies[f])));}
        for g in &cl.groups{
            let(c,d)=(g.absorbed,g.emitted);
            let mut x=v[d].adjoint_product(&rc[c]).scaled(Complex::new(-0.5,0.));x.add_scaled(Complex::new(-0.5,0.),&rc[d].adjoint_product(&v[c]));
            x.add_scaled(Complex::new(0.,-0.5),&rc[d].adjoint_product(&self.decay.product(&rc[c])));
            for &(i,j,w)in &g.entries{
                if i<n0{h[(i,j)]+=Complex::cis(w*t)*x[(cl.states[i],cl.states[j])];continue;}
                // Two explicit states: ½ Σ_b V_c^f[e,b] V_c^f[f,b]* (Re 1/(K_ee − δ_c − E_b) + Re 1/(K_ff − δ_c − E_b)).
                let(ex,fx)=(cl.excited[i-n0],cl.excited[j-n0]);let delta=self.carriers[c].detuning;
                for b in 0..self.ground_states.len(){
                    h[(i,j)]+=Complex::cis(w*t)*v[c][(ex,b)]*v[c][(fx,b)].conj()*(0.5*(e.shift_denominator(ex,b,delta)+e.shift_denominator(fx,b,delta)));
                }
            }
        }
        // The folded photons a class joins without a kept coupling between their channels: their jumps' cross
        // terms in the decay (`FoldedCross`).
        for x in &cl.folded_crosses{
            let(a,b)=(cl.states[x.row],cl.states[x.col]);
            let z=(0..3).fold(Complex::ZERO,|s,ax|{let(jd,jc)=(self.emission[ax].product(&rc[x.emitted]),self.emission[ax].product(&rc[x.absorbed]));
                x.tos.iter().fold(s,|s,&to|s+jd[(to,a)].conj()*jc[(to,b)])});
            h[(x.row,x.col)]+=Complex::cis(x.residual*t)*z*Complex::new(0.,-0.5);
        }
        // The kept drives, H's kept couplings between explicit states and the van Vleck paths of
        // their folded ones; the explicit states' dressing by those (`dressing`) and the drives it
        // brings: from ground state b through the folded excited states F, ½(H_eF A_c[F,b] + x_e†V_c^f[F,b]).
        let flags:Vec<bool>=(0..ne).map(|f|cl.excited.contains(&f)).collect();
        let dressings:Vec<Vec<Complex>>=cl.excited.iter().map(|&s|self.dressing(e,s,&flags)).collect::<Result<_,_>>()?;
        for &(i,j,c,w)in &cl.drives{let z=Complex::cis(w*t)*full[c][(cl.excited[i-n0],cl.states[j])];h[(i,j)]+=z;h[(j,i)]+=z.conj();}
        for(p,&s)in cl.excited.iter().enumerate(){let x=&dressings[p];
            for(j,&b)in cl.states.iter().enumerate(){for c in 0..self.carriers.len(){
                let z=(0..ne).fold(Complex::ZERO,|acc,f|acc-e.excited[(s,f)]*rc[c][(f,b)]*if flags[f]{0.}else{1.}+x[f].conj()*v[c][(f,b)])*0.5;
                if z==Complex::ZERO{continue;}
                let(m,mj)=(&cl.combinations[n0+p],&cl.combinations[j]);let w=self.beat(0.,&(0..m.len()).map(|i|m[i]-mj[i]-i32::from(i==c)).collect::<Vec<i32>>());
                let z=z*Complex::cis(w*t);h[(n0+p,j)]+=z;h[(j,n0+p)]+=z.conj();
            }}
        }
        for &(i,j)in &cl.excited_statics{h[(i,j)]+=Complex::cis((cl.frame(i)-cl.frame(j))*t)*e.excited[(cl.excited[i-n0],cl.excited[j-n0])];}
        for &(i,j,g)in &cl.excited_paths{let(s,u)=(cl.excited[i-n0],cl.excited[j-n0]);
            h[(i,j)]+=e.excited[(s,g)]*e.excited[(g,u)]*(0.5*(1./(e.excited_energies[s]-e.excited_energies[g])+1./(e.excited_energies[u]-e.excited_energies[g])));}
        // The jumps (with the overall sign of the module documentation): a folded photon's C_a R_c, an
        // explicit state's −C_a.
        let jumps:Vec<[Dense;3]>=cl.classes.iter().map(|class|{let target=&self.clusters[class.target];
            std::array::from_fn(|x|{let mut m=Dense::zeros(target.states.len(),n);
                for &(p,w)in &class.photons{let j=self.emission[x].product(&rc[p.carrier]);
                    m[(target.position(p.to).unwrap(),cl.position(p.from).unwrap())]+=Complex::cis(w*t)*j[(p.to,p.from)];}
                for &(i,to,w)in &class.direct{let(s,d)=(cl.excited[i-n0],&dressings[i-n0]);
                    let c=self.emission[x][(to,s)]+(0..ne).fold(Complex::ZERO,|acc,f|acc+self.emission[x][(to,f)]*d[f]);
                    m[(target.position(to).unwrap(),i)]-=Complex::cis(w*t)*c;}
                m})}).collect();
        // The decay of the explicit states: −(i/2) Σ_φ J_φ†J_φ on their rows and columns (between ground
        // states it is the folded drives', X's above).
        if n>n0{
            let mut decay=Dense::zeros(n,n);for j in &jumps{for m in j{decay.add_scaled(Complex::ONE,&m.adjoint_product(m));}}
            for i in 0..n{for j in 0..n{if i>=n0||j>=n0{h[(i,j)]+=decay[(i,j)]*Complex::new(0.,-0.5);}}}
        }
        let mut admixture=Dense::zeros(n,n);let mut excited_readout=vec![Dense::zeros(n,n);ne];
        for group in &cl.coherent{for &(x,wx)in group{for &(y,wy)in group{
            // The excited amplitude of channel x, e^{−i ε̃_x t} R_c x; Y and E_i their coherent sums.
            let(i,j)=(cl.position(y.state).unwrap(),cl.position(x.state).unwrap());let phase=Complex::cis((wy-wx)*t);
            for q in 0..ne{let z=rc[y.carrier][(q,y.state)].conj()*rc[x.carrier][(q,x.state)]*phase;admixture[(i,j)]+=z;excited_readout[q][(i,j)]+=z;}
        }}}
        for(p,&s)in cl.excited.iter().enumerate(){excited_readout[s][(n0+p,n0+p)]+=Complex::ONE;}
        Ok(Local{effective:h,jumps,admixture,excited_readout})
    }

    /// The 32 populations (in the request's order) of a cluster's density ρ (n × n) at a point: the
    /// bare ground ones, of ρ − ½{Y, ρ}, and the excited ones, tr(E_i ρ).
    pub fn populations(&self,k:usize,local:&Local,rho:&Dense)->Vec<f64>{
        let y=&local.admixture;let(left,right)=(y.product(rho),rho.product(y));
        let mut out=vec![0.;STATES];
        for(i,&b)in self.clusters[k].states.iter().enumerate(){out[self.ground_states[b]]=rho[(i,i)].re-0.5*(left[(i,i)].re+right[(i,i)].re);}
        for(e,&state)in local.excited_readout.iter().zip(&self.excited_states){
            out[state]=(0..rho.rows).map(|j|(0..rho.rows).map(|k|(e[(j,k)]*rho[(k,j)]).re).sum::<f64>()).sum();
        }
        out
    }

    /// Beam l's response R_l to all orders in an excited-energy shift w (rad/us): once the rule has made the
    /// clusters, its blocks' terms (`Block`, `Mode`) each solved with its resolvent shifted, (K + Σ_b − z_j + w)⁻¹;
    /// before, column b with (K − E_b − δ_c + w)⁻¹; K decoupled from the cluster's explicit states. R_l − w R2_l is
    /// its first order.
    pub fn response(&self,elimination:&Elimination,beam:usize,shift:f64)->Result<Dense,String>{
        let c=self.carriers.iter().position(|c|c.beams.contains(&beam)).unwrap();let carrier=&self.carriers[c];let coupling=&self.folded[beam];
        let mut out=Dense::zeros(coupling.rows,coupling.cols);
        let sources=self.sources(&elimination.excited,&elimination.excited_hamiltonian);
        if self.dressing{
            for(k,cl)in self.clusters.iter().enumerate(){let Some(block)=&elimination.blocks[c][k] else{continue};
                let kc=self.cluster_hamiltonian(&elimination.excited_hamiltonian,cl);
                for(m,mode)in block.modes.iter().enumerate(){
                    let z=mode.energy;
                    let plain=if mode.plain.is_empty(){None}else{Some(denominator(&kc,shift-z)?)};
                    let dressed=mode.dressed.iter().map(|d|self.term_denominator(&elimination.excited_hamiltonian,cl,&d.sigma,shift-z)).collect::<Result<Vec<Lu>,String>>()?;
                    let refs:Vec<&Lu>=dressed.iter().collect();
                    for(first,_,weights)in mode_terms(cl,block,m,&sources[beam],plain.as_ref(),&refs,false){spread(cl,&first,&weights,&mut out);}
                }
            }
            return Ok(out);
        }
        for b in (0..coupling.cols).filter(|&b|self.couples(c,b)){
            let mut k=elimination.excited_hamiltonian.clone();let z=elimination.energies[b]+carrier.detuning;
            if let Some(p)=self.explicit_of.get(b).filter(|p|p.iter().any(|&x|x)){k=self.decoupled(&k,p);}
            let mut column=sources[beam][b].clone();denominator(&k,shift-z)?.solve(&mut column);for i in 0..coupling.rows{out[(i,b)]=column[i];}
        }
        Ok(out)
    }

    /// The far components' fold, Σ m² of the folded ones; the couplings', Σ m² of the folded pairs of
    /// channels and static couplings (the jumps' photons split by frequency are exact); and the
    /// largest estimate of a folded static coupling between excited states (the drives' fold is its
    /// charge, `Ground::charge`).
    pub fn fold_estimates(&self)->(f64,f64,f64){
        let far=self.components.iter().filter(|c|c.folded&&c.size>0.).map(|c|mixing(c.nu,c.size,self.l0).powi(2)).sum();
        let near=self.items.iter().filter(|i|i.folded&&matches!(i.kind,Kind::Static{..}|Kind::Raman{..})).map(|i|mixing(i.frequency,i.size,self.generator_scale).powi(2)).sum();
        let statics=self.excited_items.iter().filter(|x|x.folded&&matches!(x.kind,ExcitedKind::Static{..})).map(|x|x.estimate).fold(0.,f64::max);
        (far,near,statics)
    }
    /// A carrier's saturation on a cluster's ground states over the light's pattern: the largest of their
    /// folded drives' totals (`ExcitedItem::mean`).
    pub fn carrier_saturation(&self,c:usize,cl:&Cluster)->f64{
        cl.states.iter().map(|&b|self.excited_items.iter().filter(|x|x.folded).filter_map(|x|match x.kind{ExcitedKind::Drive(d)if d.carrier==c&&d.ground==b=>Some(x.mean),_=>None}).sum::<f64>()).fold(0.,f64::max)
    }
    /// Within what a residual of a cluster is static and two are one: `same_frame`'s tolerance at its largest frame and detuning, the round-off of the frames' differences.
    pub(crate) fn residual_round_off(&self,cl:&Cluster)->f64{
        64.*f64::EPSILON*2.*(cl.frames.iter().chain(&cl.excited_frames).map(|f|f.abs()).fold(0.,f64::max)+self.carriers.iter().map(|c|c.detuning.abs()).fold(0.,f64::max))
    }
    /// The carriers the fourth order takes on a cluster, those beyond the budget's square (a carrier within it
    /// adds at most that relatively to the others' rates, its bound in the ledger's quartic item, model.rs
    /// `Model::quartic`), and the size of its H^(4), b = S l0 (S their saturations' sum, l0 the eliminated
    /// generator's scale).
    pub fn fourth_carriers(&self,cl:&Cluster)->(Vec<usize>,f64){
        let nc=self.carriers.len();
        let mut on:Vec<usize>=(0..nc).filter(|&c|self.carrier_saturation(c,cl)>BUDGET*BUDGET).collect();
        // And every carrier whose photons share a class with theirs: a class's jump adds its photons' amplitudes, so
        // the fourth order's corrections of one meet the other's second order in the jumps' products, which the
        // decay holds only between the fourth order's carriers.
        if !on.is_empty(){loop{
            let more:Vec<usize>=(0..nc).filter(|c|!on.contains(c)&&cl.classes.iter().any(|k|k.photons.iter().any(|(p,_)|p.carrier==*c)&&k.photons.iter().any(|(p,_)|on.contains(&p.carrier)))).collect();
            if more.is_empty(){break;}on.extend(more);
        }on.sort_unstable();}
        let b=on.iter().map(|&c|self.carrier_saturation(c,cl)).sum::<f64>()*self.generator_scale;(on,b)
    }
    /// Whether the fourth order keeps a cluster's entry (x, y) at its beat Ω (# Fourth order): where the second
    /// order would keep a coupling at it -- static to the frames' round-off, below the cut, at a kept coupling's
    /// residual on the entry, or one the rule does not let fold (|Ω| < FOLD_DISTANCE l0, or its mixing
    /// b/(|Ω| − l0) beyond the budget, b the size of H^(4)).
    pub fn fourth_keeps(&self,cl:&Cluster,x:usize,y:usize,omega:f64,b:f64)->bool{
        let w=omega.abs();let l0=self.generator_scale;let round_off=self.residual_round_off(cl);
        w<=round_off||w<self.cut||!(w>=FOLD_DISTANCE*l0&&mixing(omega,b,l0).powi(2)<=BUDGET)
            ||cl.groups.iter().any(|gr|gr.entries.iter().any(|&(a,c,r)|a==x&&c==y&&(r-omega).abs()<=round_off))
    }
    /// The fourth-order fold of a cluster of ground states at r and t (module documentation, # Fourth
    /// order), dense: on the slow invariant subspace Ψ = (x, W x) of the no-jump generator, W = −R + W3,
    /// W3 per triple of carriers (c1, c2, c4) at ω = δ_c1 − δ_c2 + δ_c4, W3[:,b] = G_b(ω) Σ_b′ R_c1[:,b′](V_c2†R_c4)_b′b,
    /// G_b(ω) = (K − E_b − ω)⁻¹; in y = M^{1/2} x, M = 1 + X with X = R†R on the cluster's coherent entries (the
    /// readout's metric; the rest beats and averages out): H̃'s Hermitian part that of H_eff + V†W3 − ½[X, V†R]
    /// on the entries the second order keeps at their beat (model.rs `FourthLayout`), its anti-Hermitian
    /// part −(i/2) Σ L̃†L̃ of the jumps L̃ = C_a (R_c (1 − ½X) − W3^c) (W3^c the triples at δ_c; the jumps' sign
    /// the module's); after a jump into this cluster the slow part Π = 1 + ½X + YW of the bare state,
    /// (YW)_ab = −Σ_{c1,c2} V_c1[:,a]† G_a(δ_c1) R_c2[:,b]. Everything in the cluster's frames, each term at
    /// its beat e^{iΩt}.
    pub fn fourth(&self,e:&Elimination,k:usize,r:[f64;3],t:f64)->Result<Fourth,String>{
        let cl=&self.clusters[k];let(n,nc,ne)=(cl.states.len(),self.carriers.len(),self.excited_states.len());
        if !cl.excited.is_empty(){return Err("The fourth-order fold takes a cluster of ground states".into());}
        let local=self.local_at(e,k,r,t)?;let a=self.amplitudes_at(r);
        let v:Vec<Dense>=self.carriers.iter().map(|c|combine(&c.beams,&a,&self.folded)).collect();
        let rc:Vec<Dense>=self.carriers.iter().map(|c|combine(&c.beams,&a,&e.responses)).collect();
        let delta=|c:usize|self.carriers[c].detuning;let st=&cl.states;let theta=&cl.frames;
        // The carriers the fourth order takes (`fourth_carriers`, the tables' alike); an entry's beat Ω is kept where
        // the second order would keep a coupling at it (`fourth_keeps`).
        let(on,size)=self.fourth_carriers(cl);let keep=|i:usize,j:usize,omega:f64|self.fourth_keeps(cl,i,j,omega,size);
        // W3 per triple on the cluster's columns (ne × n), less what the second order's dressing holds: the
        // triples (d, d, c) through b′ = a for (d, a) a far channel of (c, b) -- Σ's first order -- and the light
        // shift of the cluster's states by d, the Hermitian part of (V_d†R_d) between them, in the triples (c, d, d),
        // d ≠ c -- c's block's (`Ground::excluded`); G(ω) at a photon's carrier its block's (`fourth_column`).
        let mut w3:Vec<Dense>=vec![];
        for c1 in 0..nc{for c2 in 0..nc{for c4 in 0..nc{
            let w=self.frequency(&[c1,c4],&[c2]);let inner=v[c2].adjoint_product(&rc[c4]);let mut out=Dense::zeros(ne,n);
            if !(on.contains(&c1)&&on.contains(&c2)&&on.contains(&c4)){w3.push(out);continue;}
            for(j,&b)in st.iter().enumerate(){
                let mut between:Vec<Complex>=(0..self.ground_states.len()).map(|bp|inner[(bp,b)]).collect();self.excluded(c1,c2,c4,b,&mut between,&|bp|inner[(b,bp)]);
                let col:Vec<Complex>=(0..ne).map(|i|between.iter().enumerate().fold(Complex::ZERO,|s,(bp,&x)|s+rc[c1][(i,bp)]*x)).collect();
                for(x,o)in self.fourth_column(e,k,j,w,&col,false)?.into_iter().enumerate(){for i in 0..ne{out[(i,x)]+=o[i];}}
            }
            w3.push(out);
        }}}
        let triple=|c1:usize,c2:usize,c4:usize|&w3[(c1*nc+c2)*nc+c4];
        // X on the cluster at t, and its parts (row, column, beat, value).
        let mut xs=Dense::zeros(n,n);let mut parts:Vec<(usize,usize,f64,Complex)>=vec![];
        for group in &cl.coherent{for &(y,wy)in group{for &(x,wx)in group{
            if !(on.contains(&y.carrier)&&on.contains(&x.carrier)){continue;}
            let(i,j)=(cl.position(y.state).unwrap(),cl.position(x.state).unwrap());
            let value=(0..ne).fold(Complex::ZERO,|s,q|s+rc[y.carrier][(q,y.state)].conj()*rc[x.carrier][(q,x.state)]);
            xs[(i,j)]+=Complex::cis((wy-wx)*t)*value;parts.push((i,j,wy-wx,value));
        }}}
        // H^(4) on the cluster: V_c3†W3 per triple and −½[X, V†R] per carrier pair, on the kept entries.
        let mut h4=Dense::zeros(n,n);
        for c1 in 0..nc{for c2 in 0..nc{for c4 in 0..nc{let w=triple(c1,c2,c4);let omega=self.frequency(&[c1,c4],&[c2]);
            for &c3 in &on{for i in 0..n{for j in 0..n{let beat=theta[i]-theta[j]+delta(c3)-omega;if !keep(i,j,beat){continue;}
                h4[(i,j)]+=Complex::cis(beat*t)*(0..ne).fold(Complex::ZERO,|s,q|s+v[c3][(q,st[i])].conj()*w[(q,j)]);}}}
        }}}
        for &d in &on{for &c in &on{let vr=v[d].adjoint_product(&rc[c]);let nu=delta(d)-delta(c);
            for &(i,q,w,x)in &parts{for j in 0..n{let beat=w+theta[q]-theta[j]+nu;if keep(i,j,beat){h4[(i,j)]-=Complex::cis(beat*t)*x*vr[(st[q],st[j])]*0.5;}}}
            for &(q,j,w,x)in &parts{for i in 0..n{let beat=theta[i]-theta[q]+nu+w;if keep(i,j,beat){h4[(i,j)]+=Complex::cis(beat*t)*vr[(st[i],st[q])]*x*0.5;}}}
        }}
        // The jumps: each class's photons from R_c on (1 − ½X)ψ and −W3^c on ψ, the fourth order's carriers' (the
        // others' stay the second order's).
        let mut w3c:Vec<Dense>=vec![Dense::zeros(ne,n);nc];
        for c in 0..nc{for c1 in 0..nc{for c2 in 0..nc{for c4 in 0..nc{if same_frame(self.frequency(&[c1,c4],&[c2]),delta(c)){w3c[c].add_scaled(Complex::ONE,triple(c1,c2,c4));}}}}}
        let bend=Dense::from_fn(n,n,|i,j|(if i==j{Complex::ONE}else{Complex::ZERO})-xs[(i,j)]*0.5);
        let jumps:Vec<[Dense;3]>=cl.classes.iter().map(|class|{let target=&self.clusters[class.target];
            std::array::from_fn(|x|{let mut m=Dense::zeros(target.dimension(),n);
                for &(p,w)in &class.photons{
                    let(row,col)=(target.position(p.to).unwrap(),cl.position(p.from).unwrap());let phase=Complex::cis(w*t);
                    let r=(0..ne).fold(Complex::ZERO,|s,q|s+self.emission[x][(p.to,q)]*rc[p.carrier][(q,p.from)]);
                    let r4=(0..ne).fold(Complex::ZERO,|s,q|s+self.emission[x][(p.to,q)]*w3c[p.carrier][(q,col)]);
                    if on.contains(&p.carrier){for j in 0..n{m[(row,j)]+=phase*r*bend[(col,j)];}}else{m[(row,col)]+=phase*r;}
                    m[(row,col)]-=phase*r4;
                }
                m})}).collect();
        // H̃: the Hermitian parts of the second order's and of H^(4), and −(i/2) Σ L̃†L̃.
        let effective_hermitian=Dense::from_fn(n,n,|i,j|(local.effective[(i,j)]+local.effective[(j,i)].conj())*0.5+(h4[(i,j)]+h4[(j,i)].conj())*0.5);
        let mut effective=effective_hermitian;
        let mut decay=Dense::zeros(n,n);for j in &jumps{for m in j{decay.add_scaled(Complex::ONE,&m.adjoint_product(m));}}
        effective.add_scaled(Complex::new(0.,-0.5),&decay);
        // Π = 1 + ½X + YW on the cluster.
        let mut slow=Dense::from_fn(n,n,|i,j|(if i==j{Complex::ONE}else{Complex::ZERO})+xs[(i,j)]*0.5);
        // (YW)_ax = −Σ_b′ Σ_E P_E[a,b′] V_c1[:,b′]† G_b′(z_E) R_c2[:,x], c1's block's terms (`fourth_column` at each place b′,
        // its rows a by the projector's transpose).
        for &c1 in &on{for &c2 in &on{for(j,&b)in st.iter().enumerate(){
            let col:Vec<Complex>=(0..ne).map(|q|rc[c2][(q,b)]).collect();
            for(p,&bp)in st.iter().enumerate(){
                for(i,img)in self.fourth_column(e,k,p,delta(c1),&col,true)?.iter().enumerate(){
                    let z=(0..ne).fold(Complex::ZERO,|s,q|s+v[c1][(q,bp)].conj()*img[q]);if z==Complex::ZERO{continue;}
                    slow[(i,j)]-=Complex::cis((theta[i]-theta[j]+delta(c1)-delta(c2))*t)*z;}
            }
        }}}
        // The readout: M = 1 + Σ_g W_g†W_g over the coherent groups g (W_g's columns its channels' R_c − W3^c at
        // their beats), M^{−1/2} = 1 − ½Δ + ⅜Δ²; the excited populations from the same amplitudes.
        let amplitudes:Vec<Dense>=cl.coherent.iter().map(|group|{let mut m=Dense::zeros(ne,n);
            for &(ch,w)in group{let col=cl.position(ch.state).unwrap();let phase=Complex::cis(-w*t);for q in 0..ne{m[(q,col)]+=phase*(rc[ch.carrier][(q,ch.state)]-w3c[ch.carrier][(q,col)]);}}
            m}).collect();
        let mut metric=Dense::zeros(n,n);for m in &amplitudes{metric.add_scaled(Complex::ONE,&m.adjoint_product(m));}
        let mut root=Dense::from_fn(n,n,|i,j|if i==j{Complex::ONE}else{Complex::ZERO});root.add_scaled(Complex::new(-0.5,0.),&metric);root.add_scaled(Complex::new(0.375,0.),&metric.product(&metric));
        let excited=(0..ne).map(|q|{let mut m=Dense::zeros(n,n);
            for a in &amplitudes{let row=Dense::from_fn(1,n,|_,j|a[(q,j)]);m.add_scaled(Complex::ONE,&row.adjoint_product(&row));}
            root.product(&m).product(&root)}).collect();
        // The transient after a jump into the cluster: per coherent group X (ne × n) its excited states' surplus.
        let mut surplus:Vec<(usize,Dense)>=(0..ne).map(|i|(i,Dense::zeros(n,n))).collect();
        for group in &cl.coherent{
            let mut x=Dense::zeros(ne,n);
            for &(ch,w)in group{let col=cl.position(ch.state).unwrap();let phase=Complex::cis(-w*t);for i in 0..ne{x[(i,col)]+=phase*rc[ch.carrier][(i,ch.state)];}}
            let(ch,_)=group[0];
            for i in 0..ne{let(k,gamma)=(self.state_factor(e,i,ch.carrier,ch.state),self.decay[(i,i)].re);
                let row=Dense::from_fn(1,n,|_,j|x[(i,j)]);surplus[i].1.add_scaled(Complex::new(k/gamma,0.),&row.adjoint_product(&row));}
        }
        Ok(Fourth{effective,jumps,slow,root,excited,surplus})
    }
    /// The carrier whose frequency a fourth-order triple's ω is (a photon's), if any.
    pub(crate) fn photon_carrier(&self,w:f64)->Option<usize>{(0..self.carriers.len()).find(|&c|same_frame(self.carriers[c].detuning,w))}
    /// The fourth order's G(ω) on a source s at the cluster's place b (# Fourth order): at a carrier c's frequency -- a
    /// photon's triples, and Π's YW -- c's block's dressed resolvent, as the second order's response takes it (# The
    /// static ground block): out[x] = Σ_E G_b(z_E) s P_E[b, x] over the cluster's places x, G_b(z_E) the eigen-energy's
    /// (K − z_E)⁻¹ or b's own (K + Σ_b(z_E) − z_E)⁻¹ (`Mode`), a place the block leaves out nothing; without a block
    /// (no carrier at ω, or none on the cluster) the bare (K − E_b − ω)⁻¹ on column b alone. `transpose`: the weights
    /// P_E[x, b] instead, Y's rows (Π's YW).
    pub(crate) fn fourth_column(&self,e:&Elimination,k:usize,b:usize,w:f64,s:&[Complex],transpose:bool)->Result<Vec<Vec<Complex>>,String>{
        let(n,ne)=(self.clusters[k].states.len(),self.excited_states.len());let mut out=vec![vec![Complex::ZERO;ne];n];
        if let Some(block)=self.photon_carrier(w).and_then(|c|e.blocks.get(c)).and_then(|x|x.get(k)).and_then(|x|x.as_ref()){
            for(m,mode)in block.modes.iter().enumerate(){
                let lu=if mode.plain.contains(&b){mode.lu.as_ref()}else{mode.dressed.iter().find(|d|d.place==b).and_then(|d|d.lu.as_ref())};
                let Some(lu)=lu else{continue};let mut g=s.to_vec();lu.solve(&mut g);
                for(x,o)in out.iter_mut().enumerate(){let p=if transpose{block.projector(m,x,b)}else{block.projector(m,b,x)};if p==Complex::ZERO{continue;}for(y,&z)in o.iter_mut().zip(&g){*y+=z*p;}}
            }
            return Ok(out);
        }
        let mut g=s.to_vec();denominator(&e.excited_hamiltonian,-e.energies[self.clusters[k].states[b]]-w)?.solve(&mut g);out[b]=g;Ok(out)
    }
    /// The fourth order's W3 triple (c1, c2, c4) on column b: its middle factor (V_c2†R_c4)[b′, b] over every ground state
    /// b′, less what the second order's dressing already holds (# Fourth order, module documentation # The near light's
    /// dressing): through a far channel (c2, b′) of (c4, b) with c1 = c2, the path R′'s Σ resums -- in W3 its
    /// Rayleigh–Schrödinger form, Σ's first order less the intermediate's admixture, which the dressing's
    /// Brillouin–Wigner form takes whole --; and with c2 = c4 ≠ c1 the light shift by c2 between the cluster's states b′
    /// and b, the Hermitian part of (V_c2†R_c2)[b′, b] (`column(b′)` its entry (b, b′)), which c1's block holds -- an
    /// unmixed block (`Ground::unmixed`) at b′ = b alone, b's energy, its off-diagonal part W3's --; the anti-Hermitian
    /// part, b's scattering by c2, stays.
    pub(crate) fn excluded(&self,c1:usize,c2:usize,c4:usize,b:usize,between:&mut [Complex],column:&dyn Fn(usize)->Complex){
        let k=self.cluster[b];
        if c1==c2{for via in &self.vias[c4][b]{if via.carrier==c2{between[via.state]=Complex::ZERO;}}}
        else if c2==c4{let unmixed=self.unmixed.get(c1).is_some_and(|x|x[k]);
            for &bp in self.clusters[k].states.iter().filter(|&&bp|!unmixed||bp==b){between[bp]=(between[bp]-column(bp).conj())*0.5;}}
    }
    /// An excited state's post-jump trace factor κ_i = 1 − Γ_i²/(Δ_i² + Γ_i²/4) (# Fourth order) from channel (c, b): Δ_i
    /// its detuning from the channel's dressed energy, E_i + Re Σ_ii − z_E at the block's eigen-energy E of b's largest
    /// weight P_E[b,b], Σ b's own self-energy there (`Mode`; E_i − E_b − δ_c without a block).
    pub fn state_factor(&self,e:&Elimination,i:usize,c:usize,b:usize)->f64{
        let gamma=self.decay[(i,i)].re;let k=self.cluster[b];let x=self.clusters[k].position(b);
        let delta=match(e.blocks.get(c).and_then(|y|y.get(k)).and_then(|y|y.as_ref()),x){
            (Some(block),Some(x))if !block.modes.is_empty()=>{
                let m=(0..block.modes.len()).max_by(|&p,&q|block.projector(p,x,x).re.total_cmp(&block.projector(q,x,x).re)).unwrap();let mode=&block.modes[m];
                e.excited_energies[i]+mode.dressed.iter().find(|d|d.place==x).map_or(0.,|d|d.sigma[(i,i)].re)-mode.energy}
            _=>e.excited_energies[i]-e.energies[b]-self.carriers[c].detuning};
        1.-gamma*gamma/(delta*delta+0.25*gamma*gamma)
    }
    /// H's excited block at the units' intensities, decoupled from a cluster's explicit states as its folded responses are (`decoupled`): where a folded amplitude rings after a jump (engine.rs, # The transient).
    pub fn ringing_block(&self,intensities:&[f64],cluster:usize)->Dense{
        self.cluster_hamiltonian(&sub(&self.hamiltonian_with(intensities),&self.excited_states,&self.excited_states),&self.clusters[cluster])
    }
    /// What the transient's delay leaves of its interference at an elimination (engine.rs, # The transient), relative to the photons' weight: the largest
    /// over the clusters of the spread of ξ_i = Γ_i/z_i over the excited amplitudes their channels drive, per coherent group about its mean, √(Σ w_i |ξ_i − ξ̄|²/Σ w_i), every
    /// near beam at its peak, w_i = Σ |R[i,b]|² (ξ_i the channels' mean, −iΓ_i Σ R2[i,b] R[i,b]*/w_i: R2 = G R, z = i G⁻¹).
    pub fn ringing_spread(&self,e:&Elimination)->f64{
        let ne=self.excited_states.len();
        self.clusters.iter().map(|cl|{let(mut spread,mut total)=(0.,0.);
            for group in &cl.coherent{
                let(mut w,mut xi)=(vec![0.;ne],vec![Complex::ZERO;ne]);
                for &(ch,_)in group{for &l in &self.carriers[ch.carrier].beams{for i in 0..ne{let r=e.responses[l][(i,ch.state)];w[i]+=r.norm_sqr();xi[i]+=e.derivatives[l][(i,ch.state)]*r.conj();}}}
                let sum:f64=w.iter().sum();if !(sum>0.){continue;}
                for i in 0..ne{if w[i]>0.{xi[i]=Complex::new(0.,-self.decay[(i,i)].re/w[i])*xi[i];}}
                let mean=(0..ne).fold(Complex::ZERO,|s,i|s+xi[i]*w[i])*(1./sum);
                spread+=(0..ne).map(|i|w[i]*(xi[i]-mean).norm_sqr()).sum::<f64>();total+=sum;
            }
            if total>0.{(spread/total).sqrt()}else{0.}}).fold(0.,f64::max)
    }
    /// The 32 populations of a cluster of ground states' density in the fourth order's coordinates
    /// (`Fourth`): the bare ground ones (M^{−1/2} ρ M^{−1/2})_aa and the excited ones tr(E_i ρ).
    pub fn populations_fourth(&self,k:usize,f:&Fourth,rho:&Dense)->Vec<f64>{
        let cl=&self.clusters[k];let mut out=vec![0.;STATES];let bare=f.root.product(rho).product(&f.root.adjoint());
        for(i,&s)in cl.states.iter().enumerate(){out[self.ground_states[s]]=bare[(i,i)].re;}
        for(i,m)in f.excited.iter().enumerate(){out[self.excited_states[i]]+=(0..m.rows).map(|a|(0..m.cols).map(|b|m[(a,b)]*rho[(b,a)]).fold(Complex::ZERO,|x,y|x+y).re).sum::<f64>();}
        out
    }
    /// Whether a cluster has explicit excited states.
    pub fn mixed(&self,k:usize)->bool{!self.clusters[k].excited.is_empty()}
}

#[cfg(test)]
pub(crate) mod tests{
    use super::*;
    use crate::linear::uniform;
    use crate::Matrix;
    use serde_json::Value;

    /// The descent (module documentation, # Where the trap is): on a standing wave across a Gaussian profile from every side of its sites, at a tie either way, the well nearest the start; a bowl at the start its very bits, one off it
    /// its bottom; on a potential without a well, the start.
    #[test]
    fn a_descent_reaches_the_nearest_well_or_none(){
        let(k,h)=(8.,difference(1.));let everywhere=|_:[f64;3]|true;
        let wave=|phase:f64|move|r:[f64;3]|{let x=(k*r[2]+0.5*phase).cos();-x*x*(-2.*(r[0]*r[0]+r[1]*r[1])).exp()/(1.+r[2]*r[2]/25.)};
        assert_eq!(descent(&wave(0.),[0.;3],h,&everywhere),[0.;3]);
        for degrees in [30.,-60.,89.,91.,-120.,150.,-179.]{
            let phase=f64::to_radians(degrees);let r=descent(&wave(phase),[0.;3],h,&everywhere);let site=-0.5*phase/k;
            assert!((r[2]-site).abs()<=2e-3&&r[0].abs()<=1e-6&&r[1].abs()<=1e-6,"{degrees}: {r:?} against the site at {site}");
        }
        let r=descent(&wave(std::f64::consts::PI),[0.;3],h,&everywhere);assert!((r[2].abs()-0.5*std::f64::consts::PI/k).abs()<=2e-3,"the tie: {r:?}");
        let bowl=|r:[f64;3]|(r[0]-0.3).powi(2)+2.*(r[1]+0.1).powi(2)+3.*(r[2]-r[0]).powi(2);let r=descent(&bowl,[0.;3],h,&everywhere);
        assert!((r[0]-0.3).abs()<=1e-6&&(r[1]+0.1).abs()<=1e-6&&(r[2]-0.3).abs()<=1e-6,"{r:?}");
        let hill=|r:[f64;3]|(-2.*(r[0]*r[0]+r[1]*r[1])).exp()/(1.+r[2]*r[2]/25.);
        assert_eq!(descent(&hill,[0.;3],h,&everywhere),[0.;3]);assert_eq!(descent(&hill,[0.;3],h,&|r|r[0]*r[0]+r[1]*r[1]+r[2]*r[2]<=1.),[0.;3]);
    }

    /// The free-space 1D PGC (the PGC example before the tweezer example) as the app sent it, and
    /// gme.py's numbers for it (tests/fixtures/ground_gme.json).
    fn fixture()->Value{serde_json::from_str(include_str!("../tests/fixtures/ground_gme.json")).unwrap()}
    fn index(v:&Value)->usize{v.as_u64().unwrap() as usize}
    fn float(v:&Value)->f64{v.as_f64().unwrap()}
    /// A rows × cols matrix from its nonzero entries [row, column, re, im].
    fn entries(v:&Value,rows:usize,cols:usize)->Dense{
        let mut m=Dense::zeros(rows,cols);
        for e in v.as_array().unwrap(){m[(index(&e[0]),index(&e[1]))]=Complex::new(float(&e[2]),float(&e[3]));}
        m
    }
    fn matrix(v:&Value)->Matrix{let m=entries(v,STATES,STATES);Matrix{real:m.entries.iter().map(|z|z.re).collect(),imag:m.entries.iter().map(|z|z.im).collect()}}
    fn floats(v:&Value)->Vec<f64>{v.as_array().unwrap().iter().map(float).collect()}
    /// A request from a fixture: the matrices its beams lack (the counter-rotating coupling, the outside shift) are zero.
    pub(crate) fn request(v:&Value)->Request{
        let or_zero=|m:&Value|if m.is_null(){Matrix{real:vec![0.;STATES*STATES],imag:vec![0.;STATES*STATES]}}else{matrix(m)};
        let beams=v["beams"].as_array().unwrap().iter().map(|b|{let d=floats(&b["direction"]);Beam{id:b["id"].as_str().unwrap().into(),name:b["name"].as_str().unwrap().into(),
            direction:[d[0],d[1],d[2]],wave_number_per_um:float(&b["waveNumberPerUm"]),waist_um:b["waistUm"].as_f64(),intensity_mw_cm2:float(&b["intensityMWcm2"]),
            frequency_index:b["frequencyIndex"].as_array().unwrap().iter().map(|n|n.as_i64().unwrap() as i32).collect(),coupling:matrix(&b["coupling"]),
            counter_coupling:or_zero(&b["counterCoupling"]),outside_shift:or_zero(&b["outsideShift"])}}).collect::<Vec<Beam>>();
        // The trap beam: the focused one (none in the free-space fixture).
        let trap_beam=beams.iter().position(|b|b.waist_um.is_some()).unwrap_or(0);
        let e=&v["emission"];
        let frames=if v["frameFrequenciesMHz"].is_null(){vec![0.;STATES]}else{floats(&v["frameFrequenciesMHz"])};
        Request{states:v["states"].as_array().unwrap().iter().map(|s|s.as_str().unwrap().into()).collect(),hamiltonian:matrix(&v["hamiltonian"]),
            frame_frequencies_mhz:frames,frequencies_mhz:floats(&v["frequenciesMHz"]),beams,
            emission:[matrix(&e[0]),matrix(&e[1]),matrix(&e[2])],field_vector_g:[0.;3],mass_kg:float(&v["massKg"]),
            trap_beam,initial_density:vec![Complex::ZERO;STATES*STATES],temperature_uk:30.,times_us:vec![]}
    }
    fn mismatch(a:&Dense,b:&Dense)->(f64,f64){let mut d=a.clone();d.add_scaled(-Complex::ONE,b);(d.largest(),b.largest())}

    /// The free-space PGC perturbed at random: the ground energies split as by a field, every beam's
    /// coupling scaled by a complex factor, an oblique plane wave added to the cooling carrier and a
    /// beam focused at the atom across the line, at a frequency of its own.
    fn perturbed(seed:u64)->Request{
        let mut next=uniform(seed);let mut r=request(&fixture()["request"]);
        let scale=|m:&mut Matrix,z:Complex|for(re,im)in m.real.iter_mut().zip(m.imag.iter_mut()){let x=z*Complex::new(*re,*im);(*re,*im)=(x.re,x.im);};
        for i in (0..STATES).filter(|&i|r.states[i].starts_with("ground-")){r.hamiltonian.real[i*STATES+i]+=next();}
        for beam in &mut r.beams{scale(&mut beam.coupling,Complex::new(1.+next(),next()));}
        let cooling=r.beams.iter().find(|b|b.id=="cool-plus").unwrap().clone();
        let mut oblique=cooling.clone();(oblique.id,oblique.name,oblique.direction)=("oblique".into(),"斜光".into(),[0.6,0.,0.8]);
        scale(&mut oblique.coupling,Complex::new(next(),next()));
        let mut focused=cooling;(focused.id,focused.name,focused.direction,focused.waist_um)=("focused".into(),"聚焦光".into(),[0.,1.,0.],Some(1.5));
        focused.frequency_index=focused.frequency_index.iter().map(|n|2*n).collect();scale(&mut focused.coupling,Complex::new(3.+next(),next()));
        r.beams.extend([oblique,focused]);r.trap_beam=r.beams.len()-1;r
    }
    /// Points over a few wavelengths and the focused beam's waist (um).
    fn points(seed:u64)->Vec<[f64;3]>{let mut next=uniform(seed);(0..3).map(|_|[next(),2.*next(),4.*next()]).collect()}

    /// The trace is kept: at every point and time each cluster's jumps carry its whole decay,
    /// Σ_φ L_φ†L_φ = i(H_eff − H_eff†).
    #[test]
    fn the_jumps_carry_the_whole_decay(){
        for seed in 1..=4{
            let r=perturbed(seed);let g=Ground::new(&r).unwrap();
            for(p,&x)in points(seed).iter().enumerate(){
                let e=g.eliminate(&g.intensities_at(x)).unwrap();
                for k in 0..g.clusters.len(){
                    let local=g.local_at(&e,k,x,0.37*p as f64).unwrap();let n=g.clusters[k].dimension();
                    let mut sum=Dense::zeros(n,n);for j in &local.jumps{for m in j{sum.add_scaled(Complex::ONE,&m.adjoint_product(m));}}
                    // The whole decay is the jumps', the folded pairs' far channels in the dressed
                    // responses (module documentation, # The near light's dressing).
                    let h=&local.effective;
                    let decay=Dense::from_fn(n,n,|i,j|(h[(i,j)]-h[(j,i)].conj())*Complex::new(0.,1.));
                    let(error,scale)=mismatch(&sum,&decay);
                    assert!(error<=1e-12*scale,"seed {seed}, cluster {k}: {error:e} of {scale:e}");
                }
            }
        }
    }

    /// The readout keeps the trace: Σ_i E_i = Y + Π_P (the explicit excited states'), and the 32
    /// populations of a density sum to one.
    #[test]
    fn the_populations_sum_to_one(){
        for seed in 1..=4{
            let r=perturbed(seed);let g=Ground::new(&r).unwrap();let mut next=uniform(seed);
            for x in points(seed){
                let e=g.eliminate(&g.intensities_at(x)).unwrap();
                for k in 0..g.clusters.len(){
                    let n=g.clusters[k].dimension();let local=g.local_at(&e,k,x,1.3).unwrap();
                    let a=Dense::from_fn(n,n,|_,_|Complex::new(next(),next()));let mut rho=a.product(&a.adjoint());
                    let trace:f64=(0..n).map(|i|rho[(i,i)].re).sum();rho=rho.scaled(Complex::new(1./trace,0.));
                    // Σ_i E_i = Y + the explicit excited states' projector.
                    let mut total=Dense::zeros(n,n);for readout in &local.excited_readout{total.add_scaled(Complex::ONE,readout);}
                    let mut expected=local.admixture.clone();for i in g.clusters[k].states.len()..n{expected[(i,i)]+=Complex::ONE;}
                    let(error,scale)=mismatch(&total,&expected);assert!(error<=1e-13*scale.max(1e-300),"seed {seed}: {error:e} of {scale:e}");
                    let sum:f64=g.populations(k,&local,&rho).iter().sum();assert!((sum-1.).abs()<=1e-13,"seed {seed}: {sum}");
                }
            }
        }
    }

    /// R − w R2 is the exact response to first order: column b's remainder is w² (K − E_b − δ_c)⁻² R(w)_b,
    /// at most (w/pole)² of R(w)_b. At w = 0 the exact response is R itself.
    #[test]
    fn the_first_order_response_leaves_a_second_order_remainder(){
        let r=perturbed(1);let g=Ground::new(&r).unwrap();let e=g.eliminate(&g.intensities_at([0.;3])).unwrap();let pole=g.pole;
        let norm=|m:&Dense,b:usize|(0..m.rows).map(|i|m[(i,b)].norm_sqr()).sum::<f64>().sqrt();
        for l in g.carriers.iter().flat_map(|c|c.beams.iter().copied()){
            assert_eq!(g.response(&e,l,0.).unwrap(),e.responses[l]);
            for w in [pole/8.,-pole/3.]{
                let exact=g.response(&e,l,w).unwrap();let mut remainder=exact.clone();
                remainder.add_scaled(-Complex::ONE,&e.responses[l]);remainder.add_scaled(Complex::new(w,0.),&e.derivatives[l]);
                for b in 0..g.ground_states.len(){
                    let bound=(w/pole).powi(2)*norm(&exact,b)+f64::EPSILON*(norm(&e.responses[l],b)+w.abs()*norm(&e.derivatives[l],b));
                    assert!(norm(&remainder,b)<=bound,"beam {l}, w {w}, column {b}: {} > {bound}",norm(&remainder,b));
                }
            }
        }
    }

    /// The app example's tweezer alone (tests/fixtures/tweezer_focus.json): its co- and counter-rotating
    /// components both fold, so it is no carrier, and at the focus its fold and outside shift give the
    /// pipeline study's numbers: |2,+2⟩ 21.733 MHz deep, the D2 stretched resonance 24.04 MHz up, the
    /// radial trap frequency 100.55 kHz.
    #[test]
    fn the_tweezer_folds_into_the_study_s_trap(){
        let f:Value=serde_json::from_str(include_str!("../tests/fixtures/tweezer_focus.json")).unwrap();
        let r=request(&f["request"]);let g=Ground::new(&r).unwrap();
        assert!(g.components.iter().all(|c|c.folded)&&g.carriers.is_empty(),"{:?}",g.components);
        let(focus,far)=(g.hamiltonian_at([0.;3]),g.hamiltonian_at([0.,0.,1e6]));
        let state=|label:&str|r.states.iter().position(|s|s==label).unwrap();
        let shift=|i:usize|(focus[(i,i)].re-far[(i,i)].re)/TAU;
        let(depth,resonance)=(shift(state("ground-2-2")),shift(state("D2-3-3"))-shift(state("ground-2-2")));
        // The radial curvature of |2,+2⟩'s energy at the focus, by central differences across the tweezer.
        let h=1e-3;let energy=|x:f64|g.hamiltonian_at([x,0.,0.])[(state("ground-2-2"),state("ground-2-2"))].re;
        let curvature=(energy(h)+energy(-h)-2.*energy(0.))/(h*h);
        let trap=(curvature*1.0545718176461565e-34/r.mass_kg*1e6).sqrt()/TAU*1e3;
        assert!((depth+21.733).abs()<=1e-3&&(resonance-24.04).abs()<=1e-2&&(trap-100.55).abs()<=1e-2,"depth {depth} MHz, resonance {resonance} MHz, trap {trap} kHz");
    }

    /// Without a field the sublevels of a hyperfine level are degenerate and the light couples them
    /// -- the free-space PGC's lin⊥lin pairs along z by Δm = 0, ±2, so each level splits into its
    /// even and odd m (the jumps' photons keep the parity too) -- and the cooler and the repumper far
    /// from Raman resonance leave F = 1 and F = 2 apart: clusters of 1, 2, 2 and 3 states. The cooler
    /// moved to a Raman resonance with the repumper (both on F′ = 2: a Λ scheme) joins F = 1 and F = 2
    /// within each parity: two clusters of four, each one's frame making its Raman pairs' beat
    /// static, F = 2 offset from F = 1 by the Raman detuning. In the PGC the cooler is −f0 from the
    /// repumper (frequency axis 0), so f0 = (ν_2 − ν_1)/2π − δ_R puts the Raman detuning at δ_R.
    #[test]
    fn a_raman_resonant_lambda_scheme_joins_the_hyperfine_levels(){
        let pgc=request(&fixture()["request"]);let g=Ground::new(&pgc).unwrap();
        let sizes=|g:&Ground|{let mut s:Vec<usize>=g.clusters.iter().map(|c|c.states.len()).collect();s.sort();s};
        assert_eq!(sizes(&g),vec![1,2,2,3]);
        assert!(g.clusters.iter().all(|c|c.states.iter().all(|&b|g.levels[b]==g.levels[c.states[0]])),"F = 1 and F = 2 apart");
        let e=g.eliminate(&g.intensities_at([0.;3])).unwrap();
        let level=|f:u8|(0..8).filter(|&b|g.levels[b]==f).map(|b|e.energies[b]).sum::<f64>()/if f==1{3.}else{5.};
        let hyperfine=(level(2)-level(1))/TAU;
        for detuning in [0.,0.1]{
            let mut lambda=pgc.clone();lambda.frequencies_mhz[0]=hyperfine-detuning;
            let g=Ground::new(&lambda).unwrap();assert_eq!(sizes(&g),vec![4,4],"Raman detuning {detuning} MHz");
            for cl in &g.clusters{
                let offset=|f:u8|{let s:Vec<f64>=cl.states.iter().zip(&cl.frames).filter(|x|g.levels[*x.0]==f).map(|(&b,&theta)|e.energies[b]-theta).collect();s.iter().sum::<f64>()/s.len() as f64};
                assert!(((offset(2)-offset(1))/TAU-detuning).abs()<=1e-6,"{} MHz against {detuning} MHz",(offset(2)-offset(1))/TAU);
            }
        }
    }
}
