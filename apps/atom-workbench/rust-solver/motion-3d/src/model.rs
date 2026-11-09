//! The 3D model: the eliminated generator of `ground` on each cluster at any point r and momentum
//! ħκ, from tables over the focused beams' intensities and the near carriers' fields at r.
//!
//! # Where the generator depends on r
//! H(r) = H0 + Σ_u (s_u(r) − s_u(0)) O_u + Σ_u s_u(r) F_u (ground.rs) depends on r only through
//! the units' intensities s_u(r) -- a beam's |a_l(r)|², or the summed field's of beams of one
//! frequency and polarization, their standing wave --, and so does the elimination: the energies, K
//! and every near beam's responses. The near beams enter through their amplitudes a_l(r) (phase and
//! profile, a Gaussian beam's exact), linearly per carrier. The tables hold the elimination on a
//! uniform grid of the focused units' intensities s ∈ [0, 1]^D, cubic in each s_f, the grid doubled
//! while a finer grid costs less than the trajectories its smaller error saves (`tabulate`), the error
//! the interpolation's at the new nodes where the atoms go, against the exact elimination, relative to
//! every table's scale there. The focused units are the trap beam's and
//! every varying unit whose profile matters (`Model::new`): one whose intensity changes over the
//! atom's reach by more than the budget leaves -- its shifts O_u + F_u against the excited poles --
//! gets an axis; the others' shifts are taken at the focus.
//!
//! # A carrier's field basis
//! Each near beam's coupling V_l is the dipole operator contracted with the beam's field, so the
//! couplings of one carrier span at most three matrices. An orthonormal basis U_k of that span
//! (Gram–Schmidt, Frobenius inner product) gives V_l = Σ_k c_lk U_k and -- the responses being linear
//! in the coupling column by column -- R_l = Σ_k c_lk T_k, T_k the responses of U_k's folded part
//! U^f_k (the kept drives' entries zeroed, the same for every beam of the carrier, ground.rs). At r
//! the carrier drives V_c = Σ_k E_k U_k and responds R_c = Σ_k E_k T_k(s), E_k(r) = Σ_l a_l(r) c_lk:
//! the carrier's field on its basis, three numbers whatever its number of beams.
//!
//! # The generator of a cluster at r and t
//! With Φ_k′k = E^d_k′* E^c_k for the kept couplings absorbing c and emitting d (`ground::Group`),
//! S_k′k = ½(U^fd_k′† T^c_k + T^d_k′† U^fc_k) and Q_k′k = T^d_k′† D T^c_k on the group's entries
//! between ground states,
//!
//! ```text
//! H_eff = h0 − Σ_groups e^{iΩ̃t} Σ_k′k Φ_k′k (S_k′k + (i/2) Q_k′k)
//!       + Σ_drives e^{iΩ̃t} Σ_k E_k U_k[e,b] |e⟩⟨b| + h.c.  + (i/2) Σ_crosses e^{iΩ̃t} Σ_k E_k X_k |e⟩⟨b| + h.c.  − (i/2) Σ_decays
//! ```
//!
//! h0 holding the states' energies less their frames, the kept static couplings and the folded
//! static couplings' van Vleck shift -- for the explicit excited states too, their energies in H(r)
//! (the trap's anti-trapping of the excited states among them). A group of one carrier with itself
//! and no beat is Hermitian in (k′, k), S_kk′ = S_k′k†, Q_kk′ = Q_k′k†: its tables hold k′ ≤ k, the
//! pairs k′ < k standing for both. Between two explicit states its S is their folded drives' van
//! Vleck shift, −½ Σ_b U^f_k[e,b] U^f_k′[e′,b]* (Re 1/(K_ee − δ − E_b) + Re 1/(K_e′e′ − δ − E_b)) (ground.rs `Elimination::shift_denominator`), and Q zero.
//! The kept drives are linear in the fields and need no table: U_k at the focus is exact. The decay
//! of the explicit states, −(i/2) Σ_φ J_φ†J_φ on their rows and columns, is constant between two of
//! them (`decays`) and linear in the fields between one and a ground state (`crosses`: the folded
//! amplitude A_c = −R_c of the class's photons, X_k = w·T_k[:,b] with w = Σ_{to,a} C_a[to,e]* C_a[to,·]
//! over the photons the two share). Only the Hermitian part acts on the motion: the Ehrenfest force
//! is −∇Re⟨ψ|H_eff|ψ⟩/⟨ψ|ψ⟩, with the kept drives' gradients Σ_k ∇E_k U_k the stimulated force, beam by beam on the atom's path (below). The
//! decay is the jumps' product by construction: where a class joins two folded photons of channels
//! without a kept coupling (through an explicit state's photon of their frequency), their cross term
//! Q = Σ_{to,a} (C_a T^d)[to,a]†(C_a T^c)[to,b] over the landing states they share enters alone, in a
//! group of its own without S (ground.rs `FoldedCross`).
//!
//! # The generator's beats
//! Every term's residual Ω̃ is a beat of the generator: H_eff(t) = H₀ + Σ_j e^{iΩ_j t} H_j (`Components`), at the frequency Ω̃ + q·v on the atom's path (below),
//! H₀ the static part less its scalar (a phase), H_j the terms at Ω_j, the gradients alike. Residuals
//! within `same_frame`'s tolerance at the cluster's largest frame and detuning (the frames' round-off)
//! are static, and are one beat where within it of each other; the beats come in pairs ±Ω (a term and
//! its adjoint), and the gradient's Hermitian part pairs them, G_{−Ω} = G_Ω†. A cluster with beats or explicit
//! excited states takes the exact step, unless its beats are all within the exponentials' share wherever the atom goes
//! (engine.rs `quiet_clusters`); `local` is the plain step's, its beats at the step's middle.
//!
//! # The atom's path
//! Over an exact step (engine.rs) the atom moves on a straight line through the step's middle point r_h at the time t_m, r_h + v(t − t_m), so that every beam's plane wave advances as e^{ik_l·v(t − t_m)}: the
//! state sees each beam at its Doppler-shifted frequency, within the step, whatever the step's size. (Held at r_h, a kept drive had no Doppler shift within a step, only phase steps between steps, which a
//! jump's bare state forgets: the anti-friction or friction of its line lost where the step did not resolve the internal state's memory: engine.rs, # First order in the Doppler shift.)
//! A linear term of the explicit states -- a kept drive, a decay cross term, a dressing's drive -- is a sum over its carrier's beams, so each beam's share (`Tone`) goes into the beat of the term's
//! residual and of the beam's wave vector, of frequency Ω̃ + k_l·v at that atom's velocity (`on_path`): the beats are the cluster's, their matrices the point's, their frequencies a step's; a beam and
//! its reverse are one pair ±Ω, each the other's mirror. The states take wave vectors K_s as they take frames: ψ_s = e^{iK_s·v(t − t_m)} x_s, x_s under the diagonal shift K_s·v, a term's beat then at
//! q = k_l + K_b − K_e. Along a spanning tree of the kept drives, the strongest first (`state_waves`), the tree's beams stand at q = 0, so that where the drives are a tree -- one k per state among them -- no beat is left
//! and the step is static, at no cost; where a state is driven by beams of different k their (k_l − k_l′)·v are real beats, on the grid and under the rule that leaves out small ones like any other's (engine.rs `Segment::resolve`).
//! The states take wave vectors where that leaves fewer beats than none: on a 3D molasses' explicit rung the six beams' 4 pairs of beats (±k along each axis and the probe's) stand at 73 along a tree.
//! A beam within the single precision of the cluster's largest coupling has no phase of its own (`Tone`): what the kernel's generator cannot hold, a term's share it goes along with.
//!
//! On the path the terms are the field's at the point times their phases, the rest held; left out, each of second order about the middle:
//! - a beam's envelope along the path, (v·∇ln|a_l|) τ of its amplitude, as the tables' intensities and the first order's denominators: a tweezer's within v τ/w of a waist w;
//! - the velocity's change within the step, a phase k·(F/m) τ²/2, the flow–kick–flow splitting's own order, which the step-doubling check measures;
//! - the groups' couplings Φ_k′k = E^d_k′* E^c_k, held at r_h, whose beam pairs' (k_c − k_d)·v have no in-step phase, as in a cluster of ground states. They stand between states of one wave vector by construction (`state_waves`: what they couple has one), so that they stay static in any gauge.
//! The path's velocity is the step's middle's, of κ_h and half the kick I the step takes, known from the state at the step's start (engine.rs `Engine::on_path`). That keeps the step reversible: reversed, it takes the same line back only at the mean of the momenta before and
//! after its kick. At κ_h the impulse along the path depends on the momentum it changes, and phase space grows where the force is a potential's: for a force of wave vector q the kick's factor from the path falls with κ as (q (ħ/m) κ τ)²/24 and is read half a kick off the middle, an area
//! of 1 + (q (ħ/m) I τ)²/24 per step over an orbit. PGC1D's rung of 3+3 and 2+2 explicit states, a lattice of its kept drives (4 ms, 2048 atoms): T_x 13.4 ± 0.6 uK at 1.25 us and 7.2 ± 0.3 at 0.31 us at κ_h, at the middle's momentum 7.2 ± 0.5 and 7.4 ± 0.3, where the phases held in the step
//! read 5.1 ± 0.4, 5.5 ± 0.3 and at 0.078 us 7.1 ± 0.5. A kept standing wave's energy over 240 steps of 1 us, against a sum of ½(ħ/m)I² of 5.3 rad/us: moved by 0.30 at κ_h, by −0.007 at the middle's momentum (tests/engine.rs).
//! Nothing counts twice with the fold's first order: R − wR₂ (below, # The photons and the Doppler term) shifts the denominators of the amplitudes the fold eliminates, in the groups and in a cross term's X2; the path takes the phases of a term's own
//! field, the numerators. The kinetic shift's recoil part ħk²/2m stays the first order's: a kept drive's frequency takes k_l·v of the classical velocity, whose momentum the stimulated force moves as the state goes up.
//!
//! # The near light's dressing
//! Every near carrier's response is dressed by the other near light (ground.rs, # The near light's
//! dressing): the elimination at each node takes it, each near carrier's field at its units'
//! intensities there, and the tables hold the dressed responses -- the folded pairs' far channels in
//! their denominators, where the release added their first order as van Vleck paths. What the
//! dressing leaves out (a carrier's beams' interference, its next order) is bounded at each node
//! against the pole, its largest over the tables in the ledger (`Model::dressing`).
//!
//! # The photons and the Doppler term
//! The photon of axis x scattered from carrier c leaves the atom in C_x A_c ψ = −C_x R_c ψ, from the
//! jump tables J_xk = C_x T_k (all ground rows: the class chooses the target there); an explicit
//! state's photon in C_x ψ_e, the two added in one class. Absorbing from beam l
//! raises the excited state's energy by its kinetic shift w_l = (ħ/m) k_l·κ + ħ|k_l|²/2m; to first
//! order R_l → R_l − w_l R2_l (ground.rs), so the photon is C_x Σ_k (E_k T_k − E^D_k T2_k) ψ with
//! J2_xk = C_x T2_k and E^D_k = Σ_l a_l w_l c_lk, and the generator's responses are the same (`first_shift` keeps w_l within
//! half the carrier's pole, where a beam's scattering 1 − 2 w Re G stays a decay, far beyond the first order's allowance).
//! With A^D = Σ_k E^D_k T2_k its first order is
//!
//! ```text
//! H^(1) = ½ (V†A^D + A^D†V) + (i/2)(R†D A^D + A^D†D R) = ½ Σ_k′k (ξ_k′k W^S_k′k + h.c.) + (i/2) Σ_k′k (ξ_k′k W^Q_k′k + h.c.)
//! ξ_k′k = E_k′* E^D_k        W^S_k′k = U_k′†T2_k        W^Q_k′k = T_k′†D T2_k
//! ```
//!
//! per pair of carriers (d, c) at its beat, on the group's entries of ground states; a class's folded crosses in W^Q over the
//! landing states they share, a decay cross term with an explicit state in X2_k. Its parts are not corrections of the populations alone, odd in v
//! and so of second order in the ensemble: a rate first order in v, times the kick, is a force first order in v, and
//! F·v does not average out.
//! - The decay is the jumps' Σ L†L to first order: on a running wave the force's whole slope, ħk ∂R/∂v. Its first order moves the
//!   populations too, and their rest force is of the same order: σ⁺ and σ⁻ beams pumping two ground states into each other
//!   balance at p₁ = R₋/(R₊ + R₋) and no force at all, where a jump's mean reweighted by the moving rate (correct on one
//!   state: R₀ ρ b̄ = R(v) b̄) leaves p₁ = ½ and the running waves' friction (tests/engine.rs).
//! - The Hermitian part's gradient is a force of the same order as the jumps' first order: on a standing wave the jumps' mean gives
//!   the Doppler friction's spatial mean everywhere and −∇H^(1) the rest, 8 k² v g² σ Γ |G|⁴ sin²(kz) as the optical Bloch equations:
//!   nothing at an antinode, for an atom a trap holds within the wavelength. In a state's evolution it is the response's delay, the
//!   fields taken at r − v Im G, and the light shift's Doppler shift.
//! - The photon's gradient (`Photons::absorption`) takes the Doppler term's own, so that a beam's part f_l = a_l z_l moves κ by its
//!   own ∇a_l/a_l.
//!
//! With the three the mean force on a state is the adiabatic expansion's to first order in v, −2 Re⟨c_e|∇V|c_g⟩ with c_e = −(R − A^D) c_g,
//! less the time derivative of the momentum the excited amplitude holds (of mean zero) -- the folded model's mean force
//! exact at that order whatever the beams. What is left out: the second order in w (the decay's A^D†D A^D, (kv/σ)² of
//! the rate, booked, engine.rs); the amplitude's lag behind the denominators and the envelopes it moves through,
//! i G (v·∇_env) R, of the Doppler term's order times ∇_env/k (the tweezer's gradient across a shallow trap's near-resonant shell, the control
//! cycle's: the odd rate +0.003 of 0.62/us at z = 0.4 um, the loss at 300 us +0.001 ± 0.002, T within 0.1 uK); a drive the dressing brings and two
//! explicit states' shift through the folded drives, first order on what is small already.
//!
//! W needs no tables of its own: per group and node it is a number times the decay's own table, W^S ≈ λ^S Q, W^Q ≈ λ^Q Q, the first order a modified field in the second
//! order's products. Q = T†D T and W share their poles' weights: with a resolvent G of one energy, T = G U and D = i(G⁻¹ − G⁻¹†), so that W^Q = −i W^S + (i/Γ) Q where D = Γ, and on
//! a pole's term Q = Γ G*G M, W^S = G² M, W^Q = Γ G*G² M: λ^Q = G the pole's, λ^S = G/(Γ G*). A pole far off weighs in Q and W alike as its G², where it weighs in S and T as G: on the 3D
//! molasses W^S on Q leaves 0.8 % and W^Q 8 %, the excited states' tensor shift against the pole (T2 = γ T leaves 14-34 %). So, Y = Σ_k′k ξ_k′k Q_k′k,
//!
//! ```text
//! H^(1) = μ Y + ν Y†        μ = ½(λ^S + iλ^Q), ν = ½(λ^S* + iλ^Q*)        F_a ∋ −Re Σ_k′k λ^S ∇_aξ_k′k ⟨Q_k′k⟩
//! ```
//!
//! Y's and Y†'s coefficients join Φ's on the Q the generator gathers anyway. λ on a group of one pair and one entry is exact, a resonant shell's among them; what it leaves elsewhere --
//! a group of states of different energies, whose poles differ (B ≠ 0), the far share where no pole is near -- is kept entry by entry where it can reach `FIRST_SHARE` of T
//! (`Model::transfer_tables`: the friction its error makes against the recoil heating, 2 k_B T/ħ |ΔW|/|Q|), W − λ Q in a table of those entries alone, and the rest bounded (`Model::first_error`).
//! The tables' slope of W in the force, −Re ξ ∇s_f ⟨∂_f W^S⟩, is left out, and with it in the photon's gradient the slope of J2: on a pole's term the two sum to −2 |u|² Re(ξ G*∂_f G) ∇s_f, for one beam's ξ = w |E|² −w ∇ρ_e,
//! the excited population's gradient times the shift, which does no net work over a path that returns (the adiabatic expansion's force holds no derivative of the denominators: the rest is the amplitude's lag above, of
//! its order). Measured on the control cycle, on one seed's streams: L at 300 us −0.0010 ± 0.0002, T's second half within 0.12 %.
//!
//! # The trap
//! The trap beam's focus is the origin. Its trap is each state's potential with the near beams'
//! interference left out but within a unit of the trap's light (`Model::potential`; its standing wave is in it, the fields of a near carrier's beams as of a folded one's); its
//! Hessian at the well of that light (ground.rs # Where the trap is, `Trap::centre`) gives the trap's principal axes and frequencies (`Trap`). An atom is held while its
//! mechanical energy, ħ²κ²/2m plus its potential relative to far from the trap, is negative -- or,
//! unbound, until it leaves the trap's region (engine.rs # Loss): the potential the engine's force
//! derives from, Re⟨ψ|H_eff|ψ⟩/⟨ψ|ψ⟩ of its current internal state, less the same with the trap's light
//! off (`Model::trap_potential`) -- the trap beam and the beams of its mode, along its axis with its
//! waist and frequency (its reflections, `Model::trap_beams`).
use crate::ground::{self,Ground};
use crate::ledger::BUDGET;
use crate::linear::{hermitian_eigen_within,hermitian_eigenvalues,round_off,single_round_off,symmetric_eigen,unit_round_off,Dense};
use crate::cost::{advance_work,pays,COMPLEX_MULTIPLY_ADD_NS};
use crate::solve::{NUMERICAL_SHARE,POPULATION_TARGET};
use crate::stats::KT_PER_UK;
use crate::{Complex,Request,STATES};
use rayon::prelude::*;

pub const HBAR:f64=1.0545718176461565e-34;
pub const BOLTZMANN:f64=1.380649e-23;
/// The most focused beams the tables span (one axis of intensity each).
pub const MOST_FOCUSED:usize=3;
/// The ground manifold's states (5S₁/₂, F = 1 and 2): a photon's amplitude on each.
pub const GROUND:usize=8;
/// The most states of a cluster of ground states, and its generator's interleaved size.
pub const MOST_STATES:usize=8;
/// The most states of a cluster with explicit excited states: the 32 of the model.
pub const MOST_MIXED:usize=STATES;
/// The largest rank of a carrier's couplings (the dipole operator's three components).
pub const MOST_RANK:usize=3;
/// The coarsest grid of each intensity axis (nodes), the fewest a cubic stencil takes doubled once.
const FIRST_NODES:usize=5;
/// An instant at which the generator's beats take generic phases (us, the golden ratio's inverse: no
/// rational multiple of a period), where `Model::lindblad` checks it besides t = 0.
const GENERIC:f64=0.618_033_988_749_894_8;
/// What a carrier's coupling leaves off its basis, of its norm, and is on it: no new matrix of the span (`Model::laid_out`, Gram-Schmidt). Literal, no derivation; far above `round_off`.
pub(crate) const OFF_THE_BASIS:f64=1e-10;
/// What a beam's coupling may leave off its carrier's span, of its norm, before the model is refused. Literal, no derivation.
pub(crate) const OFF_THE_SPAN:f64=1e-9;
/// The head of a model's refusal for a generator that gains (`Model::lindblad`): solve.rs stops
/// with it where the ladder's first rung, every drive folded, meets it.
pub const NOT_LINDBLAD:&str="消去激发态后的生成元不是 Lindblad 形式";
/// The most a model's tables may take while a grid is refined, the coarse grid's and the fine one's
/// together (bytes), and the most nodes one grid may have: the model's work bounds, as `MOST_STEPS`
/// and `MOST_TRAJECTORIES` are a solve's (solve.rs). A resonance across the focused units'
/// intensities -- a near probe crossing its line where far beams shift it, its ridge a tenth of the
/// intensities wide -- can ask for a grid finer than a machine holds. A grid beyond either bound is
/// not made, whatever it would save (`tabulate`): the tables keep the last one, its measured
/// interpolation error booked (`Model::table_error`, the ledger's), the same on any machine.
pub const MOST_TABLE_BYTES:usize=1<<32;
pub const MOST_TABLE_NODES:usize=1<<20;
/// The doublings above their smallest error that stop the tables' refinement (`Model::tabulate`): a grid eight times finer that has not come below it. A line the grids do not yet resolve moves the error about one
/// on its way: the retro-reflected tweezer with a resonant probe at a mirror phase of 30°, its rung keeping the probe, 0.60, 1.10, 0.75 and 0.012 from 9 to 65 nodes, and every drive folded 0.84, 1.10, 1.44, 0.29,
/// then 0.069, 0.011 and 0.0014 at 513; on a line of half the width 0.87, 1.0, 0.82, 0.35, 0.059 and 0.0064 from 33 to 1025. Stopped at the first rise, it booked 110 % on every mirror phase but 0, with its T and
/// populations what the finer grids give.
const STALLED_DOUBLINGS:usize=3;
/// The nodes of a grid computed at once, written into its tables before the next: a grid's nodes
/// are held once, in its tables.
const NODE_BATCH:usize=4096;
/// What the first order's transfer may leave out on T, relative (`Model::transfer_tables`): the numerical tolerances' share of T's target (solve.rs, # Shares).
const FIRST_SHARE:f64=crate::solve::NUMERICAL_SHARE*crate::solve::TEMPERATURE_TARGET;
/// What a coupling within an excited manifold may move a ringing photon's amplitude by and be left out (`Model::ringing`): the numerical tolerances' share of the rule's budget.
pub(crate) const RINGING_SHARE:f64=crate::solve::NUMERICAL_SHARE*BUDGET;

/// A beam's wave vector k_l (rad/um) on the lab axes, from its direction and |k|.
pub fn wave_vector(direction:[f64;3],wave_number:f64)->[f64;3]{direction.map(|n|n*wave_number)}
/// A term's Doppler shift q·v (rad/us) at the velocity v (um/us): q its wave vector, the k of the beams it absorbs less those it emits (`wave_vector`) -- a kept drive's beam's k_l, a ground
/// Raman pair's k_c − k_d per pair of their beams. In the frame of an atom moving at v the term's frequency is Ω̃ + q·v, its residual's in the cluster's frames and this: the phase e^{i(Ω̃ + q·v)t}
/// it takes along the atom's straight path (module documentation, # The atom's path; the exact step's beats, `on_path`).
pub fn doppler(q:[f64;3],v:[f64;3])->f64{q[0]*v[0]+q[1]*v[1]+q[2]*v[2]}

/// A beam's geometry: its direction and wave number, and its waist when Gaussian.
#[derive(Clone,Debug)]
pub struct Geometry{pub direction:[f64;3],pub wave_number:f64,pub waist:Option<f64>}
impl Geometry{
    /// The wave vector (rad/um): `wave_vector`'s.
    pub fn wave_vector(&self)->[f64;3]{wave_vector(self.direction,self.wave_number)}
    /// A unit vector across the beam: a lab axis less its part along the beam.
    pub(crate) fn across(&self)->[f64;3]{
        let n=self.direction;let pick=if n[0].abs()<0.9{[1.,0.,0.]}else{[0.,1.,0.]};let along=pick[0]*n[0]+pick[1]*n[1]+pick[2]*n[2];
        let v:[f64;3]=std::array::from_fn(|i|pick[i]-along*n[i]);let l=(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]).sqrt();v.map(|x|x/l)
    }
    /// a(r) and ∇a(r) (1/um): the plane wave e^{ikζ}, or the paraxial Gaussian beam focused at the
    /// origin e^{ikζ} exp(−ρ²/(w² q))/q, q = 1 + iζ/z_R, z_R = k w²/2 (request.rs `Beam::amplitude_at`),
    /// whose ∇ln a = ik n − 2(r − ζn)/(w² q) + i(ρ²/(w² q²) − 1/q) n / z_R.
    pub fn field(&self,r:[f64;3])->(Complex,[Complex;3]){
        let(a,log)=self.field_log(r);(a,log.map(|x|a*x))
    }
    /// The analytic logarithmic slope stays finite when the Gaussian amplitude underflows.
    fn field_log(&self,r:[f64;3])->(Complex,[Complex;3]){
        let(k,n)=(self.wave_number,self.direction);let zeta=r[0]*n[0]+r[1]*n[1]+r[2]*n[2];let wave=Complex::cis(k*zeta);
        match self.waist{
            None=>(wave,n.map(|x|Complex::new(0.,k*x))),
            Some(w)=>{
                let rayleigh=k*w*w/2.;let q=Complex::new(1.,zeta/rayleigh);let inverse=Complex::ONE/q;
                let rho2=(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]-zeta*zeta).max(0.);
                let a=wave*(Complex::new(-rho2/(w*w),0.)*inverse).exp()*inverse;
                let along=Complex::new(0.,k)+(Complex::new(0.,rho2/(w*w))*inverse*inverse-Complex::new(0.,1.)*inverse)*(1./rayleigh);
                let across=inverse*(-2./(w*w));
                (a,std::array::from_fn(|i|along*n[i]+across*(r[i]-zeta*n[i])))
            }
        }
    }
    /// The intensity |a(r)|².
    pub fn intensity(&self,r:[f64;3])->f64{self.field(r).0.norm_sqr()}
}

/// A carrier: its beams, their couplings' basis and coefficients, its photons' wave number.
#[derive(Clone,Debug)]
pub struct Carrier{
    pub beams:Vec<usize>,
    pub rank:usize,
    /// c_lk per beam of the carrier (in `beams` order).
    pub coefficients:Vec<[Complex;MOST_RANK]>,
    pub wave_number:f64,
    /// Its beams' places (j, m), j ≠ m, whose fields add in the trap's potential (`Model::level`): both of one unit of the trap's light.
    pub interfering:Vec<(usize,usize)>,
}

/// A table group's place in a node's slice: a kept group of `ground::Group` with one residual beat,
/// its field pairs (k′ ≤ k for a Hermitian one), its entries -- the (row, column) places of the
/// cluster's generator its couplings at this beat reach; S and Q vanish elsewhere -- and where its S
/// and Q lie (the entries' values per pair). A `decay` group holds the decay alone, its S zero: the
/// cross terms of folded photons a class joins without a kept coupling (`ground::FoldedCross`). Its first
/// order in the kinetic shifts (module documentation, # The photons and the Doppler term): every pair
/// (k′, k) -- where the group's entries hold ground states --, its transfer λ^S and λ^Q at `transfer`, what the transfer leaves
/// beyond its share (`kept`), and while the tables are refined at `w` per pair its W^S and W^Q on the entries.
#[derive(Clone,Debug)]
pub(crate) struct GroupLayout{pub(crate) absorbed:usize,pub(crate) emitted:usize,pub(crate) residual:f64,pub(crate) hermitian:bool,pub(crate) pairs:Vec<(usize,usize)>,pub(crate) entries:Vec<(usize,usize)>,pub(crate) s:usize,pub(crate) q:usize,
    pub(crate) decay:bool,pub(crate) first:Vec<(usize,usize)>,pub(crate) transfer:usize,pub(crate) kept:Vec<Kept>,w:usize}
/// What the first order's transfer leaves of one field pair (k′, k) beyond its share (`Model::transfer_tables`): the entries (row, column)
/// and where their W^S − λ^S Q, then their W^Q − λ^Q Q lie.
#[derive(Clone,Debug)]
pub(crate) struct Kept{pub(crate) pair:(usize,usize),pub(crate) entries:Vec<(usize,usize)>,pub(crate) at:usize}
/// Where a carrier's photon tables lie: J and J2 (3 axes × rank × ng × n0), T (rank × ne × n0) and, right after it, its columns' norms
/// ‖T_k[:,b]‖ (rank × n0, `Model::ringing_bound`), on the cluster's n0 ground states.
#[derive(Clone,Copy,Debug)]
pub(crate) struct CarrierLayout{pub(crate) j:usize,pub(crate) j2:usize,pub(crate) t:usize,pub(crate) norm:usize}
/// A beam's share of a term of the explicit states (module documentation, # The atom's path): the beam's place among its carrier's beams; the beam whose wave vector it takes, its own or, within the single precision of
/// the cluster's largest coupling, its term's strongest, none where that too lies within it; the part of the term's residual at that wave vector (`Beats::of`), the part of their negatives, and of a kept drive the beam's coupling Σ_k c_lk U_k[e,b].
#[derive(Clone,Copy,Debug)]
pub(crate) struct Tone{pub(crate) beam:usize,lead:Option<usize>,pub(crate) part:usize,pub(crate) mirror:usize,pub(crate) u:Complex}
/// A kept drive: row (explicit state), column (ground state), carrier, residual beat, U_k[e,b]
/// per rank at the focus and its beams' tones.
#[derive(Clone,Debug)]
pub(crate) struct DriveLayout{pub(crate) row:usize,pub(crate) col:usize,pub(crate) carrier:usize,pub(crate) residual:f64,pub(crate) u:[Complex;MOST_RANK],pub(crate) tones:Vec<Tone>}
/// A decay cross term between an explicit state (row) and a ground state (column) whose folded
/// photons share a class with its: carrier, residual beat, the landing states the two share, and
/// where its X_k = w·T_k[:,b] and X2_k = w·T2_k[:,b] lie (rank values each), w = Σ_{to,a} ĉ_a[to]* C_a[to,·]
/// with the explicit state's dressed photon ĉ (`Layout::emission`), and its beams' tones.
#[derive(Clone,Debug)]
struct CrossLayout{row:usize,col:usize,carrier:usize,residual:f64,tos:Vec<usize>,at:usize,at2:usize,tones:Vec<Tone>}
/// The decay between two explicit states of a class, −(i/2) Σ_{to,a} ĉ_a[to]* ĉ′_a[to] over the
/// landing states they share: row, column, residual beat, those states, where its value lies, its part among the beats.
#[derive(Clone,Debug)]
struct DecayLayout{row:usize,col:usize,residual:f64,tos:Vec<usize>,at:usize,part:usize}
/// A drive the dressing brings, from ground state b (column) to explicit state e (row) through the
/// folded excited states F: ½(H_eF A_c[F,b] + x_e†V^f_c[F,b]) = Σ_k E_k Y_k, carrier, residual beat,
/// where Y_k lie (rank values), its beams' tones.
#[derive(Clone,Debug)]
struct EffectiveLayout{row:usize,col:usize,carrier:usize,residual:f64,at:usize,tones:Vec<Tone>}
/// Where one cluster's tables lie in a node's slice, and its explicit states' terms: the kept drives
/// (without a table), the decays, the cross terms, the dressing's drives and the explicit states'
/// dressed photons ĉ_a[to] (explicit × 3 axes × ng, at `emission`); its states' wave vectors K_s on the atom's path where they leave fewer beats (module documentation, # The atom's path; `gauged`, else none).
#[derive(Clone,Debug)]
pub(crate) struct Layout{pub(crate) n:usize,pub(crate) n0:usize,pub(crate) size:usize,base:usize,pub(crate) h0:usize,pub(crate) groups:Vec<GroupLayout>,pub(crate) carriers:Vec<CarrierLayout>,crosses:Vec<CrossLayout>,pub(crate) drives:Vec<DriveLayout>,
    decays:Vec<DecayLayout>,effective:Vec<EffectiveLayout>,effective_dropped:f64,emission:usize,pub(crate) fourth:Option<FourthLayout>,pub(crate) beats:Beats,pub(crate) labels:Vec<[f64;3]>,pub(crate) gauged:bool}
/// A cluster's beats (module documentation, # The generator's beats): the distinct residuals and wave vectors of its terms, by the residuals ascending and closed under negation -- (−Ω_max, −q) … (−Ω_min, −q′),
/// (Ω_min, q′) … (Ω_max, q), a wave vector's sign its residual's, of a residual zero its first component's --; the round-offs within which a residual or a wave vector is static or one of them.
#[derive(Clone,Debug,Default)]
pub(crate) struct Beats{pub(crate) omegas:Vec<f64>,pub(crate) waves:Vec<[f64;3]>,round_off:f64,wave_round_off:f64}
impl Beats{
    /// The beats of terms (residual, wave vector): those beyond the round-offs, one where within them of each other.
    fn new(terms:impl Iterator<Item=(f64,[f64;3])>,round_off:f64,wave_round_off:f64)->Beats{
        let mut positive:Vec<(f64,[f64;3])>=terms.filter_map(|(w,q)|Beats::positive(w,q,round_off,wave_round_off).map(|x|(x.1,x.2))).collect();
        positive.sort_by(|a,b|a.0.total_cmp(&b.0).then_with(||(0..3).map(|i|a.1[i].total_cmp(&b.1[i])).find(|o|o.is_ne()).unwrap_or(std::cmp::Ordering::Equal)));
        let mut merged:Vec<(f64,[f64;3])>=vec![];
        for x in positive{if !merged.iter().rev().take_while(|m|x.0-m.0<=round_off).any(|m|(0..3).all(|i|(x.1[i]-m.1[i]).abs()<=wave_round_off)){merged.push(x);}}
        let omegas:Vec<f64>=merged.iter().rev().map(|m|-m.0).chain(merged.iter().map(|m|m.0)).collect();
        let waves:Vec<[f64;3]>=merged.iter().rev().map(|m|m.1.map(|x|0.-x)).chain(merged.iter().map(|m|m.1)).collect();
        Beats{omegas,waves,round_off,wave_round_off}
    }
    /// A term's beat with the sign of the pair's positive one: whether it is the negative of that, the residual and the wave vector, none where both lie within the round-offs of zero.
    fn positive(w:f64,q:[f64;3],round_off:f64,wave_round_off:f64)->Option<(bool,f64,[f64;3])>{
        let w=if w.abs()<=round_off{0.}else{w};let q=q.map(|x|if x.abs()<=wave_round_off{0.}else{x});
        if w==0.&&q==[0.;3]{return None;}
        let negative=w<0.||(w==0.&&q.iter().find(|x|**x!=0.).is_some_and(|x|*x<0.));
        Some(if negative{(true,-w,q.map(|x|0.-x))}else{(false,w,q)})
    }
    /// The beat of a term: None where it is static, else its index in `omegas`.
    fn of(&self,residual:f64,wave:[f64;3])->Option<usize>{
        let(negative,w,q)=Beats::positive(residual,wave,self.round_off,self.wave_round_off)?;let half=self.omegas.len()/2;
        let i=(half..self.omegas.len()).position(|j|(w-self.omegas[j]).abs()<=self.round_off&&(0..3).all(|a|(q[a]-self.waves[j][a]).abs()<=self.wave_round_off)).expect("every term has its beat");
        Some(if negative{half-1-i}else{half+i})
    }
}
/// The fourth order's tables of a cluster of ground states (ground.rs, # Fourth order), on
/// the carriers that drive it, `fields` their field indices (carrier, k):
/// - H^(4) per monomial E*_p E*_p′ E_q E_q′ (`QuarticSlab`: the conjugated fields' pair p ≤ p′, the plain
///   ones' q ≤ q′; the monomial is symmetric in either, so one carrier of rank r has (r(r+1)/2)² slabs and a
///   carrier pair's two orders share theirs), n × n each: the Hermitian part of the van Vleck form
///   V†W3 − ½[X, V†R] and −i/2 times the fourth order of Σ L̃†L̃, the jumps' own (X the metric's coherent
///   part, the readout's);
/// - per carrier pair (c1, c2) R_c1†R_c2 on the coherent entries (X's) and YW_c1c2 on all (`SlowSlab`,
///   r_c1 r_c2 × n × n each);
/// - for the jumps' W3^c at a carrier c's frequency, per carrier c1 inside it C_a G(δ_c) T^c1_k[:,b′] from each source
///   place b to each column y, c's block's (`JumpSlab`, r_c1 × 3 × ng × ng × n × n: k, axis, row, b′, y, b) and the
///   middle factors of the triples (c1, c2, c4)
///   it takes, each P^{c2c4} = U^c2†T^c4 less what the dressing holds (`Middle`, r_c2 r_c4 × ng × n, at `p`: one
///   table per pair and kind);
/// - where two photon channels' jumps meet: per pair of channels (carrier, place) the landing states their photons
///   share a class in (`landing`, a bit per ground state), where the fourth order's decay takes their products.
/// Every G(ω) at a carrier c's frequency is c's block's dressed resolvent (ground.rs `fourth_column`).
/// An entry (x, y) of a slab of lab frequency ν beats at Ω = θ_x − θ_y + ν in the cluster's frames. H^(4)
/// keeps it where the second order keeps a coupling at that beat (below the cut, or at a kept coupling's
/// residual on the entry); the others are zero in the tables, their mixings' squares booked (`dropped`).
#[derive(Clone,Debug)]
pub(crate) struct FourthLayout{pub(crate) carriers:Vec<usize>,pub(crate) fields:Vec<(usize,usize)>,pub(crate) quartic:Vec<QuarticSlab>,pub(crate) slow:Vec<SlowSlab>,pub(crate) jumps:Vec<JumpSlab>,
    pub(crate) p:Vec<(usize,usize,Middle,usize)>,landing:Vec<u32>,pub(crate) dropped:f64}
/// A W3 triple (c1, c2, c4)'s middle factor (V_c2†R_c4)[b′, b] (ground.rs `Ground::excluded`): whole; less the far channels
/// of (c4, b) through c2 (c1 = c2, Σ's first order); or less c2's light shift (c2 = c4 ≠ c1) that c1's block holds:
/// b's own, an unmixed block's (`Shift`, ground.rs `Ground::unmixed`), or between the cluster's states (`Shifts`).
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub(crate) enum Middle{Whole,Vias,Shift,Shifts}
impl Middle{
    fn of(c1:usize,c2:usize,c4:usize,unmixed:bool)->Middle{if c1==c2{Middle::Vias}else if c2!=c4{Middle::Whole}else if unmixed{Middle::Shift}else{Middle::Shifts}}
}
/// One monomial's slab of H^(4): its conjugated and plain field pairs (into `FourthLayout::fields`), its
/// lab frequency ν (rad/us), its kept entries, their parts among the cluster's beats (`Model::components`:
/// 0 the static part, 1 + j beat j's) and where it lies.
#[derive(Clone,Debug)]
pub(crate) struct QuarticSlab{pub(crate) conj:(usize,usize),pub(crate) plain:(usize,usize),pub(crate) nu:f64,pub(crate) keep:Vec<bool>,pub(crate) part:Vec<usize>,pub(crate) at:usize}
impl FourthLayout{
    /// Its slabs' kept entries' beats Ω = θ_x − θ_y + ν in a cluster's frames.
    fn residuals(&self,frames:&[f64])->Vec<f64>{
        let n=frames.len();self.quartic.iter().flat_map(|q|(0..n*n).filter(|&i|q.keep[i]).map(|i|frames[i/n]-frames[i%n]+q.nu).collect::<Vec<f64>>()).collect()
    }
    /// Each slab's kept entries' parts among the cluster's beats.
    fn place(&mut self,frames:&[f64],beats:&Beats){
        let n=frames.len();
        for q in &mut self.quartic{q.part=(0..n*n).map(|i|if q.keep[i]{beats.of(frames[i/n]-frames[i%n]+q.nu,[0.;3]).map_or(0,|j|j+1)}else{0}).collect();}
    }
}
/// One carrier pair's slabs: R_c1†R_c2 on the coherent entries (`coherent`) at `rr`, YW_c1c2 at `yw`; their
/// lab frequency δ_c1 − δ_c2.
#[derive(Clone,Debug)]
pub(crate) struct SlowSlab{pub(crate) carriers:(usize,usize),pub(crate) nu:f64,coherent:Vec<bool>,pub(crate) rr:usize,pub(crate) yw:usize}
/// The jumps' W3 at a photon carrier's frequency through one inner carrier c1: its table -- each column from every source
/// place, or from its own alone where the photon carrier's block is unmixed (`diagonal`, ground.rs `Ground::unmixed`) --
/// and the middle factors of the triples (c1, c2, c4) at that frequency (into `FourthLayout::p`).
#[derive(Clone,Debug)]
pub(crate) struct JumpSlab{pub(crate) photon:usize,pub(crate) inner:usize,pub(crate) pairs:Vec<usize>,pub(crate) diagonal:bool,pub(crate) at:usize}
impl JumpSlab{
    /// The source places per column: all n of the cluster's, or the column's own.
    fn sources(&self,n:usize)->usize{if self.diagonal{1}else{n}}
}

/// e^{iωt} of a run of residuals ω at one time, a value's rotation computed once while it repeats (the
/// photons of a class share theirs): the same rotation as Complex::cis of each.
struct Rotation{t:f64,last:u64,value:Complex}
impl Rotation{
    fn new(t:f64)->Rotation{Rotation{t,last:f64::NAN.to_bits(),value:Complex::ONE}}
    fn at(&mut self,w:f64)->Complex{if w.to_bits()!=self.last{self.last=w.to_bits();self.value=Complex::cis(w*self.t);}self.value}
}

/// A cluster's generator against its jumps at a point and time (`Model::decay_balance`): D = i(H − H†) − Σ L†L
/// (at fourth order less the corrections' δ†δ), the jumps' whole Σ L̃†L̃, D's smallest eigenvalue, Σ L†L's largest
/// and D's round-off scale.
struct Balance{d:Dense,jumps:Dense,low:f64,rate:f64,scale:f64}
/// The largest ε with −D ≤ ε M, M = J + floor (Hermitian, J ⪰ 0): the largest eigenvalue of L⁻¹(−D − round)L⁻†,
/// M = LL† (Cholesky); 0 where none is positive, and infinite where M is not positive definite.
fn excess(d:&Dense,j:&Dense,round:f64,floor:f64)->f64{
    let n=d.rows;let mut l=Dense::zeros(n,n);
    for c in 0..n{
        let mut diagonal=j[(c,c)].re+floor;for k in 0..c{diagonal-=l[(c,k)].norm_sqr();}
        if !(diagonal>0.){return f64::INFINITY;}let diagonal=diagonal.sqrt();l[(c,c)]=Complex::new(diagonal,0.);
        for i in c+1..n{let mut z=j[(i,c)];for k in 0..c{z-=l[(i,k)]*l[(c,k)].conj();}l[(i,c)]=z*(1./diagonal);}
    }
    // L⁻¹ B, column by column (forward substitution).
    let solve=|b:&Dense|->Dense{let mut x=b.clone();for c in 0..n{for i in 0..n{let mut z=x[(i,c)];for k in 0..i{z-=l[(i,k)]*x[(k,c)];}x[(i,c)]=z*(1./l[(i,i)].re);}}x};
    let a=Dense::from_fn(n,n,|i,k|-d[(i,k)]-if i==k{Complex::new(round,0.)}else{Complex::ZERO});
    let y=solve(&a);let x=solve(&y.adjoint()).adjoint();
    hermitian_eigenvalues(&x).into_iter().fold(0_f64,f64::max)
}

/// A cluster's work per call in complex multiply-adds (`Model::work`), and its table pairs.
#[derive(Clone,Copy,Debug)]
pub struct Work{pub generator:f64,pub force:f64,pub photons:f64,pub absorption:f64,pub pairs:usize}

/// The interpolation stencil at s: per focused beam the first of its four nodes, their cubic
/// weights and the weights' derivatives in s_f; the nodes are their tensor product.
#[derive(Clone,Copy,Debug)]
pub struct Stencil{pub dimensions:usize,pub first:[usize;MOST_FOCUSED],pub weights:[[f64;4];MOST_FOCUSED],pub derivatives:[[f64;4];MOST_FOCUSED],pub stride:[usize;MOST_FOCUSED]}
impl Stencil{
    /// The number of nodes (4^D).
    pub fn count(&self)->usize{1<<(2*self.dimensions)}
    /// Node i's index, its weight and its weights' derivatives in each s_f.
    pub fn node(&self,i:usize)->(usize,f64,[f64;MOST_FOCUSED]){
        let(mut index,mut weight,mut derivative)=(0,1.,[1.;MOST_FOCUSED]);
        for f in 0..self.dimensions{let m=(i>>(2*f))&3;index+=(self.first[f]+m)*self.stride[f];weight*=self.weights[f][m];
            for g in 0..self.dimensions{derivative[g]*=if g==f{self.derivatives[f][m]}else{self.weights[f][m]};}}
        (index,weight,derivative)
    }
}

/// The fields of one carrier at (r, κ): E_k, ∇E_k (per k, per axis), E^D_k and ∇E^D_k.
#[derive(Clone,Copy,Debug,Default)]
pub struct CarrierField{pub e:[Complex;MOST_RANK],pub grad:[[Complex;3];MOST_RANK],pub doppler:[Complex;MOST_RANK],pub doppler_grad:[[Complex;3];MOST_RANK]}

/// The most carriers (frequencies of near light) and beams a scene may have.
pub const MOST_CARRIERS:usize=8;
pub const MOST_BEAMS:usize=32;

/// Everything at (r, κ) that does not depend on the internal state: the stencils at the focused
/// beams' intensities there and with the trap beam off, their gradients and the carriers' fields.
#[derive(Clone,Copy,Debug)]
pub struct Point{
    pub r:[f64;3],
    pub kappa:[f64;3],
    pub stencil:Stencil,
    pub far:Stencil,
    /// ∇s_f (1/um).
    pub intensity_gradient:[[f64;3];MOST_FOCUSED],
    pub fields:[CarrierField;MOST_CARRIERS],
    /// The fields' E^D stand: the first order's shifts act on its terms (`Model::balance` takes them off).
    pub shifted:bool,
}

/// A cluster's generator at a point and a time: H_eff less `scalar`, the scalar part of its
/// Hermitian part (a phase); and what the gradient of its Hermitian part takes, for the force on the
/// state of the step's middle (`Model::force`): per group its beat's phase and the first order's λ^S (module documentation, # The photons and the Doppler term), then per table
/// pair its S, every ∂S/∂s_f and its Q at the point on the group's entries (`values`, re/im interleaved) and its coefficients −∇_aΦ and −Φ of S (each twice for a pair standing for two), and
/// after the pairs what the transfer leaves of W^S; after the groups h0's ∂/∂s_f (n × n); and ∇s_f. A cluster of ground states' (`Model::local`; its gradients as matrices `Model::gradients`).
#[derive(Clone,Debug)]
pub struct Local{pub hamiltonian:[Complex;MOST_STATES*MOST_STATES],pub scalar:f64,values:Vec<f64>,coefficients:Vec<[Complex;4]>,intensity_gradient:[[f64;3];MOST_FOCUSED]}
thread_local!{
    /// A thread's last `Local`'s buffers, for its next: a generator per step allocates nothing. (On the 8-state gray molasses a generator's 20 KB lies beyond the system heap's fast sizes,
    /// and sixteen threads at its lock took four times the CPU of one thread's work.)
    static LOCAL_BUFFERS:std::cell::Cell<(Vec<f64>,Vec<[Complex;4]>)>=const{std::cell::Cell::new((Vec::new(),Vec::new()))};
}
impl Drop for Local{
    fn drop(&mut self){let buffers=(std::mem::take(&mut self.values),std::mem::take(&mut self.coefficients));let _=LOCAL_BUFFERS.try_with(|b|b.set(buffers));}
}
/// A cluster's generator at a point and a time on any number of states, row-major n × n: H_eff
/// less `scalar` (the scalar part of its Hermitian part) and the gradient of its Hermitian part per
/// axis -- the explicit excited states' clusters', and any cluster's in the exact step (engine.rs).
#[derive(Clone,Debug)]
pub struct Generator{pub n:usize,pub hamiltonian:Vec<Complex>,pub scalar:f64,pub gradient:[Vec<Complex>;3]}
impl Generator{
    /// −Re⟨ψ|∇H|ψ⟩/⟨ψ|ψ⟩: the Ehrenfest force on ħκ (rad/us per um).
    pub fn force(&self,psi:&[Complex])->[f64;3]{
        let n=self.n;let norm:f64=psi[..n].iter().map(|z|z.norm_sqr()).sum();
        std::array::from_fn(|a|-hermitian_form(&self.gradient[a],psi,n)/norm)
    }
}
/// A cluster's generator at a point split by its beats (`Model::components`): H(t) = H₀ + Σ_j e^{iΩ_j t} H_j
/// and the Hermitian part's gradient G(t) = G₀ + Σ_j e^{iΩ_j t} G_j, Hermitian at every t (G_{−Ω} = G_Ω†); H₀
/// less the static part's scalar. Flat, as the kernel's scratch, n × n each: `matrices` H_q at q n² and, past the `parts`, G_q per axis at (parts + 3q + axis) n², q = 0 the static part;
/// `omegas` the frequencies of the beats still kept (the exact step leaves some out) and `places` their q. Along the atom's path (`Model::on_path`, # The atom's path) the frequencies take the beats' q·v, and where the
/// cluster's states have wave vectors K_s ψ_s = e^{iK_s·v(t − `middle`)} x_s: `gauge`, K_s·v per state, which x_s evolves under (`gauged`; none at rest).
#[derive(Clone,Debug)]
pub struct Components{pub n:usize,pub scalar:f64,pub(crate) parts:usize,pub(crate) omegas:Vec<f64>,pub(crate) places:Vec<usize>,pub(crate) matrices:Vec<Complex>,pub(crate) gauge:[f64;MOST_MIXED],pub(crate) middle:f64,pub(crate) gauged:bool}
/// One beat of a generator: its frequency Ω (rad/us) and its parts of H and of the gradient (n × n).
#[derive(Clone,Copy,Debug)]
pub struct Beat<'a>{pub omega:f64,pub hamiltonian:&'a [Complex],pub gradient:[&'a [Complex];3]}
thread_local!{
    /// A thread's last `Components`' buffers, for its next (as `LOCAL_BUFFERS`): the exact step's generator per step allocates nothing, where a beating cluster's parts took six allocations each.
    static COMPONENT_BUFFERS:std::cell::Cell<(Vec<Complex>,Vec<f64>,Vec<usize>)>=const{std::cell::Cell::new((Vec::new(),Vec::new(),Vec::new()))};
}
impl Drop for Components{
    fn drop(&mut self){let buffers=(std::mem::take(&mut self.matrices),std::mem::take(&mut self.omegas),std::mem::take(&mut self.places));let _=COMPONENT_BUFFERS.try_with(|b|b.set(buffers));}
}
impl Components{
    pub fn n(&self)->usize{self.n}
    /// H₀.
    pub fn hamiltonian(&self)->&[Complex]{let nn=self.n*self.n;&self.matrices[..nn]}
    /// G₀ along an axis.
    pub fn gradient(&self,axis:usize)->&[Complex]{let nn=self.n*self.n;let at=(self.parts+axis)*nn;&self.matrices[at..at+nn]}
    /// The frequencies of the beats kept, ascending and closed under negation.
    pub fn omegas(&self)->&[f64]{&self.omegas}
    /// The kept beat j.
    pub fn beat(&self,j:usize)->Beat<'_>{
        let(q,nn)=(self.places[j],self.n*self.n);
        Beat{omega:self.omegas[j],hamiltonian:&self.matrices[q*nn..(q+1)*nn],gradient:std::array::from_fn(|a|{let at=(self.parts+3*q+a)*nn;&self.matrices[at..at+nn]})}
    }
    /// The beats kept.
    pub fn beats(&self)->impl Iterator<Item=Beat<'_>>{(0..self.omegas.len()).map(|j|self.beat(j))}
    /// Only the beats `keep` says, by their index among the kept.
    pub(crate) fn retain(&mut self,keep:impl Fn(usize)->bool){
        let mut left=0;for j in 0..self.omegas.len(){if keep(j){self.omegas[left]=self.omegas[j];self.places[left]=self.places[j];left+=1;}}
        self.omegas.truncate(left);self.places.truncate(left);
    }
    /// The Ehrenfest force on ħκ on a state at the time t: −Re⟨ψ|G(t)|ψ⟩/⟨ψ|ψ⟩, G(t) = G₀ + 2 Re Σ_{Ω>0} e^{iΩt} G_Ω.
    pub fn force_at(&self,t:f64,psi:&[Complex])->[f64;3]{
        let n=self.n;let norm:f64=psi[..n].iter().map(|z|z.norm_sqr()).sum();let half=self.omegas.len()/2;
        let form=|g:&[Complex]|{let mut s=Complex::ZERO;for a in 0..n{let mut r=Complex::ZERO;for b in 0..n{r+=g[a*n+b]*psi[b];}s+=psi[a].conj()*r;}s};
        std::array::from_fn(|a|{let mut f=form(self.gradient(a)).re;for j in half..self.omegas.len(){let b=self.beat(j);f+=2.*(Complex::cis(b.omega*t)*form(b.gradient[a])).re;}-f/norm})
    }
    /// The generator at time t.
    pub fn at(&self,t:f64)->Generator{
        let mut g=Generator{n:self.n,hamiltonian:self.hamiltonian().to_vec(),scalar:self.scalar,gradient:std::array::from_fn(|a|self.gradient(a).to_vec())};
        for b in self.beats(){
            let phase=Complex::cis(b.omega*t);
            for(o,x)in g.hamiltonian.iter_mut().zip(b.hamiltonian){*o+=phase**x;}
            for a in 0..3{for(o,x)in g.gradient[a].iter_mut().zip(b.gradient[a]){*o+=phase**x;}}
        }
        g
    }
}

/// Re⟨ψ|X|ψ⟩ of a Hermitian n × n matrix X (row-major).
#[inline]
pub fn hermitian_form(x:&[Complex],psi:&[Complex],n:usize)->f64{
    let mut s=0.;
    for a in 0..n{let p=psi[a];s+=x[a*n+a].re*p.norm_sqr();for b in a+1..n{let z=x[a*n+b]*psi[b];s+=2.*(p.re*z.re+p.im*z.im);}}
    s
}

/// out = the weighted sum of four nodes' interleaved complex slices (`Model::interpolate`).
#[inline(always)]
fn four_nodes(w:[f64;4],[a,b,c,e]:[&[f64];4],out:&mut [Complex]){
    for(j,o)in out.iter_mut().enumerate(){let(x,y)=(2*j,2*j+1);o.re=w[0]*a[x]+w[1]*b[x]+w[2]*c[x]+w[3]*e[x];o.im=w[0]*a[y]+w[1]*b[y]+w[2]*c[y]+w[3]*e[y];}
}
#[target_feature(enable="avx2,fma")]
unsafe fn four_nodes_wide(w:[f64;4],nodes:[&[f64];4],out:&mut [Complex]){four_nodes(w,nodes,out)}
/// acc += c x for interleaved complex vectors of W/2 entries.
#[inline(always)]
fn caxpy<const W:usize>(c:Complex,x:&[f64;W],acc:&mut [f64;W]){
    for j in (0..W).step_by(2){let(u,v)=(x[j],x[j+1]);acc[j]+=c.re*u-c.im*v;acc[j+1]+=c.re*v+c.im*u;}
}
/// ⟨ψ|X|ψ⟩ for an interleaved N × N matrix X.
#[inline(always)]
fn quadratic<const N:usize>(x:&[f64],psi:&[Complex])->Complex{
    let mut s=Complex::ZERO;
    for a in 0..N{let mut row=Complex::ZERO;for b in 0..N{let k=2*(a*N+b);row+=Complex::new(x[k],x[k+1])*psi[b];}s+=psi[a].conj()*row;}
    s
}

/// One node of the tables: every cluster's tables there, and its bounds: the dressing's (what it leaves
/// out, `Model::dressing`), the elimination's (`Model::resonance`) and the transient's (`Model::ringing_spread`).
type Node=(Vec<Vec<Complex>>,[f64;3]);

/// The tables on one grid of n nodes per focused unit: per cluster, node-major, its layout's size per
/// node (re/im interleaved); per node its bounds (`Node`); and each bound's largest over the nodes
/// inside the reach (`Model::dressing`, `Model::resonance`, `Model::ringing_spread`); the work its computed
/// nodes took, counted (linear.rs `counted`: complex multiply-adds).
#[derive(Clone,Debug)]
pub(crate) struct Tables{pub(crate) nodes:usize,pub(crate) data:Vec<Vec<f64>>,bounds:Vec<[f64;3]>,reach:[f64;3],work:f64}

/// The trap of the trap beam for a reference state: its place, the well's bottom (ground.rs # Where the trap is), its principal axes (unit rows on the lab axes) and frequencies
/// (rad/us) there, and the frequencies per lab axis.
#[derive(Clone,Copy,Debug)]
pub struct Trap{pub centre:[f64;3],pub axes:[[f64;3];3],pub frequencies:[f64;3],pub lab:[f64;3],pub lattice:Option<Lattice>}
impl Trap{
    /// Preparation locates the well. The illuminated potential supplies its
    /// curvature in rad/µs per µm². Both stages use this definition of the trap.
    pub(crate) fn from_potentials(
        ground:&Ground,beam:usize,preparation:impl Fn([f64;3])->f64,
        illuminated:impl Fn([f64;3],[f64;3])->f64,lattice:impl Fn([f64;3])->Option<Lattice>,
    )->Trap{
        let centre=ground.well(beam,&preparation);
        let step=ground::difference(ground.beams[beam].waist_um.unwrap());
        let(_,_,hessian)=ground::slopes(&|r|illuminated(r,centre),centre,step);
        let(values,axes)=symmetric_eigen(hessian);
        let hbar_over_mass=HBAR/ground.mass_kg*1e6;
        let frequency=|curvature:f64|(curvature*hbar_over_mass).max(0.).sqrt();
        Trap{centre,axes,frequencies:values.map(frequency),lab:std::array::from_fn(|a|frequency(hessian[a][a])),lattice:lattice(centre)}
    }
    /// No trap: no beam focuses.
    pub(crate) fn none()->Trap{Trap{centre:[0.;3],axes:[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],frequencies:[0.;3],lab:[0.;3],lattice:None}}
}

/// A table axis (module documentation, # Where the generator depends on r): field `unit`'s chart coordinate `coordinate` (ground.rs
/// `Unit::chart`: 0 its intensity, then its polarization's) on [lo, hi], the table's coordinate s = (c − lo)/(hi − lo): the intensity on
/// [0, 1], its every value no beam beyond its peak; a polarization's on its range over the reach (ratios of the field's components, which an
/// atom beyond its site may take without bound: there the table takes the reach's edge, its error measured along the trajectories,
/// `Model::profile_error`).
#[derive(Clone,Copy,Debug)]
pub struct Axis{pub unit:usize,pub coordinate:usize,pub lo:f64,pub hi:f64}

/// The trap's standing wave, its beam and reflection one unit (ground.rs `Unit`): on the beam's axis the field is
/// γ₀ a + γ₁ ā, a = e^{ikz}/(1 + iz/z_R), its antinodes -- the sites -- where Φ(z) = kz − atan(z/z_R) + ½ arg(γ₀ γ̄₁)
/// is a multiple of π, its nodes half way between. Its axis (a unit vector), k, z_R and that phase; the envelope's inverse 1 + z²/z_R² at the trap's centre, where the trap's frequencies are read.
#[derive(Clone,Copy,Debug)]
pub struct Lattice{pub axis:[f64;3],pub wave_number:f64,pub rayleigh:f64,pub phase:f64,pub central:f64}
impl Lattice{
    /// The stiffness at a site at z on the axis relative to the trap's: the envelope's there over its own at the trap's centre.
    pub fn share(&self,z:f64)->f64{self.central/(1.+(z/self.rayleigh).powi(2))}
    /// The nearest site's number at z on the axis, and Φ's distance from it there, within ±π/2.
    pub fn nearest(&self,z:f64)->(f64,f64){let phase=self.wave_number*z-(z/self.rayleigh).atan()+self.phase;let n=(phase/std::f64::consts::PI).round();(n,phase-n*std::f64::consts::PI)}
    /// Site n's place on the axis: Φ's root by Newton's steps, from a start that depends on n alone.
    pub fn site(&self,n:f64)->f64{
        let(k,target)=(self.wave_number,n*std::f64::consts::PI-self.phase);let mut z=target/k;
        loop{let x=z/self.rayleigh;let dz=(k*z-x.atan()-target)/(k-1./(self.rayleigh*(1.+x*x)));z-=dz;if dz.abs()<=4.*f64::EPSILON*(z.abs()+1./k){return z;}}
    }
}

/// The 3D model of a request.
#[derive(Clone,Debug)]
pub struct Model{
    pub ground:Ground,
    pub geometry:Vec<Geometry>,
    /// The trap beam and the trap's beams (the trap beam and the beams of its mode, `trap_beams`), and the tables' axes
    /// (`Axis`), the trap beam's field's first: each one's field (`focused`, indices into `Ground::units`), its direction among the
    /// field's coordinates (`axes`); the coordinates no axis takes, at the focus, the axes' share of it removed (`held`); and each
    /// coordinate's weight, its error per unit changed (`choose_profiles`).
    pub trap_beam:usize,
    pub trap_beams:Vec<usize>,
    /// The trap's light (`trap_light`): the initial ensemble's potential (`trap_level`).
    pub trap_light:Vec<usize>,
    pub focused:Vec<usize>,
    pub axes:Vec<Axis>,
    pub held:Vec<f64>,
    pub weights:Vec<f64>,
    /// Each varying field whose coordinates the axes leave, at the focus: the field, its estimated error over the reach and its
    /// largest weight (`choose_profiles`).
    pub profiles:Vec<(usize,f64,f64)>,
    pub carriers:Vec<Carrier>,
    /// ħ/m (um²/us).
    pub hbar_over_mass:f64,
    pub mass_kg:f64,
    pub(crate) layouts:Vec<Layout>,
    pub(crate) tables:Tables,
    /// The interpolation's measured largest relative error where the atoms go (`tabulate`), and where the
    /// work bounds -- or `STALLED_DOUBLINGS` above its smallest error (`table_stalled`) -- stopped the grid's refinement the nodes per axis it kept.
    pub table_error:f64,
    pub table_limit:Option<usize>,
    pub table_stalled:bool,
    /// The near light's dressing (ground.rs `Ground::dressing_bounds`), relative to the generator's
    /// scale, its largest over the tables' nodes inside the reach's intensities (as `resonance`): the bound
    /// of what it leaves out.
    pub dressing:f64,
    /// The fourth order's H^(4) entries left out at beats the second order folds: their mixings' squares
    /// (`FourthLayout`).
    pub fourth_dropped:f64,
    /// The dressing's drives left out at beats the rule folds (`layouts`): their mixings' squares.
    pub effective_dropped:f64,
    /// The largest error on T, relative and per µK of T, of what the first order's transfer leaves out (`transfer_tables`): at the request's T within `FIRST_SHARE`.
    pub first_error:f64,
    /// The elimination's a-priori bound over the atoms' reach: the largest excited weight a ground
    /// state b can take, Σ_l ‖R_l[:,b]‖² over the near beams at their peak amplitude (each one's
    /// (Ω_l/2σ)², the beams' interference averaged out), over the tables' nodes inside the reach's
    /// intensities (`reach`). Where it exceeds the population target the eliminated model is outside
    /// its validity somewhere the atoms go -- light near resonance, which the excited population
    /// measured along the trajectories can miss when the atoms sit in dark states (solve.rs `with_open` books it for what the atoms read leave open).
    pub resonance:f64,
    /// What the transient's delay leaves of its interference (engine.rs, # The transient), relative to the photons' weight: the spread of
    /// Γ_i/z_i over the excited amplitudes a cluster's channels drive, every near beam at its peak, its largest over the tables' nodes inside
    /// the reach's intensities (as `resonance`).
    pub ringing_spread:f64,
    /// The reach the profiles and the bound are judged over: the trap's centre (ground.rs `Ground::well`) and the points ±reach_a along
    /// each principal axis of the trap (`choose_profiles`).
    reach:Vec<[f64;3]>,
    /// The processor has AVX2 and FMA: the generator is built with them.
    wide:bool,
    /// Its clusters of ground states folded at fourth order (`Model::with_order`).
    pub fourth:bool,
    /// The kinetic shift of absorption (`kinetic_shift`), always on in a solve; off only where a
    /// validation compares with the recoil-heating formula, which has no Doppler friction.
    pub doppler:bool,
}

fn frobenius_inner(a:&Dense,b:&Dense)->Complex{a.entries.iter().zip(&b.entries).fold(Complex::ZERO,|s,(x,y)|s+x.conj()**y)}

/// The interpolation weights at s on a uniform grid of `nodes` points on [0, 1], on four consecutive nodes from `first`,
/// and their derivatives in s: the cubic Hermite interpolant with Catmull-Rom's central slopes, (f_{i+1} − f_{i−1})/2h,
/// and at the grid's ends the second-order one-sided ones -- C¹, so that the force the tables' derivative gives is
/// continuous. (The piecewise 4-point Lagrange cubic before it was C⁰: its force jumped at every node by O(h³), and a
/// step that straddles a jump errs at first order in dt -- the 3D PGC's cold atoms crossed so many that the step
/// doubling read twice T's step error on 65 nodes per axis, its excess over 513 nodes falling as n⁻².) Its error is
/// O(h³) (`tabulate`'s first prediction 2⁻³ per doubling).
fn cubic(nodes:usize,s:f64)->(usize,[f64;4],[f64;4]){
    let h=1./(nodes-1) as f64;let x=s.clamp(0.,1.)/h;let i=(x.floor() as usize).min(nodes-2);let u=x-i as f64;
    // The Hermite basis on the interval [i, i + 1] and its derivatives in u.
    let(h00,h10,h01,h11)=(2.*u*u*u-3.*u*u+1.,u*u*u-2.*u*u+u,-2.*u*u*u+3.*u*u,u*u*u-u*u);
    let(d00,d10,d01,d11)=(6.*u*u-6.*u,3.*u*u-4.*u+1.,-6.*u*u+6.*u,3.*u*u-2.*u);
    // The values' weights on the stencil from the end slopes h m_i: the first interval's (−3f₀ + 4f₁ − f₂)/2 and
    // (f₂ − f₀)/2, an inner one's central pair, the last one's (f₃ − f₁)/2 and (3f₃ − 4f₂ + f₁)/2.
    let weights=|a:f64,b:f64,c:f64,d:f64|->[f64;4]{if i==0{[a-1.5*b-0.5*d,2.*b+c,-0.5*b+0.5*d,0.]}
        else if i==nodes-2{[0.,-0.5*b+0.5*d,a-2.*d,0.5*b+c+1.5*d]}else{[-0.5*b,a-0.5*d,c+0.5*b,0.5*d]}};
    let first=if i==0{0}else if i==nodes-2{nodes-4}else{i-1};
    (first,weights(h00,h10,h01,h11),weights(d00,d10,d01,d11).map(|x|x/h))
}

/// The first order's H^(1) = μ Y + ν Y† of its transfer, Y = Σ ξ Q (module documentation, # The photons and the Doppler term): μ = ½(λ^S + iλ^Q), ν = ½(λ^S* + iλ^Q*).
#[inline(always)]
fn first_coefficients(lambda_s:Complex,lambda_q:Complex)->(Complex,Complex){
    let i=Complex::new(0.,0.5);(lambda_s*0.5+i*lambda_q,lambda_s.conj()*0.5+i*lambda_q.conj())
}

/// What a booked population error e costs a run, in runs: its trajectories go as the inverse square of the room its estimates leave the statistics, 1/(1 − e/ε)² of itself against the
/// whole target ε (2e/ε at first order); no number of them reaches the target past it.
fn room_cost(e:f64)->f64{let room=1.-e/POPULATION_TARGET;if room>0.{1./(room*room)}else{f64::INFINITY}}

/// The pairs k′ ≤ k of a Hermitian group of rank r, or every (k′, k) of ranks (r′, r).
/// Each state's wave vector on the atom's path (module documentation, # The atom's path): zero at the first state of each tree, one along what `held` couples, and across each of the `edges` (size, row, column, beam's wave vector), the strongest
/// first, K_row = K_col + k. A spanning tree of the kept couplings, as the frames' (ground.rs `Ground::partition`).
fn state_waves(n:usize,held:&[(usize,usize)],edges:&[(f64,usize,usize,[f64;3])])->Vec<[f64;3]>{
    let mut labels:Vec<Option<[f64;3]>>=vec![None;n];
    let spread=|labels:&mut Vec<Option<[f64;3]>>|loop{
        let mut grew=false;
        for &(a,b)in held{match(labels[a],labels[b]){(Some(x),None)=>{labels[b]=Some(x);grew=true;}(None,Some(x))=>{labels[a]=Some(x);grew=true;}_=>{}}}
        if !grew{break;}
    };
    for s in 0..n{
        if labels[s].is_some(){continue;}
        labels[s]=Some([0.;3]);spread(&mut labels);
        while let Some(&(_,row,col,k))=edges.iter().find(|x|labels[x.1].is_some()!=labels[x.2].is_some()){
            match(labels[row],labels[col]){
                (None,Some(b))=>labels[row]=Some(std::array::from_fn(|a|b[a]+k[a])),
                (Some(e),None)=>labels[col]=Some(std::array::from_fn(|a|e[a]-k[a])),
                _=>{}
            }
            spread(&mut labels);
        }
    }
    labels.into_iter().map(|x|x.unwrap()).collect()
}
fn field_pairs(hermitian:bool,emitted:usize,absorbed:usize)->Vec<(usize,usize)>{
    if hermitian{(0..absorbed).flat_map(|k1|(k1..absorbed).map(move|k|(k1,k))).collect()}else{(0..emitted).flat_map(|k1|(0..absorbed).map(move|k|(k1,k))).collect()}
}

impl Model{
    /// The model of a request with every drive folded (`Model::with` at an infinite allowance).
    pub fn new(request:&Request)->Result<Model,String>{Model::with(request,f64::INFINITY)}
    /// The model of a request: its front end, its drives kept until each ground state's folded rest is
    /// within `allowance` (ground.rs `Ground::with`), the focused beams, the carriers' bases and the
    /// tables.
    pub fn with(request:&Request,allowance:f64)->Result<Model,String>{Model::with_order(request,allowance,false)}
    /// The same with its clusters of ground states folded at fourth order where `fourth` (ground.rs,
    /// # Fourth order).
    pub fn with_order(request:&Request,allowance:f64,fourth:bool)->Result<Model,String>{Model::bounded(request,allowance,fourth,MOST_TABLE_BYTES,MOST_TABLE_NODES)}
    /// The same within the given work bounds on its tables (`MOST_TABLE_BYTES`, `MOST_TABLE_NODES`).
    pub(crate) fn bounded(request:&Request,allowance:f64,fourth:bool,most_bytes:usize,most_nodes:usize)->Result<Model,String>{Model::built(request,Ground::with(request,allowance)?,fourth,most_bytes,most_nodes)}
    /// The model of a request on its front end at an allowance (ground.rs `Ground::with`), which the orders of an allowance share (solve.rs: a solve builds it once).
    pub(crate) fn of_ground(request:&Request,ground:Ground,fourth:bool)->Result<Model,String>{Model::built(request,ground,fourth,MOST_TABLE_BYTES,MOST_TABLE_NODES)}
    fn built(request:&Request,ground:Ground,fourth:bool,most_bytes:usize,most_nodes:usize)->Result<Model,String>{
        let(mut model,bases)=Model::laid_out(request,ground,fourth)?;
        model.prune(&bases)?;
        let stake=model.stake(request);model.tabulate(&bases,most_bytes,most_nodes,stake)?;
        model.transfer_tables(request.temperature_uk);
        model.lindblad()?;
        Ok(model)
    }
    /// Each cluster's counted work (`work`) in the model on this front end at this order, from its layout alone: its tables
    /// unbuilt (solve.rs: the fourth order's tables are worth their build only for a saving beyond it).
    pub(crate) fn layout_work(request:&Request,ground:Ground,fourth:bool)->Result<Vec<Work>,String>{
        let(model,_)=Model::laid_out(request,ground,fourth)?;Ok((0..model.clusters()).map(|k|model.work(k)).collect())
    }
    /// The model on this front end at this order but for its tables, and each carrier's coefficients on its basis.
    fn laid_out(request:&Request,ground:Ground,fourth:bool)->Result<(Model,Vec<Vec<Vec<Complex>>>),String>{
        let geometry:Vec<Geometry>=ground.beams.iter().map(|b|Geometry{direction:b.direction,wave_number:b.wave_number_per_um,waist:b.waist_um}).collect();
        let trap_beam=request.trap_beam;
        if trap_beam>=geometry.len()||geometry[trap_beam].waist.is_none(){return Err("光镊（trapBeamId 所指的光束）需要束腰：3D 运动计算聚焦在原子处的光镊中的单个原子".into());}
        let hbar_over_mass=HBAR/request.mass_kg*1e6;
        let trap_unit=ground.units.iter().position(|u|u.beams.contains(&trap_beam)).unwrap();let trap_beams=ground.trap_beams(trap_beam);
        // Each carrier's couplings: an orthonormal basis and the beams' coefficients on it.
        let mut carriers=vec![];let mut bases:Vec<Vec<Vec<Complex>>>=vec![];
        for c in &ground.carriers{
            let couplings:Vec<&Dense>=c.beams.iter().map(|&l|&ground.couplings[l]).collect();
            let(mut basis,mut mixing):(Vec<Dense>,Vec<Vec<Complex>>)=(vec![],vec![]);
            for(j,v)in couplings.iter().enumerate(){
                let mut u=(*v).clone();let mut g=vec![Complex::ZERO;couplings.len()];g[j]=Complex::ONE;
                // Twice, for orthogonality to round-off.
                for _ in 0..2{for(k,b)in basis.iter().enumerate(){
                    let x=frobenius_inner(b,&u);u.add_scaled(-x,b);for(gi,mi)in g.iter_mut().zip(&mixing[k]){*gi-=x**mi;}
                }}
                let norm=frobenius_inner(&u,&u).re.sqrt();let scale=frobenius_inner(v,v).re.sqrt();
                if norm<=OFF_THE_BASIS*scale{continue;}
                basis.push(u.scaled(Complex::new(1./norm,0.)));mixing.push(g.into_iter().map(|x|x*(1./norm)).collect());
            }
            if basis.len()>MOST_RANK{return Err(format!("A carrier's couplings span {} matrices; the dipole operator has three components",basis.len()));}
            let coefficients:Vec<[Complex;MOST_RANK]>=couplings.iter().map(|v|std::array::from_fn(|k|if k<basis.len(){frobenius_inner(&basis[k],v)}else{Complex::ZERO})).collect();
            // The beams are on the span to round-off.
            for(v,c)in couplings.iter().zip(&coefficients){
                let mut d=(*v).clone();for(k,b)in basis.iter().enumerate(){d.add_scaled(-c[k],b);}
                let(e,s)=(frobenius_inner(&d,&d).re.sqrt(),frobenius_inner(v,v).re.sqrt());
                if e>OFF_THE_SPAN*s{return Err(format!("A beam's coupling is off its carrier's span by {:.1e} of its norm",e/s));}
            }
            carriers.push(Carrier{beams:c.beams.clone(),rank:basis.len(),coefficients,wave_number:c.wave_number,interfering:vec![]});bases.push(mixing);
        }
        if carriers.len()>MOST_CARRIERS||geometry.len()>MOST_BEAMS{return Err(format!("3D 运动最多支持 {MOST_CARRIERS} 个近共振频率、{MOST_BEAMS} 束光"));}
        if ground.ground_states.len()!=GROUND{return Err(format!("The ground manifold has {} states, not {GROUND}",ground.ground_states.len()));}
        if ground.clusters.iter().any(|c|c.excited.is_empty()&&c.states.len()>MOST_STATES){return Err("A cluster of ground states has more than eight states".into());}
        let trap_light=ground.trap_light(&trap_beams);
        for c in &mut carriers{let n=c.beams.len();let of=|j:usize|ground.units.iter().position(|u|u.beams.contains(&c.beams[j]));
            c.interfering=(0..n*n).map(|x|(x/n,x%n)).filter(|&(j,m)|j!=m&&of(j)==of(m)&&trap_light.contains(&c.beams[j])&&trap_light.contains(&c.beams[m])).collect();}
        let centre=ground.trap.centre;
        let mut model=Model{ground,geometry,trap_beam,trap_beams,trap_light,focused:vec![trap_unit],axes:vec![],held:vec![],weights:vec![],profiles:vec![],carriers,hbar_over_mass,mass_kg:request.mass_kg,
            layouts:vec![],tables:Tables{nodes:1,data:vec![],bounds:vec![],reach:[0.;3],work:0.},table_error:0.,table_limit:None,table_stalled:false,dressing:0.,fourth_dropped:0.,effective_dropped:0.,first_error:0.,resonance:0.,ringing_spread:0.,reach:vec![centre],
            wide:std::arch::is_x86_feature_detected!("avx2")&&std::arch::is_x86_feature_detected!("fma"),doppler:true,fourth};
        model.choose_profiles(request)?;
        model.layouts=model.layouts(&bases);model.fourth_dropped=model.layouts.iter().filter_map(|l|l.fourth.as_ref()).map(|f|f.dropped).sum();model.effective_dropped=model.layouts.iter().map(|l|l.effective_dropped).sum();
        Ok((model,bases))
    }

    /// What the run costs (ns), the tables' stake (`tabulate`): a run's first batch, the trajectories nearly every
    /// run takes (solve.rs `FIRST_GROUPS`), each the pilot's first steps over the run (the light's pattern, solve.rs `run`)
    /// at an advance's counted work in the model's dearest cluster (cost.rs `advance_work`). A run that takes more loses only the
    /// 2e/R the refinement stopped at (`room_cost`; 129 nodes on 3D PGC's tables at the most a run takes, an error of 5e-7 against its a-priori 2 %: 33 nodes hold 3e-4),
    /// each cutting the error about 2³-fold at 2^D times the work of the last.
    fn stake(&self,request:&Request)->f64{
        let thermal=(KT_PER_UK*request.temperature_uk*self.hbar_over_mass).sqrt();
        let pattern=self.carriers.iter().map(|c|c.wave_number).fold(0.,f64::max)*thermal;
        let rate=if pattern>0.{pattern}else{self.ground.trap.frequencies.iter().copied().fold(0.,f64::max)};
        let steps=(self.ground.duration*rate/crate::solve::first_phase()).max(1.);
        let advance=(0..self.layouts.len()).map(|k|advance_work(&self.work(k))).fold(0.,f64::max);
        (crate::solve::FIRST_GROUPS*crate::TASK) as f64*steps*advance
    }

    /// The tables' axes (`Axis`). Each varying field's chart is judged over the atom's reach (ground.rs `Ground::trap`, the region of the
    /// trap within ln(1/budget) k_B T of its bottom: the harmonic trap's thermal ellipsoid holding all but the budget of a Boltzmann ensemble),
    /// T the request's -- at its ends along the principal axes, and inside it for interfering beams (ground.rs `reach_points`): a chart
    /// coordinate taken at the focus errs by what that moves the field's T there, Σ_i w_i |ΔT_i|, w_i its T coordinate's weight, its shifts'
    /// size ‖O_i + F_i‖ against the nearest excited pole and the generator's scale -- a field of rank one its intensity's change times its
    /// weight. The trap beam's field takes its intensity; then, the largest error first, the coordinates beyond the budget, at most
    /// `MOST_FOCUSED` in all. The rest stay at the focus, each field's error booked (`profiles`).
    fn choose_profiles(&mut self,request:&Request)->Result<(),String>{
        let g=&self.ground;let trap=g.trap;
        let reach:[f64;3]=std::array::from_fn(|a|ground::thermal_reach(trap.frequencies[a],request.temperature_uk,self.hbar_over_mass).unwrap_or(0.));
        self.reach.extend(ground::reach_points(trap.centre,&trap.axes,reach,g.units.iter().any(|u|u.beams.len()>1)));
        let factor=(1./g.pole.max(f64::MIN_POSITIVE)).max(1./g.generator_scale.max(f64::MIN_POSITIVE));
        self.weights=g.outside.iter().zip(&g.folds).map(|(o,f)|{let mut x=o.clone();x.add_scaled(Complex::ONE,f);x.spectral_norm()*factor}).collect();
        let focus=g.focus_intensities();let trap_unit=self.focused[0];
        // Per field its chart at the reach's points, and the error of holding some of its coordinates at the focus.
        let charts:Vec<Vec<Vec<f64>>>=(0..g.units.len()).map(|u|self.reach.iter().map(|&r|self.chart_at(u,|l|self.geometry[l].field(r).0)).collect()).collect();
        let held=|u:usize,c:&[f64],which:&dyn Fn(usize)->bool|->f64{self.chart_error(u,c,&self.chart_held(u,c,which))};
        let error=|u:usize,which:&dyn Fn(usize)->bool|charts[u].iter().map(|c|held(u,c,which)).fold(0_f64,f64::max);
        // A rank-one field's error: its intensity's largest change over the reach times its weight.
        let one=|u:usize|if g.units[u].rank==1{Some(charts[u].iter().map(|c|(c[0]-focus[u]).abs()).fold(0_f64,f64::max)*self.weights[u])}else{None};
        let mut errors:Vec<(usize,usize,f64)>=(0..g.units.len()).filter(|&u|g.units[u].varies(&g.beams)).flat_map(|u|(0..2*g.units[u].rank-1).map(move|c|(u,c)))
            .filter(|&(u,c)|!(u==trap_unit&&c==0)).map(|(u,c)|(u,c,one(u).unwrap_or_else(||error(u,&|k|k==c)))).collect();
        errors.sort_by(|a,b|b.2.total_cmp(&a.2));
        let mut axes=vec![(trap_unit,0)];
        while let Some(&(u,c,e))=errors.first(){if !(e>BUDGET&&axes.len()<MOST_FOCUSED){break;}errors.remove(0);axes.push((u,c));}
        // Each field with coordinates held: its error with all of them held, at its first place among them.
        let mut profiles:Vec<(usize,f64,f64)>=vec![];
        for &(u,_,_)in &errors{if profiles.iter().any(|p|p.0==u){continue;}
            let free=|k:usize|axes.contains(&(u,k));let e=one(u).unwrap_or_else(||error(u,&|k|!free(k)));
            profiles.push((u,e,self.places(u).iter().map(|&i|self.weights[i]).fold(0_f64,f64::max)));}
        self.profiles=profiles;
        // The axes on their ranges: an intensity on [0, 1], a polarization's coordinate on its range over the reach and at the focus, where the
        // search for each state's trap starts (`trap`).
        self.axes=axes.iter().map(|&(u,c)|{let at=self.ground.chart_of(u,&focus)[c];
            let(lo,hi)=if c==0{(0.,1.)}else{charts[u].iter().fold((at,at),|(lo,hi),x|(lo.min(x[c]),hi.max(x[c])))};
            Axis{unit:u,coordinate:c,lo,hi}}).collect();
        self.focused=self.axes.iter().map(|a|a.unit).collect();self.held=focus;
        Ok(())
    }
    /// The places of field u's T coordinates' weights (`weights`, ground.rs `Ground::outside`'s order).
    fn places(&self,u:usize)->Vec<usize>{let n=self.ground.units.len();let start=n+self.ground.units[..u].iter().map(|v|v.rank*v.rank-1).sum::<usize>();
        std::iter::once(u).chain(start..start+self.ground.units[u].rank*self.ground.units[u].rank-1).collect()}
    /// Field u's chart at the beams' amplitudes a (ground.rs `Unit::chart`).
    pub fn chart_at(&self,u:usize,a:impl Fn(usize)->Complex)->Vec<f64>{let unit=&self.ground.units[u];unit.chart(&unit.fields(a),None)[..2*unit.rank-1].to_vec()}
    /// Field u's chart c with the coordinates `which` takes at the focus.
    fn chart_held(&self,u:usize,c:&[f64],which:&dyn Fn(usize)->bool)->Vec<f64>{let focus=self.ground.chart_of(u,&self.ground.focus_intensities());
        c.iter().enumerate().map(|(k,&x)|if which(k){focus[k]}else{x}).collect()}
    /// What field u's chart b moves its static shift by against its chart a: Σ_i w_i |T_i(a) − T_i(b)| over its T coordinates.
    fn chart_error(&self,u:usize,a:&[f64],b:&[f64])->f64{let unit=&self.ground.units[u];let(ta,tb)=(unit.t_coordinates(a),unit.t_coordinates(b));
        self.places(u).iter().enumerate().map(|(i,&p)|(ta[i]-tb[i]).abs()*self.weights[p]).sum()}
    /// The profile error at r: over the fields with coordinates held, the largest of what holding them and the edges of the axes' ranges
    /// move each one's static shift by (`chart_error`) -- a field of rank one its intensity's change times its weight.
    pub fn profile_error(&self,r:[f64;3])->f64{
        let error=self.profiles.iter().map(|&(u,_,weight)|{
            if self.ground.units[u].rank==1{return (self.unit_intensity(u,r)-self.ground.units[u].focus).abs()*weight;}
            let c=self.chart_at(u,|l|self.geometry[l].field(r).0);let free=|k:usize|self.axes.iter().any(|a|a.unit==u&&a.coordinate==k);
            self.chart_error(u,&c,&self.chart_held(u,&c,&|k|!free(k)))}).fold(0.,f64::max);
        // An atom beyond a polarization axis's range: the table at its edge.
        self.axes.iter().filter(|a|a.coordinate>0).fold(error,|e,a|{let c=self.chart_at(a.unit,|l|self.geometry[l].field(r).0);let mut edge=c.clone();
            edge[a.coordinate]=c[a.coordinate].clamp(a.lo,a.hi);if edge==c{e}else{e.max(self.chart_error(a.unit,&c,&edge))}})
    }
    /// A unit's intensity at r (ground.rs `Unit`), from the beams' geometry.
    pub fn unit_intensity(&self,u:usize,r:[f64;3])->f64{
        self.ground.units[u].intensity(|l|self.geometry[l].field(r).0)
    }
    /// Axis f's table coordinate s at the beams' amplitudes a (`Axis`).
    pub fn axis_with(&self,f:usize,a:impl Fn(usize)->Complex)->f64{
        let axis=&self.axes[f];(self.chart_at(axis.unit,a)[axis.coordinate]-axis.lo)/(axis.hi-axis.lo)
    }
    /// Axis f's table coordinate s at r.
    pub fn axis_value(&self,f:usize,r:[f64;3])->f64{self.axis_with(f,|l|self.geometry[l].field(r).0)}

    /// The bottom of the reference state's well with every drive folded (`Ground::trap`): where what no state's own trap decides is taken: the explicit states' decay.
    pub fn centre(&self)->[f64;3]{self.ground.trap.centre}
    /// The trap's standing wave, where the trap beam's unit is it and its reflection, its envelope at the trap's centre.
    fn lattice(&self,centre:[f64;3])->Option<Lattice>{
        let unit=&self.ground.units[self.focused[0]];if unit.rank!=1||unit.beams.len()!=2||!unit.beams.iter().all(|l|self.trap_beams.contains(l)){return None;}
        let(a,b)=(&self.geometry[unit.beams[0]],&self.geometry[unit.beams[1]]);let waist=a.waist?;
        if (0..3).map(|i|a.direction[i]*b.direction[i]).sum::<f64>()>0.{return None;}
        let z=unit.weights[0]*unit.weights[1].conj();
        let(rayleigh,along)=(a.wave_number*waist*waist/2.,(0..3).map(|i|a.direction[i]*centre[i]).sum::<f64>());
        Some(Lattice{axis:a.direction,wave_number:a.wave_number,rayleigh,phase:0.5*z.im.atan2(z.re),central:1.+(along/rayleigh).powi(2)})
    }
    /// The trap of the model's tables for a cluster's state: at the well of what holds it before the near light comes on (`trap_level`, ground.rs `Ground::well`), its Hessian in its potential
    /// (`potential`), the near light's own shifts in it.
    pub fn trap(&self,cluster:usize,x:usize)->Trap{
        Trap::from_potentials(&self.ground,self.trap_beam,|r|self.trap_level(cluster,x,r),
            |r,centre|self.potential(cluster,x,r,centre),|centre|self.lattice(centre))
    }

    /// Each cluster's table layout: h0, its groups split by residual beat, its carriers' photon
    /// tables (on its ground states) and its decay cross terms; and its explicit states' terms
    /// without a table, the kept drives (U_k at the focus) and the decay between two of them.
    fn layouts(&self,mixing:&[Vec<Vec<Complex>>])->Vec<Layout>{
        let g=&self.ground;let(ng,ne)=(g.ground_states.len(),g.excited_states.len());
        // Per carrier its basis U_k at the focus (ne × ng), the kept drives' entries.
        let basis:Vec<Vec<Dense>>=(0..self.carriers.len()).map(|c|mixing[c].iter().map(|g_k|{let mut m=Dense::zeros(ne,ng);
            for(j,&l)in self.carriers[c].beams.iter().enumerate(){m.add_scaled(g_k[j],&g.couplings[l]);}m}).collect()).collect();
        g.clusters.iter().enumerate().map(|(k,cl)|{
            let(n,n0)=(cl.dimension(),cl.states.len());let mut size=n*n;let h0=0;let mut groups=vec![];
            for gr in &cl.groups{
                let mut residuals:Vec<f64>=gr.entries.iter().map(|e|e.2).collect();residuals.sort_by(f64::total_cmp);residuals.dedup();
                for residual in residuals{
                    let hermitian=gr.absorbed==gr.emitted&&residual==0.;
                    let pairs=field_pairs(hermitian,self.carriers[gr.emitted].rank,self.carriers[gr.absorbed].rank);
                    let mut entries:Vec<(usize,usize)>=gr.entries.iter().filter(|x|x.2==residual).map(|x|(x.0,x.1)).collect();
                    entries.sort();entries.dedup();
                    let square=pairs.len()*entries.len();let first=if entries.iter().any(|e|e.0<n0){field_pairs(false,self.carriers[gr.emitted].rank,self.carriers[gr.absorbed].rank)}else{vec![]};
                    let transfer=size+2*square;groups.push(GroupLayout{absorbed:gr.absorbed,emitted:gr.emitted,residual,hermitian,pairs,entries,s:size,q:size+square,decay:false,first,transfer,kept:vec![],w:0});size=transfer+2;
                }
            }
            // The folded photons a class joins without a kept coupling (ground.rs `FoldedCross`): their decay,
            // in groups of their own per carrier pair and beat.
            let mut keyed:Vec<(usize,usize,f64,Vec<(usize,usize)>)>=vec![];
            for x in &cl.folded_crosses{match keyed.iter_mut().find(|k|k.0==x.absorbed&&k.1==x.emitted&&k.2==x.residual){
                Some(k)=>k.3.push((x.row,x.col)),None=>keyed.push((x.absorbed,x.emitted,x.residual,vec![(x.row,x.col)]))}}
            for(absorbed,emitted,residual,mut entries)in keyed{
                entries.sort();entries.dedup();let hermitian=absorbed==emitted&&residual==0.;
                let pairs=field_pairs(hermitian,self.carriers[emitted].rank,self.carriers[absorbed].rank);
                let square=pairs.len()*entries.len();let first=field_pairs(false,self.carriers[emitted].rank,self.carriers[absorbed].rank);
                let transfer=size+2*square;groups.push(GroupLayout{absorbed,emitted,residual,hermitian,pairs,entries,s:size,q:size+square,decay:true,first,transfer,kept:vec![],w:0});size=transfer+2;
            }
            let carriers=self.carriers.iter().map(|c|{let jump=3*c.rank*ng*n0;let l=CarrierLayout{j:size,j2:size+jump,t:size+2*jump,norm:size+2*jump+c.rank*ne*n0};size+=2*jump+c.rank*ne*n0+c.rank*n0;l}).collect();
            // The explicit states' decay, −(i/2) Σ_φ J_φ†J_φ on their rows: between two of them, and
            // between one and a ground state whose folded photons share its class and landing state.
            let(mut decays,mut crosses):(Vec<DecayLayout>,Vec<CrossLayout>)=(vec![],vec![]);
            for class in &cl.classes{for &(i,to,wi)in &class.direct{
                for &(j,to2,wj)in &class.direct{if to2!=to{continue;}let residual=wj-wi;
                    match decays.iter_mut().find(|x|x.row==i&&x.col==j&&x.residual==residual){Some(x)=>x.tos.push(to),None=>decays.push(DecayLayout{row:i,col:j,residual,tos:vec![to],at:0,part:0})}
                }
                for &(p,wp)in &class.photons{if p.to!=to{continue;}
                    let(col,residual)=(cl.position(p.from).unwrap(),wp-wi);
                    match crosses.iter_mut().find(|x|x.row==i&&x.col==col&&x.carrier==p.carrier&&x.residual==residual){
                        Some(x)=>x.tos.push(to),None=>crosses.push(CrossLayout{row:i,col,carrier:p.carrier,residual,tos:vec![to],at:0,at2:0,tones:vec![]})}
                }
            }}
            for x in crosses.iter_mut(){x.at=size;x.at2=size+self.carriers[x.carrier].rank;size+=2*self.carriers[x.carrier].rank;}
            for x in decays.iter_mut(){x.at=size;size+=1;}
            // The dressing's drives: from a ground state of the cluster through a folded excited state
            // H couples to an explicit one, per carrier that drives it there. They take the rule of every coupling (ground.rs): one above the cut whose mixing fits folds (`foldable`), left out with what it leaves booked,
            // as the fourth order's fast entries (`fourth_keeps`) -- its size H's coupling times the folded state's admixture. Left out, not folded, its second order goes too: its rate over the run and its shift against the state's width, each within the numerical share of the
            // population target (a closer one holds a rate between sublevels of a PGC's probe's rung, 0.011 /us off of 0.30 with it, 0.017 without). Kept, a beat of 13 THz, an 808 nm tweezer's distance from the D2 line, at 550 rad/us of a retro-reflected tweezer's tensor mixing, took
            // the grid to 64 million nodes for a step of 0.7 us, at a mixing² of 4e-11.
            let folded_static=|a:usize,b:usize|g.dressed[a].contains(&b);
            let coupling=|a:usize,b:usize|g.excited_items.iter().find(|x|matches!(x.kind,ground::ExcitedKind::Static{a:p,b:q}if (p==a&&q==b)||(p==b&&q==a))).map_or(0.,|x|x.size);
            let admixture=|c:usize,b:usize,e:usize|g.excited_items.iter().find(|x|matches!(x.kind,ground::ExcitedKind::Drive(d)if d==ground::Drive{carrier:c,excited:e,ground:b})).map_or(0.,|x|x.size/x.distance.max(f64::MIN_POSITIVE));
            let mut effective=vec![];let mut effective_dropped=0.;
            for(p,&e)in cl.excited.iter().enumerate(){
                let reached:Vec<usize>=(0..ne).filter(|&f|!cl.excited.contains(&f)&&folded_static(e,f)).collect();
                if reached.is_empty(){continue;}
                for(x,&b)in cl.states.iter().enumerate(){for(c,carrier)in self.carriers.iter().enumerate(){
                    if !reached.iter().any(|&f|carrier.beams.iter().any(|&l|g.folded[l][(f,b)]!=Complex::ZERO)){continue;}
                    let(m,mx)=(&cl.combinations[n0+p],&cl.combinations[x]);
                    let residual=g.beat(0.,&(0..m.len()).map(|i|m[i]-mx[i]-i32::from(i==c)).collect::<Vec<i32>>());
                    let strength:f64=reached.iter().map(|&f|coupling(e,f)*admixture(c,b,f)).sum();
                    let width=g.decay[(e,e)].re.max(f64::MIN_POSITIVE);let left=ground::mixing(strength,residual.abs(),g.generator_scale,f64::MIN_POSITIVE).powi(2)*(1.+width*g.duration)+strength*strength/residual.abs().max(f64::MIN_POSITIVE)/width;
                    if residual.abs()>=g.cut&&ground::foldable(residual,strength,g.generator_scale)&&left<=crate::solve::NUMERICAL_SHARE*crate::solve::POPULATION_TARGET{effective_dropped+=left;continue;}
                    effective.push(EffectiveLayout{row:n0+p,col:x,carrier:c,residual,at:size,tones:vec![]});size+=carrier.rank;
                }}
            }
            let emission=size;size+=3*ng*cl.excited.len();
            let mut drives:Vec<DriveLayout>=cl.drives.iter().map(|&(row,col,c,residual)|DriveLayout{row,col,carrier:c,residual,
                u:std::array::from_fn(|k|if k<self.carriers[c].rank{basis[c][k][(cl.excited[row-n0],cl.states[col])]}else{Complex::ZERO}),tones:vec![]}).collect();
            // The fourth order: a cluster of ground states, on the carriers whose folded drives reach it.
            let mut fourth=(self.fourth&&cl.excited.is_empty()).then(||self.fourth_layout(k,&mut size));
            // The beams that act through each linear term (module documentation, # The atom's path): a kept drive's coupling and a cross term's or a dressing's drive's folded column, by size; a beam within the
            // single precision of the cluster's largest has no phase of its own, what the kernel's generator cannot hold (`Tone`).
            let column=|l:usize,b:usize|(0..ne).map(|f|g.folded[l][(f,cl.states[b])].abs()).fold(0.,f64::max);
            let coupling=|d:&DriveLayout,j:usize|{let carrier=&self.carriers[d.carrier];(0..carrier.rank).fold(Complex::ZERO,|s,k|s+carrier.coefficients[j][k]*d.u[k])};
            let largest_coupling=drives.iter().flat_map(|d|(0..self.carriers[d.carrier].beams.len()).map(move|j|(d,j))).map(|(d,j)|coupling(d,j).abs())
                .chain(crosses.iter().map(|x|(x.carrier,x.col)).chain(effective.iter().map(|x|(x.carrier,x.col))).flat_map(|(c,b)|self.carriers[c].beams.iter().map(move|&l|(l,b))).map(|(l,b)|column(l,b))).fold(0.,f64::max);
            let floor=single_round_off(largest_coupling);
            let new_tones=|list:Vec<(usize,f64,Complex)>|->Vec<Tone>{
                let strongest=list.iter().filter(|x|x.1>floor).max_by(|a,b|a.1.total_cmp(&b.1)).map(|x|x.0);
                list.into_iter().map(|(beam,size,u)|Tone{beam,lead:if size>floor{Some(beam)}else{strongest},part:0,mirror:0,u}).collect()};
            let columns=|c:usize,b:usize|self.carriers[c].beams.iter().enumerate().map(|(j,&l)|(j,column(l,b),Complex::ZERO)).collect::<Vec<_>>();
            for d in drives.iter_mut(){let list=(0..self.carriers[d.carrier].beams.len()).map(|j|{let u=coupling(d,j);(j,u.abs(),u)}).filter(|x|x.1>0.).collect();d.tones=new_tones(list);}
            for x in crosses.iter_mut(){x.tones=new_tones(columns(x.carrier,x.col));}
            for x in effective.iter_mut(){x.tones=new_tones(columns(x.carrier,x.col));}
            // The states' wave vectors along a spanning tree of the kept drives' beams, the strongest first; what H and the held groups' couplings hold together has one (`state_waves`).
            let held:Vec<(usize,usize)>=groups.iter().flat_map(|gl|gl.entries.iter().copied()).chain(cl.statics.iter().copied()).chain(cl.static_paths.iter().map(|x|(x.0,x.1)))
                .chain(cl.excited_statics.iter().copied()).chain(cl.excited_paths.iter().map(|x|(x.0,x.1))).filter(|x|x.0!=x.1).collect();
            let wave=|c:usize,t:&Tone|t.lead.map_or([0.;3],|j|self.geometry[self.carriers[c].beams[j]].wave_vector());
            let mut edges:Vec<(f64,usize,usize,[f64;3])>=drives.iter().flat_map(|d|d.tones.iter().filter(|t|t.lead==Some(t.beam)).map(move|t|(t.u.abs(),d.row,d.col,wave(d.carrier,t)))).collect();edges.sort_by(|a,b|b.0.total_cmp(&a.0));
            let labels=state_waves(n,&held,&edges);let none=vec![[0.;3];n];
            // The beats of every term's residual and wave vector, a residual static within `same_frame`'s tolerance at the
            // cluster's largest frame and detuning (the round-off of the frames' differences), a wave vector within the same share of the largest; the fourth order's
            // kept entries beat at θ_a − θ_b + ν (`FourthLayout::residuals`). The states' wave vectors where they leave fewer beats than none.
            let round_off=g.residual_round_off(cl);
            let wave_round_off=crate::linear::round_off(2.*self.carriers.iter().map(|c|c.wave_number).fold(0.,f64::max));
            let quartic:Vec<f64>=fourth.as_ref().map_or(vec![],|f|f.residuals(&cl.frames));
            let beats_of=|k:&[[f64;3]]|{
                let across=|row:usize,col:usize,c:usize,t:&Tone|if t.lead.is_none(){[0.;3]}else{let(q,a,b)=(wave(c,t),k[row],k[col]);std::array::from_fn::<f64,3,_>(|x|q[x]+b[x]-a[x])};
                let terms:Vec<(f64,[f64;3])>=groups.iter().map(|x|(x.residual,[0.;3]))
                    .chain(drives.iter().flat_map(|x|x.tones.iter().map(move|t|(x.residual,across(x.row,x.col,x.carrier,t)))))
                    .chain(crosses.iter().flat_map(|x|x.tones.iter().map(move|t|(x.residual,across(x.row,x.col,x.carrier,t)))))
                    .chain(effective.iter().flat_map(|x|x.tones.iter().map(move|t|(x.residual,across(x.row,x.col,x.carrier,t)))))
                    .chain(decays.iter().map(|x|(x.residual,std::array::from_fn::<f64,3,_>(|a|k[x.col][a]-k[x.row][a])))).chain(quartic.iter().map(|&w|(w,[0.;3]))).collect();
                Beats::new(terms.into_iter(),round_off,wave_round_off)};
            let(plain,along)=(beats_of(&none),beats_of(&labels));let gauged=along.omegas.len()<plain.omegas.len();
            let(beats,labels)=if gauged{(along,labels)}else{(plain,none)};
            let part=|w:f64,q:[f64;3]|beats.of(w,q).map_or(0,|j|j+1);
            let across=|row:usize,col:usize,c:usize,t:&Tone|if t.lead.is_none(){[0.;3]}else{let(q,a,b)=(wave(c,t),labels[row],labels[col]);std::array::from_fn::<f64,3,_>(|x|q[x]+b[x]-a[x])};
            let place=|t:&mut Tone,row:usize,col:usize,c:usize,w:f64|{let q=across(row,col,c,t);t.part=part(w,q);t.mirror=part(-w,q.map(|x|-x));};
            for x in drives.iter_mut(){let(row,col,c,w)=(x.row,x.col,x.carrier,x.residual);for t in x.tones.iter_mut(){place(t,row,col,c,w);}}
            for x in crosses.iter_mut(){let(row,col,c,w)=(x.row,x.col,x.carrier,x.residual);for t in x.tones.iter_mut(){place(t,row,col,c,w);}}
            for x in effective.iter_mut(){let(row,col,c,w)=(x.row,x.col,x.carrier,x.residual);for t in x.tones.iter_mut(){place(t,row,col,c,w);}}
            for x in decays.iter_mut(){x.part=part(x.residual,std::array::from_fn(|a|labels[x.col][a]-labels[x.row][a]));}
            if let Some(f)=fourth.as_mut(){f.place(&cl.frames,&beats);}
            // The first order's W^S and W^Q, last in the slice: `transfer_tables` takes them out once the grid is settled.
            let base=size;for gl in groups.iter_mut(){gl.w=size;size+=2*gl.first.len()*gl.entries.len();}
            Layout{n,n0,size,base,h0,groups,carriers,crosses,drives,decays,effective,effective_dropped,emission,fourth,beats,labels,gauged}
        }).collect()
    }

    /// Every unit's intensity with the focused ones at s, the others at the focus.
    pub fn unit_intensities(&self,s:&[f64])->Vec<f64>{
        let mut x=self.held.clone();
        for(axis,&s)in self.axes.iter().zip(s){let at=if axis.coordinate==0{axis.unit}else{self.ground.extras(axis.unit).start+axis.coordinate-1};x[at]=axis.lo+s*(axis.hi-axis.lo);}
        x
    }
    /// H at the focused units' intensities s, the others at the focus.
    pub fn hamiltonian_at(&self,s:&[f64])->Dense{self.ground.hamiltonian_with(&self.unit_intensities(s))}

    /// Every cluster's tables at the intensities s, and its bounds there (`Node`).
    fn node(&self,s:&[f64],mixing:&[Vec<Vec<Complex>>])->Result<Node,String>{
        let g=&self.ground;let intensities=self.unit_intensities(s);let e=g.eliminate(&intensities)?;let(ne,ng)=(g.excited_states.len(),g.ground_states.len());
        // Per carrier: U^f_k (the folded basis), T_k, T2_k (ne × ng).
        let combine=|c:usize,matrices:&[Dense]|->Vec<Dense>{mixing[c].iter().map(|g_k|{let mut m=Dense::zeros(ne,ng);
            for(j,&l)in self.carriers[c].beams.iter().enumerate(){m.add_scaled(g_k[j],&matrices[l]);}m}).collect()};
        let u:Vec<Vec<Dense>>=(0..self.carriers.len()).map(|c|combine(c,&g.folded)).collect();
        let t:Vec<Vec<Dense>>=(0..self.carriers.len()).map(|c|combine(c,&e.responses)).collect();
        let t2:Vec<Vec<Dense>>=(0..self.carriers.len()).map(|c|combine(c,&e.derivatives)).collect();
        let jump=|m:&Dense|->[Dense;3]{std::array::from_fn(|a|g.emission[a].product(m))};
        let(j,j2):(Vec<Vec<[Dense;3]>>,Vec<Vec<[Dense;3]>>)=(t.iter().map(|ts|ts.iter().map(jump).collect()).collect(),t2.iter().map(|ts|ts.iter().map(jump).collect()).collect());
        let dt2:Vec<Vec<Dense>>=t2.iter().map(|ts|ts.iter().map(|m|g.decay.product(m)).collect()).collect();
        let mut out=vec![];
        for(cluster,(cl,layout))in g.clusters.iter().zip(&self.layouts).enumerate(){
            let(n,n0,states)=(layout.n,layout.n0,&cl.states);let mut node=vec![Complex::ZERO;layout.size];
            // h0: the energies less the frames, the kept static couplings (their residual zero) and
            // the folded static couplings' van Vleck shift, of the ground and the explicit states.
            for x in 0..n0{node[layout.h0+x*n+x]=Complex::new(e.energies[states[x]]-cl.frames[x],0.);}
            for(p,&f)in cl.excited.iter().enumerate(){node[layout.h0+(n0+p)*n+n0+p]=Complex::new(e.excited_energies[f]-cl.excited_frames[p],0.);}
            for &(x,y)in &cl.statics{node[layout.h0+x*n+y]+=e.ground[(states[x],states[y])];}
            for &(x,y,f)in &cl.static_paths{let(a,b)=(states[x],states[y]);
                node[layout.h0+x*n+y]+=e.ground[(a,f)]*e.ground[(f,b)]*(0.5*(1./(e.energies[a]-e.energies[f])+1./(e.energies[b]-e.energies[f])));}
            for &(x,y)in &cl.excited_statics{node[layout.h0+x*n+y]+=e.excited[(cl.excited[x-n0],cl.excited[y-n0])];}
            for &(x,y,f)in &cl.excited_paths{let(a,b)=(cl.excited[x-n0],cl.excited[y-n0]);let ee=&e.excited_energies;
                node[layout.h0+x*n+y]+=e.excited[(a,f)]*e.excited[(f,b)]*(0.5*(1./(ee[a]-ee[f])+1./(ee[b]-ee[f])));}
            // The kept groups: S and Q on their entries; the decay groups: Q, the photons' product over the landing
            // states their class shares.
            for gl in layout.groups.iter().filter(|g|g.decay){
                let(c,d)=(gl.absorbed,gl.emitted);
                for(p,&(k1,k0))in gl.pairs.iter().enumerate(){let base=p*gl.entries.len();
                    for(at,&(x,y))in gl.entries.iter().enumerate(){
                        let tos=&cl.folded_crosses.iter().find(|i|i.row==x&&i.col==y&&i.absorbed==c&&i.emitted==d&&i.residual==gl.residual).unwrap().tos;let(a,b)=(states[x],states[y]);
                        node[gl.q+base+at]=(0..3).fold(Complex::ZERO,|s,ax|tos.iter().fold(s,|s,&to|s+j[d][k1][ax][(to,a)].conj()*j[c][k0][ax][(to,b)]));
                    }
                }
            }
            for gl in layout.groups.iter().filter(|g|!g.decay){
                let(c,d)=(gl.absorbed,gl.emitted);
                let entries:Vec<(usize,usize)>=cl.groups.iter().find(|x|x.absorbed==c&&x.emitted==d).unwrap().entries.iter().filter(|x|x.2==gl.residual).map(|x|(x.0,x.1)).collect();
                for(p,&(k1,k0))in gl.pairs.iter().enumerate(){
                    let base=p*gl.entries.len();
                    for &(x,y)in &entries{
                        if x>=n0{
                            // Two explicit states: their folded drives' van Vleck shift (c = d, no Q).
                            let(ea,eb,delta)=(cl.excited[x-n0],cl.excited[y-n0],g.carriers[c].detuning);let mut sv=Complex::ZERO;
                            for b in 0..ng{sv-=u[c][k0][(ea,b)]*u[c][k1][(eb,b)].conj()*(0.5*(e.shift_denominator(ea,b,delta)+e.shift_denominator(eb,b,delta)));}
                            node[gl.s+base+gl.entries.binary_search(&(x,y)).unwrap()]=sv;continue;
                        }
                        let(a,b)=(states[x],states[y]);let(mut sv,mut qv)=(Complex::ZERO,Complex::ZERO);
                        for i in 0..ne{sv+=(u[d][k1][(i,a)].conj()*t[c][k0][(i,b)]+t[d][k1][(i,a)].conj()*u[c][k0][(i,b)])*0.5;}
                        let mut dt=vec![Complex::ZERO;ne];
                        for i in 0..ne{for i2 in 0..ne{dt[i]+=g.decay[(i,i2)]*t[c][k0][(i2,b)];}}
                        for i in 0..ne{qv+=t[d][k1][(i,a)].conj()*dt[i];}
                        let j=gl.entries.binary_search(&(x,y)).unwrap();node[gl.s+base+j]=sv;node[gl.q+base+j]=qv;
                    }
                }
            }
            // The first order in the kinetic shifts: per pair (k′, k), W^S = U^d_k′† T2^c_k and W^Q = T^d_k′† D T2^c_k on the
            // group's entries of ground states; a decay group's W^Q over the landing states its class shares, as its Q.
            for gl in &layout.groups{
                let(c,d,count)=(gl.absorbed,gl.emitted,gl.entries.len());
                for(p,&(k1,k0))in gl.first.iter().enumerate(){
                    let base=gl.w+2*p*count;
                    for(at,&(x,y))in gl.entries.iter().enumerate(){
                        if x>=n0{continue;}
                        let(a,b)=(states[x],states[y]);
                        if gl.decay{
                            let tos=&cl.folded_crosses.iter().find(|i|i.row==x&&i.col==y&&i.absorbed==c&&i.emitted==d&&i.residual==gl.residual).unwrap().tos;
                            node[base+count+at]=(0..3).fold(Complex::ZERO,|s,ax|tos.iter().fold(s,|s,&to|s+j[d][k1][ax][(to,a)].conj()*j2[c][k0][ax][(to,b)]));
                        }else{
                            let(mut s,mut q)=(Complex::ZERO,Complex::ZERO);
                            for i in 0..ne{let(ua,ta)=(u[d][k1][(i,a)].conj(),t[d][k1][(i,a)].conj());s+=ua*t2[c][k0][(i,b)];q+=ta*dt2[c][k0][(i,b)];}
                            node[base+at]=s;node[base+count+at]=q;
                        }
                    }
                }
            }
            for(c,cl2)in layout.carriers.iter().enumerate(){
                for kk in 0..self.carriers[c].rank{for ax in 0..3{for row in 0..ng{for y in 0..n0{
                    let at=((kk*3+ax)*ng+row)*n0+y;node[cl2.j+at]=j[c][kk][ax][(row,states[y])];node[cl2.j2+at]=j2[c][kk][ax][(row,states[y])];
                }}}}
                for kk in 0..self.carriers[c].rank{for y in 0..n0{let mut norm=0.;for i in 0..ne{let z=t[c][kk][(i,states[y])];node[cl2.t+(kk*ne+i)*n0+y]=z;norm+=z.norm_sqr();}node[cl2.norm+kk*n0+y]=Complex::new(norm.sqrt(),0.);}}
            }
            // The explicit states: their dressing x (ground.rs `Ground::dressing`) and dressed photons
            // ĉ_a[to] = C_a[to,e] + Σ_f C_a[to,f] x_f; the decays and cross terms of those; the drives
            // the dressing brings, Y_k = ½(−H_eF T_k[F,b] + x_F† U^f_k[F,b]).
            if !cl.excited.is_empty(){
                let flags:Vec<bool>=(0..ne).map(|f|cl.excited.contains(&f)).collect();
                let dressings:Vec<Vec<Complex>>=cl.excited.iter().map(|&f|g.dressing(&e,f,&flags)).collect::<Result<_,_>>()?;
                let photon:Vec<[Vec<Complex>;3]>=cl.excited.iter().zip(&dressings).map(|(&f,x)|std::array::from_fn(|a|(0..ng).map(|to|
                    g.emission[a][(to,f)]+(0..ne).fold(Complex::ZERO,|s,q|s+g.emission[a][(to,q)]*x[q])).collect())).collect();
                for(p,ph)in photon.iter().enumerate(){for a in 0..3{for to in 0..ng{node[layout.emission+(p*3+a)*ng+to]=ph[a][to];}}}
                for x in &layout.decays{let(i,j)=(x.row-n0,x.col-n0);
                    node[x.at]=x.tos.iter().fold(Complex::ZERO,|s,&to|s+(0..3).fold(Complex::ZERO,|s,a|s+photon[i][a][to].conj()*photon[j][a][to]))*Complex::new(0.,-0.5);}
                for x in &layout.crosses{let i=x.row-n0;
                    let w:Vec<Complex>=(0..ne).map(|q|x.tos.iter().fold(Complex::ZERO,|s,&to|s+(0..3).fold(Complex::ZERO,|s,a|s+photon[i][a][to].conj()*g.emission[a][(to,q)]))).collect();
                    for kk in 0..self.carriers[x.carrier].rank{node[x.at+kk]=(0..ne).fold(Complex::ZERO,|s,q|s+w[q]*t[x.carrier][kk][(q,states[x.col])]);
                        node[x.at2+kk]=(0..ne).fold(Complex::ZERO,|s,q|s+w[q]*t2[x.carrier][kk][(q,states[x.col])]);}
                }
                for x in &layout.effective{let(p,b)=(x.row-n0,states[x.col]);let f=cl.excited[p];let d=&dressings[p];
                    for kk in 0..self.carriers[x.carrier].rank{
                        node[x.at+kk]=(0..ne).filter(|&q|!flags[q]).fold(Complex::ZERO,|s,q|s-e.excited[(f,q)]*t[x.carrier][kk][(q,b)]+d[q].conj()*u[x.carrier][kk][(q,b)])*0.5;
                    }
                }
            }
            if let Some(f)=&layout.fourth{self.fourth_node(&e,cluster,f,&u,&t,&mut node)?;}
            out.push(node);
        }
        // The elimination's excited weight per ground state, every near beam at its peak (`resonance`).
        let resonance=(0..ng).map(|b|self.carriers.iter().flat_map(|c|c.beams.iter()).map(|&l|(0..ne).map(|i|e.responses[l][(i,b)].norm_sqr()).sum::<f64>()).sum::<f64>()).fold(0.,f64::max);
        let dressing=g.dressing_bounds(&e,&intensities)?;
        Ok((out,[dressing,resonance,g.ringing_spread(&e)]))
    }

    /// A cluster of ground states' fourth-order layout (`FourthLayout`), its tables from `size` on: H^(4)'s
    /// entries kept where the second order would keep a coupling at their beat Ω (ground.rs `fourth_keeps`);
    /// another one's mixing m = b/(|Ω| − l0) booked, b = S l0 the size of H^(4).
    fn fourth_layout(&self,k:usize,size:&mut usize)->FourthLayout{
        let g=&self.ground;let cl=&g.clusters[k];let(n,ng)=(cl.states.len(),g.ground_states.len());let unmixed=|c:usize|g.unmixed.get(c).is_some_and(|x|x[k]);let r=|c:usize|self.carriers[c].rank;let delta=|c:usize|g.carriers[c].detuning;
        let(on,b)=g.fourth_carriers(cl);let l0=g.generator_scale;
        let fields:Vec<(usize,usize)>=on.iter().flat_map(|&c|(0..r(c)).map(move|k|(c,k))).collect();
        let pairs:Vec<(usize,usize)>=(0..fields.len()).flat_map(|i|(i..fields.len()).map(move|j|(i,j))).collect();
        let round_off=g.residual_round_off(cl);
        let mut left:Vec<(usize,usize,f64)>=vec![];
        let mut keep=|nu:f64|->Vec<bool>{(0..n*n).map(|i|{
            let(x,y)=(i/n,i%n);let omega=cl.frames[x]-cl.frames[y]+nu;
            if g.fourth_keeps(cl,x,y,omega,b){return true;}
            let(a,c)=(x.min(y),x.max(y));if !left.iter().any(|&(p,q,w)|p==a&&q==c&&(w-omega.abs()).abs()<=round_off){left.push((a,c,omega.abs()));}
            false}).collect()};
        let mut quartic=vec![];
        for &(p1,p2)in &pairs{for &(q1,q2)in &pairs{
            let nu=g.frequency(&[fields[p1].0,fields[p2].0],&[fields[q1].0,fields[q2].0]);let k=keep(nu);
            if k.iter().any(|&x|x){quartic.push(QuarticSlab{conj:(p1,p2),plain:(q1,q2),nu,keep:k,part:vec![],at:*size});*size+=n*n;}
        }}
        let dropped=left.iter().fold(0.,|s,&(_,_,w)|s+crate::ground::mixing(b,w,l0,f64::MIN_POSITIVE).powi(2));
        // The metric's and Π's slabs per carrier pair, the metric on the coherent entries (the readout's).
        let group=|c:usize,s:usize|cl.coherent.iter().position(|gr|gr.iter().any(|x|x.0.carrier==c&&x.0.state==s));
        let mut slow=vec![];
        for &c1 in &on{for &c2 in &on{
            let coherent=(0..n*n).map(|i|{let(x,y)=(cl.states[i/n],cl.states[i%n]);let gx=group(c1,x);gx.is_some()&&gx==group(c2,y)}).collect();
            let square=r(c1)*r(c2)*n*n;slow.push(SlowSlab{carriers:(c1,c2),nu:delta(c1)-delta(c2),coherent,rr:*size,yw:*size+square});*size+=2*square;
        }}
        // The jumps' W3 at each carrier's frequency, through the triples (c1, c2, c4) at it, each middle factor
        // less what the dressing holds (`Middle`).
        let(mut jumps,mut p)=(vec![],Vec::<(usize,usize,Middle,usize)>::new());
        for &c in &on{for &c1 in &on{
            let triples:Vec<(usize,usize)>=on.iter().flat_map(|&c2|on.iter().map(move|&c4|(c2,c4))).filter(|&(c2,c4)|crate::ground::same_frame(g.frequency(&[c1,c4],&[c2]),delta(c))).collect();
            if triples.is_empty(){continue;}
            let pairs=triples.into_iter().map(|(c2,c4)|{let kind=Middle::of(c1,c2,c4,unmixed(c1));
                p.iter().position(|x|x.0==c2&&x.1==c4&&x.2==kind).unwrap_or_else(||{p.push((c2,c4,kind,*size));*size+=r(c2)*r(c4)*ng*n;p.len()-1})}).collect();
            let slab=JumpSlab{photon:c,inner:c1,pairs,diagonal:unmixed(c),at:*size};*size+=r(c1)*3*ng*ng*n*slab.sources(n);jumps.push(slab);
        }}
        // Where two photon channels' jumps meet: the landing states their photons share a class in (ground.rs `Class`)
        // -- a coherent pair's every one, a folded cross's (`FoldedCross`) --, as the jumps make their products.
        let nc=g.carriers.len();let mut landing=vec![0u32;nc*n*nc*n];
        for class in &cl.classes{for &(x,_)in &class.photons{for &(y,_)in &class.photons{if x.to!=y.to{continue;}
            let(a,b)=(cl.position(x.from).unwrap(),cl.position(y.from).unwrap());landing[((x.carrier*n+a)*nc+y.carrier)*n+b]|=1<<x.to;}}}
        FourthLayout{carriers:on,fields,quartic,slow,jumps,p,landing,dropped}
    }
    /// A cluster of ground states' fourth-order tables at a node (`FourthLayout`), from the elimination e
    /// there and the carriers' folded bases u and responses t (ne × ng per field index). With G_b(ω) the
    /// fourth order's resolvents (ground.rs `fourth_column`: at a carrier's frequency its block's dressed eigen-energy
    /// terms, from each source place to every column), the fields f3, f1, f2, f4 (carriers c3, c1, c2, c4), ω = δ_c1 − δ_c2 + δ_c4, every ground
    /// state b′ between and the middle factor M_f2f4 = U_f2†T_f4 less what the dressing holds (`Middle`): the van
    /// Vleck form's U_f3[:,a]† G_b(ω) T_f1[:,b′] M_f2f4[b′,b] and −½[X_f3f1, U_f2†T_f4] (X = T†T on the cluster's
    /// coherent entries), its Hermitian half on the slab of the conjugated pair {f3, f2} and the plain {f1, f4}
    /// and the adjoint's on the swapped one; the jumps' fourth order of Σ L̃†L̃ -- −(C T_f3)[to,a]* (C G_b(ω) T_f1
    /// M_f2f4)[to,b] at a carrier's frequency ω and its adjoint, −½{X, T†C†C T} --, over the landing states the two
    /// photons share a class in (`FourthLayout::landing`), times −i/2, on the same slabs; R_c1†R_c2 on the
    /// coherent entries and YW_c1c2 = −U_c1[:,a]† G_a(δ_c1) T_c2[:,b]; and the jumps' tables.
    fn fourth_node(&self,e:&ground::Elimination,k:usize,f:&FourthLayout,u:&[Vec<Dense>],t:&[Vec<Dense>],node:&mut [Complex])->Result<(),String>{
        let g=&self.ground;let cl=&g.clusters[k];let(n,ng,ne,nc)=(cl.states.len(),g.ground_states.len(),g.excited_states.len(),g.carriers.len());let st=&cl.states;let fl=&f.fields;let nf=fl.len();
        let delta=|c:usize|g.carriers[c].detuning;let field=|c:usize,k:usize|fl.iter().position(|&x|x==(c,k)).unwrap();
        let photon=|w:f64|f.carriers.iter().any(|&c|crate::ground::same_frame(delta(c),w));
        // The triples' frequencies, each once, and G_b(ω) T_f[:,b′] for the cluster's b and every b′.
        let mut omegas:Vec<f64>=vec![];
        for &c1 in &f.carriers{for &c2 in &f.carriers{for &c4 in &f.carriers{let w=g.frequency(&[c1,c4],&[c2]);if !omegas.iter().any(|&x|crate::ground::same_frame(x,w)){omegas.push(w);}}}}
        let omega_of=|w:f64|omegas.iter().position(|&x|crate::ground::same_frame(x,w)).unwrap();
        // G(ω) on T_f[:,b′] from each source place b to every place x (ground.rs `fourth_column`: at a photon's carrier its
        // block's eigen-energy terms, P_E[b,x]; else the bare G_b(ω) on x = b alone).
        let gt:Vec<Vec<Vec<Vec<Vec<Vec<Complex>>>>>>=omegas.iter().map(|&w|(0..n).map(|b|fl.iter().map(|&(c,kk)|(0..ng).map(|bp|
            g.fourth_column(e,k,b,w,&(0..ne).map(|i|t[c][kk][(i,bp)]).collect::<Vec<_>>(),false)).collect::<Result<Vec<_>,_>>()).collect::<Result<Vec<_>,_>>()).collect::<Result<Vec<_>,_>>())
            .collect::<Result<Vec<_>,_>>()?;// [ω][b][f][b′][x] → ne
        // The photons' amplitudes on every axis and landing state (axis-major, 3 ng): C T_f[:,s], and at a carrier's
        // frequency C G_b(ω) T_f[:,b′].
        let emit=|v:&[Complex]|->Vec<Complex>{(0..3*ng).map(|q|{let(ax,to)=(q/ng,q%ng);(0..ne).fold(Complex::ZERO,|s,i|s+g.emission[ax][(to,i)]*v[i])}).collect()};
        let ct:Vec<Vec<Vec<Complex>>>=fl.iter().map(|&(c,k)|(0..ng).map(|s|emit(&(0..ne).map(|i|t[c][k][(i,s)]).collect::<Vec<_>>())).collect()).collect();// [f][s]
        let cgt:Vec<Vec<Vec<Vec<Vec<Vec<Complex>>>>>>=omegas.iter().enumerate().map(|(w,&om)|if photon(om){
            (0..n).map(|b|(0..nf).map(|fi|(0..ng).map(|bp|(0..n).map(|x|emit(&gt[w][b][fi][bp][x])).collect()).collect()).collect()).collect()}else{vec![]}).collect();// [ω][b][f][b′][x]
        let full=|x:&Dense,y:&Dense|->Vec<Complex>{let mut m=vec![Complex::ZERO;ng*ng];for i in 0..ng{for j in 0..ng{m[i*ng+j]=(0..ne).fold(Complex::ZERO,|s,q|s+x[(q,i)].conj()*y[(q,j)]);}}m};
        let ut:Vec<Vec<Vec<Complex>>>=fl.iter().map(|&(c,k)|fl.iter().map(|&(c2,k2)|full(&u[c][k],&t[c2][k2])).collect()).collect();
        let tt:Vec<Vec<Vec<Complex>>>=fl.iter().map(|&(c,k)|fl.iter().map(|&(c2,k2)|full(&t[c][k],&t[c2][k2])).collect()).collect();
        // Two photon channels' product in the decay: over the landing states their photons share a class in, as the
        // jumps make it (`FourthLayout::landing`).
        let shared=|ca:usize,a:usize,cb:usize,b:usize|f.landing[((ca*n+a)*nc+cb)*n+b];
        let meet=|mask:u32,x:&[Complex],y:&[Complex]|->Complex{(0..3*ng).filter(|q|(mask>>(q%ng))&1==1).fold(Complex::ZERO,|s,q|s+x[q].conj()*y[q])};
        let tdt:Vec<Vec<Vec<Complex>>>=(0..nf).map(|i|(0..nf).map(|j|(0..n*n).map(|q|{let(a,b)=(q/n,q%n);
            meet(shared(fl[i].0,a,fl[j].0,b),&ct[i][st[a]],&ct[j][st[b]])}).collect()).collect()).collect();// [f][f′][a n + b]
        // X on the cluster: T_f†T_f′ on the coherent entries of the carriers' pair.
        let coherent=|c1:usize,c2:usize|&f.slow.iter().find(|s|s.carriers==(c1,c2)).unwrap().coherent;
        let xs:Vec<Vec<Vec<Complex>>>=(0..nf).map(|i|(0..nf).map(|j|{let mask=coherent(fl[i].0,fl[j].0);
            (0..n*n).map(|q|if mask[q]{tt[i][j][st[q/n]*ng+st[q%n]]}else{Complex::ZERO}).collect()}).collect()).collect();
        // The middle factor of a triple (c1, c2, c4) for the fields f2, f4 on column b, over b′ (`Middle`).
        let middle=|kind:Middle,f2:usize,f4:usize,b:usize|->Vec<Complex>{let s=st[b];let(c2,c4)=(fl[f2].0,fl[f4].0);
            let mut m:Vec<Complex>=(0..ng).map(|bp|ut[f2][f4][bp*ng+s]).collect();
            match kind{Middle::Whole=>{},
                Middle::Vias=>{for via in &g.vias[c4][s]{if via.carrier==c2{m[via.state]=Complex::ZERO;}}},
                // The field's Hermitian part of c2's light shift, (V_c2†R_c2)[b′,b], that c1's block holds, out: b's own (an unmixed
                // block's), or between the cluster's states; its anti-Hermitian part stays.
                Middle::Shift=>{m[s]=(ut[f2][f4][s*ng+s]-ut[f4][f2][s*ng+s].conj())*0.5;},
                Middle::Shifts=>{for &bp in st{m[bp]=(ut[f2][f4][bp*ng+s]-ut[f4][f2][s*ng+bp].conj())*0.5;}}}
            m};
        // A(ω)[f3][f1][((x n + a) n + b) ng + b′] = U_f3[:,a]† (G(ω) T_f1[:,b′] from place b)[x].
        let a_t:Vec<Vec<Vec<Vec<Complex>>>>=(0..omegas.len()).map(|w|(0..nf).map(|f3|{let(c3,k3)=fl[f3];(0..nf).map(|f1|{let mut m=vec![Complex::ZERO;n*n*n*ng];
            for x in 0..n{for a in 0..n{for b in 0..n{for bp in 0..ng{let v=&gt[w][b][f1][bp][x];m[((x*n+a)*n+b)*ng+bp]=(0..ne).fold(Complex::ZERO,|s,i|s+u[c3][k3][(i,st[a])].conj()*v[i]);}}}}
            m}).collect()}).collect()).collect();
        // The slabs by their pairs' indices (i ≤ j of nf fields).
        let pair=|i:usize,j:usize|{let(i,j)=(i.min(j),i.max(j));i*nf-i*(i+1)/2+j};let np=nf*(nf+1)/2;
        let mut index=vec![usize::MAX;np*np];for(s,q)in f.quartic.iter().enumerate(){index[pair(q.conj.0,q.conj.1)*np+pair(q.plain.0,q.plain.1)]=s;}
        let mut acc:Vec<Vec<Complex>>=vec![vec![Complex::ZERO;n*n];f.quartic.len()];
        let adjoint=|x:&[Complex]|->Vec<Complex>{(0..n*n).map(|q|x[(q%n)*n+q/n].conj()).collect()};
        let add=|acc:&mut Vec<Vec<Complex>>,conj:usize,plain:usize,x:&[Complex],z:Complex|{let s=index[conj*np+plain];if s!=usize::MAX{for q in 0..n*n{acc[s][q]+=x[q]*z;}}};
        let(mut x,mut l,mut m)=(vec![Complex::ZERO;n*n],vec![Complex::ZERO;n*n],vec![Complex::ZERO;n*n]);
        for &c1 in &f.carriers{for &c2 in &f.carriers{for &c4 in &f.carriers{
            let w=omega_of(g.frequency(&[c1,c4],&[c2]));let jumps=photon(omegas[w]);let kind=Middle::of(c1,c2,c4,g.unmixed.get(c1).is_some_and(|x|x[k]));
            // The photon's carrier at the triple's frequency (the jumps' W3^c).
            let carrier=f.carriers.iter().copied().find(|&c|crate::ground::same_frame(delta(c),omegas[w]));
            for f1 in (0..nf).filter(|&i|fl[i].0==c1){for f2 in (0..nf).filter(|&i|fl[i].0==c2){for f4 in (0..nf).filter(|&i|fl[i].0==c4){
                let p24=&ut[f2][f4];let middles:Vec<Vec<Complex>>=(0..n).map(|b|middle(kind,f2,f4,b)).collect();
                // C G(ω) T_f1 M on column x: this triple's W3 photons, from every source place b.
                let photons:Vec<Vec<Complex>>=if jumps{(0..n).map(|x|{let mut v=vec![Complex::ZERO;3*ng];
                    for b in 0..n{for(bp,&z)in middles[b].iter().enumerate(){if z==Complex::ZERO{continue;}for(o,&y)in v.iter_mut().zip(&cgt[w][b][f1][bp][x]){*o+=y*z;}}}v}).collect()}else{vec![]};
                for f3 in 0..nf{
                    let(conj,plain)=(pair(f3,f2),pair(f1,f4));
                    if index[conj*np+plain]==usize::MAX&&index[plain*np+conj]==usize::MAX{continue;}
                    for a in 0..n{for b in 0..n{
                        // Entry (a, b): U_f3[:,a]† W3[:,b], W3's column b from every source place s.
                        let mut z=Complex::ZERO;
                        for(src,mid)in middles.iter().enumerate(){let at=&a_t[w][f3][f1][((b*n+a)*n+src)*ng..][..ng];for(bp,&y)in mid.iter().enumerate(){z+=at[bp]*y;}}
                        let(mut zx,mut zl)=(Complex::ZERO,Complex::ZERO);
                        for q in 0..n{
                            z-=(xs[f3][f1][a*n+q]*p24[st[q]*ng+st[b]]-p24[st[a]*ng+st[q]]*xs[f3][f1][q*n+b])*0.5;
                            zx+=xs[f3][f1][a*n+q]*tdt[f2][f4][q*n+b]+tdt[f3][f1][a*n+q]*xs[f2][f4][q*n+b];
                        }
                        if let Some(c)=carrier.filter(|_|jumps){zl=meet(shared(fl[f3].0,a,c,b),&ct[f3][st[a]],&photons[b]);}
                        x[a*n+b]=z;m[a*n+b]=zx;l[a*n+b]=zl;
                    }}
                    // The Hermitian half: ½X here, ½X† on the swapped slab; −(i/2) of Σ L̃†L̃'s −½{X, T†DT} here.
                    add(&mut acc,conj,plain,&x,Complex::new(0.5,0.));add(&mut acc,plain,conj,&adjoint(&x),Complex::new(0.5,0.));
                    add(&mut acc,conj,plain,&m,Complex::new(0.,0.25));
                    // −(i/2) of Σ L̃†L̃'s −R†D W3 here and −W3†D R on the swapped slab.
                    if jumps{add(&mut acc,conj,plain,&l,Complex::new(0.,0.5));add(&mut acc,plain,conj,&adjoint(&l),Complex::new(0.,0.5));}
                }
            }}}
        }}}
        for(q,v)in f.quartic.iter().zip(&acc){for i in 0..n*n{if q.keep[i]{node[q.at+i]=v[i];}}}
        // The metric's and Π's slabs: YW_c1c2[a,x] = −Σ_b′ Σ_E P_E[a,b′] U_c1[:,b′]† G_b′(z_E) T_c2[:,x], c1's block's terms
        // (ground.rs `fourth_column`, its rows by the projector's transpose).
        for sl in &f.slow{let(c1,c2)=sl.carriers;let r2=self.carriers[c2].rank;
            for k1 in 0..self.carriers[c1].rank{for k2 in 0..r2{let(f1,f2)=(field(c1,k1),field(c2,k2));let base=(k1*r2+k2)*n*n;
                for a in 0..n{for b in 0..n{if sl.coherent[a*n+b]{node[sl.rr+base+a*n+b]=tt[f1][f2][st[a]*ng+st[b]];}}}
                for x in 0..n{let col:Vec<Complex>=(0..ne).map(|i|t[c2][k2][(i,st[x])]).collect();
                    for p in 0..n{for(a,img)in g.fourth_column(e,k,p,delta(c1),&col,true)?.iter().enumerate(){
                        node[sl.yw+base+a*n+x]-=(0..ne).fold(Complex::ZERO,|s,i|s+u[c1][k1][(i,st[p])].conj()*img[i]);}}}
            }}
        }
        // The jumps' tables: C G(δ_c) T_f[:,b′] from each source place b to each column y (a diagonal slab's y from y alone), and the
        // middle factors.
        for js in &f.jumps{let w=omega_of(delta(js.photon));let s=js.sources(n);
            for kk in 0..self.carriers[js.inner].rank{let fi=field(js.inner,kk);for ax in 0..3{for row in 0..ng{for bp in 0..ng{for y in 0..n{for b in 0..s{
                node[js.at+((((kk*3+ax)*ng+row)*ng+bp)*n+y)*s+b]=cgt[w][if js.diagonal{y}else{b}][fi][bp][y][ax*ng+row];
            }}}}}}
        }
        for &(c2,c4,kind,at)in &f.p{let r4=self.carriers[c4].rank;for k2 in 0..self.carriers[c2].rank{for k4 in 0..r4{
            let(f2,f4)=(field(c2,k2),field(c4,k4));let base=at+(k2*r4+k4)*ng*n;
            for b in 0..n{let m=middle(kind,f2,f4,b);for bp in 0..ng{node[base+bp*n+b]=m[bp];}}
        }}}
        Ok(())
    }
    /// Node i's intensities on a grid of n nodes per focused unit (node order: the first unit's
    /// fastest).
    fn grid_point(&self,n:usize,i:usize)->[f64;MOST_FOCUSED]{
        std::array::from_fn(|f|if f<self.focused.len(){((i/n.pow(f as u32))%n) as f64/(n-1).max(1) as f64}else{0.})
    }
    /// The tables of a grid of n nodes per focused unit: the nodes it shares with the grid of
    /// (n + 1)/2 before it (`coarse`: every other node along each axis, at the same intensities to the
    /// last bit) taken from that grid's tables, the others computed, `NODE_BATCH` at a time written
    /// into the tables; and the elimination's bound over the nodes inside the reach's intensities --
    /// each focused unit's from its smallest over the reach (less one node, the interval holding it)
    /// to 1.
    fn grid_tables(&self,n:usize,mixing:&[Vec<Vec<Complex>>],coarse:Option<&Tables>)->Result<Tables,String>{
        let(d,m)=(self.focused.len(),n.div_ceil(2));let count=n.pow(d as u32);
        let mut data:Vec<Vec<f64>>=self.layouts.iter().map(|l|Vec::with_capacity(2*count*l.size)).collect();let mut bounds=Vec::with_capacity(count);
        // A node of the coarse grid it shares, or computed (with its counted work: a node is one thread's).
        enum Made{Shared(usize),Computed(Node,f64)}let mut work=0.;
        for first in (0..count).step_by(NODE_BATCH){
            let batch:Vec<Made>=(first..(first+NODE_BATCH).min(count)).into_par_iter().map(|i|{
                let digits:Vec<usize>=(0..d).map(|f|(i/n.pow(f as u32))%n).collect();
                match coarse{
                    Some(_)if digits.iter().all(|x|x%2==0)=>Ok(Made::Shared(digits.iter().enumerate().map(|(f,x)|x/2*m.pow(f as u32)).sum())),
                    _=>{let before=crate::linear::counted();self.node(&self.grid_point(n,i)[..d],mixing).map(|x|Made::Computed(x,crate::linear::counted()-before))}
                }
            }).collect::<Result<_,_>>()?;
            for made in batch{match made{
                Made::Shared(j)=>{let c=coarse.unwrap();for(k,l)in self.layouts.iter().enumerate(){data[k].extend_from_slice(&c.data[k][2*j*l.size..2*(j+1)*l.size]);}bounds.push(c.bounds[j]);}
                Made::Computed(node,w)=>{for(k,t)in node.0.iter().enumerate(){data[k].extend(t.iter().flat_map(|z|[z.re,z.im]));}bounds.push(node.1);work+=w;}
            }}
        }
        let inside=self.inside(n);
        let reach=std::array::from_fn(|k|inside.iter().map(|&i|bounds[i][k]).fold(0.,f64::max));
        Ok(Tables{nodes:n,data,bounds,reach,work})
    }
    /// The nodes of a grid of n per focused unit inside the reach's intensities: each focused unit's from its
    /// smallest over the reach (less one node, the interval holding it) to 1.
    fn inside(&self,n:usize)->Vec<usize>{
        let h=1./(n-1).max(1) as f64;
        let low:Vec<f64>=(0..self.axes.len()).map(|f|self.reach.iter().map(|&r|self.axis_value(f,r).min(1.)).fold(1.,f64::min)-h).collect();
        // A field of rank one takes every intensity from the reach's least to its peak, which its atoms pass on their way out of the trap;
        // a field of higher rank its coordinates' range over the reach.
        let high:Vec<f64>=(0..self.axes.len()).map(|f|if self.ground.units[self.axes[f].unit].rank==1{f64::INFINITY}else{self.reach.iter().map(|&r|self.axis_value(f,r)).fold(0.,f64::max)+h}).collect();
        (0..n.pow(self.focused.len() as u32)).filter(|&i|self.grid_point(n,i).iter().zip(low.iter().zip(&high)).all(|(x,(l,h))|x>=l&&x<=h)).collect()
    }
    /// The coarse grid's (`self.tables`) largest interpolation error at a finer grid's nodes where the atoms go,
    /// relative to the scale of each cluster's table kind there: the nodes inside the reach's intensities (`inside`),
    /// as the elimination's bounds are taken (`grid_tables`). Beyond them -- a Gaussian beam's edge, where its
    /// intensity falls to 0 and the function bends its sharpest (the 3D PGC's error 2e-4 at 257 nodes there, 8e-10
    /// inside; F9's 7.5e-4 at 513 nodes per axis, 1.4e-12 inside) -- an atom is far beyond its thermal
    /// ellipsoid, and the estimates over the reach do not take it either.
    fn interpolation_error(&self,fine:&Tables)->f64{
        let(ng,ne)=(self.ground.ground_states.len(),self.ground.excited_states.len());let d=self.focused.len();
        let inside=self.inside(fine.nodes);let mut worst=0_f64;
        for(b,layout)in self.layouts.iter().enumerate(){
            let n=layout.n;let mut kinds:Vec<(usize,usize)>=vec![(layout.h0,layout.h0+n*n)];
            // The first order's tables stay out: their error enters the generator times the small w/σ, and J2's, measured, bounds T2's.
            for gl in &layout.groups{let square=gl.pairs.len()*gl.entries.len();kinds.extend([(gl.s,gl.s+square),(gl.q,gl.q+square)]);}
            for(c,cl)in layout.carriers.iter().enumerate(){let r=self.carriers[c].rank;let(jump,response)=(3*r*ng*layout.n0,r*ne*layout.n0);
                kinds.extend([(cl.j,cl.j+jump),(cl.j2,cl.j2+jump),(cl.t,cl.t+response)]);}
            // The explicit states' tables, each kind one block (its entries' common scale: a decay
            // between two states with orthogonal photons is zero to rounding).
            let block=|slots:Vec<(usize,usize)>|slots.iter().map(|x|x.0).min().zip(slots.iter().map(|x|x.0+x.1).max());
            kinds.extend(block(layout.crosses.iter().map(|x|(x.at,self.carriers[x.carrier].rank)).collect()));
            kinds.extend(block(layout.decays.iter().map(|x|(x.at,1)).collect()));
            kinds.extend(block(layout.effective.iter().map(|x|(x.at,self.carriers[x.carrier].rank)).collect()));
            if layout.n>layout.n0{kinds.push((layout.emission,layout.emission+3*ng*(layout.n-layout.n0)));}
            if let Some(f)=&layout.fourth{
                let r=|c:usize|self.carriers[c].rank;let sq=n*n;
                // H^(4)'s slabs one kind (one scale), the metric's and Π's per pair, the jumps' tables.
                if let(Some(a),Some(b))=(f.quartic.first(),f.quartic.last()){kinds.push((a.at,b.at+sq));}
                for sl in &f.slow{let square=r(sl.carriers.0)*r(sl.carriers.1)*sq;kinds.push((sl.rr,sl.rr+square));kinds.push((sl.yw,sl.yw+square));}
                for js in &f.jumps{kinds.push((js.at,js.at+r(js.inner)*3*ng*ng*n*js.sources(n)));}
                for &(c2,c4,_,at)in &f.p{kinds.push((at,at+r(c2)*r(c4)*ng*n));}
            }
            // Per kind its scale and the largest error, the fine grid's nodes on the threads.
            for(start,end)in kinds{
                let exact=|i:usize|&fine.data[b][2*(i*layout.size+start)..2*(i*layout.size+end)];
                let modulus=|x:&[f64]|x.chunks_exact(2).map(|z|z[0].hypot(z[1])).fold(0_f64,f64::max);
                let scale=inside.par_iter().map(|&i|modulus(exact(i))).reduce(||0.,f64::max);
                if scale==0.{continue;}
                let e=inside.par_iter().map_init(||vec![Complex::ZERO;end-start],|out,&i|{
                    self.interpolate(b,&self.stencil(&self.grid_point(fine.nodes,i)[..d]),None,start,out);
                    out.iter().zip(exact(i).chunks_exact(2)).fold(0_f64,|m,(x,y)|m.max((x.re-y[0]).hypot(x.im-y[1])))
                }).reduce(||0.,f64::max)/scale;
                worst=worst.max(e);
            }
        }
        worst
    }
    /// The groups' entries that hold nothing out of the clusters and the layouts: those whose term, its fields at their peak times its S or Q over the first grid, stays within the round-off of the
    /// cluster's largest. What the dressed responses leave between channels a selection rule keeps apart is no term, whatever its beat: 1e-23 rad/us on a resonant probe's rung of 3+10 against its 8, 24 of its
    /// 36 groups, all computed in every generator call, as is the decay's between folded photons a class joins (`Ground`'s rule keeps a pair by its undressed size; `Model::lindblad` holds a cluster's decay to its jumps at this round-off).
    fn prune(&mut self,mixing:&[Vec<Vec<Complex>>])->Result<(),String>{
        let first=self.grid_tables(FIRST_NODES,mixing,None)?;let nodes=first.nodes.pow(self.focused.len() as u32);
        let peak:Vec<[f64;MOST_RANK]>=self.carriers.iter().map(|c|std::array::from_fn(|k|c.coefficients.iter().map(|x|x[k].abs()).sum())).collect();
        for(k,layout)in self.layouts.iter().enumerate(){
            let modulus=|node:usize,x:usize|{let i=2*(node*layout.size+x);first.data[k][i].hypot(first.data[k][i+1])};
            let rates:Vec<Vec<f64>>=layout.groups.iter().map(|gl|{let count=gl.entries.len();(0..count).map(|j|gl.pairs.iter().enumerate().map(|(p,&(k1,k0))|{
                let largest=(0..nodes).map(|node|modulus(node,gl.s+p*count+j).max(modulus(node,gl.q+p*count+j))).fold(0.,f64::max);largest*peak[gl.emitted][k1]*peak[gl.absorbed][k0]}).fold(0.,f64::max)).collect()}).collect();
            let floor=unit_round_off(rates.iter().flatten().copied().fold(0.,f64::max));
            let cl=&mut self.ground.clusters[k];
            for(gl,rate)in layout.groups.iter().zip(&rates){for(&(x,y),_)in gl.entries.iter().zip(rate).filter(|(_,&r)|r<=floor){
                if gl.decay{cl.folded_crosses.retain(|c|!(c.row==x&&c.col==y&&c.absorbed==gl.absorbed&&c.emitted==gl.emitted&&c.residual==gl.residual));}
                else if let Some(g)=cl.groups.iter_mut().find(|g|g.absorbed==gl.absorbed&&g.emitted==gl.emitted){g.entries.retain(|e|!(e.0==x&&e.1==y&&e.2==gl.residual));}
            }}
            cl.groups.retain(|g|!g.entries.is_empty());
        }
        self.layouts=self.layouts(mixing);self.fourth_dropped=self.layouts.iter().filter_map(|l|l.fourth.as_ref()).map(|f|f.dropped).sum();self.effective_dropped=self.layouts.iter().map(|l|l.effective_dropped).sum();
        Ok(())
    }
    /// The tables on grids of FIRST_NODES, then doubled (each grid's nodes computed once), the finer grid
    /// kept with the coarser one's measured error against it (`interpolation_error`, an upper estimate of
    /// its own) -- once refined at least, so that the error is measured --, and doubled again while that
    /// pays (cost.rs `pays`): while the next grid's counted work -- its new nodes' eliminations at the last grid's counted
    /// work per node (linear.rs `counted`) and its error's measure, `COMPLEX_MULTIPLY_ADD_NS` each -- is
    /// below what its smaller error saves the run (`stake`'s run at its `room_cost`), with no uncertainty: no pilot has run, no need's error stands yet. Its error
    /// is predicted by the last doubling's rate, and by the cubic interpolation's order before one is seen,
    /// 2⁻³ a doubling (`cubic`), and what it saves is `room_cost`'s: an error the whole population target leaves no run its room, whatever the run's length (a request of no
    /// duration, a step's stake: 5 % at 9 nodes on the tweezer PGC). A doubling that did not shrink the error says nothing of the next: a grid short of a line across the intensity axis reads about one
    /// until it resolves it, the cubic's order the prediction there. `STALLED_DOUBLINGS` above its smallest stop it (`table_stalled`: the tabulated function is not smooth on the grids' scale where the atoms go).
    /// A grid beyond the work bounds (`MOST_TABLE_BYTES`, `MOST_TABLE_NODES`) is not made
    /// whatever it saves (`table_limit`). Its error is booked (the ledger's), the room it takes from the
    /// statistics like any estimate's (solve.rs).
    fn tabulate(&mut self,mixing:&[Vec<Vec<Complex>>],most_bytes:usize,most_nodes:usize,stake:f64)->Result<(),String>{
        let d=self.focused.len() as u32;
        let entries:usize=self.layouts.iter().map(|l|l.size).sum();
        // The first order's tables are not measured (`interpolation_error`).
        let measured:usize=self.layouts.iter().map(|l|l.size-l.groups.iter().map(|g|2*g.first.len()*g.entries.len()).sum::<usize>()).sum();
        let within=|n:usize,coarse:usize|n.pow(d)<=most_nodes&&16*entries*(n.pow(d)+coarse.pow(d))<=most_bytes;
        if !within(2*FIRST_NODES-1,FIRST_NODES){return Err(format!("The elimination's tables take {entries} complex entries per node: beyond the work bounds"));}
        self.tables=self.grid_tables(FIRST_NODES,mixing,None)?;self.dressing=self.tables.reach[0];
        // The counted work per computed node, and the errors measured so far.
        let(mut per_node,mut errors)=(self.tables.work/FIRST_NODES.pow(d) as f64,vec![]);
        loop{
            let(coarse,n)=(self.tables.nodes,2*self.tables.nodes-1);
            if let Some(&error)=errors.last(){
                let smallest=errors.iter().copied().fold(f64::INFINITY,f64::min);
                if errors.iter().rev().take_while(|&&e|e>smallest).count()>=STALLED_DOUBLINGS{self.table_stalled=true;self.table_limit=Some(coarse);break;}
                let rate:f64=if errors.len()>1&&error<errors[errors.len()-2]{error/errors[errors.len()-2]}else{0.125};
                let work=COMPLEX_MULTIPLY_ADD_NS*((n.pow(d)-coarse.pow(d)) as f64*per_node+(n.pow(d)*measured*4_usize.pow(d)) as f64);
                if room_cost(error).is_finite()&&!pays(work,stake*(room_cost(error)-room_cost(rate*error)),0.){break;}
            }
            if !within(n,coarse){self.table_limit=Some(coarse);break;}
            let fine=self.grid_tables(n,mixing,Some(&self.tables))?;let worst=self.interpolation_error(&fine);
            per_node=fine.work/(n.pow(d)-coarse.pow(d)) as f64;
            self.dressing=self.dressing.max(fine.reach[0]);
            self.tables=fine;self.table_error=worst;errors.push(worst);
            if worst==0.{break;}
        }
        self.resonance=self.tables.reach[1];self.ringing_spread=self.tables.reach[2];
        Ok(())
    }

    /// The first order's tables from W^S and W^Q on the settled grid (module documentation, # The photons and the Doppler term): per group and
    /// node λ^S and λ^Q, their least squares on Q over the group's pairs and entries, into their places; and what they leave of each pair
    /// and entry where it counts, W − λ Q, after the slice's others. An entry counts by the friction its error makes: E^d_k′* E^D_k ΔW^Q, times the kick ħk, is
    /// a force of k (kv) (|E^d_k′ E^c_k| |ΔW|) -- ΔW^S likewise, through the fields' gradient -- whose power, times v, against the recoil heating, ħk²/m per photon of the
    /// largest rate |E^d E^c Q|, is (2 k_B T/ħ) |E^d_k′ E^c_k ΔW| / |E^d E^c Q| of T relative, at most mv² = k_BT per axis and the beams' peak fields: an entry beyond `FIRST_SHARE`
    /// is kept, the largest left out `first_error`. A group whose W itself is within the share (the far light on a cluster, a Raman pair's beat of no weight) takes no first order at all. λ is W/Q: where Q falls
    /// faster than W, at the tune-out of a group's whole light between two poles, it has a pole of its own that the tables' grid, fine for Q and W, need not resolve: measured on the grid of every other node (its
    /// interpolation's error there, by that size of friction, an upper estimate of its own), a group whose λ does not hold takes its W whole, all kept that counts.
    fn transfer_tables(&mut self,temperature_uk:f64){
        let grid=self.tables.nodes;let n=grid.pow(self.focused.len() as u32);let inside=self.inside(grid);let per_uk=2.*KT_PER_UK;
        let peak:Vec<[f64;MOST_RANK]>=self.carriers.iter().map(|c|std::array::from_fn(|k|c.coefficients.iter().map(|x|x[k].abs()).sum())).collect();
        let mut first_error=0_f64;
        for(cluster,layout)in self.layouts.iter_mut().enumerate(){
            let data=std::mem::take(&mut self.tables.data[cluster]);let(size,base)=(2*layout.size,layout.base);
            let at=|node:usize,x:usize|Complex::new(data[node*size+2*x],data[node*size+2*x+1]);
            // A pair's Q at a node, entry by entry: its own or, in a Hermitian group, the adjoint of the pair it stands with.
            let q_of=|gl:&GroupLayout,node:usize,(k1,k):(usize,usize),j:usize|->Complex{let count=gl.entries.len();
                match gl.pairs.iter().position(|&x|x==(k1,k)){Some(p)=>at(node,gl.q+p*count+j),None=>{let p=gl.pairs.iter().position(|&x|x==(k,k1)).unwrap();let(a,b)=gl.entries[j];
                    gl.entries.binary_search(&(b,a)).map_or(Complex::ZERO,|m|at(node,gl.q+p*count+m).conj())}}};
            let fields=|gl:&GroupLayout,(k1,k):(usize,usize)|peak[gl.emitted][k1]*peak[gl.absorbed][k];
            // Per group and node λ^S and λ^Q.
            let mut lambdas:Vec<Vec<[Complex;2]>>=layout.groups.iter().map(|gl|(0..n).into_par_iter().map(|node|{let count=gl.entries.len();
                let(mut ws,mut wq,mut qq)=(Complex::ZERO,Complex::ZERO,0.);
                for(pi,&pair)in gl.first.iter().enumerate(){for j in 0..count{if gl.entries[j].0>=layout.n0{continue;}
                    let q=q_of(gl,node,pair,j);let place=gl.w+2*pi*count+j;ws+=q.conj()*at(node,place);wq+=q.conj()*at(node,place+count);qq+=q.norm_sqr();}}
                if qq>0.{[ws*(1./qq),wq*(1./qq)]}else{[Complex::ZERO;2]}}).collect()).collect();
            // The largest rate where the atoms go.
            let scale=layout.groups.iter().map(|gl|gl.first.iter().map(|&pair|(0..gl.entries.len()).map(|j|inside.iter().map(|&node|q_of(gl,node,pair,j).abs()).fold(0.,f64::max)).fold(0.,f64::max)*fields(gl,pair)).fold(0.,f64::max)).fold(0.,f64::max);
            // λ against its interpolation from every other node (a node's weights by the coarser grid's `cubic`): at the finer nodes among them inside the reach, by its size of friction.
            let(m,axes)=((grid+1)/2,self.focused.len());
            for(g,gl)in layout.groups.iter().enumerate(){
                let holds=inside.iter().all(|&node|{
                    let digits:Vec<usize>=(0..axes).map(|f|(node/grid.pow(f as u32))%grid).collect();
                    if digits.iter().all(|x|x%2==0){return true;}
                    let stencil:Vec<(usize,[f64;4])>=digits.iter().map(|&x|{let(first,w,_)=cubic(m,x as f64/(grid-1) as f64);(first,w)}).collect();
                    let mut held=[Complex::ZERO;2];
                    for i in 0..1usize<<(2*axes){
                        let(mut at_node,mut weight)=(0,1.);for f in 0..axes{let j=(i>>(2*f))&3;at_node+=2*(stencil[f].0+j)*grid.pow(f as u32);weight*=stencil[f].1[j];}
                        for x in 0..2{held[x]+=lambdas[g][at_node][x]*weight;}
                    }
                    let error=(held[0]-lambdas[g][node][0]).abs().max((held[1]-lambdas[g][node][1]).abs());
                    let size=gl.first.iter().map(|&pair|fields(gl,pair)*(0..gl.entries.len()).map(|j|q_of(gl,node,pair,j).abs()).fold(0.,f64::max)).fold(0.,f64::max);
                    temperature_uk*per_uk*size*error<=FIRST_SHARE*scale
                });
                if !holds{lambdas[g].fill([Complex::ZERO;2]);}
            }
            // Per group what it leaves, or none of it.
            let mut rest=base;
            for g in 0..layout.groups.len(){
                let gl=&layout.groups[g];let count=gl.entries.len();let mut kept=vec![];
                let whole=gl.first.iter().enumerate().map(|(pi,&pair)|per_uk*fields(gl,pair)*(0..2*count).map(|j|inside.iter().map(|&node|at(node,gl.w+2*pi*count+j).abs()).fold(0.,f64::max)).fold(0.,f64::max)).fold(0.,f64::max);
                if temperature_uk*whole<=FIRST_SHARE*scale{
                    if scale>0.{first_error=first_error.max(whole/scale);}
                    layout.groups[g].first.clear();lambdas[g].fill([Complex::ZERO;2]);continue;
                }
                for(pi,&pair)in gl.first.iter().enumerate(){
                    let left=|j:usize,node:usize|{let q=q_of(gl,node,pair,j);let place=gl.w+2*pi*count+j;let[ls,lq]=lambdas[g][node];
                        (at(node,place)-ls*q).abs().max((at(node,place+count)-lq*q).abs())};
                    let sizes:Vec<f64>=(0..count).map(|j|if gl.entries[j].0>=layout.n0{0.}else{per_uk*fields(gl,pair)*inside.iter().map(|&node|left(j,node)).fold(0.,f64::max)}).collect();
                    let entries:Vec<(usize,usize)>=(0..count).filter(|&j|temperature_uk*sizes[j]>FIRST_SHARE*scale).map(|j|gl.entries[j]).collect();
                    for j in 0..count{if temperature_uk*sizes[j]<=FIRST_SHARE*scale&&scale>0.{first_error=first_error.max(sizes[j]/scale);}}
                    if !entries.is_empty(){let k=entries.len();kept.push(Kept{pair,entries,at:rest});rest+=2*k;}
                }
                layout.groups[g].kept=kept;
            }
            // The slice without W, with what is kept.
            let mut out=Vec::with_capacity(2*n*rest);
            for node in 0..n{
                let start=out.len();out.extend_from_slice(&data[node*size..node*size+2*base]);
                for(g,gl)in layout.groups.iter().enumerate(){for(i,z)in lambdas[g][node].iter().enumerate(){out[start+2*(gl.transfer+i)]=z.re;out[start+2*(gl.transfer+i)+1]=z.im;}}
                for(g,gl)in layout.groups.iter().enumerate(){let count=gl.entries.len();let[ls,lq]=lambdas[g][node];
                    for k in &gl.kept{let pi=gl.first.iter().position(|&x|x==k.pair).unwrap();
                        for(lambda,table)in [(ls,0),(lq,count)]{for e in &k.entries{let j=gl.entries.binary_search(e).unwrap();
                            let z=at(node,gl.w+2*pi*count+table+j)-lambda*q_of(gl,node,k.pair,j);out.push(z.re);out.push(z.im);}}}}
            }
            layout.size=rest;self.tables.data[cluster]=out;
        }
        self.first_error=first_error;
    }

    /// The stencil at intensities s (one per focused beam).
    pub fn stencil(&self,s:&[f64])->Stencil{
        let n=self.tables.nodes;let mut out=Stencil{dimensions:self.focused.len(),first:[0;MOST_FOCUSED],weights:[[1.,0.,0.,0.];MOST_FOCUSED],derivatives:[[0.;4];MOST_FOCUSED],stride:[0;MOST_FOCUSED]};
        for f in 0..out.dimensions{let(first,w,d)=cubic(n,s[f]);out.first[f]=first;out.weights[f]=w;out.derivatives[f]=d;out.stride[f]=n.pow(f as u32);}
        out
    }

    /// out = the table slice [start, start + out.len()) of a cluster, interpolated (with the weights'
    /// derivative in s_f when `derivative` names one).
    pub fn interpolate(&self,cluster:usize,stencil:&Stencil,derivative:Option<usize>,start:usize,out:&mut [Complex]){
        let size=2*self.layouts[cluster].size;let data=&self.tables.data[cluster];let len=out.len();
        if stencil.dimensions==1&&derivative.is_none(){
            let at=stencil.first[0]*size+2*start;let nodes:[&[f64];4]=std::array::from_fn(|m|&data[at+m*size..at+m*size+2*len]);
            if self.wide{unsafe{four_nodes_wide(stencil.weights[0],nodes,out);}}else{four_nodes(stencil.weights[0],nodes,out);}
            return;
        }
        out.fill(Complex::ZERO);
        for i in 0..stencil.count(){
            let(node,w,dw)=stencil.node(i);let w=match derivative{None=>w,Some(f)=>dw[f]};let base=node*size+2*start;
            for(o,x)in out.iter_mut().zip(data[base..base+2*len].chunks_exact(2)){o.re+=w*x[0];o.im+=w*x[1];}
        }
    }
    /// The value and the derivatives in every s_f of a slice of `len` values (2 × its entries), in
    /// one pass over the stencil's nodes; the one-beam stencil unrolled.
    #[inline(always)]
    fn gather<const W:usize>(&self,cluster:usize,stencil:&Stencil,start:usize,len:usize,value:&mut [f64;W],derivatives:&mut [[f64;W];MOST_FOCUSED]){
        let size=2*self.layouts[cluster].size;let data=&self.tables.data[cluster];let base=2*start;let d=stencil.dimensions;
        if d==1{
            let(w,dw)=(stencil.weights[0],stencil.derivatives[0]);let at=stencil.first[0]*size+base;
            let(a,b,c,e)=(&data[at..at+len],&data[at+size..at+size+len],&data[at+2*size..at+2*size+len],&data[at+3*size..at+3*size+len]);
            for j in 0..len{value[j]=w[0]*a[j]+w[1]*b[j]+w[2]*c[j]+w[3]*e[j];derivatives[0][j]=dw[0]*a[j]+dw[1]*b[j]+dw[2]*c[j]+dw[3]*e[j];}
            return;
        }
        value[..len].fill(0.);for g in 0..d{derivatives[g][..len].fill(0.);}
        for i in 0..stencil.count(){
            let(node,w,dw)=stencil.node(i);let at=node*size+base;let x=&data[at..at+len];
            for j in 0..len{value[j]+=w*x[j];}for g in 0..d{for j in 0..len{derivatives[g][j]+=dw[g]*x[j];}}
        }
    }
    /// The value alone (the stencil with the trap beam off).
    #[inline(always)]
    fn gather_value(&self,cluster:usize,stencil:&Stencil,start:usize,len:usize,value:&mut [f64]){
        let size=2*self.layouts[cluster].size;let data=&self.tables.data[cluster];let base=2*start;
        if stencil.dimensions==1{
            let w=stencil.weights[0];let at=stencil.first[0]*size+base;
            let(a,b,c,e)=(&data[at..at+len],&data[at+size..at+size+len],&data[at+2*size..at+2*size+len],&data[at+3*size..at+3*size+len]);
            for j in 0..len{value[j]=w[0]*a[j]+w[1]*b[j]+w[2]*c[j]+w[3]*e[j];}
            return;
        }
        value[..len].fill(0.);
        for i in 0..stencil.count(){let(node,w,_)=stencil.node(i);if w==0.{continue;}let at=node*size+base;for(o,x)in value[..len].iter_mut().zip(&data[at..at+len]){*o+=w*x;}}
    }

    /// Everything at (r, κ) but the internal state: per focused unit its intensity s = |F|²/N
    /// (F = Σ_l a_l γ_l, ground.rs `Unit`), its gradient 2 Re(F* ∇F)/N and its intensity with the
    /// trap's light off (`trap_beams`).
    pub fn point(&self,r:[f64;3],kappa:[f64;3])->Point{if self.wide{unsafe{self.point_wide(r,kappa)}}else{self.point_body(r,kappa)}}
    #[target_feature(enable="avx2,fma")]
    unsafe fn point_wide(&self,r:[f64;3],kappa:[f64;3])->Point{self.point_body(r,kappa)}
    #[inline(always)]
    fn point_body(&self,r:[f64;3],kappa:[f64;3])->Point{
        let(mut s,mut off)=([0.;MOST_FOCUSED],[0.;MOST_FOCUSED]);let mut gradient=[[0.;3];MOST_FOCUSED];
        for(i,&u)in self.focused.iter().enumerate(){
            let axis=&self.axes[i];(s[i],off[i],gradient[i])=self.ground.units[u].axis_at(axis.coordinate,[axis.lo,axis.hi],|l|self.geometry[l].field(r),|l|self.trap_beams.contains(&l));
        }
        let mut fields=[CarrierField::default();MOST_CARRIERS];
        for(i,(c,f))in self.carriers.iter().zip(fields.iter_mut()).enumerate(){
            for(j,&l)in c.beams.iter().enumerate(){
                let(a,g)=self.geometry[l].field(r);let w=self.first_shift(i,l,kappa);
                for k in 0..c.rank{let x=c.coefficients[j][k];let ax=a*x;f.e[k]+=ax;f.doppler[k]+=ax*w;for d in 0..3{let y=g[d]*x;f.grad[k][d]+=y;f.doppler_grad[k][d]+=y*w;}}
            }
        }
        let d=self.focused.len();let stencil=self.stencil(&s[..d]);
        Point{r,kappa,stencil,far:self.stencil(&off[..d]),intensity_gradient:gradient,fields,shifted:true}
    }

    /// w_l = (ħ/m) k_l·κ + ħ|k_l|²/2m (rad/us): the kinetic energy absorbing from beam l gives.
    pub fn kinetic_shift(&self,l:usize,kappa:[f64;3])->f64{
        if !self.doppler{return 0.;}
        let g=&self.geometry[l];let k=g.wave_vector();self.hbar_over_mass*(k[0]*kappa[0]+k[1]*kappa[1]+k[2]*kappa[2]+0.5*g.wave_number*g.wave_number)
    }
    /// The kinetic shift the first order takes (module documentation, # The photons and the Doppler term): beam l's of carrier
    /// c, within half its nearest pole σ_c, far beyond the first order's allowance: a beam's scattering, ∝ 1 − 2 w Re G at
    /// ‖G‖ ≤ 1/σ_c, stays a decay however fast the atom.
    pub fn first_shift(&self,c:usize,l:usize,kappa:[f64;3])->f64{let half=0.5*self.ground.poles[c];self.kinetic_shift(l,kappa).clamp(-half,half)}
    /// The first order in the Doppler shift's expansion parameter at κ: the largest (w_l/σ_c)² over the
    /// near beams, w_l the kinetic shift and σ_c its carrier's nearest excited pole (ground.rs).
    pub fn doppler_ratio(&self,kappa:[f64;3])->f64{
        self.carriers.iter().zip(&self.ground.poles).flat_map(|(c,&pole)|c.beams.iter().map(move|&l|(l,pole))).map(|(l,pole)|(self.kinetic_shift(l,kappa)/pole).powi(2)).fold(0.,f64::max)
    }

    /// A cluster's dimension.
    pub fn dimension(&self,cluster:usize)->usize{self.layouts[cluster].n}
    /// A cluster's work per call in complex multiply-adds, from its tables' layout (the step choice's
    /// cost, solve.rs): the generator (`local`: per pair its S with the slopes and its Q gathered on the
    /// stencil's nodes, added -- twice in a Hermitian group -- and kept for the force; h0 likewise), the
    /// force (`force`: the products once per group, a form per pair, slope and, with a first order, Q, h0's slopes), a jump's
    /// photons (`photons`: per carrier, rank and axis J and J2 on the stencil and their sum, and the transient's) and the
    /// picked class's absorbed light (`Photons::absorption`: per photon and rank J and J2 along the
    /// polarization and J's slope on the stencil), its classes' mean; at fourth order H^(4)'s slabs
    /// in the generator and W3, the metric and Π in a jump.
    pub fn work(&self,cluster:usize)->Work{
        let layout=&self.layouts[cluster];let d=self.focused.len() as f64;let nodes=(1usize<<(2*self.focused.len())) as f64;
        let(n2,ng)=((layout.n*layout.n) as f64,self.ground.ground_states.len() as f64);
        let(mut generator,mut force)=(nodes*(1.+d)*n2+n2,d*n2);
        for gl in &layout.groups{
            let(e,pairs)=(gl.entries.len() as f64,gl.pairs.len() as f64);let added=if gl.hermitian{4.}else{2.};
            generator+=pairs*(nodes*(2.+d)+added+1.+d)*e;force+=e+pairs*(1.+d)*e;
            // The first order: its transfer on the stencil, each pair's coefficients, the forms of Q and their sums; what it leaves (known once the tables are made).
            if !gl.first.is_empty(){let first=gl.first.len() as f64;generator+=2.*nodes+3.*first;force+=pairs*e+8.*first;}
            for k in &gl.kept{let x=k.entries.len() as f64;generator+=2.*x*(nodes+2.)+4.;force+=x+4.;}
        }
        let mut photons=self.carriers.iter().map(|c|c.rank as f64*3.*ng*layout.n as f64*(2.*nodes+1.)).sum();
        // The transient after a jump (engine.rs `transient`, its target's taken for this cluster's): the columns' norms on the stencil and their sums.
        photons+=self.carriers.iter().map(|c|c.rank as f64*layout.n0 as f64*(nodes+1.)).sum::<f64>();
        // The fourth order: H^(4)'s slabs per generator; per jump W3's tables and forms, the metric on the
        // state and Π after the jump (`FourthLayout`).
        if let Some(f)=&layout.fourth{
            let r=|c:usize|self.carriers[c].rank as f64;let n=layout.n as f64;
            generator+=f.quartic.len() as f64*nodes*n2+n2;
            let pairs:f64=f.slow.iter().map(|s|r(s.carriers.0)*r(s.carriers.1)).sum();
            photons+=f.jumps.iter().map(|j|r(j.inner)*3.*ng*ng*n*j.sources(layout.n) as f64*(nodes+1.)).sum::<f64>()+f.p.iter().map(|&(c2,c4,_,_)|r(c2)*r(c4)*ng*n*nodes).sum::<f64>()+3.*pairs*nodes*n2;
        }
        let classes=&self.ground.clusters[cluster].classes;
        let absorption=classes.iter().map(|c|c.photons.iter().map(|(ph,_)|self.carriers[ph.carrier].rank as f64*(6.+2.*nodes*(3.+d))).sum::<f64>()).sum::<f64>()/classes.len().max(1) as f64;
        Work{generator,force,photons,absorption,pairs:layout.groups.iter().map(|g|g.pairs.len()).sum()}
    }
    /// The number of clusters.
    pub fn clusters(&self)->usize{self.layouts.len()}
    /// Whether a cluster has explicit excited states.
    pub fn mixed(&self,cluster:usize)->bool{self.ground.mixed(cluster)}
    /// Whether a cluster's generator beats: a term keeps a residual frequency in its frame.
    pub fn beating(&self,cluster:usize)->bool{!self.layouts[cluster].beats.omegas.is_empty()}
    /// Whether a cluster's generator is static: ground states alone, none beating.
    pub fn is_static(&self,cluster:usize)->bool{!self.mixed(cluster)&&!self.beating(cluster)}
    /// Whether a cluster takes the exact step (engine.rs): explicit excited states, or beats it does not leave out wherever the atom goes (`quiet`, engine.rs `quiet_clusters`).
    pub(crate) fn takes_exact(&self,cluster:usize,quiet:&[Option<f64>])->bool{self.mixed(cluster)||(self.beating(cluster)&&quiet[cluster].is_none())}
    /// The points the profiles and the bound are judged over (`reach`): the focus and ±reach_a along the trap's principal axes.
    pub(crate) fn reach_points(&self)->&[[f64;3]]{&self.reach}
    /// A cluster's beats (rad/us), ascending and closed under negation.
    pub fn beats(&self,cluster:usize)->&[f64]{&self.layouts[cluster].beats.omegas}
    /// The entries of a cluster's groups the first order's transfer leaves (`transfer_tables`).
    pub fn kept_entries(&self,cluster:usize)->usize{self.layouts[cluster].groups.iter().map(|g|g.kept.iter().map(|k|k.entries.len()).sum::<usize>()).sum()}
    /// A cluster's number of ground states (the first of its basis).
    pub fn ground_dimension(&self,cluster:usize)->usize{self.layouts[cluster].n0}
    /// The part of a cluster's components a residual goes into (`components`): 0 the static one, 1 + j beat j's.
    pub(crate) fn part(&self,cluster:usize,residual:f64)->usize{self.layouts[cluster].beats.of(residual,[0.;3]).map_or(0,|j|j+1)}
    /// A cluster's terms of its explicit states for the GPU's components (gpu/program.rs `Term`), in
    /// `components`' order: per kept drive (kind 0), decay cross term (1), decay (2) and dressing's drive
    /// (3) its row, column, carrier, a decay's part, its tables' place
    /// (a cross term's X_k, then its X2_k's), a drive's U_k and its beams' tones (their parts and the beams); and where the explicit states' dressed photons lie (`photon`).
    pub(crate) fn exact_terms(&self,cluster:usize)->(Vec<[usize;7]>,Vec<[Complex;MOST_RANK]>,Vec<Vec<[usize;3]>>,usize){
        let l=&self.layouts[cluster];let(mut terms,mut u,mut tones)=(vec![],vec![],vec![]);
        let beams=|c:usize,list:&[Tone]|->Vec<[usize;3]>{list.iter().map(|t|[self.carriers[c].beams[t.beam],t.part,t.mirror]).collect()};
        for d in &l.drives{terms.push([0,d.row,d.col,d.carrier,0,0,0]);u.push(d.u);tones.push(beams(d.carrier,&d.tones));}
        for c in &l.crosses{terms.push([1,c.row,c.col,c.carrier,0,c.at,c.at2]);u.push([Complex::ZERO;MOST_RANK]);tones.push(beams(c.carrier,&c.tones));}
        for d in &l.decays{terms.push([2,d.row,d.col,0,d.part,d.at,0]);u.push([Complex::ZERO;MOST_RANK]);tones.push(vec![]);}
        for e in &l.effective{terms.push([3,e.row,e.col,e.carrier,0,e.at,0]);u.push([Complex::ZERO;MOST_RANK]);tones.push(beams(e.carrier,&e.tones));}
        (terms,u,tones,l.emission)
    }
    /// A cluster's beats' wave vectors and its states' (`Beats`, `Layout::labels`), none where it has no gauge: gpu/program.rs's.
    pub(crate) fn beat_waves(&self,cluster:usize)->(&[[f64;3]],Option<&[[f64;3]]>){let l=&self.layouts[cluster];(&l.beats.waves,l.gauged.then_some(&l.labels[..]))}

    /// The table slice [start, start + value.len()) of a cluster at a stencil, and with `slopes` its
    /// derivatives in every s_f (each at f times the given stride), in one pass over the stencil's nodes.
    fn gather_dyn(&self,cluster:usize,stencil:&Stencil,start:usize,value:&mut [Complex],mut slopes:Option<(&mut [Complex],usize)>){
        let size=2*self.layouts[cluster].size;let data=&self.tables.data[cluster];let len=value.len();let d=stencil.dimensions;
        value.fill(Complex::ZERO);if let Some((s,stride))=&mut slopes{for g in 0..d{s[g* *stride..g* *stride+len].fill(Complex::ZERO);}}
        for i in 0..stencil.count(){
            let(node,w,dw)=stencil.node(i);let base=node*size+2*start;let x=&data[base..base+2*len];
            for(o,z)in value.iter_mut().zip(x.chunks_exact(2)){o.re+=w*z[0];o.im+=w*z[1];}
            if let Some((s,stride))=&mut slopes{for g in 0..d{for(o,z)in s[g* *stride..g* *stride+len].iter_mut().zip(x.chunks_exact(2)){o.re+=dw[g]*z[0];o.im+=dw[g]*z[1];}}}
        }
    }
    /// A cluster's generator at a point and time t on its n states (`Generator`): a static cluster of
    /// ground states' through `local`, every other one's through its beats' components (`components`).
    pub fn generator(&self,cluster:usize,p:&Point,t:f64)->Generator{
        if self.is_static(cluster){
            let local=self.local(cluster,p,t);let n=self.dimension(cluster);
            return Generator{n,hamiltonian:local.hamiltonian[..n*n].to_vec(),scalar:local.scalar,gradient:self.gradients(cluster,&local,p)};
        }
        self.components(cluster,p).at(t)
    }
    /// A cluster's generator at a point split by its beats (`Components`): its tables' terms, the kept
    /// drives, the decay cross terms and the explicit states' decay (module documentation), and a cluster
    /// of ground states' fourth order, each into the part of its residual and wave vector -- the static one, or its beat's: the atom at the point, at its beats' residuals (`on_path` moves it).
    pub fn components(&self,cluster:usize,p:&Point)->Components{
        let layout=&self.layouts[cluster];let(n,d)=(layout.n,self.focused.len());let nn=n*n;let beats=&layout.beats;let parts=1+beats.omegas.len();
        // The part of a residual: 0 the static one, 1 + j beat j's.
        let part=|w:f64|beats.of(w,[0.;3]).map_or(0,|j|j+1);
        // Flat: M_q, then H_q in its place at the end, G_q per axis (`Components`), Γ_q, a table's entries, their slopes.
        let(mut matrices,mut omegas,mut places)=COMPONENT_BUFFERS.take();crate::linear::zeroed(&mut matrices,(5*parts+1+d)*nn);
        let(m,rest)=matrices.split_at_mut(parts*nn);let(gradient,rest)=rest.split_at_mut(3*parts*nn);let(gamma,rest)=rest.split_at_mut(parts*nn);let(buf,slope)=rest.split_at_mut(nn);
        // Every near beam's amplitude and gradient at the point: a cluster with explicit states' terms take them beam by beam.
        let mut fields=[(Complex::ZERO,[Complex::ZERO;3]);MOST_BEAMS];
        if layout.n>layout.n0{for c in &self.carriers{for &l in &c.beams{fields[l]=self.geometry[l].field(p.r);}}}
        let of=|q:usize|q*nn..(q+1)*nn;let along=|q:usize,a:usize|of(3*q+a);
        // acc += c X, and with `adjoint` c X + (c X)† (a Hermitian group's pair k1 < k standing for both);
        // a group's tables hold its entries alone (`scatter`).
        let add=|c:Complex,x:&[Complex],acc:&mut [Complex],adjoint:bool|{for a in 0..n{for b in 0..n{acc[a*n+b]+=c*x[a*n+b];if adjoint{acc[a*n+b]+=(c*x[b*n+a]).conj();}}}};
        let scatter=|c:Complex,x:&[Complex],entries:&[(usize,usize)],acc:&mut [Complex],adjoint:bool|{for(j,&(a,b))in entries.iter().enumerate(){acc[a*n+b]+=c*x[j];if adjoint{acc[b*n+a]+=(c*x[j]).conj();}}};
        // H ∋ c X is 2i c X in Γ, H ∋ −(i/2) Γ.
        let into=Complex::new(0.,2.);
        for gl in &layout.groups{
            let(f,e)=(&p.fields[gl.absorbed],&p.fields[gl.emitted]);let q=part(gl.residual);let mirror=part(-gl.residual);let count=gl.entries.len();
            let mut transfer=[Complex::ZERO;2];if !gl.first.is_empty(){self.gather_dyn(cluster,&p.stencil,gl.transfer,&mut transfer,None);}
            let lambda_s=transfer[0];let(mu,nu)=first_coefficients(transfer[0],transfer[1]);
            for(pi,&(k1,k))in gl.pairs.iter().enumerate(){
                let at=pi*count;let ee=e.e[k1].conj()*f.e[k];let both=gl.hermitian&&k1<k;
                self.gather_dyn(cluster,&p.stencil,gl.s+at,&mut buf[..count],Some((&mut *slope,nn)));
                scatter(ee,&buf[..count],&gl.entries,&mut m[of(q)],both);
                // ∇H_herm ∋ −∇(Φ) S − Φ Σ_f ∂_f S ∇s_f.
                for a in 0..3{
                    let c=e.grad[k1][a].conj()*f.e[k]+e.e[k1].conj()*f.grad[k][a];scatter(-c,&buf[..count],&gl.entries,&mut gradient[along(q,a)],both);
                    for g in 0..d{scatter(-ee*p.intensity_gradient[g][a],&slope[g*nn..g*nn+count],&gl.entries,&mut gradient[along(q,a)],both);}
                }
                self.gather_dyn(cluster,&p.stencil,gl.q+at,&mut buf[..count],None);scatter(ee,&buf[..count],&gl.entries,&mut gamma[of(q)],both);
                // The first order on Q (module documentation): in Γ, H ∋ −(i/2) Γ, its coefficients μ ξ_k′k + ν ξ_kk′* and, on Q†, μ ξ_kk′ + ν ξ_k′k* or, between two
                // carriers, ν ξ* at the mirror's beat; its force's gradient once, the pairing below halves it onto the two beats.
                if gl.first.is_empty(){continue;}
                let xi=e.e[k1].conj()*f.doppler[k];let adjoint_at=|c:Complex,acc:&mut [Complex]|{for(j,&(a,b))in gl.entries.iter().enumerate(){acc[b*n+a]+=c*buf[j].conj();}};
                let slope_of=|k1:usize,k:usize,a:usize|e.grad[k1][a].conj()*f.doppler[k]+e.e[k1].conj()*f.doppler_grad[k][a];
                if gl.hermitian{
                    let other=e.e[k].conj()*f.doppler[k1];
                    scatter((mu*xi+nu*other.conj())*into,&buf[..count],&gl.entries,&mut gamma[of(q)],false);
                    if k1<k{adjoint_at((mu*other+nu*xi.conj())*into,&mut gamma[of(q)]);}
                    for a in 0..3{let mut c=lambda_s*slope_of(k1,k,a);if k1<k{c+=(lambda_s*slope_of(k,k1,a)).conj();}scatter(c,&buf[..count],&gl.entries,&mut gradient[along(q,a)],false);}
                }else{
                    scatter(mu*xi*into,&buf[..count],&gl.entries,&mut gamma[of(q)],false);adjoint_at(nu*xi.conj()*into,&mut gamma[of(mirror)]);
                    for a in 0..3{scatter(lambda_s*slope_of(k1,k,a),&buf[..count],&gl.entries,&mut gradient[along(q,a)],false);}
                }
            }
            // What the transfer leaves: ½ ξ ΔW^S + (i/2) ξ ΔW^Q and their mirrors' adjoints, in Γ; ΔW^S's gradient.
            for kept in &gl.kept{
                let(k1,k)=kept.pair;let count=kept.entries.len();let xi=e.e[k1].conj()*f.doppler[k];
                self.gather_dyn(cluster,&p.stencil,kept.at,&mut buf[..count],None);
                for(j,&(x,y))in kept.entries.iter().enumerate(){let z=xi*buf[j]*0.5;gamma[q*nn+x*n+y]+=z*into;gamma[mirror*nn+y*n+x]+=z.conj()*into;
                    for a in 0..3{gradient[along(q,a).start+x*n+y]+=(e.grad[k1][a].conj()*f.doppler[k]+e.e[k1].conj()*f.doppler_grad[k][a])*buf[j];}}
                self.gather_dyn(cluster,&p.stencil,kept.at+count,&mut buf[..count],None);
                for(j,&(x,y))in kept.entries.iter().enumerate(){let z=xi*buf[j]*Complex::new(0.,0.5);gamma[q*nn+x*n+y]+=z*into;gamma[mirror*nn+y*n+x]+=(xi*buf[j]).conj()*Complex::new(0.,0.5)*into;}
            }
        }
        self.gather_dyn(cluster,&p.stencil,layout.h0,&mut *buf,Some((&mut *slope,nn)));
        for a in 0..3{for g in 0..d{let c=Complex::new(p.intensity_gradient[g][a],0.);add(c,&slope[of(g)],&mut gradient[along(0,a)],false);}}
        // The static part less its scalar (its Hermitian part's mean diagonal, a phase); a beat's diagonal,
        // a phase that beats, stays in its part.
        let scalar=(0..n).map(|a|buf[a*n+a].re-m[a*n+a].re).sum::<f64>()/n as f64;
        let h=m;for q in 0..parts{for i in 0..nn{h[q*nn+i]=(if q==0{buf[i]}else{Complex::ZERO})-h[q*nn+i]-gamma[q*nn+i]*Complex::new(0.,0.5);}}
        for a in 0..n{h[a*n+a]-=Complex::new(scalar,0.);}
        // The fourth order (a cluster of ground states): each slab's kept entries in their beats' parts
        // (`fourth_parts`; the force keeps the second order's).
        self.fourth_parts(cluster,p,h);
        // The kept drives beam by beam, a_l Σ_k c_lk U_k e^{iΩ̃t} and its adjoint, each into its part (`Tone`); their gradients the stimulated force.
        for dr in &layout.drives{
            let carrier=&self.carriers[dr.carrier];
            for t in &dr.tones{
                let(a,g)=&fields[carrier.beams[t.beam]];
                let v=*a*t.u;h[t.part*nn+dr.row*n+dr.col]+=v;h[t.mirror*nn+dr.col*n+dr.row]+=v.conj();
                for x in 0..3{let z=g[x]*t.u;gradient[along(t.part,x).start+dr.row*n+dr.col]+=z;gradient[along(t.mirror,x).start+dr.col*n+dr.row]+=z.conj();}
            }
        }
        // The decay cross terms beam by beam, (i/2) e^{iΩ̃t} a_l Σ_k c_lk (X_k − w_l X2_k) and its mirror; the explicit states' decay.
        let(mut x,mut x2)=([Complex::ZERO;MOST_RANK],[Complex::ZERO;MOST_RANK]);
        for cr in &layout.crosses{
            let carrier=&self.carriers[cr.carrier];let rank=carrier.rank;self.gather_dyn(cluster,&p.stencil,cr.at,&mut x[..rank],None);self.gather_dyn(cluster,&p.stencil,cr.at2,&mut x2[..rank],None);
            for t in &cr.tones{
                let l=carrier.beams[t.beam];let c=&carrier.coefficients[t.beam];let w=if p.shifted{self.first_shift(cr.carrier,l,p.kappa)}else{0.};
                let z=(0..rank).fold(Complex::ZERO,|s,k|s+c[k]*(x[k]-x2[k]*w))*fields[l].0;
                h[t.part*nn+cr.row*n+cr.col]+=z*Complex::new(0.,0.5);h[t.mirror*nn+cr.col*n+cr.row]+=z.conj()*Complex::new(0.,0.5);
            }
        }
        let mut z1=[Complex::ZERO];
        for x in &layout.decays{self.gather_dyn(cluster,&p.stencil,x.at,&mut z1,None);h[x.part*nn+x.row*n+x.col]+=z1[0];}
        // The drives the dressing brings beam by beam: a_l Σ_k c_lk Y_k e^{iΩ̃t} and its adjoint, and their gradients.
        let(mut y,mut dy)=([Complex::ZERO;MOST_RANK],[Complex::ZERO;MOST_FOCUSED*MOST_RANK]);
        for x in &layout.effective{
            let carrier=&self.carriers[x.carrier];let rank=carrier.rank;
            self.gather_dyn(cluster,&p.stencil,x.at,&mut y[..rank],Some((&mut dy,MOST_RANK)));
            for t in &x.tones{
                let(a,g)=&fields[carrier.beams[t.beam]];let c=&carrier.coefficients[t.beam];
                let sum=(0..rank).fold(Complex::ZERO,|s,k|s+c[k]*y[k]);
                let v=*a*sum;h[t.part*nn+x.row*n+x.col]+=v;h[t.mirror*nn+x.col*n+x.row]+=v.conj();
                for axis in 0..3{
                    let mut z=g[axis]*sum;
                    for f in 0..d{z+=*a*(0..rank).fold(Complex::ZERO,|s,k|s+c[k]*dy[f*MOST_RANK+k])*p.intensity_gradient[f][axis];}
                    gradient[along(t.part,axis).start+x.row*n+x.col]+=z;gradient[along(t.mirror,axis).start+x.col*n+x.row]+=z.conj();
                }
            }
        }

        // The gradients' Hermitian parts (the force reads Re⟨ψ|·|ψ⟩ alone): the static one's, and per pair of
        // beats ±Ω, (G_Ω + G_{−Ω}†)/2 at Ω and its adjoint at −Ω, so that G(t) is Hermitian at every t.
        for x in 0..3{let g=&mut gradient[along(0,x)];for a in 0..n{for b in a..n{let z=(g[a*n+b]+g[b*n+a].conj())*0.5;g[a*n+b]=z;g[b*n+a]=z.conj();}}}
        let half=beats.omegas.len()/2;
        for i in 0..half{
            let(up,down)=(1+half+i,half-i);let(low,high)=gradient.split_at_mut(3*up*nn);
            for x in 0..3{let(u,v)=(&mut high[of(x)],&mut low[along(down,x)]);
                for a in 0..n{for b in 0..n{let z=(u[a*n+b]+v[b*n+a].conj())*0.5;u[a*n+b]=z;v[b*n+a]=z.conj();}}}
        }
        omegas.clear();omegas.extend_from_slice(&beats.omegas);places.clear();places.extend(1..parts);
        Components{n,scalar,parts,omegas,places,matrices,gauge:[0.;MOST_MIXED],middle:0.,gauged:false}
    }
    /// Whether a cluster's step has a path to take (`on_path`): a beat with a wave vector, or states with theirs.
    pub fn moves(&self,cluster:usize)->bool{let l=&self.layouts[cluster];self.doppler&&(l.gauged||l.beats.waves.iter().any(|w|*w!=[0.;3]))}
    /// A cluster's components at rest (`components`) along the atom's straight path through their point at the time `middle`, at the momentum κ (module documentation, # The atom's path): every beat's frequency takes its q·v and
    /// its matrices e^{−iq·v t_m}, so that e^{i(Ω̃ + q·v)t} at t_m is e^{iΩ̃t_m}, the fields' phases at the point; the states with wave vectors take their diagonal shifts K_s·v.
    pub fn on_path(&self,cluster:usize,c:&mut Components,kappa:[f64;3],middle:f64){
        if !self.doppler{return;}
        let layout=&self.layouts[cluster];let v=kappa.map(|x|x*self.hbar_over_mass);let(n,parts)=(layout.n,c.parts);let nn=n*n;
        for j in 0..parts-1{
            let rate=doppler(layout.beats.waves[j],v);if rate==0.{continue;}
            c.omegas[j]+=rate;let phase=Complex::cis(-rate*middle);let q=j+1;
            for z in &mut c.matrices[q*nn..(q+1)*nn]{*z=*z*phase;}
            let at=(parts+3*q)*nn;for z in &mut c.matrices[at..at+3*nn]{*z=*z*phase;}
        }
        if layout.gauged{
            for a in 0..n{c.gauge[a]=doppler(layout.labels[a],v);c.matrices[a*n+a]+=Complex::new(c.gauge[a],0.);}
            c.middle=middle;c.gauged=true;
        }
    }

    /// A cluster's generator at a point and time t (`Local`): a cluster of ground states' (one with
    /// explicit excited states has `generator`).
    pub fn local(&self,cluster:usize,p:&Point,t:f64)->Local{
        assert!(!self.mixed(cluster),"a cluster with explicit excited states has no Local");
        macro_rules! sized{($($n:literal)*)=>{match self.dimension(cluster){$($n=>if self.wide{unsafe{self.local_wide::<$n,{2*$n*$n}>(cluster,p,t)}}else{self.local_body::<$n,{2*$n*$n}>(cluster,p,t)},)*_=>unreachable!("a cluster has 1 to 8 states")}}}
        let mut out=sized!(1 2 3 4 5 6 7 8);
        // The fourth order: H^(4) in the cluster's frames (the force keeps the second order's: T's
        // error stays the fold's S, the ledger's).
        if let Some(h)=self.fourth_hamiltonian(cluster,p,t){let n=self.dimension(cluster);for i in 0..n*n{out.hamiltonian[i]+=h[i];}}
        out
    }
    /// A cluster's fourth-order part of H_eff at a point and time (n × n, row-major), None without its
    /// tables: over its slabs the monomial E*_p E*_p′ E_q E_q′ in its beat's phase e^{iνt} times the slab, one
    /// pass over the stencil's nodes, each entry (a, b) then in the frames' phase e^{i(θ_a − θ_b)t} (the
    /// entries left out are zero in the tables).
    pub fn fourth_hamiltonian(&self,cluster:usize,p:&Point,t:f64)->Option<[Complex;MOST_STATES*MOST_STATES]>{
        let f=self.layouts[cluster].fourth.as_ref()?;let n=self.dimension(cluster);let cl=&self.ground.clusters[cluster];
        let e=|i:usize|{let(c,k)=f.fields[i];p.fields[c].e[k]};let mut rotation=Rotation::new(t);
        let mut terms:Vec<(usize,Complex)>=Vec::with_capacity(f.quartic.len());
        for q in &f.quartic{let m=e(q.conj.0).conj()*e(q.conj.1).conj()*e(q.plain.0)*e(q.plain.1);if m!=Complex::ZERO{terms.push((q.at,m*rotation.at(q.nu)));}}
        let size=2*self.layouts[cluster].size;let data=&self.tables.data[cluster];let mut h=[Complex::ZERO;MOST_STATES*MOST_STATES];let nn=n*n;
        for i in 0..p.stencil.count(){
            let(node,w,_)=p.stencil.node(i);let row=&data[node*size..(node+1)*size];
            for &(at,m)in &terms{let c=m*w;let x=&row[2*at..2*(at+nn)];
                for(o,z)in h[..nn].iter_mut().zip(x.chunks_exact(2)){o.re+=c.re*z[0]-c.im*z[1];o.im+=c.re*z[1]+c.im*z[0];}}
        }
        for a in 0..n{for b in 0..n{if a!=b{h[a*n+b]=h[a*n+b]*Complex::cis((cl.frames[a]-cl.frames[b])*t);}}}
        Some(h)
    }
    /// A cluster's H^(4) at a point added into its beats' parts (`components`): each slab's monomial times its
    /// kept entries, each into its part (`QuarticSlab::part`).
    fn fourth_parts(&self,cluster:usize,p:&Point,h:&mut [Complex]){
        let Some(f)=self.layouts[cluster].fourth.as_ref()else{return};let nn=self.dimension(cluster).pow(2);
        let e=|i:usize|{let(c,k)=f.fields[i];p.fields[c].e[k]};let mut slab=[Complex::ZERO;MOST_STATES*MOST_STATES];let slab=&mut slab[..nn];
        for q in &f.quartic{
            let m=e(q.conj.0).conj()*e(q.conj.1).conj()*e(q.plain.0)*e(q.plain.1);if m==Complex::ZERO{continue;}
            self.interpolate(cluster,&p.stencil,None,q.at,slab);
            for i in 0..nn{if q.keep[i]{h[q.part[i]*nn+i]+=m*slab[i];}}
        }
    }
    /// out += a carrier pair's slab at `at` at a point and time (n × n): Σ E^c1*_k1 E^c2_k2 M_k1k2 in its
    /// beat's phase e^{iνt} and the frames' e^{i(θ_a − θ_b)t}.
    fn fourth_pair(&self,cluster:usize,p:&Point,t:f64,sl:&SlowSlab,at:usize,out:&mut [Complex]){
        let n=self.dimension(cluster);let cl=&self.ground.clusters[cluster];let(c1,c2)=sl.carriers;let(r1,r2)=(self.carriers[c1].rank,self.carriers[c2].rank);
        let(e1,e2)=(&p.fields[c1].e,&p.fields[c2].e);let beat=Complex::cis(sl.nu*t);let mut buf=[Complex::ZERO;MOST_STATES*MOST_STATES];
        let phases:Vec<Complex>=(0..n*n).map(|i|Complex::cis((cl.frames[i/n]-cl.frames[i%n])*t)).collect();
        for k1 in 0..r1{for k2 in 0..r2{let m=e1[k1].conj()*e2[k2]*beat;if m==Complex::ZERO{continue;}
            self.interpolate(cluster,&p.stencil,None,at+(k1*r2+k2)*n*n,&mut buf[..n*n]);
            for i in 0..n*n{out[i]+=m*buf[i]*phases[i];}}}
    }
    /// The metric's coherent part X = Σ R_c1†R_c2 on a cluster's coherent entries at a point and time (n × n,
    /// in its frames; the readout's), None without the fourth order's tables.
    pub fn coherent_metric(&self,cluster:usize,p:&Point,t:f64)->Option<Vec<Complex>>{
        let f=self.layouts[cluster].fourth.as_ref()?;let n=self.dimension(cluster);let mut x=vec![Complex::ZERO;n*n];
        for sl in &f.slow{self.fourth_pair(cluster,p,t,sl,sl.rr,&mut x);}
        Some(x)
    }
    /// After a jump into a cluster with fourth-order tables (and at t = 0), the bare state's slow part
    /// Π = 1 + ½X + YW at the point (n × n, module documentation): ψ ← Π f, its weight A = ‖Π f‖² the
    /// threshold's divisor (engine.rs). None without the tables.
    pub fn slow_part(&self,cluster:usize,p:&Point,t:f64)->Option<Vec<Complex>>{self.slow_and_metric(cluster,p,t).map(|x|x.0)}
    /// `slow_part` and the metric's coherent part X it holds (`coherent_metric`), whose form on a bare state is the ringing's probability (engine.rs, # The transient).
    pub fn slow_and_metric(&self,cluster:usize,p:&Point,t:f64)->Option<(Vec<Complex>,Vec<Complex>)>{
        let f=self.layouts[cluster].fourth.as_ref()?;let n=self.dimension(cluster);let(mut x,mut yw)=(vec![Complex::ZERO;n*n],vec![Complex::ZERO;n*n]);
        for sl in &f.slow{self.fourth_pair(cluster,p,t,sl,sl.rr,&mut x);self.fourth_pair(cluster,p,t,sl,sl.yw,&mut yw);}
        Some(((0..n*n).map(|i|x[i]*0.5+yw[i]+if i%(n+1)==0{Complex::ONE}else{Complex::ZERO}).collect(),x))
    }
    /// The fourth order's W3 part of a cluster's photons at a point: per carrier c and axis −C W3^c (ng × n,
    /// row-major), W3^c[:,b] = Σ over the triples (c1, c2, c4) at δ_c of G_b(δ_c) Σ_b′ R_c1[:,b′] (V_c2†R_c4)[b′,b]
    /// (the metric's part acts on the state, `photons`); None without the tables.
    fn fourth_photons(&self,cluster:usize,p:&Point)->Option<Vec<Vec<Complex>>>{
        let f=self.layouts[cluster].fourth.as_ref()?;let(n,ng)=(self.dimension(cluster),self.ground.ground_states.len());let r=|c:usize|self.carriers[c].rank;
        let form=|d:usize,e2:usize,at:usize|->Vec<Complex>{let(ed,ee)=(&p.fields[d].e,&p.fields[e2].e);
            let mut m=vec![Complex::ZERO;ng*n];let mut buf=vec![Complex::ZERO;ng*n];
            for k3 in 0..r(d){for k4 in 0..r(e2){let w=ed[k3].conj()*ee[k4];if w==Complex::ZERO{continue;}self.interpolate(cluster,&p.stencil,None,at+(k3*r(e2)+k4)*ng*n,&mut buf);for i in 0..ng*n{m[i]+=w*buf[i];}}}
            m};
        let forms:Vec<Vec<Complex>>=f.p.iter().map(|&(c2,c4,_,at)|form(c2,c4,at)).collect();
        let mut out=vec![vec![Complex::ZERO;3*ng*n];self.carriers.len()];let mut buf=vec![Complex::ZERO;3*ng*ng*n*n];
        for js in &f.jumps{
            let mut weight=vec![Complex::ZERO;ng*n];
            for &i in &js.pairs{for q in 0..ng*n{weight[q]+=forms[i][q];}}
            let ee=&p.fields[js.inner].e;
            let s=js.sources(n);let size=3*ng*ng*n*s;
            for k in 0..r(js.inner){if ee[k]==Complex::ZERO{continue;}
                self.interpolate(cluster,&p.stencil,None,js.at+k*size,&mut buf[..size]);
                // Column y of W3^c: its sources' places b (a diagonal slab's y alone), each through its middle factor (bp, b).
                for ax in 0..3{for row in 0..ng{for y in 0..n{
                    let z=(0..ng).fold(Complex::ZERO,|z,bp|{let x=&buf[(((ax*ng+row)*ng+bp)*n+y)*s..][..s];
                        if js.diagonal{z+x[0]*weight[bp*n+y]}else{(0..n).fold(z,|z,b|z+x[b]*weight[bp*n+b])}});
                    out[js.photon][(ax*ng+row)*n+y]-=ee[k]*z;
                }}}
            }
        }
        Some(out)
    }
    #[target_feature(enable="avx2,fma")]
    unsafe fn local_wide<const N:usize,const W:usize>(&self,cluster:usize,p:&Point,t:f64)->Local{self.local_body::<N,W>(cluster,p,t)}
    #[inline(always)]
    fn local_body<const N:usize,const W:usize>(&self,cluster:usize,p:&Point,t:f64)->Local{
        let layout=&self.layouts[cluster];let d=self.focused.len();
        let coefficients:usize=layout.groups.iter().map(|g|1+g.pairs.len()).sum();
        let values:usize=(1+d)*W+layout.groups.iter().map(|g|4*(!g.first.is_empty()) as usize+2*((3+d)*g.pairs.len()*g.entries.len()+2*g.kept.iter().map(|k|k.entries.len()).sum::<usize>())).sum::<usize>();
        let(mut buffer,mut factors)=LOCAL_BUFFERS.with(|b|b.take());buffer.clear();buffer.reserve(values);factors.clear();factors.reserve(coefficients);
        // The tables at the stencil (`Local`): per group its first order's transfer, per pair S, its slopes and Q, what the transfer leaves (ΔW^S, ΔW^Q); h0 and its slopes.
        let mut buf=[0.;W];let mut slope=[[0.;W];MOST_FOCUSED];
        for gl in &layout.groups{
            if !gl.first.is_empty(){let mut x=[0.;4];self.gather_value(cluster,&p.stencil,gl.transfer,4,&mut x);buffer.extend_from_slice(&x);}
            let(count,len)=(gl.entries.len(),2*gl.entries.len());
            for pi in 0..gl.pairs.len(){
                self.gather(cluster,&p.stencil,gl.s+pi*count,len,&mut buf,&mut slope);buffer.extend_from_slice(&buf[..len]);for g in 0..d{buffer.extend_from_slice(&slope[g][..len]);}
                self.gather_value(cluster,&p.stencil,gl.q+pi*count,len,&mut buf);buffer.extend_from_slice(&buf[..len]);
            }
            for kept in &gl.kept{let len=2*kept.entries.len();for at in [kept.at,kept.at+kept.entries.len()]{self.gather_value(cluster,&p.stencil,at,len,&mut buf);buffer.extend_from_slice(&buf[..len]);}}
        }
        self.gather(cluster,&p.stencil,layout.h0,W,&mut buf,&mut slope);buffer.extend_from_slice(&buf);for g in 0..d{buffer.extend_from_slice(&slope[g]);}
        let mut out=Local{hamiltonian:[Complex::ZERO;MOST_STATES*MOST_STATES],scalar:0.,values:buffer,coefficients:factors,intensity_gradient:p.intensity_gradient};
        self.assemble::<N,W>(cluster,&mut out,p,t);
        out
    }
    /// A generator from its tables at a stencil (`Local::values`) at a point's fields and the time t: H_eff less its scalar part, and the force's coefficients.
    #[inline(always)]
    fn assemble<const N:usize,const W:usize>(&self,cluster:usize,out:&mut Local,p:&Point,t:f64){
        let layout=&self.layouts[cluster];let d=self.focused.len();let Local{hamiltonian,scalar,values,coefficients,..}=out;coefficients.clear();
        // m, H's Hermitian part less h0; `decay`, H's part on Q: −(i/2) Γ and the first order.
        let(mut m,mut decay)=([0.;W],[0.;W]);
        // The pair (k, k1)'s tables are (k1, k)'s adjoints in a Hermitian group: c X + (c X)† for k1 < k
        // (all of c X added first). The force reads only Re⟨ψ|·|ψ⟩, so its coefficients take 2 c for
        // the two. A group's tables hold its entries alone: a group of every entry is added as
        // matrices, the others entry by entry.
        let adjoint=|c:Complex,x:&[f64;W],acc:&mut [f64;W]|{for a in 0..N{for b in 0..N{let(i,j)=(2*(a*N+b),2*(b*N+a));let(u,v)=(x[j],-x[j+1]);acc[i]+=c.re*u+c.im*v;acc[i+1]+=c.re*v-c.im*u;}}};
        let scatter=|c:Complex,x:&[f64],entries:&[(usize,usize)],mirror:bool,acc:&mut [f64;W]|{let x=&x[..2*entries.len()];
            for(j,&(a,b))in entries.iter().enumerate(){let w=c*Complex::new(x[2*j],x[2*j+1]);let i=2*(a*N+b);acc[i]+=w.re;acc[i+1]+=w.im;}
            if mirror{for(j,&(a,b))in entries.iter().enumerate(){let w=c*Complex::new(x[2*j],x[2*j+1]);let i=2*(b*N+a);acc[i]+=w.re;acc[i+1]-=w.im;}}
        };
        // c x and d x† on a group's entries.
        let both_ways=|c:Complex,d:Complex,x:&[f64],entries:&[(usize,usize)],acc:&mut [f64;W]|{let x=&x[..2*entries.len()];
            for(j,&(a,b))in entries.iter().enumerate(){let z=Complex::new(x[2*j],x[2*j+1]);let(u,v)=(c*z,d*z.conj());let(i,i2)=(2*(a*N+b),2*(b*N+a));acc[i]+=u.re;acc[i+1]+=u.im;acc[i2]+=v.re;acc[i2+1]+=v.im;}
        };
        let half_i=Complex::new(0.,-0.5);let mut at=0;
        for gl in &layout.groups{
            let(f,e)=(&p.fields[gl.absorbed],&p.fields[gl.emitted]);let phase=if gl.residual==0.{Complex::ONE}else{Complex::cis(gl.residual*t)};let(count,len)=(gl.entries.len(),2*gl.entries.len());
            // The first order's transfer (module documentation, # The photons and the Doppler term).
            let first=!gl.first.is_empty();let(mut lambda_s,mut mu,mut nu)=(Complex::ZERO,Complex::ZERO,Complex::ZERO);
            if first{let x=&values[at..at+4];at+=4;lambda_s=Complex::new(x[0],x[1]);(mu,nu)=first_coefficients(lambda_s,Complex::new(x[2],x[3]));}
            coefficients.push([phase,lambda_s*phase,Complex::ZERO,Complex::ZERO]);
            for &(k1,k)in &gl.pairs{
                let ee=phase*e.e[k1].conj()*f.e[k];let(both,mirror)=if gl.hermitian&&k1<k{(2.,true)}else{(1.,false)};
                // ∇H_herm ∋ −∇(Φ) S − Φ Σ_f ∂_f S ∇s_f: S, its slopes and the coefficients kept.
                let c=|a:usize|phase*(e.grad[k1][a].conj()*f.e[k]+e.e[k1].conj()*f.grad[k][a])*(-both);
                coefficients.push([c(0),c(1),c(2),ee*(-both)]);
                let x=&values[at..at+len];at+=(1+d)*len;
                if count==N*N{let x:&[f64;W]=x.try_into().unwrap();caxpy::<W>(ee,x,&mut m);if mirror{adjoint(ee,x,&mut m);}}else{scatter(ee,x,&gl.entries,mirror,&mut m);}
                // H ∋ −(i/2) Φ Q + μ ξ_k′k Q + ν ξ_kk′* Q and on the pair's adjoint its own; between two carriers ν ξ* Q† alone.
                let xi=phase*e.e[k1].conj()*f.doppler[k];
                let(a,b)=if gl.hermitian{let other=e.e[k].conj()*f.doppler[k1];(ee*half_i+mu*xi+nu*other.conj(),ee.conj()*half_i+mu*other+nu*xi.conj())}else{(ee*half_i+mu*xi,nu*xi.conj())};
                let mirrored=mirror||(first&&!gl.hermitian);let q=&values[at..at+len];at+=len;
                if count==N*N{let q:&[f64;W]=q.try_into().unwrap();caxpy::<W>(a,q,&mut decay);if mirrored{adjoint(b.conj(),q,&mut decay);}}
                else if mirrored{both_ways(a,b,q,&gl.entries,&mut decay);}else{scatter(a,q,&gl.entries,false,&mut decay);}
            }
            // What the transfer leaves: ½(ξ ΔW^S + h.c.) + (i/2)(ξ ΔW^Q + h.c.).
            for kept in &gl.kept{
                let(k1,k)=kept.pair;let xi=phase*e.e[k1].conj()*f.doppler[k];let len=2*kept.entries.len();
                let half=xi*0.5;both_ways(half,half.conj(),&values[at..at+len],&kept.entries,&mut decay);at+=len;
                both_ways(xi*Complex::new(0.,0.5),xi.conj()*Complex::new(0.,0.5),&values[at..at+len],&kept.entries,&mut decay);at+=len;
            }
        }
        let h0:&[f64;W]=values[at..at+W].try_into().unwrap();let el=|v:&[f64;W],a:usize,b:usize|Complex::new(v[2*(a*N+b)],v[2*(a*N+b)+1]);
        let mut s=0.;for a in 0..N{s+=h0[2*(a*N+a)]-m[2*(a*N+a)];}s/=N as f64;*scalar=s;
        for a in 0..N{for b in 0..N{
            hamiltonian[a*N+b]=el(h0,a,b)-el(&m,a,b)+el(&decay,a,b)-if a==b{Complex::new(s,0.)}else{Complex::ZERO};
        }}
    }

    /// −∇Re⟨ψ|H_eff|ψ⟩/⟨ψ|ψ⟩ (rad/us per um): the Ehrenfest force on ħκ. Its gradient's quadratic
    /// form, Σ_pairs Re(−∇Φ ⟨ψ|S|ψ⟩ − Φ Σ_f ∇s_f ⟨ψ|∂_f S|ψ⟩) + Σ_f ∇s_f ⟨ψ|∂_f h0|ψ⟩, from the forms of
    /// each pair's tables on its entries (`Local`).
    pub fn force(&self,cluster:usize,local:&Local,p:&Point,psi:&[Complex])->[f64;3]{
        macro_rules! sized{($($n:literal)*)=>{match self.dimension(cluster){$($n=>if self.wide{unsafe{self.force_wide::<$n>(cluster,local,p,psi)}}else{self.force_sized::<$n>(cluster,local,p,psi)},)*_=>unreachable!("a cluster has 1 to 8 states")}}}
        sized!(1 2 3 4 5 6 7 8)
    }
    #[target_feature(enable="avx2,fma")]
    unsafe fn force_wide<const N:usize>(&self,cluster:usize,local:&Local,p:&Point,psi:&[Complex])->[f64;3]{self.force_sized::<N>(cluster,local,p,psi)}
    #[inline(always)]
    fn force_sized<const N:usize>(&self,cluster:usize,local:&Local,p:&Point,psi:&[Complex])->[f64;3]{
        let layout=&self.layouts[cluster];let d=self.focused.len();let norm:f64=psi[..N].iter().map(|z|z.norm_sqr()).sum();
        // Per group the products ψ_a* ψ_b on its entries (row-major for a group of every entry); a
        // pair's ⟨ψ|X|ψ⟩ is then their sum weighted by its table X there.
        let mut products=[Complex::ZERO;MOST_STATES*MOST_STATES];
        let mut g=[0.;3];let(mut at,mut pair)=(0,0);
        for gl in &layout.groups{
            let count=gl.entries.len();let len=2*count;let(e,f)=(&p.fields[gl.emitted],&p.fields[gl.absorbed]);
            let[phase,lambda,_,_]=local.coefficients[pair];pair+=1;let first=!gl.first.is_empty();if first{at+=4;}
            // ⟨Q_k′k⟩ of every field pair, Q_kk′ = Q_k′k† in a Hermitian group.
            let mut rates=[[Complex::ZERO;MOST_RANK];MOST_RANK];
            if count==N*N{for a in 0..N{for b in 0..N{products[a*N+b]=psi[a].conj()*psi[b];}}}
            else{for(j,&(a,b))in gl.entries.iter().enumerate(){products[j]=psi[a].conj()*psi[b];}}
            let form=|x:&[f64]|->Complex{let mut s=Complex::ZERO;for j in 0..count{s+=products[j]*Complex::new(x[2*j],x[2*j+1]);}s};
            for &(k1,k)in &gl.pairs{
                let[c0,c1,c2,b]=local.coefficients[pair];let sigma=form(&local.values[at..at+len]);
                g[0]+=(c0*sigma).re;g[1]+=(c1*sigma).re;g[2]+=(c2*sigma).re;
                for f in 0..d{let tau=b*form(&local.values[at+(1+f)*len..at+(2+f)*len]);for a in 0..3{g[a]+=tau.re*local.intensity_gradient[f][a];}}
                at+=(1+d)*len;pair+=1;
                if first{let rate=form(&local.values[at..at+len]);rates[k1][k]=rate;if gl.hermitian&&k1<k{rates[k][k1]=rate.conj();}}
                at+=len;
            }
            if !first{at+=gl.kept.iter().map(|k|4*k.entries.len()).sum::<usize>();continue;}
            // The first order's −Re λ^S Σ_k′k ∇_aξ_k′k ⟨Q_k′k⟩, ξ_k′k = E^d_k′* E^D_k, on the fields' sums: Σ_k′ ∇_aE^d_k′* u_k′ + Σ_k ∇_aE^D_k v_k.
            let(emitted,absorbed)=(self.carriers[gl.emitted].rank,self.carriers[gl.absorbed].rank);
            let(mut u,mut v)=([Complex::ZERO;MOST_RANK],[Complex::ZERO;MOST_RANK]);
            for k1 in 0..emitted{for k in 0..absorbed{u[k1]+=rates[k1][k]*f.doppler[k];v[k]+=e.e[k1].conj()*rates[k1][k];}}
            for a in 0..3{
                let mut x=Complex::ZERO;for k1 in 0..emitted{x+=e.grad[k1][a].conj()*u[k1];}for k in 0..absorbed{x+=f.doppler_grad[k][a]*v[k];}
                g[a]+=(lambda*x).re;
            }
            for kept in &gl.kept{
                let(k1,k)=kept.pair;let values=&local.values[at..];
                let sigma=phase*kept.entries.iter().enumerate().fold(Complex::ZERO,|s,(j,&(a,b))|s+psi[a].conj()*psi[b]*Complex::new(values[2*j],values[2*j+1]));
                for a in 0..3{g[a]+=((e.grad[k1][a].conj()*f.doppler[k]+e.e[k1].conj()*f.doppler_grad[k][a])*sigma).re;}
                at+=4*kept.entries.len();
            }
        }
        at+=2*N*N;
        for f in 0..d{let eta=quadratic::<N>(&local.values[at+f*2*N*N..at+(f+1)*2*N*N],psi).re;for a in 0..3{g[a]+=local.intensity_gradient[f][a]*eta;}}
        std::array::from_fn(|a|-g[a]/norm)
    }
    /// A cluster of ground states' gradients of the Hermitian part per axis as matrices (n × n,
    /// Hermitian, row-major) from its `Local`'s pair forms -- `force`'s quadratic form, each pair's S and
    /// slopes on its entries with its coefficients, h0's slopes --, for the exact step's `Generator`.
    pub fn gradients(&self,cluster:usize,local:&Local,p:&Point)->[Vec<Complex>;3]{
        let layout=&self.layouts[cluster];let(n,d)=(layout.n,self.focused.len());
        let mut g:[Vec<Complex>;3]=std::array::from_fn(|_|vec![Complex::ZERO;n*n]);let(mut at,mut pair)=(0,0);
        for gl in &layout.groups{
            let len=2*gl.entries.len();let(e,f)=(&p.fields[gl.emitted],&p.fields[gl.absorbed]);
            let[phase,lambda,_,_]=local.coefficients[pair];pair+=1;if !gl.first.is_empty(){at+=4;}
            let value=|start:usize,j:usize|Complex::new(local.values[start+2*j],local.values[start+2*j+1]);
            let slope=|k1:usize,k:usize,a:usize|e.grad[k1][a].conj()*f.doppler[k]+e.e[k1].conj()*f.doppler_grad[k][a];
            for &(k1,k)in &gl.pairs{
                let[c0,c1,c2,b]=local.coefficients[pair];let c=[c0,c1,c2];
                // The first order's λ^S ∇_aξ on Q, the pair (k′, k) standing for (k, k′) in a Hermitian group.
                let q:[Complex;3]=std::array::from_fn(|a|{let x=lambda*slope(k1,k,a);if gl.hermitian&&k1<k{x+(lambda*slope(k,k1,a)).conj()}else{x}});
                let first=!gl.first.is_empty();
                for(j,&(x,y))in gl.entries.iter().enumerate(){
                    let s=value(at,j);
                    for a in 0..3{let mut z=c[a]*s;if first{z+=q[a]*value(at+(1+d)*len,j);}for f in 0..d{z+=b*value(at+(1+f)*len,j)*local.intensity_gradient[f][a];}g[a][x*n+y]+=z;}
                }
                at+=(2+d)*len;pair+=1;
            }
            for kept in &gl.kept{
                let(k1,k)=kept.pair;
                for(j,&(x,y))in kept.entries.iter().enumerate(){let z=phase*value(at,j);for a in 0..3{g[a][x*n+y]+=slope(k1,k,a)*z;}}
                at+=4*kept.entries.len();
            }
        }
        at+=2*n*n;
        for a in 0..3{for f in 0..d{let w=local.intensity_gradient[f][a];let h0=&local.values[at+f*2*n*n..];for k in 0..n*n{g[a][k]+=Complex::new(h0[2*k],h0[2*k+1])*w;}}}
        // Their Hermitian parts: the force reads Re⟨ψ|·|ψ⟩ alone.
        for m in g.iter_mut(){for x in 0..n{for y in x..n{let z=(m[x*n+y]+m[y*n+x].conj())*0.5;m[x*n+y]=z;m[y*n+x]=z.conj();}}}
        g
    }
    /// The trap's potential of ψ at a point (rad/us): Re⟨ψ|H_eff|ψ⟩ less the same with the trap's light
    /// off (`trap_beams`), per ⟨ψ|ψ⟩ -- (h0 − M) at the point's stencil and fields less at its far
    /// stencil with the trap beams' fields taken out of their units' and carriers': the tweezer's
    /// whole light shift whether it is folded (in h0) or a near carrier (in M, through its field) -- a
    /// tweezer alone is always one, the light's frame rotating at it.
    pub fn trap_potential(&self,cluster:usize,p:&Point,psi:&[Complex],t:f64)->f64{
        if self.mixed(cluster){return self.trap_mixed(cluster,p,psi,t,false);}
        macro_rules! sized{($($n:literal)*)=>{match self.dimension(cluster){$($n=>self.trap_sized::<$n,{2*$n*$n}>(cluster,p,psi,t,false),)*_=>unreachable!("a cluster has 1 to 8 states")}}}
        sized!(1 2 3 4 5 6 7 8)
    }
    /// An upper bound of it: h0's share, and the light shift's change at most twice its bound.
    pub fn trap_bound(&self,cluster:usize,p:&Point,psi:&[Complex])->f64{
        if self.mixed(cluster){return self.trap_mixed(cluster,p,psi,0.,true);}
        macro_rules! sized{($($n:literal)*)=>{match self.dimension(cluster){$($n=>self.trap_sized::<$n,{2*$n*$n}>(cluster,p,psi,0.,true),)*_=>unreachable!("a cluster has 1 to 8 states")}}}
        sized!(1 2 3 4 5 6 7 8)
    }
    fn trap_sized<const N:usize,const W:usize>(&self,cluster:usize,p:&Point,psi:&[Complex],t:f64,bound:bool)->f64{
        let layout=&self.layouts[cluster];let(mut here,mut far,mut difference)=([0.;W],[0.;W],[0.;W]);
        self.gather_value(cluster,&p.stencil,layout.h0,W,&mut here);self.gather_value(cluster,&p.far,layout.h0,W,&mut far);
        for j in 0..W{difference[j]=here[j]-far[j];}
        let norm:f64=psi[..N].iter().map(|z|z.norm_sqr()).sum();
        if bound{return quadratic::<N>(&difference,psi).re/norm+2.*self.ground.light;}
        let off=self.fields_off(p);
        for gl in &layout.groups{
            let(f,e)=(&p.fields[gl.absorbed],&p.fields[gl.emitted]);let(f0,e0)=(&off[gl.absorbed],&off[gl.emitted]);let phase=Complex::cis(gl.residual*t);
            let count=gl.entries.len();
            for(pi,&(k1,k))in gl.pairs.iter().enumerate(){
                // −(Φ S(s) − Φ_off S(far)) on the group's entries, the pair (k1, k) standing for (k, k1) in a
                // Hermitian group.
                let both=if gl.hermitian&&k1<k{2.}else{1.};let at=gl.s+pi*count;
                self.gather_value(cluster,&p.stencil,at,2*count,&mut here);self.gather_value(cluster,&p.far,at,2*count,&mut far);
                let(c,c0)=(-(e.e[k1].conj()*f.e[k]*both*phase),e0.e[k1].conj()*f0.e[k]*both*phase);
                for(j,&(a,b))in gl.entries.iter().enumerate(){let w=c*Complex::new(here[2*j],here[2*j+1])+c0*Complex::new(far[2*j],far[2*j+1]);let i=2*(a*N+b);difference[i]+=w.re;difference[i+1]+=w.im;}
            }
        }
        quadratic::<N>(&difference,psi).re/norm
    }

    /// An explicit state's dressed photon ĉ_a[to] at a stencil (the p-th explicit state of a cluster).
    pub fn photon(&self,cluster:usize,stencil:&Stencil,p:usize,a:usize,to:usize)->Complex{
        let mut z=[Complex::ZERO];let ng=self.ground.ground_states.len();self.gather_dyn(cluster,stencil,self.layouts[cluster].emission+(p*3+a)*ng+to,&mut z,None);z[0]
    }
    /// The carriers' fields at a point with the trap's light off (`trap_potential`): each one's less
    /// its trap beams' parts a_l(r) c_lk.
    fn fields_off(&self,p:&Point)->[CarrierField;MOST_CARRIERS]{
        let mut off=p.fields;
        for(c,carrier)in self.carriers.iter().enumerate(){
            for(j,&l)in carrier.beams.iter().enumerate(){
                if !self.trap_beams.contains(&l){continue;}
                let a=self.geometry[l].field(p.r).0;for k in 0..carrier.rank{off[c].e[k]-=a*carrier.coefficients[j][k];}
            }
        }
        off
    }
    /// `trap_sized` on any number of states.
    fn trap_mixed(&self,cluster:usize,p:&Point,psi:&[Complex],t:f64,bound:bool)->f64{
        let layout=&self.layouts[cluster];let n=layout.n;let nn=n*n;let(mut here,mut far)=(vec![Complex::ZERO;nn],vec![Complex::ZERO;nn]);
        self.gather_dyn(cluster,&p.stencil,layout.h0,&mut here,None);self.gather_dyn(cluster,&p.far,layout.h0,&mut far,None);
        let mut difference:Vec<Complex>=here.iter().zip(&far).map(|(a,b)|*a-*b).collect();
        let norm:f64=psi[..n].iter().map(|z|z.norm_sqr()).sum();
        let form=|x:&[Complex]|->f64{let mut s=Complex::ZERO;for a in 0..n{for b in 0..n{s+=psi[a].conj()*x[a*n+b]*psi[b];}}s.re};
        if bound{return form(&difference)/norm+2.*self.ground.light;}
        let off=self.fields_off(p);
        for gl in &layout.groups{
            let(f,e)=(&p.fields[gl.absorbed],&p.fields[gl.emitted]);let(f0,e0)=(&off[gl.absorbed],&off[gl.emitted]);let count=gl.entries.len();let phase=Complex::cis(gl.residual*t);
            for(pi,&(k1,k))in gl.pairs.iter().enumerate(){
                let both=if gl.hermitian&&k1<k{2.}else{1.};let at=gl.s+pi*count;
                self.gather_dyn(cluster,&p.stencil,at,&mut here[..count],None);self.gather_dyn(cluster,&p.far,at,&mut far[..count],None);
                let(c,c0)=(-(e.e[k1].conj()*f.e[k]*both*phase),e0.e[k1].conj()*f0.e[k]*both*phase);
                for(j,&(a,b))in gl.entries.iter().enumerate(){difference[a*n+b]+=c*here[j]+c0*far[j];}
            }
        }
        form(&difference)/norm
    }

    /// The photon amplitudes at a point of every class of a cluster's jumps (`Photons`), with the
    /// exact responses in a step beyond the first order's allowance (engine.rs): a folded photon's
    /// C_x A_c ψ = −C_x R_c ψ, an explicit state's C_x ψ_e.
    pub fn photons<'a>(&'a self,cluster:usize,p:&Point,psi:&[Complex],t:f64,exact:Option<Exact>)->Photons<'a>{self.photons_to(cluster,p,psi,t,exact,true)}
    /// `photons`, with the fourth order's corrections where `fourth` (else the second order's alone).
    fn photons_to<'a>(&'a self,cluster:usize,p:&Point,psi:&[Complex],t:f64,exact:Option<Exact>,fourth:bool)->Photons<'a>{
        if self.wide{unsafe{self.photons_wide(cluster,p,psi,t,exact,fourth)}}else{self.photons_body(cluster,p,psi,t,exact,fourth)}
    }
    #[target_feature(enable="avx2,fma")]
    unsafe fn photons_wide<'a>(&'a self,cluster:usize,p:&Point,psi:&[Complex],t:f64,exact:Option<Exact>,fourth:bool)->Photons<'a>{self.photons_body(cluster,p,psi,t,exact,fourth)}
    #[inline(always)]
    fn photons_body<'a>(&'a self,cluster:usize,p:&Point,psi:&[Complex],t:f64,exact:Option<Exact>,fourth:bool)->Photons<'a>{
        let g=&self.ground;let cl=&g.clusters[cluster];let layout=&self.layouts[cluster];let(n,n0)=(layout.n,layout.n0);let ng=g.ground_states.len();let size=ng*n0;
        // Per carrier, rank and axis the jump tables J_k and J2_k at the point (ng × n0), and per carrier
        // and axis M = Σ_k E_k J_k − E^D_k J2_k, or the exact responses' C_a Σ_l a_l R_l(w_l).
        let ranks:usize=self.carriers.iter().map(|c|c.rank).sum();
        let[mut jumps,mut doppler,mut matrices]=PHOTON_BUFFERS.with(|b|b.take());
        for v in [&mut jumps,&mut doppler]{if v.len()<3*ranks*size{v.resize(3*ranks*size,Complex::ZERO);}}matrices.clear();matrices.resize(3*self.carriers.len()*size,Complex::ZERO);
        let mut at=0;
        for(c,cl2)in layout.carriers.iter().enumerate(){
            let fl=&p.fields[c];
            for k in 0..self.carriers[c].rank{for a in 0..3{
                let offset=(k*3+a)*size;
                self.interpolate(cluster,&p.stencil,None,cl2.j+offset,&mut jumps[at..at+size]);self.interpolate(cluster,&p.stencil,None,cl2.j2+offset,&mut doppler[at..at+size]);
                if exact.is_none(){let m=&mut matrices[(c*3+a)*size..(c*3+a+1)*size];for i in 0..size{m[i]+=fl.e[k]*jumps[at+i]-fl.doppler[k]*doppler[at+i];}}
                at+=size;
            }}
            if let Some(exact)=&exact{
                for &l in &self.carriers[c].beams{let amplitude=self.geometry[l].field(p.r).0;
                    for a in 0..3{let m=g.emission[a].product(&exact.responses[l]);let out=&mut matrices[(c*3+a)*size..(c*3+a+1)*size];
                        for row in 0..ng{for y in 0..n0{out[row*n0+y]+=m[(row,cl.states[y])]*amplitude;}}}}
            }
        }
        // The fourth order (the Doppler's first order stays the second's): the photons' W3 part on ψ and their
        // second order on ψ − ½Xψ, X the metric's coherent part (`coherent_metric`) -- the photons of the fourth
        // order's carriers (ground.rs `fourth_carriers`), whose products the tables' decay holds.
        let w3=if fourth{self.fourth_photons(cluster,p)}else{None};
        let bends=|c:usize|layout.fourth.as_ref().is_some_and(|f|f.carriers.contains(&c));
        let bent:Option<[Complex;MOST_STATES]>=if !fourth{None}else{self.coherent_metric(cluster,p,t)}.map(|x|std::array::from_fn(|a|if a<n0{psi[a]-(0..n0).fold(Complex::ZERO,|s,b|s+x[a*n0+b]*psi[b])*0.5}else{Complex::ZERO}));
        // The classes: each photon's entry into its target state's row from its source's column.
        let amplitude=cl.classes.iter().map(|class|{let target=&g.clusters[class.target];
            let mut u=[[Complex::ZERO;MOST_STATES];3];let mut rotation=Rotation::new(t);
            for &(ph,w)in &class.photons{
                let(row,col)=(target.position(ph.to).unwrap(),cl.position(ph.from).unwrap());let phase=rotation.at(w);
                let state=match &bent{Some(b)if bends(ph.carrier)=>b[col],_=>psi[col]};
                for a in 0..3{u[a][row]+=phase*matrices[(ph.carrier*3+a)*size+ph.to*n0+col]*state;}
                if let Some(w3)=&w3{for a in 0..3{u[a][row]+=phase*w3[ph.carrier][a*size+ph.to*n0+col]*psi[col];}}
            }
            for &(i,to,w)in &class.direct{
                let row=target.position(to).unwrap();let phase=Complex::cis(w*t)*psi[i];
                for a in 0..3{u[a][row]-=phase*self.photon(cluster,&p.stencil,i-n0,a,to);}
            }
            u}).collect();
        let mut state=[Complex::ZERO;MOST_MIXED];state[..n].copy_from_slice(&psi[..n]);
        Photons{model:self,cluster,point:*p,psi:state,time:t,jumps,doppler,matrices,exact,amplitude}
    }

    /// A cluster's generator against its jumps at a point and time: D = i(H − H†) − Σ L†L over its classes'
    /// photons, which a Lindblad generator keeps positive semidefinite -- its no-jump decay at least its
    /// jumps' rate in every state -- at rest: the decay holds the photons' first order in the Doppler and recoil shifts,
    /// not its square (of second order, its own estimate). At fourth order Σ L̃†L̃ to that order, the decay's
    /// (ground.rs, # Fourth order): L̃ = L + δ, δ the fourth order's corrections, less δ†δ, the sixth order
    /// within the fold's charge. D's smallest eigenvalue, Σ L†L's largest and D's round-off scale, the largest of
    /// i(H − H†)'s and Σ L†L's eigenvalues and of H's entries with the scalar the generator leaves out as a phase: the
    /// tables' entries carry it and their rounding with it, while i(H − H†) cancels it (rad/us). (Without it a lone
    /// state, its H nothing but the rest of its entry's rounding, 3e-20 of 70 rad/us under F9's crossed far beams, read
    /// as a gain beyond its own round-off.)
    pub fn decay_balance(&self,cluster:usize,p:&Point,t:f64)->(f64,f64,f64){let b=self.balance(cluster,p,t);(b.low,b.rate,b.scale)}
    /// `decay_balance` with D and the jumps' whole Σ L̃†L̃ (`Balance`).
    fn balance(&self,cluster:usize,p:&Point,t:f64)->Balance{
        let mut p=*p;for f in p.fields.iter_mut(){f.doppler=[Complex::ZERO;MOST_RANK];f.doppler_grad=[[Complex::ZERO;3];MOST_RANK];}p.shifted=false;let p=&p;
        let n=self.dimension(cluster);let generator=self.generator(cluster,p,t);let h=generator.hamiltonian;let largest=h.iter().map(|z|z.abs()).fold(0.,f64::max)+generator.scalar.abs();
        let decay=Dense::from_fn(n,n,|a,b|(h[a*n+b]-h[b*n+a].conj())*Complex::I);
        // Column b: the photons of the basis state b.
        let columns=|fourth:bool|->Vec<Vec<[[Complex;MOST_STATES];3]>>{(0..n).map(|b|{let mut psi=[Complex::ZERO;MOST_MIXED];psi[b]=Complex::ONE;std::mem::take(&mut self.photons_to(cluster,p,&psi,t,None,fourth).amplitude)}).collect()};
        let(full,second)=(columns(true),self.layouts[cluster].fourth.as_ref().map(|_|columns(false)));
        let(mut jumps,mut whole)=(Dense::zeros(n,n),Dense::zeros(n,n));
        // A photon lands in its target's ground states.
        for(c,class)in self.ground.clusters[cluster].classes.iter().enumerate(){let nt=self.ground_dimension(class.target);
            for x in 0..3{for a in 0..n{for b in 0..n{for i in 0..nt{
                let z=full[a][c][x][i].conj()*full[b][c][x][i];jumps[(a,b)]+=z;whole[(a,b)]+=z;
                if let Some(l)=&second{let(da,db)=(full[a][c][x][i]-l[a][c][x][i],full[b][c][x][i]-l[b][c][x][i]);jumps[(a,b)]-=da.conj()*db;}
            }}}}}
        let mut d=decay.clone();d.add_scaled(-Complex::ONE,&jumps);
        let extreme=|m:&Dense|hermitian_eigenvalues(m).into_iter().fold((f64::INFINITY,f64::NEG_INFINITY),|(lo,hi),x|(lo.min(x),hi.max(x)));
        let((low,_),(_,rate),(g0,g1))=(extreme(&d),extreme(&jumps),extreme(&decay));
        Balance{d,jumps:whole,low,rate,scale:rate.abs().max(g0.abs()).max(g1.abs()).max(largest)}
    }
    /// A Lindblad generator where the atoms go, or why not (`balance` at rest, at the focus and the reach's
    /// points -- `reach`, the ends of the trap's principal axes --, at t = 0 and at a generic instant `GENERIC`,
    /// the beats' phases), the tables at their node nearest each point with the point's fields: there they are
    /// the elimination's own, while between nodes D carries their interpolation's error too, the tables' booked
    /// one. The eliminated generator is a Lindblad one by construction at either order (ground.rs: the decay its
    /// jumps' product, the folded photons' cross terms and the fourth order's alike, every frame and beat an exact
    /// combination of the carriers' detunings), so D = i(H − H†) − Σ L̃†L̃ is zero to round-off at a node: below
    /// its round-off it gains -- a decay no jump carries, the norm growing and the jumps stopping (the validation's
    /// F10 before the near light's dressing; b1/109's fourth order before the folded crosses met its W3, 2.4e-3
    /// rad/us) --, a defect of the model, not an approximation the ledger could take: it is refused
    /// (`NOT_LINDBLAD`), with its relative error, the largest ε with −D ≤ ε (Σ L̃†L̃ + 1/T) beyond D's round-off
    /// (`excess`), and the fold's ladder (solve.rs) passes over it -- the solve stops where it is the ladder's first
    /// rung, every drive folded. Between nodes the interpolation's D the solve meets along its pilot's states is
    /// booked (solve.rs `gain`). So is a gain within what the norm it grows over the run can show, the exponentials' share of the population target: a beat's own rounding (a frame's sum of detunings, ε times hundreds of
    /// THz) parts the decay's phases from the jumps' as t goes on -- 1e-12 to 1e-9 of the jump rate at the generic instant, none at t = 0, smooth in the beams' amplitudes -- and that alone refused a D1 scene of three
    /// colours at 12.7 G at twice the round-off here, and a neighbour of it at 575 times, with a norm of 1e-7 over its 100 us.
    fn lindblad(&self)->Result<(),String>{
        let last=(self.tables.nodes-1).max(1) as f64;
        for &r in &self.reach{
            let mut p=self.point(r,[0.;3]);
            let s:Vec<f64>=(0..p.stencil.dimensions).map(|f|{let w=&p.stencil.weights[f];
                let k=(0..4).max_by(|&a,&b|w[a].abs().total_cmp(&w[b].abs())).unwrap();(p.stencil.first[f]+k) as f64/last}).collect();
            p.stencil=self.stencil(&s);
            for t in [0.,GENERIC]{for k in 0..self.clusters(){
                let b=self.balance(k,&p,t);let round=round_off(self.dimension(k) as f64*b.scale);
                // Its round-off, or what the norm it grows over the run cannot show.
                let limit=if self.ground.duration>0.{round.max(NUMERICAL_SHARE*POPULATION_TARGET/self.ground.duration)}else{round};
                if -b.low<=limit{continue;}
                let floor=if self.ground.duration>0.{1./self.ground.duration}else{b.rate.max(round)};
                let error=excess(&b.d,&b.jumps,round,floor);
                return Err(format!("{NOT_LINDBLAD}：子空间 {k} 在 r = ({:.2}, {:.2}, {:.2}) μm、t = {t:.3} μs 处，无跳跃的衰减低于跳跃率（i(H − H†) − ΣL†L 的最小本征值 {:.2e} rad/μs，超出舍入误差 {round:.1e} rad/μs；最大跳跃率 {:.2e} rad/μs，相对误差 {:.2e}）——这部分衰减没有跳跃承载，轨迹的范数会增长",
                    r[0],r[1],r[2],b.low,b.rate,error));
            }}
        }
        Ok(())
    }
    /// The 32 populations (request order) of a cluster's state ψ at a point and time: the bare
    /// ground ones, ρ − ½{Y, ρ}, and the excited ones, Σ |Σ_coherent e^{−iε̃t} (R ψ)_i|² and ρ_ee of
    /// the explicit ones, for ρ = ψψ†/‖ψ‖² (ground.rs's readout).
    pub fn populations(&self,cluster:usize,p:&Point,psi:&[Complex],t:f64)->[f64;STATES]{self.readout(cluster,p,psi,t).0}
    /// The populations and the folded excited population among them (the elimination's η²).
    pub fn readout(&self,cluster:usize,p:&Point,psi:&[Complex],t:f64)->([f64;STATES],f64){
        let g=&self.ground;let cl=&g.clusters[cluster];let layout=&self.layouts[cluster];let(n,n0,ne)=(layout.n,layout.n0,g.excited_states.len());
        let norm:f64=psi[..n].iter().map(|z|z.norm_sqr()).sum();
        let mut out=[0.;STATES];let mut y=[Complex::ZERO;MOST_STATES];let mut folded=0.;
        // Per carrier its response at the point, R_c = Σ_k E_k T_k (ne × n0).
        let responses=self.responses(cluster,p);
        for group in &cl.coherent{
            let mut e=vec![Complex::ZERO;ne];
            for &(x,w)in group{let col=cl.position(x.state).unwrap();let phase=Complex::cis(-w*t);for i in 0..ne{e[i]+=phase*responses[x.carrier][i*n0+col]*psi[col];}}
            for &(x,w)in group{let col=cl.position(x.state).unwrap();let phase=Complex::cis(w*t);
                for i in 0..ne{y[col]+=phase*responses[x.carrier][i*n0+col].conj()*e[i];}}
            for(i,&state)in g.excited_states.iter().enumerate(){let q=e[i].norm_sqr()/norm;out[state]+=q;folded+=q;}
        }
        for(x,&b)in cl.states.iter().enumerate(){out[g.ground_states[b]]=(psi[x].norm_sqr()-(psi[x].conj()*y[x]).re)/norm;}
        for(i,&e)in cl.excited.iter().enumerate(){out[g.excited_states[e]]+=psi[n0+i].norm_sqr()/norm;}
        (out,folded)
    }
    /// Per carrier its folded response at a point, R_c = Σ_k E_k T_k (ne × the cluster's n0 ground
    /// states, row-major).
    pub fn responses(&self,cluster:usize,p:&Point)->Vec<Vec<Complex>>{
        let layout=&self.layouts[cluster];let(n0,ne)=(layout.n0,self.ground.excited_states.len());
        layout.carriers.iter().enumerate().map(|(c,cl2)|{
            let mut rc=vec![Complex::ZERO;ne*n0];let mut buf=vec![Complex::ZERO;ne*n0];
            for k in 0..self.carriers[c].rank{self.interpolate(cluster,&p.stencil,None,cl2.t+k*ne*n0,&mut buf);for(o,x)in rc.iter_mut().zip(&buf){*o+=p.fields[c].e[k]**x;}}
            rc}).collect()
    }

    /// The post-jump transient's amplitudes of a cluster at a point and time (engine.rs, # The transient): per coherent
    /// group of channels X = −Σ e^{−iε̃t} R_c|b⟩, the folded amplitude of its ground states (ne × n0).
    pub fn transient(&self,cluster:usize,p:&Point,t:f64)->Transient{
        let g=&self.ground;let cl=&g.clusters[cluster];let(n0,ne)=(self.ground_dimension(cluster),g.excited_states.len());
        let responses=self.responses(cluster,p);
        let groups=cl.coherent.iter().map(|group|{let mut x=vec![Complex::ZERO;ne*n0];
            for &(ch,w)in group{let col=cl.position(ch.state).unwrap();let phase=-Complex::cis(-w*t);for i in 0..ne{x[i*n0+col]+=phase*responses[ch.carrier][i*n0+col];}}
            x}).collect();
        Transient{n0,ne,groups}
    }
    /// The transient's amplitudes of a ground state ψ′: per coherent group v = X ψ′ on every excited state (`transient`), through the carriers' responses at the point (`responses`).
    pub fn ringing_amplitudes(&self,cluster:usize,responses:&[Vec<Complex>],psi:&[Complex],t:f64)->Vec<Vec<Complex>>{
        let g=&self.ground;let cl=&g.clusters[cluster];let(n0,ne)=(self.ground_dimension(cluster),g.excited_states.len());let mut rotation=Rotation::new(t);
        cl.coherent.iter().map(|group|{let mut v=vec![Complex::ZERO;ne];
            for &(ch,w)in group{let col=cl.position(ch.state).unwrap();let z=-(rotation.at(-w)*psi[col]);let r=&responses[ch.carrier];for i in 0..ne{v[i]+=z*r[i*n0+col];}}
            v}).collect()
    }
    /// A bound of the ringing's probability p = Σ ‖X ψ′‖² from the columns' norms (`CarrierLayout`), Σ_groups (Σ_channels |ψ′_b| Σ_k |E_k| ‖T_k[:,b]‖)²: only a draw
    /// within it asks for the amplitudes (engine.rs `transient`).
    pub fn ringing_bound(&self,cluster:usize,p:&Point,psi:&[Complex])->f64{
        let cl=&self.ground.clusters[cluster];let layout=&self.layouts[cluster];let n0=layout.n0;let mut norms=[[Complex::ZERO;MOST_RANK*MOST_STATES];MOST_CARRIERS];
        for(c,carrier)in layout.carriers.iter().enumerate(){let rank=self.carriers[c].rank;self.interpolate(cluster,&p.stencil,None,carrier.norm,&mut norms[c][..rank*n0]);}
        cl.coherent.iter().map(|group|{let mut s=0.;
            for &(ch,_)in group{let col=cl.position(ch.state).unwrap();let a=psi[col].abs();
                for k in 0..self.carriers[ch.carrier].rank{s+=a*p.fields[ch.carrier].e[k].abs()*norms[ch.carrier][k*n0+col].re;}}
            s*s}).sum()
    }
    /// How a folded excited amplitude rings after a jump into a cluster at r (engine.rs, # The transient), on one excited manifold (ground.rs `excited_groups`):
    /// the eigenstates of H's block there, decoupled from the cluster's explicit states as its folded responses are, as far as a coupling moves a photon's amplitude
    /// over the ringing's width beyond the numerical share.
    pub fn ringing(&self,cluster:usize,r:[f64;3],manifold:usize)->Ringing{
        let g=&self.ground;let block=g.ringing_block(&self.intensities_at_point(r),cluster);let levels=&g.excited_groups[manifold];
        let width=levels.iter().map(|&i|g.decay[(i,i)].re).sum::<f64>()/levels.len() as f64;
        let(energies,vectors)=hermitian_eigen_within(&Dense::from_fn(levels.len(),levels.len(),|a,b|block[(levels[a],levels[b])]),width,RINGING_SHARE);
        Ringing{levels:levels.clone(),energies,vectors,width}
    }
    /// The photon's operators of one manifold's ringing τ after a jump at t (engine.rs, # The transient), per axis
    /// (every ground state × every excited state, its other manifolds' columns zero): C_a e^{−iHτ} on the manifold, each row
    /// at its cluster's frame at t + τ, relative to its cluster's first state.
    pub fn ringing_photon(&self,ring:&Ringing,tau:f64,t:f64)->[Dense;3]{
        let g=&self.ground;let(ng,ne,m)=(g.ground_states.len(),g.excited_states.len(),ring.levels.len());
        let phases:Vec<Complex>=ring.energies.iter().map(|&e|Complex::cis(-e*tau)).collect();
        let frame:Vec<Complex>=(0..ng).map(|to|{let cl=&g.clusters[g.cluster[to]];Complex::cis((cl.frames[cl.position(to).unwrap()]-cl.frames[0])*(t+tau))}).collect();
        std::array::from_fn(|a|{let mut out=Dense::zeros(ng,ne);let mut row=vec![Complex::ZERO;m];
            for to in 0..ng{
                row.fill(Complex::ZERO);let mut reached=false;
                for(j,&i)in ring.levels.iter().enumerate(){let c=g.emission[a][(to,i)];if c==Complex::ZERO{continue;}reached=true;for l in 0..m{row[l]+=c*ring.vectors[(j,l)];}}
                if !reached{continue;}
                for l in 0..m{row[l]=row[l]*phases[l];}
                for(j,&i)in ring.levels.iter().enumerate(){out[(to,i)]=frame[to]*(0..m).fold(Complex::ZERO,|s,l|s+row[l]*ring.vectors[(j,l)].conj());}
            }
            out})
    }
    /// The ringing photon's landing from a density σ on a cluster's ground states, dense (examples/frozen.rs, its ensemble's mean): per cluster of
    /// ground states Σ_a C_a Z C_a† over the manifolds and the coherent groups, Z the ringing's whole, ∫ e^{−iKτ} X σ X† e^{iK†τ} dτ, each pair of eigenstates' (and of landing states' frames') coherence Γ/(Γ + iδ).
    pub fn ringing_landing(&self,transient:&Transient,rings:&[Ringing],sigma:&Dense,t:f64)->Vec<Dense>{
        let g=&self.ground;let(n0,ne)=(transient.n0,transient.ne);let mut out:Vec<Dense>=g.clusters.iter().map(|c|Dense::zeros(c.states.len(),c.states.len())).collect();
        for x in &transient.groups{
            let x=Dense::from_fn(ne,n0,|i,b|x[i*n0+b]);let rho=x.product(sigma).product(&x.adjoint());
            for ring in rings{
                let m=ring.levels.len();
                // ρ on the manifold's eigenstates, the photon's amplitude per axis, landing state and eigenstate.
                let block=Dense::from_fn(m,m,|a,b|rho[(ring.levels[a],ring.levels[b])]);let turned=ring.vectors.adjoint().product(&block).product(&ring.vectors);
                let photon:[Dense;3]=std::array::from_fn(|a|Dense::from_fn(g.ground_states.len(),m,|to,l|ring.levels.iter().enumerate().fold(Complex::ZERO,|s,(j,&i)|s+g.emission[a][(to,i)]*ring.vectors[(j,l)])));
                for(c,cluster)in g.clusters.iter().enumerate(){for(r,&to)in cluster.states.iter().enumerate(){for(q,&from)in cluster.states.iter().enumerate(){
                    let apart=cluster.frames[r]-cluster.frames[q];let mut z=Complex::ZERO;
                    for w in &photon{for j in 0..m{for k in 0..m{z+=w[(to,j)]*turned[(j,k)]*w[(from,k)].conj()/Complex::new(ring.width,ring.energies[j]-ring.energies[k]-apart);}}}
                    out[c][(r,q)]+=z*Complex::cis(apart*t);
                }}}
            }
        }
        out
    }
    /// A jump's map on the bare density σ it leaves on cluster b at r and t, dense: per cluster its image, the mean of the engine's transient (engine.rs, # The transient), `q` and
    /// `rings` the cluster's there (`transient`, `ringing`), `slow` its Π at fourth order. The ringing's landing (`ringing_landing`) and at fourth order what the engine's delay stands for, the
    /// interference −Σ_a C_a (a v† + v a†) C_a†, a = i R2 ψ′, with the slow part Π σ Π† and the trace's balance ½{1 − Π†Π − Y′, σ}, Y′ the landing's trace (O(S²)); else what keeps the trace
    /// linearly, σ − ½{Y, σ}, where the engine's atom stays in its state.
    pub fn after_jump(&self,e:&ground::Elimination,r:[f64;3],b:usize,sigma:&Dense,q:&Transient,rings:&[Ringing],slow:Option<&Dense>,t:f64)->Vec<Dense>{
        let g=&self.ground;let cluster=&g.clusters[b];let(nt,ne)=(q.n0,q.ne);let mut out=self.ringing_landing(q,rings,sigma,t);
        let square=|x:&[Complex],rows:usize,cols:usize|Dense::from_fn(rows,cols,|i,j|x[i*cols+j]);
        let mut y=square(&q.admixture(),nt,nt);
        let mut kept=match slow{
            Some(slow)=>{
                let a=g.amplitudes_at(r);
                for(x,group)in q.groups.iter().zip(&cluster.coherent){
                    let x=square(x,ne,nt);let mut x2=Dense::zeros(ne,nt);
                    for &(ch,w)in group{let col=cluster.position(ch.state).unwrap();let phase=Complex::cis(-w*t);for &l in &self.carriers[ch.carrier].beams{for i in 0..ne{x2[(i,col)]+=phase*a[l]*e.derivatives[l][(i,ch.state)];}}}
                    let cross=x2.scaled(Complex::I).product(sigma).product(&x.adjoint());
                    let share=x.adjoint().product(&g.decay).product(&x2).scaled(Complex::I);y.add_scaled(-Complex::ONE,&share);y.add_scaled(-Complex::ONE,&share.adjoint());
                    for c in &g.emission{let m=c.product(&cross).product(&c.adjoint());
                        for(k,cl)in g.clusters.iter().enumerate(){for(i,&to)in cl.states.iter().enumerate(){for(j,&from)in cl.states.iter().enumerate(){out[k][(i,j)]-=m[(to,from)]+m[(from,to)].conj();}}}}
                }
                let mut d=Dense::from_fn(nt,nt,|i,j|if i==j{Complex::ONE}else{Complex::ZERO});d.add_scaled(-Complex::ONE,&slow.adjoint_product(slow));d.add_scaled(-Complex::ONE,&y);
                let mut kept=slow.product(sigma).product(&slow.adjoint());kept.add_scaled(Complex::new(0.5,0.),&d.product(sigma));kept.add_scaled(Complex::new(0.5,0.),&sigma.product(&d));kept
            }
            None=>{let mut kept=sigma.clone();kept.add_scaled(Complex::new(-0.5,0.),&y.product(sigma));kept.add_scaled(Complex::new(-0.5,0.),&sigma.product(&y));kept}
        };
        std::mem::swap(&mut out[b],&mut kept);out[b].add_scaled(Complex::ONE,&kept);
        out
    }
    /// The ringing photon of a coherent group from the ground state ψ′ (engine.rs `ringing_photon`), for the kicks of a
    /// moving atom: on `target`'s ground states, through the photon's operators on `levels` (`ringing_photon`), its amplitude
    /// Σ_a ops_a (Σ e^{−iε̃t} R ψ′), the amplitude's gradient [axis a][lab axis] through the fields and the intensities, and each beam's part of it.
    pub fn transient_photon(&self,cluster:usize,p:&Point,t:f64,coherent:usize,levels:&[usize],target:usize,ops:&[Dense;3],psi:&[Complex])->([[Complex;MOST_STATES];3],[[[Complex;MOST_STATES];3];3],Vec<(usize,[[Complex;MOST_STATES];3])>){
        let g=&self.ground;let cl=&g.clusters[cluster];let layout=&self.layouts[cluster];let(n0,ne)=(layout.n0,g.excited_states.len());
        let group=&cl.coherent[coherent];let target=&g.clusters[target];
        // T_k (and ∂_f T_k) at the point, per carrier and rank.
        let table=|c:usize,k:usize,f:Option<usize>|->Vec<Complex>{let mut buf=vec![Complex::ZERO;ne*n0];self.interpolate(cluster,&p.stencil,f,layout.carriers[c].t+k*ne*n0,&mut buf);buf};
        let tables:Vec<Vec<Vec<Complex>>>=(0..self.carriers.len()).map(|c|(0..self.carriers[c].rank).map(|k|table(c,k,None)).collect()).collect();
        // The photon's amplitude per axis and row of Σ e^{−iε̃t} Σ_k w_k M_k[i, b] ψ_b on the levels, over the group's channels (the jumps' sign, ground.rs), M the tables or their slopes.
        let photon=|weights:&dyn Fn(usize,usize)->Complex,matrices:&[Vec<Vec<Complex>>]|->[[Complex;MOST_STATES];3]{
            let mut folded=vec![Complex::ZERO;levels.len()];
            for &(ch,w)in group{let col=cl.position(ch.state).unwrap();let phase=Complex::cis(-w*t)*psi[col];
                for k in 0..self.carriers[ch.carrier].rank{let wk=weights(ch.carrier,k);if wk==Complex::ZERO{continue;}let z=phase*wk;let m=&matrices[ch.carrier][k];
                    for(o,&i)in folded.iter_mut().zip(levels){*o+=z*m[i*n0+col];}}}
            std::array::from_fn(|a|{let mut u=[Complex::ZERO;MOST_STATES];
                for(row,&to)in target.states.iter().enumerate(){for(o,&i)in folded.iter().zip(levels){u[row]+=ops[a][(to,i)]**o;}}u})};
        let u=photon(&|c,k|p.fields[c].e[k],&tables);
        let mut du=[[[Complex::ZERO;MOST_STATES];3];3];
        let slopes:Vec<Vec<Vec<Vec<Complex>>>>=(0..self.focused.len()).map(|f|(0..self.carriers.len()).map(|c|(0..self.carriers[c].rank).map(|k|table(c,k,Some(f))).collect()).collect()).collect();
        for x in 0..3{
            let field=photon(&|c,k|p.fields[c].grad[k][x],&tables);
            for a in 0..3{for r in 0..MOST_STATES{du[a][x][r]+=field[a][r];}}
            for(f,slope)in slopes.iter().enumerate(){
                let part=photon(&|c,k|p.fields[c].e[k]*p.intensity_gradient[f][x],slope);
                for a in 0..3{for r in 0..MOST_STATES{du[a][x][r]+=part[a][r];}}
            }
        }
        let beams:Vec<(usize,[[Complex;MOST_STATES];3])>={
            let mut carriers:Vec<usize>=group.iter().map(|x|x.0.carrier).collect();carriers.sort();carriers.dedup();
            carriers.iter().flat_map(|&c|self.carriers[c].beams.iter().enumerate().map(move|(j,&l)|(c,j,l))).map(|(c,j,l)|{
                let amplitude=self.geometry[l].field(p.r).0;let coefficients=self.carriers[c].coefficients[j];
                (l,photon(&|d,k|if d==c{amplitude*coefficients[k]}else{Complex::ZERO},&tables))}).collect()
        };
        (u,du,beams)
    }

    /// The near beams' field products ā_l a_m at r (`level`): a beam's own intensity, the interference of two.
    fn products(&self,r:[f64;3])->impl Fn(usize,usize)->Complex+'_{
        move|l,m|if l==m{Complex::new(self.geometry[l].intensity(r),0.)}else{self.geometry[l].field(r).0.conj()*self.geometry[m].field(r).0}
    }
    /// A cluster state's potential at r relative to the trap's centre `centre` (rad/us): the diagonal of h0 (the units'
    /// intensities there, a standing wave's in it) and of every near beam's own light shift,
    /// |a_l(r)|² Σ c_lk′* c_lk S_k′k -- the near beams' interference (their lattice) left out, but for a unit of the trap's light's (`level`), its standing wave whether it folds or is a near carrier.
    pub fn potential(&self,cluster:usize,x:usize,r:[f64;3],centre:[f64;3])->f64{
        self.level(cluster,x,&|f|self.axis_value(f,r),&self.products(r))-self.level(cluster,x,&|f|self.axis_value(f,centre),&self.products(centre))
    }
    /// A cluster state's trap depth (rad/us): the level its potential tends to far from every
    /// focused beam -- the Gaussian beams' amplitudes zero, the plane waves' as at the focus --,
    /// relative to the trap's centre; positive for a state the trap holds.
    pub fn depth(&self,cluster:usize,x:usize,centre:[f64;3])->f64{
        let far=|l:usize|if self.geometry[l].waist.is_some(){Complex::ZERO}else{self.geometry[l].field([0.;3]).0};
        let units=|f:usize|self.axis_with(f,far);
        self.level(cluster,x,&units,&|l,m|if l==m{Complex::new(far(l).norm_sqr(),0.)}else{far(l).conj()*far(m)})-self.level(cluster,x,&|f|self.axis_value(f,centre),&self.products(centre))
    }
    /// A cluster state's potential in the trap's light at r (rad/us): `level` with every beam at r less the same with
    /// the trap's light off (`trap_light`), the near light off in both -- the shifts of the trap's beams and of every
    /// far focused beam there, zero where those beams' fields are, −D_T at the trap's centre (ensemble.rs: the initial
    /// ensemble's energies). The near light comes on at t = 0: its fold's shift, whose denominator a focused trap's
    /// shift can close on a shell about the focus (PGCP_10's probe in its all-folded model: 137 k_BT there against the
    /// focus's 34), is no part of what holds the atoms before it. Far from a resonance, with a trap beam alone, the
    /// loss's trap potential (`trap_potential`) without the near light, and the trap's potential relative to its far
    /// level (`potential`, `depth`).
    pub fn trap_level(&self,cluster:usize,x:usize,r:[f64;3])->f64{
        let off=|f:usize|self.axis_with(f,|l|if self.trap_light.contains(&l){Complex::ZERO}else{self.geometry[l].field(r).0});
        let near=self.products(r);let beams=|l:usize,m:usize|if self.trap_light.contains(&l){near(l,m)}else{Complex::ZERO};
        self.level(cluster,x,&|f|self.axis_value(f,r),&beams)-self.level(cluster,x,&off,&|_,_|Complex::ZERO)
    }
    /// The envelope's modes (ensemble.rs): one beam of each mode -- waist, wave number and axis either way, the same
    /// intensity profile -- of the trap's light in the focused units, the trap beam's first.
    pub fn trap_modes(&self)->Vec<usize>{
        let same=|a:&Geometry,b:&Geometry|{let c:f64=(0..3).map(|i|a.direction[i]*b.direction[i]).sum();a.waist==b.waist&&a.wave_number==b.wave_number&&(c.abs()-1.).abs()<=1e-12};
        let mut modes=vec![self.trap_beam];
        for &u in &self.focused{for &l in &self.ground.units[u].beams{
            if self.trap_light.contains(&l)&&!modes.iter().any(|&m|same(&self.geometry[m],&self.geometry[l])){modes.push(l);}}}
        modes
    }
    /// The trap's largest depth per unit of its light's intensity (`trap_level`), relative to `depth`, its depth at
    /// its centre: a bound of c = sup −V_T/(D_T L) over the intensities of the units of the trap's light in (0, 1], L
    /// their mean weighted by each one's share of the depth at the centre, every other focused unit's in [0, 1]. Every
    /// point's intensities lie in that set and a
    /// field's intensity lies below B_u times the largest of its beams' profiles A_l(r) (`Unit::reach`: B = 1 where its beams' fields are
    /// parallel, 1/2 for two crossed at right angles), so −V_T(r) ≤ c D_T L ≤ c D_T max_u B_u max_l A_l(r) everywhere: the initial ensemble's
    /// envelope (ensemble.rs), c max_u B_u returned.
    ///
    /// Proven, not sampled: on each cell of the tables' grid in the units' intensities `level` is a polynomial -- `cubic`'s weights in
    /// each unit's intensity, times its s where the trap's own near beams scale with it: quartic in those, cubic in the
    /// others' --, −V_T too, zero where the trap's light is off. A polynomial lies below the largest of its Bernstein
    /// coefficients on the cell and L is linear, so −V_T ≤ R L on the cell wherever each coefficient of −V_T is at most
    /// R times L at its control point: the cell's R, from −V_T at the cell's equally spaced points (to the evaluations'
    /// round-off). A trap with no unit of its own (its shift linear in its intensity) takes L = 1. An error where a
    /// unit of the trap's light holds a beam not of it: its interference with the trap's field falls off as √A, and
    /// the trap's bound states fill an unbounded phase space.
    pub fn trap_slope(&self,cluster:usize,x:usize,depth:f64,centre:[f64;3])->Result<f64,String>{
        let g=&self.ground;let trapped=|l:&usize|self.trap_light.contains(l);
        let traps:Vec<usize>=(0..self.axes.len()).filter(|&f|self.axes[f].coordinate==0&&g.units[self.focused[f]].beams.iter().any(trapped)).collect();let traps=&traps[..];
        if traps.iter().any(|&f|!g.units[self.focused[f]].beams.iter().all(trapped)){return Err("光镊与同频率、同偏振的其他光束相干叠加：其干涉项随光镊强度的平方根衰减，束缚的初始热分布无界".into());}
        let others:Vec<usize>=(0..self.axes.len()).filter(|f|!traps.contains(f)).collect();let others=&others[..];
        let t=traps.len();
        // −V_T D_T at intensities q (the trap's units', then the other units').
        let at=|q:&[f64]|->f64{
            let units=|on:bool|move|f:usize|match traps.iter().position(|&v|v==f){Some(i)=>if on{q[i]}else{0.},None=>others.iter().position(|&v|v==f).map_or(0.,|i|q[t+i])};
            // A near beam's intensity with its unit's, and two of a unit's interference there: their fields in phase, every beam's amplitude its unit's, whose summed field's intensity it is.
            let beams=|on:bool|move|l:usize,m:usize|->Complex{if !(on&&trapped(&l)){return Complex::ZERO;}
                let s=traps.iter().position(|&f|g.units[self.focused[f]].beams.contains(&l)).map_or(1.,|i|q[i]);if l==m{return Complex::new(s,0.);}
                let unit=&g.units[g.unit_of(l)];let weight=|b:usize|unit.weights[unit.beams.iter().position(|&y|y==b).unwrap()];let(a,b)=(weight(l),weight(m));
                a*b.conj()*(s/(a.abs()*b.abs()))};
            self.level(cluster,x,&units(false),&beams(false))-self.level(cluster,x,&units(true),&beams(true))};
        // L's weights: each trap field's share of the depth at the centre (none for a repulsive one); L their intensities, each at most the
        // largest of its beams' profiles (|E|²/N ≤ max_l |a_l|²).
        let there:Vec<f64>=traps.iter().chain(others).map(|&f|self.axis_value(f,centre).min(1.)).collect();
        let shares:Vec<f64>=(0..t).map(|i|{let mut q=there.clone();q[i]=0.;(at(&there)-at(&q)).max(0.)}).collect();let whole:f64=shares.iter().sum();
        let weights:Vec<f64>=shares.iter().map(|s|if whole>0.{s/whole}else{1./t as f64}).collect();
        // The dimensions: the trap's units' (quartic), the other units' (cubic). Each one's equally spaced points and
        // the inverse of Bernstein's basis there (Gauss–Jordan).
        let degrees:Vec<usize>=(0..t+others.len()).map(|i|if i<t{4}else{3}).collect();
        let inverse=|m:usize|->Vec<Vec<f64>>{
            let binomial=|j:usize|(0..j).fold(1.,|b,i|b*(m-i) as f64/(i+1) as f64);
            let mut a:Vec<Vec<f64>>=(0..=m).map(|i|{let t=i as f64/m as f64;(0..=m).map(|j|binomial(j)*t.powi(j as i32)*(1.-t).powi((m-j) as i32)).chain((0..=m).map(|k|if k==i{1.}else{0.})).collect()}).collect();
            for c in 0..=m{let p=(c..=m).max_by(|&x,&y|a[x][c].abs().total_cmp(&a[y][c].abs())).unwrap();a.swap(c,p);let v=a[c][c];for e in a[c].iter_mut(){*e/=v;}
                for r in 0..=m{if r!=c{let f=a[r][c];let row=a[c].clone();for(e,q)in a[r].iter_mut().zip(row){*e-=f*q;}}}}
            a.into_iter().map(|row|row[m+1..].to_vec()).collect()};
        let inverses:Vec<Vec<Vec<f64>>>=degrees.iter().map(|&m|inverse(m)).collect();
        let n=self.tables.nodes;let h=1./(n-1) as f64;let cells=n-1;let d=degrees.len();
        let sizes:Vec<usize>=degrees.iter().map(|m|m+1).collect();let size:usize=sizes.iter().product();
        let bounds:Vec<f64>=(0..cells.pow(d as u32)).into_par_iter().map(|c|{
            let corner=|i:usize|((c/cells.pow(i as u32))%cells) as f64*h;
            // The cell's points: index p ↦ per dimension j_i of its degree m_i, at corner + h j_i/m_i.
            let digit=|p:usize,i:usize|(p/sizes[..i].iter().product::<usize>())%sizes[i];
            let place=|p:usize,i:usize|corner(i)+h*digit(p,i) as f64/degrees[i] as f64;
            let mut v:Vec<f64>=(0..size).map(|p|{let q:Vec<f64>=(0..d).map(|i|place(p,i)).collect();at(&q)}).collect();
            for i in 0..d{let stride:usize=sizes[..i].iter().product();
                for p in 0..size{if digit(p,i)!=0{continue;}
                    let line:Vec<f64>=(0..sizes[i]).map(|k|v[p+k*stride]).collect();
                    for k in 0..sizes[i]{v[p+k*stride]=inverses[i][k].iter().zip(&line).map(|(a,b)|a*b).sum();}}}
            // The cell's R: the largest coefficient over L at its control point (none where L is 0: the trap's light off,
            // where −V_T is 0); L = 1 throughout without a unit.
            (0..size).filter_map(|p|{let l:f64=if t>0{(0..t).map(|i|weights[i]*place(p,i)).sum()}else{1.};(l>0.).then(||v[p]/l)}).fold(f64::NEG_INFINITY,f64::max)
        }).collect();
        let reach=if t>0{traps.iter().map(|&f|g.units[self.focused[f]].reach()).fold(0_f64,f64::max)}else{1.};
        Ok(bounds.into_iter().fold(f64::NEG_INFINITY,f64::max)/depth*reach)
    }
    /// The diagonal of h0 and of the near light's shift for a cluster state, the units at
    /// the intensities `units(u)` and the near beams' fields' products ā_l a_m at `beams(l, m)` (`potential`): the groups' S at the
    /// state's diagonal entry (none where it is not one of a group's), E_k′* E_k with E_k = Σ_l a_l c_lk: each beam's own shift, and the interference of a unit of the trap's light (`Carrier::interfering`).
    fn level(&self,cluster:usize,x:usize,units:&dyn Fn(usize)->f64,beams:&dyn Fn(usize,usize)->Complex)->f64{
        let layout=&self.layouts[cluster];let n=layout.n;
        let s:Vec<f64>=(0..self.axes.len()).map(|f|units(f).min(1.)).collect();let stencil=self.stencil(&s);
        let mut entry=[Complex::ZERO];self.interpolate(cluster,&stencil,None,layout.h0+x*n+x,&mut entry);let mut u=entry[0].re;
        for gl in layout.groups.iter().filter(|g|g.hermitian){
            let Ok(diagonal)=gl.entries.binary_search(&(x,x)) else{continue};
            let carrier=&self.carriers[gl.absorbed];let mut value=[Complex::ZERO];
            for(j,&l)in carrier.beams.iter().enumerate(){
                let at=beams(l,l).re;if at==0.{continue;}let coefficient=&carrier.coefficients[j];
                for(pi,&(k1,k))in gl.pairs.iter().enumerate(){
                    self.interpolate(cluster,&stencil,None,gl.s+pi*gl.entries.len()+diagonal,&mut value);
                    u-=at*(coefficient[k1].conj()*coefficient[k]*value[0]).re*if k1<k{2.}else{1.};
                }
            }
            for &(j,m)in &carrier.interfering{
                let product=beams(carrier.beams[j],carrier.beams[m]);if product==Complex::ZERO{continue;}let(a,b)=(&carrier.coefficients[j],&carrier.coefficients[m]);
                for(pi,&(k1,k))in gl.pairs.iter().enumerate(){
                    self.interpolate(cluster,&stencil,None,gl.s+pi*gl.entries.len()+diagonal,&mut value);
                    u-=(product*a[k1].conj()*b[k]*value[0]).re*if k1<k{2.}else{1.};
                }
            }
        }
        u
    }
}

/// The exact responses of every near beam at a point for a segment beyond the first order's
/// allowance (engine.rs): R_l(w_l) at the point's elimination.
#[derive(Clone,Debug)]
pub struct Exact{pub responses:Vec<Dense>}
impl Exact{
    /// The exact responses at a point and momentum.
    pub fn new(model:&Model,r:[f64;3],kappa:[f64;3])->Result<Exact,String>{
        let g=&model.ground;let e=g.eliminate(&model.intensities_at_point(r))?;
        let mut responses=vec![Dense::zeros(g.excited_states.len(),g.ground_states.len());g.beams.len()];
        for c in &g.carriers{for &l in &c.beams{responses[l]=g.response(&e,l,model.kinetic_shift(l,kappa))?;}}
        Ok(Exact{responses})
    }
}
impl Model{
    /// Every unit's intensity at a point as the tables take it: the focused ones there, the others at
    /// the focus.
    pub fn intensities_at_point(&self,r:[f64;3])->Vec<f64>{
        let s:Vec<f64>=(0..self.axes.len()).map(|f|self.axis_value(f,r).min(1.)).collect();self.unit_intensities(&s)
    }
}

/// A cluster's post-jump transient amplitudes (`Model::transient`): per coherent group of channels X (ne × n0, row-major).
#[derive(Clone,Debug)]
pub struct Transient{pub n0:usize,pub ne:usize,pub groups:Vec<Vec<Complex>>}
impl Transient{
    /// Y = Σ X†X, the readout's admixture (n0 × n0).
    pub fn admixture(&self)->Vec<Complex>{
        let(n0,ne)=(self.n0,self.ne);let mut y=vec![Complex::ZERO;n0*n0];
        for x in &self.groups{for a in 0..n0{for b in 0..n0{y[a*n0+b]+=(0..ne).fold(Complex::ZERO,|s,i|s+x[i*n0+a].conj()*x[i*n0+b]);}}}
        y
    }
}
/// One excited manifold's ringing at a point (`Model::ringing`): its states, the eigenstates of H's block on them, H = U diag(E) U†, and its width, its states' mean decay (they agree to 1e-4 of it).
#[derive(Clone,Debug)]
pub struct Ringing{pub levels:Vec<usize>,pub energies:Vec<f64>,pub vectors:Dense,pub width:f64}
/// A photon's absorbed light by its parts (`Photons::absorption`, engine.rs # Absorption): per near beam of its class's
/// carriers its folded share and its amplitude's projection on the photon, and its explicit states' part -- its share and
/// the beams' weights by their drives into the explicit states.
#[derive(Clone,Debug,Default)]
pub struct Absorption{pub beams:Vec<(usize,f64,f64)>,pub explicit:Option<(f64,Vec<Recoil>)>}
/// One driven transition's logarithmic response gradient. Independent signs on its two quadratures
/// give zero mean and covariance Re[∇ln σ* ⊗ ∇ln σ], σ = a/(δ − iΓ/2), at low saturation.
/// This extends the existing incoherent attribution of an explicit photon; it does not eliminate
/// coherences between ground states or assert a diffusion theorem at arbitrary saturation.
#[derive(Clone,Debug)]
pub struct Recoil{pub phase:[f64;3],pub amplitude:[f64;3],pub weight:f64}
impl Recoil{
    pub fn response(log_field:[Complex;3],delta:f64,half_width:f64,gradient:[f64;3],weight:f64)->Self{
        let inverse=Complex::ONE/Complex::new(delta,-half_width);
        let log:[Complex;3]=std::array::from_fn(|a|log_field[a]-inverse*gradient[a]);
        Self{phase:log.map(|z|z.im),amplitude:log.map(|z|z.re),weight}
    }
}
/// The photon amplitudes of every class of a cluster's jumps at a point: per class and axis the
/// target cluster's amplitude u_a; for a class drawn, its gradient and its absorbed light's parts
/// (`Photons::absorption`) from the jump tables at the point, kept.
pub struct Photons<'a>{
    model:&'a Model,cluster:usize,point:Point,psi:[Complex;MOST_MIXED],time:f64,
    /// Per carrier, rank and axis (carrier-major): J_k and J2_k at the point, ng × n0; the third buffer of `PHOTON_BUFFERS`, per carrier and axis their sum on the fields.
    jumps:Vec<Complex>,doppler:Vec<Complex>,matrices:Vec<Complex>,
    exact:Option<Exact>,
    pub amplitude:Vec<[[Complex;MOST_STATES];3]>,
}
impl Drop for Photons<'_>{
    fn drop(&mut self){let buffers=[std::mem::take(&mut self.jumps),std::mem::take(&mut self.doppler),std::mem::take(&mut self.matrices)];let _=PHOTON_BUFFERS.try_with(|b|b.set(buffers));}
}
thread_local!{
    /// A jump's working rows (`Photons`), kept per thread: a heap's lock and a zeroing per jump.
    static PHOTON_BUFFERS:std::cell::Cell<[Vec<Complex>;3]>=const{std::cell::Cell::new([Vec::new(),Vec::new(),Vec::new()])};
}
impl Photons<'_>{
    /// Where carrier c's rank k and axis a's tables start in `jumps`.
    fn offset(&self,c:usize,k:usize,a:usize)->usize{
        let m=self.model;let size=m.ground.ground_states.len()*m.ground_dimension(self.cluster);
        (m.carriers[..c].iter().map(|x|x.rank).sum::<usize>()*3+k*3+a)*size
    }
    /// A class's absorbed light along a photon's polarization e (`Engine::jump`), in one pass over its
    /// photons: per carrier of the class and target row, A_k = Σ e^{iω̃t} ψ[from] Σ_a e_a J_k[a][to, from],
    /// B_k the same of J2_k and ∂_f A_k its slope in each s_f (the axes' entries combined at
    /// each node of the stencil). Of them, per row the gradient of the class's amplitude along e,
    /// Σ_l ∇a_l z_l + Σ_k Σ_f E_k ∂_f A_k ∇s_f, z_l = Σ_k c_lk (A_k − w_l B_k) -- beyond the first order its exact
    /// response's Σ e^{iω̃t} ψ[from] Σ_a e_a (C_a R_l)[to, from], the intensities' slopes still the rest response's --; and the parts
    /// (`Absorption`) of the photon f along e: per beam l of the class's carriers its share ‖f_l‖² and projection
    /// Re⟨f|f_l⟩/⟨f|f⟩, f_l = a_l z_l per row, and the explicit states' part.
    pub fn absorption(&self,class:usize,e:[f64;3],f:&[Complex;MOST_STATES])->([[Complex;3];MOST_STATES],Absorption){
        if self.model.wide{unsafe{self.absorption_wide(class,e,f)}}else{self.absorption_body(class,e,f)}
    }
    #[target_feature(enable="avx2,fma")]
    unsafe fn absorption_wide(&self,class:usize,e:[f64;3],f:&[Complex;MOST_STATES])->([[Complex;3];MOST_STATES],Absorption){self.absorption_body(class,e,f)}
    #[inline(always)]
    fn absorption_body(&self,class:usize,e:[f64;3],f:&[Complex;MOST_STATES])->([[Complex;3];MOST_STATES],Absorption){
        let m=self.model;let g=&m.ground;let cl=&g.clusters[self.cluster];let class=&cl.classes[class];let target=&g.clusters[class.target];
        let(n0,nt,p,d)=(m.ground_dimension(self.cluster),target.states.len(),&self.point,m.focused.len());
        let(layout,size)=(&m.layouts[self.cluster],g.ground_states.len()*n0);let(data,slice)=(&m.tables.data[self.cluster],2*layout.size);
        let carriers:Vec<usize>={let mut v:Vec<usize>=class.photons.iter().map(|x|x.0.carrier).collect();v.sort();v.dedup();v};
        // Per carrier of the class and row, per rank: A, B and ∂_f A.
        let mut sums=vec![[[Complex::ZERO;2+MOST_FOCUSED];MOST_RANK];carriers.len()*MOST_STATES];let mut rotation=Rotation::new(self.time);
        for &(ph,w)in &class.photons{
            let(row,col)=(target.position(ph.to).unwrap(),cl.position(ph.from).unwrap());let phase=rotation.at(w)*self.psi[col];
            let sum=&mut sums[carriers.binary_search(&ph.carrier).unwrap()*MOST_STATES+row];
            let first=self.offset(ph.carrier,0,0)+ph.to*n0+col;
            for k in 0..m.carriers[ph.carrier].rank{
                let(mut a,mut b)=(Complex::ZERO,Complex::ZERO);let mut slope=[Complex::ZERO;MOST_FOCUSED];
                for x in 0..3{let at=first+(k*3+x)*size;a+=self.jumps[at]*e[x];b+=self.doppler[at]*e[x];}
                if d>0{
                    let place=(k*3,ph.to*n0+col);let cl2=&layout.carriers[ph.carrier];
                    let starts:[usize;3]=std::array::from_fn(|x|2*(cl2.j+(place.0+x)*size+place.1));
                    let along=|base:usize|(0..3).fold(Complex::ZERO,|v,x|v+Complex::new(data[base+starts[x]],data[base+starts[x]+1])*e[x]);
                    if d==1{let(node,dw)=(p.stencil.first[0],p.stencil.derivatives[0]);for i in 0..4{slope[0]+=along((node+i)*slice)*dw[i];}}
                    else{for i in 0..p.stencil.count(){
                        let(node,_,dw)=p.stencil.node(i);
                        let v=along(node*slice);
                        for f in 0..d{slope[f]+=v*dw[f];}
                    }}
                }
                sum[k][0]+=phase*a;sum[k][1]+=phase*b;for f in 0..d{sum[k][2+f]+=phase*slope[f];}
            }
        }
        // The gradient's part through the intensities: the tables' slopes.
        let mut gradient=[[Complex::ZERO;3];MOST_STATES];
        for(ci,&c)in carriers.iter().enumerate(){
            let fl=&p.fields[c];
            for(row,out)in gradient[..nt].iter_mut().enumerate(){
                let sum=&sums[ci*MOST_STATES+row];
                for x in 0..3{for k in 0..m.carriers[c].rank{for f in 0..d{out[x]+=fl.e[k]*sum[k][2+f]*p.intensity_gradient[f][x];}}}
            }
        }
        let size:f64=f[..nt].iter().map(|z|z.norm_sqr()).sum();let on_f=|row:usize,z:Complex|if size>0.{(f[row].conj()*z).re/size}else{0.};
        let mut beams:Vec<(usize,f64,f64)>=vec![];
        for(ci,&c)in carriers.iter().enumerate(){for(j,&l)in m.carriers[c].beams.iter().enumerate(){
            let(amplitude,slope)=m.geometry[l].field(p.r);
            // The beam's part per row without its amplitude, z_l: f_l = a_l z_l, its gradient through the field ∇a_l z_l.
            let rows:[Complex;MOST_STATES]=match &self.exact{
                Some(exact)=>{
                    let product:[Dense;3]=std::array::from_fn(|a|g.emission[a].product(&exact.responses[l]));let mut rows=[Complex::ZERO;MOST_STATES];let mut rotation=Rotation::new(self.time);
                    for &(ph,w)in class.photons.iter().filter(|x|x.0.carrier==c){
                        let(row,col)=(target.position(ph.to).unwrap(),cl.position(ph.from).unwrap());
                        let z=(0..3).fold(Complex::ZERO,|z,a|z+product[a][(ph.to,cl.states[col])]*e[a]);rows[row]+=rotation.at(w)*self.psi[col]*z;
                    }
                    rows
                }
                None=>{let w=m.first_shift(c,l,p.kappa);std::array::from_fn(|row|{let sum=&sums[ci*MOST_STATES+row];
                    (0..m.carriers[c].rank).fold(Complex::ZERO,|z,k|z+m.carriers[c].coefficients[j][k]*(sum[k][0]-sum[k][1]*w))})}
            };
            for row in 0..nt{for x in 0..3{gradient[row][x]+=slope[x]*rows[row];}}
            beams.push((l,rows[..nt].iter().map(|z|(amplitude*(*z)).norm_sqr()).sum(),(0..nt).map(|row|on_f(row,amplitude*rows[row])).sum()));
        }}
        // The explicit states' part of the photon: its weight along e, and the beams' weights: each explicit state's weight
        // in the photon shared by the drives into it, Σ |a_l Σ_k c_lk U_k[e,b] ψ_b|² -- a photon of that state.
        let explicit=(!class.direct.is_empty()).then(||{
            let dimension=m.dimension(self.cluster);let mut parts=vec![[Complex::ZERO;MOST_STATES];dimension];
            for &(i,to,w)in &class.direct{let row=target.position(to).unwrap();let phase=Complex::cis(w*self.time)*self.psi[i];
                for a in 0..3{parts[i][row]+=phase*m.photon(self.cluster,&p.stencil,i-n0,a,to)*e[a];}}
            let total:f64=(0..nt).map(|row|parts.iter().fold(Complex::ZERO,|s,v|s+v[row]).norm_sqr()).sum();
            let drive=|dr:&DriveLayout,j:usize,l:usize|{let carrier=&m.carriers[dr.carrier];((0..carrier.rank).fold(Complex::ZERO,|s,k|s+carrier.coefficients[j][k]*dr.u[k])*m.geometry[l].field(p.r).0*self.psi[dr.col]).norm_sqr()};
            let mut into=vec![0.;dimension];
            for dr in &m.layouts[self.cluster].drives{for(j,&l)in m.carriers[dr.carrier].beams.iter().enumerate(){into[dr.row]+=drive(dr,j,l);}}
            let mut diagonal=vec![(0.,[0.;3]);dimension];
            for(i,out)in diagonal.iter_mut().enumerate(){
                let(mut value,mut slope)=([Complex::ZERO],[Complex::ZERO;MOST_FOCUSED]);
                m.gather_dyn(self.cluster,&p.stencil,layout.h0+i*dimension+i,&mut value,Some((&mut slope,1)));
                *out=(value[0].re,std::array::from_fn(|a|(0..d).map(|f|slope[f].re*p.intensity_gradient[f][a]).sum()));
            }
            let mut driven=vec![];
            for dr in &layout.drives{for &l in &m.carriers[dr.carrier].beams{if !driven.contains(&l){driven.push(l);}}}
            let mut drives=vec![];
            for l in driven{for dr in &layout.drives{
                let Some(j)=m.carriers[dr.carrier].beams.iter().position(|&x|x==l)else{continue};
                let weight:f64=parts[dr.row][..nt].iter().map(|z|z.norm_sqr()).sum();
                let weight=if into[dr.row]>0.{weight*drive(dr,j,l)/into[dr.row]}else{0.};
                if weight<=0.{continue;}
                let(e,b)=(diagonal[dr.row],diagonal[dr.col]);
                let delta=e.0-b.0+dr.residual+m.kinetic_shift(l,p.kappa);
                let half_width=0.5*g.decay[(cl.excited[dr.row-n0],cl.excited[dr.row-n0])].re;
                let(_,log_field)=m.geometry[l].field_log(p.r);
                drives.push(Recoil::response(log_field,delta,half_width,std::array::from_fn(|a|e.1[a]-b.1[a]),weight));
            }}
            (total,drives)});
        (gradient,Absorption{beams,explicit})
    }
}

#[cfg(test)]
mod tests{
    use super::{cubic,Beats,Model,MOST_TABLE_BYTES,MOST_TABLE_NODES,NOT_LINDBLAD};

    #[test]
    fn a_vanishing_drive_has_a_finite_logarithmic_response(){
        let geometry=super::Geometry{direction:[0.,0.,1.],wave_number:8.,waist:Some(1.)};
        let(field,log)=geometry.field_log([1000.,0.,0.]);assert_eq!(field,crate::Complex::ZERO);
        let response=super::Recoil::response(log,0.,19.,[31.,-7.,11.],0.);
        assert!(response.phase.iter().chain(&response.amplitude).all(|x|x.is_finite()));
        assert_eq!(response.amplitude[0],-2000.);
    }

    /// The interpolant (`cubic`) is C¹ -- its derivative, the force the tables give, the same from either side of every
    /// node, the grid's ends included --, reproduces a quadratic exactly, and converges at third order: a smooth
    /// function's largest error falls about eightfold per doubling of the grid.
    #[test]
    fn the_interpolant_is_c1_and_third_order(){
        let at=|n:usize,s:f64,f:&dyn Fn(f64)->f64|{let(first,w,d)=cubic(n,s);let h=1./(n-1) as f64;
            (0..4).fold((0.,0.),|(v,g),k|{let y=f((first+k) as f64*h);(v+w[k]*y,g+d[k]*y)})};
        let quadratic=|x:f64|3.*x*x-2.*x+0.5;
        for n in [4,5,9,17]{let h=1./(n-1) as f64;
            for i in 0..=200{let s=i as f64/200.;let(v,g)=at(n,s,&quadratic);assert!((v-quadratic(s)).abs()<1e-12&&(g-(6.*s-2.)).abs()<1e-10,"{n} {s}: {v} {g}");}
            for node in 1..n-1{let s=node as f64*h;let f=|x:f64|(7.*x).sin()+x*x*x;
                let(left,right)=(at(n,s-1e-12,&f).1,at(n,s+1e-12,&f).1);assert!((left-right).abs()<1e-8*(1.+left.abs()),"{n} node {node}: {left} {right}");}
        }
        let worst=|n:usize|(0..=1000).map(|i|{let s=i as f64/1000.;(at(n,s,&|x:f64|(5.*x).sin()).0-(5.*s).sin()).abs()}).fold(0.,f64::max);
        let(a,b,c)=(worst(33),worst(65),worst(129));assert!(a/b>6.&&b/c>6.&&a/b<10.&&b/c<10.,"{a} {b} {c}");
    }

    /// The tables stop refining at the work bounds (F9): the tweezer PGC's (its tweezer one intensity
    /// axis) within nine nodes, or within the bytes of the grids of nine and five, keep the grid of
    /// nine, its error measured against the five before it and booked -- where the unbounded tables
    /// refine on to a smaller one, the next grid paying for itself: nine nodes' error takes the whole population target, even at a run of no duration --; a bound below the first
    /// refinement leaves no measured error, and is refused.
    #[test]
    fn the_tables_stop_at_the_work_bounds_with_their_error_measured(){
        let f:serde_json::Value=serde_json::from_str(include_str!("../tests/fixtures/tweezer_pgc.json")).unwrap();
        let r=crate::ground::tests::request(&f["request"]);
        let free=Model::new(&r).unwrap();
        assert!(free.table_limit.is_none()&&!free.table_stalled&&free.tables.nodes>9,"{:?} at {} nodes: {:e}",free.table_limit,free.tables.nodes,free.table_error);
        // A node's entries while the tables are refined: W^S and W^Q among them (`transfer_tables`).
        let entries:usize=Model::laid_out(&r,crate::ground::Ground::new(&r).unwrap(),false).unwrap().0.layouts.iter().map(|l|l.size).sum();
        for(bytes,nodes)in [(MOST_TABLE_BYTES,9),(16*entries*(9+5),MOST_TABLE_NODES)]{
            let m=Model::bounded(&r,f64::INFINITY,false,bytes,nodes).unwrap();
            assert!(m.table_limit==Some(9)&&m.tables.nodes==9&&m.table_error>free.table_error,"{:?} at {} nodes: {:e}",m.table_limit,m.tables.nodes,m.table_error);
        }
        assert!(Model::bounded(&r,f64::INFINITY,false,MOST_TABLE_BYTES,5).is_err());
    }
    /// A generator whose no-jump decay misses its jumps' rate beyond round-off is refused (`Model::lindblad`): the
    /// tweezer PGC's model, a Lindblad one to round-off, with one kept group's decay Q a millionth smaller at every
    /// node of its tables -- a decay no jump carries, a millionth of that group's rate.
    #[test]
    fn a_generator_whose_decay_misses_its_jumps_is_refused(){
        let f:serde_json::Value=serde_json::from_str(include_str!("../tests/fixtures/tweezer_pgc.json")).unwrap();
        let r=crate::ground::tests::request(&f["request"]);
        let mut model=Model::new(&r).unwrap();assert!(model.lindblad().is_ok());
        let(k,q,square)=model.layouts.iter().enumerate().find_map(|(k,l)|l.groups.iter().find(|g|!g.decay).map(|g|(k,g.q,g.pairs.len()*g.entries.len()))).unwrap();
        let size=model.layouts[k].size;
        for node in 0..model.tables.data[k].len()/(2*size){for i in q..q+square{for c in 0..2{model.tables.data[k][2*(node*size+i)+c]*=1.-1e-6;}}}
        let e=model.lindblad().expect_err("a gain of a millionth of a rate");assert!(e.starts_with(NOT_LINDBLAD),"{e}");
        // Over a run the same gain is refused only where the norm it grows can show: not over a nanosecond, over ten seconds.
        model.ground.duration=1e-3;assert!(model.lindblad().is_ok());
        model.ground.duration=1e7;assert!(model.lindblad().is_err());
    }
    /// Residuals within the round-off of zero are static and of each other one beat; the beats come in
    /// pairs ±Ω.
    #[test]
    fn the_beats_merge_the_round_off(){
        let w=228.702;let round_off=1e-10;
        let still=|x:Vec<f64>,r:f64|Beats::new(x.into_iter().map(|w|(w,[0.;3])),r,1e-12);
        let b=still(vec![0.,-5.7e-14,w,w*(1.+1e-15),-w,2.*w,-2.*w+3e-13,5.*w,0.],round_off);
        assert_eq!(b.omegas.len(),6);
        for(x,y)in b.omegas.iter().zip([-5.*w,-2.*w,-w,w,2.*w,5.*w]){assert!((x-y).abs()<=round_off,"{:?}",b.omegas);}
        assert_eq!(b.of(1e-13,[0.;3]),None);assert_eq!(b.of(-2.*w,[0.;3]),Some(1));assert_eq!(b.of(5.*w*(1.-1e-16),[0.;3]),Some(5));
        assert!(still(vec![0.,1e-11],1e-10).omegas.is_empty());
        // A beam's wave vector and its negative at a residual and its negative are one pair; a term with a wave vector alone is a beat.
        let k=[0.,0.,8.];let m=[0.,0.,-8.];let b=Beats::new([(0.,k),(0.,m),(w,k),(-w,m),(w,m),(0.,[1e-13,0.,0.])].into_iter(),1e-10,1e-12);
        assert_eq!(b.omegas.len(),6);
        assert_eq!(b.of(0.,k),Some(3));assert_eq!(b.of(0.,m),Some(2));assert_eq!(b.of(-w,k),Some(1));assert_eq!(b.of(w,m),Some(4));assert_eq!(b.of(w,k),Some(5));assert_eq!(b.of(0.,[0.;3]),None);
    }
}
