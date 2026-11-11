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
//! exact: with ψ_s = e^{iK_s·v(t − t_m)} x_s in the cluster's Doppler frame (`Cluster::waves`) x evolves under Ĥ(t), the
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
//!   of Ĥ₀'s states that turns a radian and more between two nodes follows the change (a polynomial in the time, to
//!   the second order: the generator's own eigenvectors along the path); the slower pairs, with the change's own light
//!   shift, are two hops of the interaction picture a step, the first Magnus term with its integral over the pair's
//!   turn exact, and a quintic between the nodes (spectral.rs). The propagation's work is the same whatever Ĥ₀'s
//!   frequencies times the step are. The gradient's slow change is taken through the same three nodes (`propagate`).
//! The step's time integrals -- the kick, the mean populations -- are read on that state at the nodes of a
//! Gauss–Legendre quadrature at the bandwidth of the forms ⟨x|·|x⟩ on a state Ĥ₀ propagates (`quadrature`: to
//! round-off), and the norm's root between them. These nodes meet no model and no hop. Their number still follows the
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
//! step's doubling reads all of it (adapt.rs, # The read).
//!
//! The impulse is exact in its decomposition about the state at the step's middle, held in the frame, x_m = x(t_m)
//! (= ψ(t_m), U(t_m) = 1):
//! - the shear: the conservative force at the fixed r_h on x_m, −∫⟨x_m|G(r_h, t)|x_m⟩dt/‖x_m‖². At a fixed point the
//!   gradient beats in time alone, G(r_h, t) = Σ_m G_m e^{iΩ_m(t − t_m)} over its tones (`Middle::tones`), each
//!   τ sinc(Ω_mτ/2) (`shear`). Of a state that does not change it depends on r_h alone: the conservative kick is a shear
//!   and the step symplectic (propagation.md §2.3);
//! - what the state's change along the path adds: the evolved state's force along the path less x_m's held in the frame,
//!   −∫(⟨x(t)|Ĝ(t)|x(t)⟩/‖x(t)‖² − ⟨x_m|Ĝ(t)|x_m⟩/‖x_m‖²)dt, Ĝ(t) the gradient on the path, one quadrature of the
//!   difference on the propagation's nodes (B1's `force_change`): it vanishes at every node where the state does not
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
//! step's own error (tests). B1's PGC read the same T on the middle state as on the start state (propagation.md §2.3).
//!
//! # The path's momentum
//! v_m = (ħ/m)κ_m, κ_m the momentum at the step's middle time: the mean of the momenta before and after the kick,
//! κ_h + Δκ/2 (propagation.md §2.3). It depends on the kick the step makes, so it is predicted: κ_m = κ_h + K(ψ₀)/2, K(ψ)
//! the kick of a state that does not change -- its shear at κ_h and the reference's τ∇V_ref(r_h) -- on the step's start
//! state (`predicted`). The generator and the path are both taken at κ_m: one middle, one propagation a step.
//! - The prediction errs by half of what the state's change adds to the kick, τ ΔF/2 (ΔF the force's change over the
//!   step through the state): O(τ²) where the step resolves the internal state. The kick's sensitivity to v_m,
//!   (τ/2)|∂Δκ/∂κ_m|, is O(τ), so the kick errs by O(τ³), the scheme's own local order; the step's doubling reads it
//!   with the rest of the step's error (adapt.rs, # The read: the halves predict from their own starts).
//! - The reverse step predicts from its own start, the step's end: its path's momentum is κ_h + Δκ − K(ψ₁)/2 against the
//!   step's κ_h + K(ψ₀)/2. They differ by (K(ψ₀) + K(ψ₁))/2 − Δκ, the held kick's trapezoid against the kick: O(τ³), so a
//!   step and its reverse miss the start by O(τ⁴), beyond the scheme's order (tests), and by nothing where the kick
//!   does not depend on the path (one internal state: a conservative trap's step is its own inverse to round-off).
//! - Where the step does not resolve the internal state (a jump's bare state relaxing in 1/Γ ≪ τ) ΔF is O(1) and the
//!   path's velocity errs in first order over that step; the doubling reads that too.
//!
//! # Jumps (SPEC §1.6)
//! ‖ψ‖² falls monotonically between jumps (the decay is −(i/2)ΣJ†J by the model's construction). A jump is made at the
//! end of a step whose own propagation reaches the trajectory's threshold θ there (`jump`). A step tried for it
//! (`Midway::attempt`) says where its propagation reaches θ: the root s* of ln‖ψ(s)‖² = ln θ on the step's own path,
//! by bracketed inverse interpolation to the norm's rounding (`root`) -- within the step, or beyond its end, where the
//! propagation is followed on the same path as far as the caller's reach. A root within the caller's slack of the step's
//! end is the step's: it is the step to the jump. A root elsewhere is only a length to try next: a step's generator sits
//! at its own middle, so the norm along a step of another length is another curve, and the time of a jump is never read
//! off a step that is not taken (adapt.rs, # The jumps' times). A jump draws from its own stream
//! (`random::Stream::jump`) in a fixed order: the next threshold, the class, the polarization, and the direction's
//! rejection loop last, so that the loop's length moves no other draw.
//!
//! # The absorbed light's momentum (SPEC §1.6)
//! Per unit time the light an atom scatters gives its momentum a mean and a second moment that are smooth in the
//! position and finite where the light interferes away -- the local Kramers–Moyal coefficients Σ Im⟨u|∇u⟩ and
//! Σ Re⟨∇u|⊗|∇u⟩ of the jumps' amplitudes u = Jψ, neither divided by the jumps' rate (model/photon.rs). They are
//! delivered so that no increment of the momentum is a quotient by a rate:
//! - a jump carries the emitted photon's recoil −k_φ n and its class's conditional mean as far as one absorbed photon's
//!   momentum goes (`Dynamics::kick`: at most k_φ + k_c in all);
//! - a step carries the rest at its middle, on the state x_m there (`Dynamics::scattering`): the force F_s the jumps do
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

use std::sync::OnceLock;

use crate::decide::COMPLEX;
use crate::linalg::{self, inner, norm_sqr, C64, Mat};
use crate::model::{Cluster, Photon, Point, Quadratic, Scattering, STATES};
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

    /// The photon classes of the cluster's state ψ (unnormalized) at a point, their amplitudes (`Model::photons_of`): what
    /// a jump's draw of its class, direction and polarization reads.
    fn photons(&self, cluster: usize, at: &Point, psi: &[C64]) -> Vec<Photon>;

    /// The absorbed light's part of a jump's kick (1/µm): the conditional mean of the class of `photon`, one of the
    /// state's photons there, as far as one absorbed photon's momentum goes (`Photon::kick`).
    fn kick(&self, cluster: usize, at: &Point, psi: &[C64], photon: &Photon) -> [f64; 3];
    /// Where a jump's bare state `f` lands in a cluster at a point: the state the trajectory goes on in and the
    /// probability p of one more photon at once, before it (a fold's bare state is its slow state and a part that
    /// decays within the excited state's lifetime; model.rs `Model::landed`). The state itself and p = 0 where the model
    /// folds nothing of that: then `jump` draws nothing for it.
    fn landed(&self, _cluster: usize, _at: &Point, f: &[C64]) -> (Internal, f64) {
        (f.iter().copied().collect(), 0.0)
    }

    /// What the light the state scatters gives the momentum per unit time beside the Ehrenfest force and the jumps'
    /// kicks (`Model::scattering_of`): the force and the second moment's rate the jumps do not carry, of the state ψ at
    /// a step's middle, with the states' populations averaged over the step (of unit sum) for what goes with an
    /// explicit state's decay -- its rate is the step's mean, not the middle's.
    fn scattering(&self, cluster: usize, at: &Point, psi: &[C64], populations: &[f64]) -> Scattering;

    /// The loss energy E_s = (ħ/m)κ²/2 + V_run(r; p(ψ)) of the cluster's state at a point (`Model::energy`, rad/µs).
    fn energy(&self, cluster: usize, at: &Point, psi: &[C64]) -> f64;

    /// The loss test at a point (`Model::lost`): outside the region and E_s above its far value.
    fn lost(&self, cluster: usize, at: &Point, psi: &[C64]) -> bool;

    /// The displayed rows of the cluster's state at a point (`Model::readout`).
    fn readout(&self, cluster: usize, at: &Point, psi: &[C64]) -> [f64; STATES];

    /// The counted work of one middle (`decide::Work`, real multiply-adds), made once per run: the price of a step's
    /// middle beside its own work and its nodes' (`step_work`).
    fn assembly_work(&self) -> f64;

    /// The counted real multiply-adds of one jump's photons (`Dynamics::photons`), made once per run: `jump` adds it to the
    /// trajectory's work.
    fn photon_work(&self) -> f64;
}

/// A cluster's generator at a point (r_h, t_m) (module documentation, # The kick): what of it depends on neither the
/// momentum nor the time is made with it; a path through the point adds the momentum's part; and the generator and its
/// gradient's forms are asked at times along that path. In the cluster's Doppler frames throughout.
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
    /// The steps taken (their parts of the Brownian path are the path's own, adapt.rs `Path`)
    /// (adapt.rs).
    pub taken: u32,
    /// The time the atom was lost (SPEC §1.7: checked at every step's end and after every jump); None while held.
    pub lost: Option<f64>,
    /// The run's seed and the trajectory's number: its jumps' streams.
    pub seed: u64,
    pub index: u64,
    /// The step control's own (`adapt`): its step (µs), its count of steps, the time its cap was met, and the hazard's
    /// rate over the last step taken (1/µs: that step's fall of ln‖ψ‖² over its length; none before the first).
    pub tau: f64,
    pub steps: u64,
    pub capped: Option<f64>,
    pub rate: f64,
    /// The work counted (`decide::Work`, real multiply-adds) of the integrator's own algorithms over the trajectory --
    /// the generator's decomposition, its following and hops, the states at the kick's nodes and their sums, with the model's generators
    /// and forces at the step's nodes, the doubling (`adapt` adds each step's) --, the middles it asked the model for
    /// (each costed by
    /// `Dynamics::assembly_work`) and its jumps' photons (`Dynamics::photon_work`).
    pub work: f64,
    pub assemblies: u64,
}

impl Trajectory {
    /// A trajectory at a basis state (a ground state of the model basis) at (r, κ) and time 0: ψ the unit vector of that
    /// state on its cluster, the first threshold from jump 0's stream (SPEC §2.4); the step control's own unset.
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
            taken: 0,
            lost: None,
            seed,
            index,
            tau: f64::INFINITY,
            steps: 0,
            capped: None,
            rate: 0.0,
            work: 0.0,
            assemblies: 0,
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
    fn reach(&self, (r, kappa): ([f64; 3], [f64; 3]), t: f64) -> (([f64; 3], [f64; 3]), [f64; 2]) {
        let hm = self.hbar_over_mass;
        let d = [r[0] - self.centre[0], r[1] - self.centre[1], r[2] - self.centre[2]];
        let (mut r1, mut k1) = (self.centre, [0.0; 3]);
        let mut sizes = [self.centre.iter().fold(0.0, |m: f64, x| m.max(x.abs())), 0.0];
        for i in 0..3 {
            let e = self.axes[i];
            let (y, k, lambda, g) = (linalg::dot3(e, d), linalg::dot3(e, kappa), self.curvatures[i], self.gradient[i]);
            let (c0, c1, c2) = stumpff(hm * lambda * t * t);
            let position = [y * c0, hm * k * t * c1, -hm * g * t * t * c2];
            let momentum = [k * c0, -(lambda * y + g) * t * c1];
            let (y1, k_1) = (position[0] + position[1] + position[2], momentum[0] + momentum[1]);
            sizes[0] += position.iter().map(|x| x.abs()).sum::<f64>();
            sizes[1] += momentum.iter().map(|x| x.abs()).sum::<f64>();
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

/// The rules of 2 to 64 nodes (two nodes at least: exact for the cubics a free flight's κ² is).
fn rules() -> &'static [Rule] {
    static RULES: OnceLock<Vec<Rule>> = OnceLock::new();
    RULES.get_or_init(|| (2..=64).map(Rule::new).collect())
}

/// The rule and the panels over a width of an entire integrand of bandwidth ν to round-off: the fewest nodes, each panel
/// of width w within its rule's phase, νw ≤ x_m. Its node count is the propagation's states and the kick's sums
/// (`propagate`, `count::impulse`).
fn plan(width: f64, bandwidth: f64) -> (&'static Rule, usize) {
    rules()
        .iter()
        .map(|rule| (rule, ((bandwidth * width / rule.phase).ceil() as usize).max(1)))
        .min_by_key(|(rule, panels)| rule.nodes.len() * panels)
        .unwrap()
}

/// The nodes and weights over [a, b] of an entire integrand of bandwidth ν to round-off (`plan`), in their order from a
/// to b (the rule's nodes are held from the largest down), panel by panel as they are read: a stiff generator's
/// millions of nodes are never held at once.
fn quadrature(a: f64, b: f64, bandwidth: f64) -> impl Iterator<Item = (f64, f64)> {
    let (rule, panels) = plan((b - a).abs(), bandwidth);
    let w = (b - a) / panels as f64;
    (0..panels).flat_map(move |p| {
        let centre = a + (p as f64 + 0.5) * w;
        rule.nodes.iter().zip(&rule.weights).rev().map(move |(x, weight)| (centre + 0.5 * w * x, 0.5 * w * weight))
    })
}

// ---------------------------------------------------------------------------------------------------------------------
// The step's path and its moments.

/// One step's path (`step`): the A flow from its start to the middle t_m, the kick there, the A flow from the kicked
/// middle to its end -- the splitting's continuous path, second order in τ like the step. Its window integrals of κ
/// (`Segment::moments`) are T's window means' (SPEC §1.7), exact on this path.
#[derive(Clone, Debug, PartialEq)]
pub struct Segment {
    /// The step's start time and its length (µs; negative for a reverse step).
    pub start: f64,
    pub tau: f64,
    /// The reference flow of the step.
    pub flow: Flow,
    /// (r, κ) at the start, and at the middle after the kick.
    pub from: ([f64; 3], [f64; 3]),
    pub kicked: ([f64; 3], [f64; 3]),
    /// The moments over the whole step, made where first asked (`moments`): an output's window and a read's trail ask
    /// them of the same step.
    whole: std::cell::OnceCell<([f64; 3], [f64; 3])>,
}

impl Segment {
    /// The path of a step of τ from the time `start`: its reference flow, its start and its kicked middle.
    fn new(start: f64, tau: f64, flow: Flow, from: ([f64; 3], [f64; 3]), kicked: ([f64; 3], [f64; 3])) -> Segment {
        Segment { start, tau, flow, from, kicked, whole: std::cell::OnceCell::new() }
    }
    /// The middle's time t_m.
    fn middle(&self) -> f64 {
        self.start + 0.5 * self.tau
    }
    /// ∫κ_a dt and ∫κ_a² dt (1/µm·µs, 1/µm²·µs) per lab axis over [from, to] within the step, on its path (`integrals`);
    /// over the whole step, from its start to its end, made once.
    pub fn moments(&self, from: f64, to: f64) -> ([f64; 3], [f64; 3]) {
        if (from, to) == (self.start, self.start + self.tau) {
            *self.whole.get_or_init(|| self.integrals(from, to))
        } else {
            self.integrals(from, to)
        }
    }
    /// `moments` over [from, to]: per side of the middle, Gauss–Legendre to round-off on the flow, κ² of bandwidth twice
    /// the flow's rate (`quadrature`).
    fn integrals(&self, from: f64, to: f64) -> ([f64; 3], [f64; 3]) {
        let (m, (lo, hi)) = (self.middle(), (from.min(to), from.max(to)));
        let sign = if to >= from { 1.0 } else { -1.0 };
        let (first, last) = (self.start.min(self.start + self.tau), self.start.max(self.start + self.tau));
        let bandwidth = 2.0 * self.flow.rate();
        let (mut one, mut two) = ([0.0; 3], [0.0; 3]);
        // The step's two sides in time order, each with the flow's start and its time origin.
        let sides = if self.tau >= 0.0 {
            [(first, m, self.from, self.start), (m, last, self.kicked, m)]
        } else {
            [(first, m, self.kicked, m), (m, last, self.from, self.start)]
        };
        for (a, b, origin, at) in sides {
            let (a, b) = (a.max(lo), b.min(hi));
            if !(b > a) {
                continue;
            }
            for (t, w) in quadrature(a, b, bandwidth) {
                let k = self.flow.advance(origin, t - at).1;
                for x in 0..3 {
                    one[x] += sign * w * k[x];
                    two[x] += sign * w * k[x] * k[x];
                }
            }
        }
        (one, two)
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// The generator along the path (B).

/// The cluster's generator along the step's path in its Doppler frame (module documentation, # The kick): the model's
/// generator at the middle (`Middle`) with the path through it, and per state the gauge's phase over half the step --
/// the frame's state at the step's start is x₀ = e^{iK_s·v τ/2} ψ₀, and at its end ψ = e^{iK_s·v τ/2} x.
struct Along<'a, M: Middle> {
    middle: &'a M,
    path: M::Path,
    half: [C64; STATES],
    n: usize,
}

impl<'a, M: Middle> Along<'a, M> {
    /// The path through a middle at the velocity v, its generator at the momentum κ, for a step of τ.
    fn new(middle: &'a M, kappa: [f64; 3], velocity: [f64; 3], tau: f64) -> Along<'a, M> {
        let (n, path) = (middle.states(), middle.path(kappa, velocity));
        let mut gauge = [0.0; STATES];
        middle.gauge(&path, &mut gauge[..n]);
        let mut half = [C64::ZERO; STATES];
        for s in 0..n {
            half[s] = C64::cis(0.5 * gauge[s] * tau);
        }
        Along { middle, path, half, n }
    }
}

/// The upper triangles (a ≤ b, row by row in n × n) of Hermitian sums, each with its weight times a state's density
/// more: sums[j] += w_j |x⟩⟨x|/⟨x|x⟩ (`mirror` makes them whole).
fn accumulate(x: &[C64], weights: &[f64], sums: &mut [C64]) {
    let (n, scale) = (x.len(), 1.0 / norm_sqr(x));
    let mut row = [C64::ZERO; STATES];
    for (a, left) in x.iter().enumerate() {
        let left = *left * scale;
        for (z, right) in row[a..n].iter_mut().zip(&x[a..]) {
            *z = left * right.conj();
        }
        for (sum, w) in sums.chunks_mut(n * n).zip(weights) {
            for (sum, z) in sum[a * n + a..(a + 1) * n].iter_mut().zip(&row[a..n]) {
                *sum += *z * *w;
            }
        }
    }
}

/// A Hermitian matrix whole from its upper triangle.
fn mirror(sum: &mut [C64], n: usize) {
    for a in 0..n {
        for b in 0..a {
            sum[a * n + b] = sum[b * n + a].conj();
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
/// its following, the state's path), and one array carved (`Scratch::carve`) into the middle state's density and the nodes' weighted
/// sums of theirs (`propagate`). Every use writes what it reads: nothing a step left in it is read by another.
#[derive(Default)]
pub struct Scratch {
    picture: Picture,
    rest: Vec<C64>,
}

struct Carved<'a> {
    picture: &'a mut Picture,
    held: &'a mut [C64],
    sums: &'a mut [C64],
}

impl Scratch {
    /// The scratch for a cluster of n states.
    fn fit(&mut self, n: usize) {
        self.picture.fit(n);
        self.rest.resize(4 * n * n, C64::ZERO);
    }
    /// The scratch's parts.
    fn carve(&mut self) -> Carved<'_> {
        let n = self.picture.states();
        let (held, sums) = self.rest.split_at_mut(n * n);
        Carved { picture: &mut self.picture, held, sums }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// The propagations.

/// What a propagation of the frame's state over the step made: ψ at the step's end (lab; the cluster's states first),
/// the state at the step's middle x_m (the frame's, which is the lab's there), the states' populations averaged over
/// the step, the kick's part of the state, its counted work, and whether ‖ψ‖² reaches the watched threshold
/// at the step's end (`Watch`: within its slack).
struct Propagated {
    end: [C64; STATES],
    middle: [C64; STATES],
    populations: [f64; STATES],
    impulse: [f64; 3],
    work: f64,
    reached: bool,
}

/// What a propagation looks at of a jump's threshold (`Midway::attempt`): the threshold θ of ‖ψ‖²; the slack (µs) --
/// a crossing within it of the step's end is the step's end --; and the reach (µs) to which the propagation is followed
/// on its path beyond the step's end where it ends above θ.
#[derive(Clone, Copy, Debug)]
struct Watch {
    theta: f64,
    slack: f64,
    reach: f64,
}

/// A watched propagation whose norm reaches the threshold elsewhere than at its end: the first time (from the step's
/// start) at which ‖ψ‖² reaches it -- before the end less the slack, where the propagation ends, or beyond the end plus
/// the slack, within the reach --, and the work counted. Its end and its impulse are not made: the step only proposes
/// that length (adapt.rs, # The jumps' times), and nothing else of it is read.
struct Crossed {
    at: f64,
    work: f64,
}

/// The counted work (`decide::Work`, real multiply-adds) of the integrator's operations on n states, one formula each,
/// the model's own counts of its generator, its force and its tones at a node with them (`Middle::work`): what the
/// propagation adds up and what `step_work` prices a step by.
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
    let (rule, panels) = plan(tau.abs(), bandwidth);
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

/// The propagation of B over the step on the path through a middle at the momentum κ, the generator taken at it (module
/// documentation, # The kick): the generator's picture launched from the step's start (`spectral::Picture::launch`:
/// the state is then known at any time of the step), the frame's state read at the nodes of a quadrature that
/// resolves the generator's own frequencies -- Gauss–Legendre over the step at the bandwidth of the forms on a state it
/// propagates (`quadrature`) --, at the step's middle and at its end (`spectral::Picture::advance`); and the kick's
/// part of the state by that quadrature.
///
/// The kick is exact in its decomposition about the middle state x_m held in the frame: its shear at r_h, each tone at
/// its beat alone (`shear`), and −∫Re tr(Ĝ(u)(ρ(u) − ρ_m))du of the state's change along the path. Ĝ(u) changes along
/// the path only as the generator's slow part does, so it is taken through the step's three nodes u_j = −h, 0, h, their
/// Lagrange polynomials ℓ_j: the integral is Σ_j Re tr(Ĝ(u_j) Q_j) with Q_j = ∫ℓ_j(u)(ρ(u) − ρ_m)du, three forms of the
/// model, and the quadrature's nodes meet no model at all. Q_j is summed as Σ_k w_k ℓ_j(u_k) ρ_k less its weights' sum
/// times ρ_m (the middle state is not known before the first half is through): of a state that does not change it is
/// the rounding of that difference, and nothing where there is one state. Where Ĥ is still along every path Ĝ is one
/// matrix, and the kick its one form with ∫ρ(u)du.
///
/// Watched for a threshold (`Watch`), it says whether ‖ψ‖² reaches it at the step's end, and ends where it reaches it
/// elsewhere (`Crossed`): the norm's root between the nodes that bracket it, the state read at the root's times
/// (`spectral::Picture::between`; `root`) -- within the step, or, where the step ends above the threshold, on further
/// nodes of the same quadrature on the same path followed beyond its end (`spectral::Picture::extend`), as far as the
/// reach.
fn propagate<M: Middle>(scratch: &mut Scratch, middle: &M, psi: &[C64], kappa: [f64; 3], hbar_over_mass: f64, tau: f64, watch: Option<Watch>) -> Result<Propagated, Crossed> {
    let along = Along::new(middle, kappa, kappa.map(|k| hbar_over_mass * k), tau);
    let (n, h, still, model) = (along.n, 0.5 * tau, middle.still(), middle.work());
    let Carved { picture, held, sums } = scratch.carve();
    pictured(&along, picture, tau);
    let (bandwidth, watch) = (picture.bandwidth, watch.filter(|_| tau > 0.0));
    let mut x = [C64::ZERO; STATES];
    for (x, (z, phase)) in x.iter_mut().zip(psi.iter().zip(&along.half)) {
        *x = *z * *phase;
    }
    picture.launch(&x[..n]);
    // The norm at a time from the step's start, between the last two stops.
    let between = |picture: &mut Picture, time: f64| {
        let mut z = [C64::ZERO; STATES];
        picture.between(time - h, &mut z[..n]);
        norm_sqr(&z[..n])
    };
    let (mut ends, mut weights, mut counted) = ([[C64::ZERO; STATES]; 2], [0.0; 3], 0);
    let (mut at, mut norm, mut reached) = (-h, norm_sqr(psi), false);
    sums.fill(C64::ZERO);
    // The stops in the path's order: the quadrature's nodes, each with its weight, and the step's middle and end, where
    // the state is kept -- a node within the time's round-off of the middle (an odd rule's) is the middle.
    let (mut nodes, mut kept) = (quadrature(-h, h, bandwidth).peekable(), [0.0, h].into_iter().enumerate().peekable());
    let ahead = |u: f64| (u + h) / tau;
    loop {
        let (u, weight, keep) = match (nodes.peek().copied(), kept.peek().copied()) {
            (Some((u, w)), Some((end, stop))) if (u - stop).abs() <= f64::EPSILON * h.abs() => {
                nodes.next();
                kept.next();
                (stop, Some(w), Some(end))
            }
            (Some((u, w)), stop) if stop.map_or(true, |(_, stop)| ahead(u) < ahead(stop)) => {
                nodes.next();
                (u, Some(w), None)
            }
            (_, Some((end, stop))) => {
                kept.next();
                (stop, None, Some(end))
            }
            (_, None) => break,
        };
        let bound = picture.advance(u, &mut x[..n]);
        let now = norm_sqr(&x[..n]);
        if let Some(watch) = watch.filter(|watch| !reached && now <= watch.theta) {
            let floor = 2.0 * (picture.rounding + bound) + rounding(n + 1);
            let crossing = root(|time| between(picture, time), (at + h, norm), (u + h, now), watch.theta, floor);
            if crossing < tau - watch.slack {
                return Err(Crossed { at: crossing, work: count::propagation(model, still, picture.work) });
            }
            reached = true;
        }
        if let Some(w) = weight {
            counted += 1;
            if still {
                accumulate(&x[..n], &[w], sums);
            } else {
                let s = u / h;
                let lagrange = [0.5 * s * (s - 1.0), 1.0 - s * s, 0.5 * s * (s + 1.0)].map(|l| w * l);
                for (sum, l) in weights.iter_mut().zip(lagrange) {
                    *sum += l;
                }
                accumulate(&x[..n], &lagrange, sums);
            }
        }
        if let Some(end) = keep {
            ends[end] = x;
        }
        (at, norm) = (u, now);
    }
    // Above the threshold at its end: the same path followed beyond it, as far as the reach.
    if let Some(watch) = watch.filter(|watch| !reached && watch.reach > 0.0) {
        picture.extend(h + watch.reach);
        for u in quadrature(h, h + watch.reach, bandwidth).map(|(u, _)| u).chain(std::iter::once(h + watch.reach)) {
            let bound = picture.advance(u, &mut x[..n]);
            let now = norm_sqr(&x[..n]);
            if now <= watch.theta {
                let floor = 2.0 * (picture.rounding + bound) + rounding(n + 1);
                let crossing = root(|time| between(picture, time), (at + h, norm), (u + h, now), watch.theta, floor);
                if crossing <= tau + watch.slack {
                    reached = true;
                } else if crossing <= tau + watch.reach {
                    return Err(Crossed { at: crossing, work: count::propagation(model, still, picture.work) });
                }
                break;
            }
            (at, norm) = (u, now);
        }
    }
    let [middle, mut end] = ends;
    let mut populations = [0.0; STATES];
    let impulse = if still {
        let sum = &mut sums[..n * n];
        mirror(sum, n);
        for (s, p) in populations[..n].iter_mut().enumerate() {
            *p = sum[s * n + s].re / tau;
        }
        along.middle.force(&along.path, 0.0, sum).map(|f| -f)
    } else {
        let mut out = shear(&along, held, &middle[..n], tau);
        for (j, at) in [-h, 0.0, h].into_iter().enumerate() {
            let sum = &mut sums[j * n * n..(j + 1) * n * n];
            mirror(sum, n);
            for (s, p) in populations[..n].iter_mut().enumerate() {
                *p += sum[s * n + s].re / tau;
            }
            for (q, rho) in sum.iter_mut().zip(held.iter()) {
                *q -= *rho * weights[j];
            }
            let form = along.middle.force(&along.path, at, sum);
            for a in 0..3 {
                out[a] -= form[a];
            }
        }
        out
    };
    for (z, phase) in end[..n].iter_mut().zip(&along.half) {
        *z *= *phase;
    }
    let work = count::propagation(model, still, picture.work) + count::impulse(n, model, counted, still);
    Ok(Propagated { end, middle, populations, impulse, work, reached })
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

/// One step of τ from a trajectory's state (`step`).
#[derive(Clone, Debug)]
pub struct Step {
    /// The state at the step's end; jumps, the loss and the trajectory's counters are the caller's.
    pub end: Trajectory,
    /// The step's path.
    pub segment: Segment,
    /// The step's counted work (real multiply-adds: its own and its nodes' generators and forces). The middle it went
    /// through is counted where it is made (`Midway`: one a step, or one for the steps that share it).
    pub work: f64,
    /// The factor Λ of the scattered light's diffusion the step put its part of the Brownian path through (`factor`;
    /// 1/µm/√µs), and the kinetic energy that diffusion puts in over the step per lab axis in the mean, (ħ/m) 2D_aa
    /// |τ|/2 of the step's own 2D at its middle (rad/µs).
    pub spread: [[f64; 3]; 3],
    pub heating: [f64; 3],
}

/// Why a step or a jump could not be made: the reference's fixed point is not a contraction at this step (`middle`); the
/// state left the floating-point range; or the norm reached its threshold with no photon to emit (`jump`: the model's
/// decay and its photons contradict each other). The step control halves τ at the first two (its more accurate level).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Refused {
    Reference,
    Overflow,
    Emission,
}

/// The path's momentum of the step of τ through a middle from a state (module documentation, # The path's momentum),
/// predicted: κ_h and half the kick of the start state held there -- its shear at κ_h and the reference's τ∇V_ref(r_h).
fn predicted<M: Middle>(scratch: &mut Scratch, middle: &M, psi: &[C64], kappa_h: [f64; 3], reference: [f64; 3], tau: f64) -> [f64; 3] {
    let held = shear(&Along::new(middle, kappa_h, [0.0; 3], tau), scratch.carve().held, psi, tau);
    std::array::from_fn(|a| kappa_h[a] + 0.5 * (held[a] + tau * reference[a]))
}

/// A step tried for its state's jump (`Midway::attempt`), by where its own propagation reaches the state's threshold
/// (module documentation, # Jumps).
#[derive(Clone, Debug)]
pub enum Attempt {
    /// Nowhere within the step, nor within the reach beyond it: the step, and no jump.
    Whole(Step),
    /// At its end, within the slack: the step to the jump, which is made at its end.
    Reached(Step),
    /// Elsewhere -- before its end, where the propagation ended, or beyond it within the reach --: the time from the
    /// step's start, the length of the step to try next, and the work counted.
    Proposes { length: f64, work: f64 },
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
    /// any internal state of the cluster --: the path's momentum predicted from the state (`predicted`), the propagation
    /// at it and on its path (`propagate`), the kick -- the propagation's impulse, the reference's τ∇V_ref(r_h), and
    /// the scattered light's force and its diffusion's part of the Brownian path `brownian` (module documentation, # The
    /// absorbed light's momentum) -- and the second half flow. The flow between jumps: the state's threshold is not
    /// looked at (`attempt` looks). `scratch` is the trajectory's.
    ///
    /// `spread` is the factor the path's part goes through in the stead of the step's own (None: its own): a step read
    /// against its halves gives them its own factor (adapt.rs, # The read), so that the three are one realization of
    /// the diffusion and differ by the scheme's error alone.
    pub fn step(&self, from: &Trajectory, brownian: &Brownian, spread: Option<&[[f64; 3]; 3]>, scratch: &mut Scratch) -> Result<Step, Refused> {
        match self.tried(from, brownian, spread, None, scratch)? {
            Attempt::Whole(step) => Ok(step),
            Attempt::Reached(_) | Attempt::Proposes { .. } => unreachable!("a step without a threshold meets no crossing"),
        }
    }

    /// The same step tried for the state's jump (module documentation, # Jumps): where its own propagation reaches
    /// the state's threshold -- nowhere (within the step and `reach` µs beyond its end): the step; at its end within
    /// `slack` µs: the step to the jump; elsewhere: the length to try next (`Attempt`).
    pub fn attempt(&self, from: &Trajectory, brownian: &Brownian, slack: f64, reach: f64, scratch: &mut Scratch) -> Result<Attempt, Refused> {
        self.tried(from, brownian, None, Some(Watch { theta: from.threshold, slack, reach }), scratch)
    }

    /// `step`, or watched for the state's threshold `attempt`.
    fn tried(&self, from: &Trajectory, brownian: &Brownian, spread: Option<&[[f64; 3]; 3]>, watch: Option<Watch>, scratch: &mut Scratch) -> Result<Attempt, Refused> {
        let (tau, hm, n) = (self.tau, self.hbar_over_mass, from.psi.len());
        let (flow, (r_h, kappa_h), reference, generator) = (self.flow, self.reached, self.reference, &self.generator);
        scratch.fit(n);
        let kappa_m = predicted(scratch, generator, &from.psi, kappa_h, reference, tau);
        let shear = count::shear(n, generator.work());
        let made = match propagate(scratch, generator, &from.psi, kappa_m, hm, tau, watch) {
            Ok(made) => made,
            Err(crossed) => return Ok(Attempt::Proposes { length: crossed.at, work: crossed.work + shear }),
        };
        // The scattered light at the middle, on the state there: its force over the step, and its diffusion's factor on
        // the Brownian path's increment (the momentum) and on its first moment about the middle (the position).
        let scattered = self.model.scattering(from.cluster, &Point { r: r_h, kappa: kappa_m, t: from.time + 0.5 * tau }, &made.middle[..n], &made.populations[..n]);
        let spread = spread.copied().unwrap_or_else(|| factor(&scattered.diffusion, scattered.size, scattered.links));
        let heating = std::array::from_fn(|a| 0.5 * hm * scattered.diffusion[a][a] * tau.abs());
        let through = |w: &[f64; 3]| -> [f64; 3] { std::array::from_fn(|a| (0..3).map(|c| spread[a][c] * w[c]).sum()) };
        let (push, shift) = spreading(&flow, tau, through(&brownian.increment), through(&brownian.moment));
        let kicked = (
            std::array::from_fn(|a| r_h[a] - hm * shift[a]),
            std::array::from_fn(|a| kappa_h[a] + made.impulse[a] + tau * (reference[a] + scattered.force[a]) + push[a]),
        );
        let (r1, kappa1) = flow.advance(kicked, 0.5 * tau);
        if !(r1.iter().chain(&kappa1).all(|x| x.is_finite()) && made.end[..n].iter().all(|z| z.is_finite())) {
            return Err(Refused::Overflow);
        }
        let mut end = from.clone();
        end.r = r1;
        end.kappa = kappa1;
        end.psi.copy_from_slice(&made.end[..n]);
        end.time = from.time + tau;
        let step = Step { end, segment: Segment::new(from.time, tau, flow, (from.r, from.kappa), kicked), work: made.work + shear, spread, heating };
        Ok(if made.reached { Attempt::Reached(step) } else { Attempt::Whole(step) })
    }
}

/// One step of τ (negative: the reverse step) from a trajectory's state (module documentation): its middle
/// (`Midway::new`) and the step through it (`Midway::step`) on a Brownian path that does not move (`Brownian::STILL`)
/// -- the step's drift alone, the scattered light's diffusion left out.
pub fn step<D: Dynamics>(model: &D, from: &Trajectory, tau: f64, scratch: &mut Scratch) -> Result<Step, Refused> {
    Midway::new(model, from, tau)?.step(from, &Brownian::STILL, None, scratch)
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
    let along = Along::new(middle, kappa, velocity, tau);
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
/// class ∝ Σ_a‖u_a‖², the polarization across the direction by ‖Σ_a ε_a u_a‖², and the direction from the dipole pattern
/// tr G − n·Re(G)·n (at most tr G, as Re G ≥ 0) by exact rejection from the uniform sphere; then ψ ← f/‖f‖ on the landing
/// cluster and κ += −k_φ n + s b, the emitted photon's recoil and the class's conditional mean of the absorbed momentum
/// as far as one photon's goes (`Dynamics::kick`; module documentation, # The absorbed light's momentum) -- at most
/// k_φ + k_c, whatever the light does at the point; then the loss test (`test_loss`). A norm that reached its threshold
/// with no photon to emit -- every class's weight zero, or not a number -- is refused (`Refused::Emission`) with the
/// trajectory as it was: the model's decay and its photons contradict each other, and no jump is made of that.
///
/// The landing (`Dynamics::landed`): the bare state f/‖f‖ lands with a probability p of one more photon at once. Where
/// p = 0 the trajectory goes on in the bare state and nothing is drawn. Else one more uniform number of the jump's
/// stream decides: below p the bare state emits a photon as any state does -- its class, polarization and direction
/// drawn anew, its recoil and its bounded kick with them -- and lands again; else the trajectory goes on in the slow
/// state the model gives, normalized.
pub fn jump<D: Dynamics>(model: &D, trajectory: &mut Trajectory) -> Result<(), Refused> {
    let j = trajectory.jumps + 1;
    let mut stream = Stream::jump(trajectory.seed, trajectory.index, j);
    let threshold = stream.open_uniform();
    let (mut class, mut polarization) = (stream.uniform(), stream.uniform());
    let weight = |p: &Photon| -> f64 { p.amplitudes.iter().map(|u| norm_sqr(u)).sum() };
    let mut first = true;
    loop {
        let at = trajectory.point();
        let photons = model.photons(trajectory.cluster, &at, &trajectory.psi);
        let total: f64 = photons.iter().map(weight).sum();
        if !(total > 0.0) {
            // A norm at its threshold with no photon to emit is refused; a bare state with none goes on as it is.
            if first {
                return Err(Refused::Emission);
            }
            break;
        }
        if first {
            trajectory.jumps = j;
            trajectory.threshold = threshold;
        }
        trajectory.work += model.photon_work();
        let mut x = class * total;
        let mut pick = photons.len() - 1;
        for (i, w) in photons.iter().map(weight).enumerate() {
            if x < w {
                pick = i;
                break;
            }
            x -= w;
        }
        let photon = &photons[pick];
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
        let absorbed = model.kick(trajectory.cluster, &at, &trajectory.psi, photon);
        for a in 0..3 {
            trajectory.kappa[a] += -photon.wave_number * direction[a] + absorbed[a];
        }
        let scale = 1.0 / w.sqrt();
        trajectory.psi = f.iter().map(|z| *z * scale).collect();
        trajectory.cluster = photon.landing;
        first = false;
        // The landing: the bare state as it is, its slow state, or one more photon from it.
        let at = trajectory.point();
        let (slow, prompt) = model.landed(trajectory.cluster, &at, &trajectory.psi);
        if !(prompt > 0.0) {
            break;
        }
        if stream.uniform() >= prompt {
            let size = norm_sqr(&slow);
            if size > 0.0 {
                trajectory.psi = slow.iter().map(|z| *z * (1.0 / size.sqrt())).collect();
            }
            break;
        }
        (class, polarization) = (stream.uniform(), stream.uniform());
    }
    test_loss(model, trajectory);
    Ok(())
}

/// The loss test at a trajectory's state (SPEC §1.7: at every step's end and after every jump), the one definition: its
/// time once the model finds it lost (`Dynamics::lost`).
pub fn test_loss<D: Dynamics>(model: &D, trajectory: &mut Trajectory) {
    if trajectory.lost.is_none() && model.lost(trajectory.cluster, &trajectory.point(), &trajectory.psi) {
        trajectory.lost = Some(trajectory.time);
    }
}

#[cfg(test)]
mod tests;
