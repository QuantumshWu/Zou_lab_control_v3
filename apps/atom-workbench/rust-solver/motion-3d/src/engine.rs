//! Trajectories: a classical point (r, ħκ) and the internal state ψ on one cluster (ground.rs),
//! unnormalized between quantum jumps (the waiting-time unraveling of the cluster's generator).
//!
//! # One step (flow–kick–flow)
//! The trap's harmonic part along each of its principal axes, ½ m ω_i² x_i² (`Settings::axes`,
//! `Settings::frequencies`), is moved exactly: (r, κ) turns in phase space for dt/2 to (r_h, κ_h);
//! there the cluster's H_eff at the step's middle time gives U = exp(−i H_eff dt/2); ψ_h = U ψ;
//! κ_h += dt (F − F_harmonic) with F the Ehrenfest force −∇Re⟨H_eff⟩ (the trap's full force and
//! the light's) and F_harmonic its harmonic part; ψ ← U ψ_h; and the harmonic flow for dt/2 again.
//! Only the anharmonic rest and the light are split, so the step is not bound to the trap period.
//! The scalar part of H_herm (the trap's depth) is a phase and left out of U.
//!
//! # The exact step (explicit excited states, beats)
//! An explicit excited state rotates at its detuning in the frame, and its saturation, Rabi
//! oscillation and decay set the internal state's pace; a beating generator (two colours on one line,
//! a term whose residual its frame keeps: model.rs `Components`) changes within the step. Holding
//! either at the step's middle ties the motion's step to the internal state's pace, so the step takes
//! the internal state exactly instead -- multirate: the motion's step keeps its own size. The flow for
//! dt/2 to (r_h, κ_h); there, the position held, every cluster the trajectory visits in the step takes
//! its generator's components once (`Segment`), H(t) = H₀ + Σ_j e^{iΩ_j t} H_j, and ψ is propagated
//! through the whole step on a grid of spacing δ from the step's start, every jump at the root of
//! ‖ψ(t)‖² = θ (safeguarded Newton on ln‖ψ‖², whose derivative is −⟨ψ|i(H − H†)|ψ⟩/‖ψ‖²), the
//! trajectory going on from the state the jump left in the cluster it landed in:
//! - a static generator through its eigendecomposition H = R Λ R⁻¹ (eigen.rs), exact at any time,
//!   where its rounding -- R's condition times the double precision, grown by the phases over the
//!   step -- is within the exponentials' share (`Settings::truncation`); near an exceptional point by
//!   the propagator exp(−iHδ) (its Taylor series to double precision) and the series on the vector off
//!   the grid;
//! - a beating one by CF4, the fourth-order commutator-free Magnus scheme (`cf4`, exact where H is
//!   static), over each spacing; beats that are harmonics of one frequency (a period T, model.rs) take
//!   the grid T/k, and the period's k propagators and node gradients are built once where the step
//!   holds more than k n spacings. A pair of beats ±Ω whose whole effect on the step is within the
//!   exponentials' share is left out (`Segment::resolve`: Duhamel's formula integrated by parts bounds
//!   it, its boundary terms paid once over the time to the next jump, `Settings::horizon`, and its
//!   integral per step) -- a gray molasses' explicit states keep terms of 10⁻⁹ of its generator at the
//!   hyperfine splitting, a red molasses' of 10⁻⁵; the solve books them in its ledger.
//! The impulse is the conditional force's time integral along that evolution, ∫ ⟨ψ|F|ψ⟩/⟨ψ|ψ⟩ dt
//! (composite Simpson on the grid, its error within `QUADRATURE` of the impulse: (ν·2δ)⁴/2880 bounds
//! it, ν the integrand's largest frequency -- 2‖H‖₁, and with beats 2(‖H₀‖₁ + Σ_j ‖H_j‖₁) + max|Ω_j|;
//! through the eigendecomposition each frequency λ_a* − λ_b weighted by its amplitude in the state,
//! `Segment::weighted`), applied with the
//! jumps' kicks and less dt F_harmonic(r_h) before the flow for dt/2 again. CF4's own error at that
//! grid is of fourth order in the beats' derivatives, far below the quadrature's (validated against a
//! fine CF4, the master equation and the population core: examples/multirate.rs). What is left is the
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
//! own recoil). The photon f = Σ_a ε_a u_a is a sum of parts: each near beam's folded amplitude f_l (model.rs
//! `Photons::absorption`) and the explicit states' v. Beams that do not interfere share it by ‖f_l‖², the
//! incoherent attribution, whose mean recoil is then the photon's exact conditional mean b = Im⟨f|∇f⟩/⟨f|f⟩
//! (the jump term's first Wigner moment; ∇f_l = i k_l f_l for a plane wave, and an explicit state's amplitude
//! does not depend on r). Interference -- overlapping parts, Re⟨f_l|f_l′⟩ ≠ 0 -- moves b off the shares' mean,
//! and the draw takes the shares tilted to b by the least information (`tilt`, q_l ∝ ‖f_l‖² e^{λ·k_l}): the
//! jumps carry the exact mean force of the absorbed light, and per jump the beams' incoherent second moment
//! about it, Σ_l q_l k_l k_lᵀ -- the shares' where nothing interferes, k_z² wherever every beam's recoil has
//! the same projection k_z (a 1D molasses, a tetrahedral MOT's axis), a running wave's ħk per photon. A mean
//! beyond every mixture of the beams' recoils (a superoscillation by a dark point) is reached from the face of
//! their hull toward it, the rest a deterministic kick (`absorbed_recoil`). Adding b to the drawn kick less
//! the shares' mean instead counted the interference's mean on top of the incoherent spread: on a tetrahedral
//! MOT's z it gave 0.142 µK per photon against 0.120 incoherent and 0.083 localized, and T_z 13 % above the
//! explicit model's. The localized second moment, ‖∇f‖²/⟨f|f⟩ (the second-order Wigner value at a point), is
//! not taken: against the tweezer PGC's 1D quantum reference (TW2, x alone) it cools 2–4 % further below it at
//! 4–5 ħω, where the incoherent one is 4–8 % low (# What the kicks leave out).
//! An explicit state's absorbed light acts through the Ehrenfest force -- the stimulated force of its kept
//! drives, −∇Re⟨ψ|V|ψ⟩ --, which carries its mean: in the tilt its part's recoil has mean zero, and its kick is
//! the recoil of a beam drawn by its drives' weights (Σ |a_l Σ_k c_lk U_k ψ_b|²) less their mean a, plus the
//! mean's shot noise ±a with equal chance -- per photon the beams' second moment, without the mean twice. A
//! running wave's beam axis takes R E_r (1 + ⟨n²⟩) so; with the spread alone it took R E_r ⟨n²⟩. The draw's
//! uniform picks the part, its leftover the explicit part's beam and then its sign: no further deviate. The
//! momentum diffusion beyond these kicks is the hook `Diffusion`'s.
//!
//! # The transient after a jump into a cluster with explicit states
//! A jump leaves the atom in a bare ground state, while the folded drives' amplitude of the state it
//! landed in, X ψ′ (X = Σ e^{−iε̃t} A_c per coherent group of channels, model.rs `Transient`), is
//! dressed: it builds up from zero, and with probability ‖X ψ′‖² the transient really excites the
//! folded states and they emit one more photon at once (their decay damps the rest). Where the jump
//! lands in a cluster with explicit excited states, the map after it is ρ → ρ − ½{Y, ρ} + Σ_s S_s ρ S_s†,
//! S_s = P_B C_a D_g^{−1/2} X_g per group g of nearly degenerate excited states (which ring apart),
//! Σ_s S_s†S_s = Y: with probability ψ′†Yψ′ one more photon from the S_s (its class by ‖S_s ψ′‖², its
//! kicks as a jump's), else ψ′ ← (1 − ½Y)ψ′; its draws come from the jump's stream before the
//! directions' loops. In a cluster of ground states alone the transient is left out: it is of the
//! order of the fold's own error, which the fold's charge bounds as validated without it (ground.rs),
//! and the engine there is the release's, draw for draw (the GPU's kernel, gpu/mod.rs).
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
//! The folded photon amplitudes take the kinetic shift of absorption to first order, R − w R2
//! (model.rs), good to (kv/σ)², the largest (w_l/σ_c)² over the near beams, σ_c the nearest excited
//! pole of beam l's carrier (model.rs `Model::doppler_ratio`). A step whose (kv/σ)² exceeds the
//! solve's allowance (`Settings::doppler`) takes the exact response at its jumps (model.rs `Exact`);
//! the largest first-order (kv/σ)² a trajectory used is its share of the ledger
//! (`Trajectory::doppler`). An explicit drive needs neither: the atom's motion through e^{ik_l·r} is
//! its Doppler shift.
//!
//! # Loss
//! An atom is lost when it is no longer bound to the trap beam and has left the trap's region: its
//! mechanical energy, ħ²κ²/2m plus its potential (model.rs `Model::trap_potential`: Re⟨ψ|H_eff|ψ⟩ of its
//! current internal state less the same with the trap beam off, so U → 0 far from it), is positive, and
//! the trap's harmonic part there lies above the region's depth (`Settings::region`, a solve's the
//! reference state's depth: the region its initial positions are drawn in). An unbound atom inside the
//! region is held until it leaves -- it may be bound again before it does, as an atom that a photon's
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
//! the absorbed beam (and into a cluster with explicit states the transient's four) first and the
//! directions' rejection loops last, so that
//! a trajectory's twin on the same streams (control.rs) keeps its thresholds and choices whatever its
//! loops take.
//!
//! # A held atom
//! A held twin (control.rs) does not move and its generator is static: its engine (`Engine::held`)
//! takes every cluster's generator and jump operators at its point once, steps by the
//! half step's propagators, and draws a jump's direction and polarization without the kicks.
use crate::eigen::{eigen,Eigen};
use crate::model::{Absorption,Components,Exact,Local,Model,Point,MOST_MIXED,MOST_STATES};
use crate::rng::{Purpose,Rng};
use crate::Complex;
use atom_population_core::lattice::accuracy::A_PRIORI_SHARE;

/// The impulse's quadrature share in the exact step: its grid has ν δ ≤ 2 `node_phase`, ν the force's
/// largest frequency, so that composite Simpson's error, (ν·2δ)⁴/2880 of the impulse at most, is within it.
pub const QUADRATURE:f64=A_PRIORI_SHARE*A_PRIORI_SHARE*crate::solve::TEMPERATURE_TARGET;
pub(crate) fn node_phase()->f64{(2880.*QUADRATURE).powf(0.25)/4.}

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
    /// The largest (kv/σ_min)² of its first-order steps.
    pub doppler:f64,
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
    /// The motion's sums (T, ⟨v⟩, n̄, ⟨r²⟩) averaged over one trap period of each axis centred on
    /// each output time (lib.rs `simulate`, a solve's), or taken at the output time.
    pub average:bool,
    /// The loss by the mechanical energy (a solve's); off where a validation holds the atom.
    pub loss:bool,
    /// The largest (kv/σ_min)² the first order in the Doppler shift is kept to.
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
    /// the trap's harmonic part (`axes`, `frequencies`) lies above this depth (rad/us) -- a solve's is its
    /// initial positions' region, U_h ≤ D (lib.rs `thermal_position`). 0, the default, is none: an atom
    /// is lost as soon as it is unbound.
    pub region:f64,
}
impl Settings{
    /// The settings of a trap (model.rs `Trap`), every axis free, the loss on, the first order in
    /// the Doppler shift everywhere.
    pub fn new(dt:f64,trap:&crate::model::Trap,truncation:f64,average:bool)->Settings{
        Settings{dt,free:[true;3],axes:trap.axes,frequencies:trap.frequencies,lab:trap.lab,truncation,average,loss:true,doppler:f64::INFINITY,beats:truncation,horizon:dt,region:0.}
    }
}

/// A momentum diffusion scheme beyond the per-beam kicks: the hook for one (none by default). It
/// may change κ at the end of each exact step's internal evolution, given the step's point and
/// length, and at each photon, given its amplitude f (the target cluster's ground states) and the
/// kick the engine has drawn.
pub trait Diffusion:Sync{
    fn step(&self,_model:&Model,_t:&mut Trajectory,_point:&Point,_tau:f64){}
    fn photon(&self,_model:&Model,_t:&Trajectory,_point:&Point,_f:&[Complex],_kick:&mut [f64;3]){}
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
/// in sub-steps of ‖H‖τ ≤ ½ (‖H‖ the largest column sum of |re| + |im|, above the 1-norm), each
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
/// The series of exp(−iHτ) for a generator of that norm (`propagate`): its sub-steps of ‖H‖τ ≤ ½ and
/// each one's terms, through the first whose bound y^m/m! is within the sub-step's share of the
/// truncation bound.
pub fn series(norm:f64,tau:f64,truncation:f64)->(usize,usize){
    let x=norm*tau.abs();let parts=((x/0.5).ceil() as usize).max(1);
    let y=x/parts as f64;let share=truncation/parts as f64;let mut terms=1;let mut size=y;while size>share&&terms<40{terms+=1;size*=y/terms as f64;}
    (parts,terms)
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

/// ‖H‖∞ bounded above: the largest row sum (with `column_norm` above the spectral norm, √(‖H‖₁‖H‖∞)).
fn row_norm(h:&[Complex],n:usize)->f64{(0..n).map(|a|(0..n).map(|b|h[a*n+b].re.abs()+h[a*n+b].im.abs()).sum::<f64>()).fold(0.,f64::max)}
/// ψ ← exp(−i H τ) ψ on any number of states, to double precision: `propagate`'s series.
pub fn propagate_dyn(h:&[Complex],n:usize,tau:f64,psi:&mut [Complex]){
    let x=column_norm(h,n)*tau.abs();let parts=((x/0.5).ceil() as usize).max(1);let step=tau/parts as f64;
    let y=x/parts as f64;let share=f64::EPSILON/parts as f64;let mut terms=1;let mut size=y;while size>share&&terms<40{terms+=1;size*=y/terms as f64;}
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
/// ⟨ψ|X|ψ⟩ of an n × n matrix X (row-major).
fn form(x:&[Complex],psi:&[Complex],n:usize)->Complex{
    let mut s=Complex::ZERO;for a in 0..n{let mut row=Complex::ZERO;for b in 0..n{row+=x[a*n+b]*psi[b];}s+=psi[a].conj()*row;}s
}
/// The product a b of n × n matrices (row-major), a's zero entries skipped (a beat's matrix is sparse).
fn product(a:&[Complex],b:&[Complex],n:usize)->Vec<Complex>{
    let mut out=vec![Complex::ZERO;n*n];for i in 0..n{for k in 0..n{let x=a[i*n+k];if x==Complex::ZERO{continue;}for j in 0..n{out[i*n+j]+=x*b[k*n+j];}}}out
}
/// The product a b with b's zero entries skipped.
fn product_right(a:&[Complex],b:&[Complex],n:usize)->Vec<Complex>{
    let mut out=vec![Complex::ZERO;n*n];for k in 0..n{for j in 0..n{let y=b[k*n+j];if y==Complex::ZERO{continue;}for i in 0..n{out[i*n+j]+=a[i*n+k]*y;}}}out
}

/// A segment's state between its grid's nodes: ψ, or a spectral segment's coordinates R⁻¹ψ.
type State=[Complex;MOST_MIXED];

/// One cluster's internal evolution in an exact step (module documentation): its generator's components
/// at the held point, the step's start (the beats' phases are absolute), the grid's spacing δ -- its nodes
/// at start + iδ --, how it propagates, and the bound of the beats it left out.
struct Segment{components:Components,start:f64,spacing:f64,how:How,dropped:f64}
/// How a segment propagates over its grid.
enum How{
    /// A static generator through its eigendecomposition H = R Λ R⁻¹ (eigen.rs), in the coordinates
    /// d = R⁻¹ψ: e^{−iλδ} per spacing; the norm's, the force's and the norm's rate's forms in them,
    /// R†R, R†G_aR and i(λ_a* − λ_b)(R†R)_ab; per force form and the norm's, |X_ab| and |X_ab| |λ_a* − λ_b|⁴
    /// (`Segment::weighted`).
    Spectral{values:Vec<Complex>,vectors:Vec<Complex>,inverse:Vec<Complex>,phases:Vec<Complex>,gram:Vec<Complex>,forces:[Vec<Complex>;3],rate:Vec<Complex>,moments:[[Vec<f64>;2];4]},
    /// A static generator whose eigenvectors are too ill conditioned (near an exceptional point): the
    /// propagator over a spacing and the decay operator i(H − H†).
    Taylor{propagator:Vec<Complex>,decay:Vec<Complex>},
    /// A beating generator: CF4 over each spacing, and with a period its cache.
    Beating{cache:Option<Cache>},
}
/// One period of a beating segment's grid: each spacing's CF4 propagator and each node's G(t).
struct Cache{propagators:Vec<Vec<Complex>>,gradients:Vec<[Vec<Complex>;3]>}

impl Segment{
    /// The segment of a cluster's components over a step of τ from `start`, its exponentials within
    /// `truncation` and the beats it leaves out within `beats` over `horizon` (`Settings`).
    fn new(mut components:Components,start:f64,tau:f64,truncation:f64,beats:f64,horizon:f64)->Segment{
        let dropped=Segment::resolve(&mut components,tau,beats,horizon);
        let mut segment=Segment::build(components,start,tau,truncation);segment.dropped=dropped;segment
    }
    fn build(components:Components,start:f64,tau:f64,truncation:f64)->Segment{
        let n=components.n();
        if components.beats.is_empty(){
            let h=&components.statics.hamiltonian;
            // The eigendecomposition where its rounding -- R's condition times the double precision, grown
            // by the phases over the step -- is within the exponentials' share.
            if let Some(e)=eigen(h,n){
                let spread=e.values.iter().flat_map(|a|e.values.iter().map(move|b|(a.conj()-*b).abs())).fold(0.,f64::max);
                if e.condition*f64::EPSILON*(1.+spread*tau)<=truncation{return Segment::spectral(components,start,tau,e,spread);}
            }
            let decay:Vec<Complex>=(0..n*n).map(|k|{let(a,b)=(k/n,k%n);(h[a*n+b]-h[b*n+a].conj())*Complex::I}).collect();
            let steps=((column_norm(h,n)*tau/node_phase()).ceil() as usize).max(1);let spacing=tau/steps as f64;
            let propagator=exponential(h,n,spacing);
            return Segment{components,start,spacing,how:How::Taylor{propagator,decay},dropped:0.};
        }
        // The force's frequencies are within 2‖H(t)‖ + max|Ω|, ‖H(t)‖ ≤ ‖H₀‖ + Σ‖H_j‖.
        let norm=column_norm(&components.statics.hamiltonian,n)+components.beats.iter().map(|b|column_norm(&b.hamiltonian,n)).sum::<f64>()
            +0.5*components.beats.iter().map(|b|b.omega.abs()).fold(0.,f64::max);
        let widest=node_phase()/norm;
        // Beats of one period T: the grid T/k, and the period's k propagators of n × n cached where the step
        // holds more than k n spacings (a propagator costs about n of CF4's steps on the vector).
        if let Some(period)=components.period{
            let k=(period/widest).ceil() as usize;let spacing=period/k as f64;
            if tau/spacing>(k*n) as f64{let cache=Segment::cache(&components,start,spacing,k);return Segment{components,start,spacing,how:How::Beating{cache:Some(cache)},dropped:0.};}
        }
        let steps=((tau/widest).ceil() as usize).max(1);
        Segment{components,start,spacing:tau/steps as f64,how:How::Beating{cache:None},dropped:0.}
    }
    /// The beats a step resolves: the pairs ±Ω whose cost on the step (`bounds`) is within the share are
    /// left out, the cheapest first while their costs' sum stays within it -- on ‖ψ‖ and on the static
    /// force's integral ‖G₀‖τ. Their cost.
    fn resolve(c:&mut Components,tau:f64,truncation:f64,horizon:f64)->f64{
        let half=c.beats.len()/2;let(mut pairs,scale)=Segment::bounds(c,tau,horizon);
        pairs.sort_by(|a,b|a.1.total_cmp(&b.1));
        let(mut moved,mut pushed)=(0.,0.);let mut dropped=vec![false;half];
        for &(i,x,f)in &pairs{if moved+x<=truncation&&pushed+f<=truncation*scale{moved+=x;pushed+=f;dropped[i]=true;}else{break;}}
        let beats=std::mem::take(&mut c.beats);
        c.beats=beats.into_iter().enumerate().filter(|&(j,_)|!dropped[if j<half{half-1-j}else{j-half}]).map(|(_,b)|b).collect();
        moved
    }
    /// Per pair of beats ±Ω (`resolve`): its index and what leaving it out costs a step, on the state and
    /// on the force's integral; and the static force's integral ‖G₀‖τ. Dropping B e^{iΩt} from H moves the
    /// propagator by ‖∫ U_rest(τ,s) B U(s) e^{iΩs} ds‖ (Duhamel's formula); integrated by parts it splits
    /// into boundary terms, which successive steps' cancel -- a trajectory pays them once over its horizon H,
    /// until a jump --, and an integral that accumulates: the cost is boundary·τ/H + integral, the smaller
    /// of two bounds. Once: boundary 2b/|Ω|, integral 2hbτ/|Ω|; twice: boundary 2b/|Ω| + 2x/Ω² and integral
    /// τy/Ω², x = ‖[H₀,B]‖ + 3b b′, y = ‖[H₀,[H₀,B]]‖ + 3‖[H₀,B]‖ b′ + 6h b b′ + 3b b′² + 2b h″ + |Ω| b b′ --
    /// b = ‖B‖, h the generator's bound, b′ all the beats' sizes, h″ the other beats' Σ|Ω_k|‖H_k‖ (norms
    /// above the spectral one). The force's integral alike, its integrand's derivatives ⟨ψ|G|ψ⟩/⟨ψ|ψ⟩
    /// within 4hg and (24h² + 4h″)g (h″ every beat's).
    fn bounds(c:&Components,tau:f64,horizon:f64)->(Vec<(usize,f64,f64)>,f64){
        let n=c.n();let size=|h:&[Complex]|column_norm(h,n).max(row_norm(h,n));let half=c.beats.len()/2;let h0=&c.statics.hamiltonian;
        let all=c.beats.iter().map(|b|size(&b.hamiltonian)).sum::<f64>();let bound=size(h0)+all;
        let pace=c.beats.iter().map(|b|b.omega.abs()*size(&b.hamiltonian)).sum::<f64>();
        let scale=(0..3).map(|a|size(&c.statics.gradient[a])).fold(0.,f64::max)*tau;let once=(tau/horizon).min(1.);
        let commutator=|x:&[Complex]|->Vec<Complex>{let(a,b)=(product_right(h0,x,n),product(x,h0,n));a.iter().zip(&b).map(|(p,q)|*p-*q).collect()};
        let state=|b:&[Complex],w:f64,others:f64|->f64{
            let m=size(b);let c1=commutator(b);let(x1,x2)=(size(&c1),size(&commutator(&c1)));
            let x=x1+3.*m*all;let y=x2+3.*x1*all+6.*bound*m*all+3.*m*all*all+2.*m*others+w*m*all;
            (2.*m/w*once+2.*bound*m*tau/w).min((2.*m/w+2.*x/(w*w))*once+tau*y/(w*w))};
        let force=|g:f64,w:f64|(2.*g/w*once+4.*bound*g*tau/w).min((2.*g/w+8.*bound*g/(w*w))*once+tau*(24.*bound*bound+4.*pace)*g/(w*w));
        ((0..half).map(|i|{let(down,up)=(&c.beats[half-1-i],&c.beats[half+i]);let w=up.omega;
            let others=pace-w*(size(&down.hamiltonian)+size(&up.hamiltonian));
            (i,state(&down.hamiltonian,w,others)+state(&up.hamiltonian,w,others),(0..3).map(|a|force(size(&down.gradient[a]),w)+force(size(&up.gradient[a]),w)).fold(0.,f64::max))}).collect(),scale)
    }
    /// A static segment through its eigendecomposition: Simpson's frequencies are the spectrum's
    /// differences λ_a* − λ_b, so the grid's spacing has (spread·2δ)⁴/2880 within `QUADRATURE`.
    fn spectral(components:Components,start:f64,tau:f64,e:Eigen,spread:f64)->Segment{
        let n=e.n;let r=&e.vectors;
        let steps=((spread*tau/(2.*node_phase())).ceil() as usize).max(1);let spacing=tau/steps as f64;
        // R† X R.
        let sandwich=|x:&[Complex]|->Vec<Complex>{let xr=product(x,r,n);let mut out=vec![Complex::ZERO;n*n];
            for i in 0..n{for k in 0..n{let y=r[k*n+i].conj();for j in 0..n{out[i*n+j]+=y*xr[k*n+j];}}}out};
        let identity:Vec<Complex>=(0..n*n).map(|k|if k/n==k%n{Complex::ONE}else{Complex::ZERO}).collect();
        let gram=sandwich(&identity);let forces=std::array::from_fn(|a|sandwich(&components.statics.gradient[a]));
        let rate=(0..n*n).map(|k|{let(a,b)=(k/n,k%n);Complex::I*(e.values[a].conj()-e.values[b])*gram[k]}).collect();
        let phases=e.values.iter().map(|&l|rotation(l,spacing)).collect();
        let moments=std::array::from_fn(|f|{let m=if f<3{&forces[f]}else{&gram};let size:Vec<f64>=m.iter().map(|z|z.abs()).collect();
            let fourth=(0..n*n).map(|k|size[k]*(e.values[k/n].conj()-e.values[k%n]).abs().powi(4)).collect();[size,fourth]});
        Segment{components,start,spacing,how:How::Spectral{values:e.values,vectors:e.vectors,inverse:e.inverse,phases,gram,forces,rate,moments},dropped:0.}
    }
    /// One period's cache from `start`: k spacings' CF4 propagators (each a product of two exponentials of
    /// n × n) and the gradient at each node.
    fn cache(components:&Components,start:f64,spacing:f64,k:usize)->Cache{
        let n=components.n();let(c,w)=cf4();let mut b=vec![Complex::ZERO;n*n];
        let propagators=(0..k).map(|i|{
            let t=start+i as f64*spacing;let times=[t+c[0]*spacing,t+c[1]*spacing];
            Segment::combine(components,times,w,&mut b);let first=exponential(&b,n,spacing);
            Segment::combine(components,times,[w[1],w[0]],&mut b);let second=exponential(&b,n,spacing);
            product(&second,&first,n)}).collect();
        let gradients=(0..k).map(|i|components.at(start+i as f64*spacing).gradient).collect();
        Cache{propagators,gradients}
    }
    /// out = w₁H(t₁) + w₂H(t₂) of a beating generator.
    fn combine(components:&Components,times:[f64;2],w:[f64;2],out:&mut [Complex]){
        let x=w[0]+w[1];for(o,h)in out.iter_mut().zip(&components.statics.hamiltonian){*o=*h*x;}
        for beat in &components.beats{
            let z=Complex::cis(beat.omega*times[0])*w[0]+Complex::cis(beat.omega*times[1])*w[1];
            for(o,h)in out.iter_mut().zip(&beat.hamiltonian){*o+=z**h;}
        }
    }
    fn n(&self)->usize{self.components.n()}
    /// A spectral segment's grid spacing for the evolution of the state x over `remaining` (`threshold`
    /// the norm it may stop at), each frequency weighted by its amplitude: in the eigenbasis every
    /// component of the force's forms and of the norm's, c = (a, b), is A_c e^{iω_c s}, ω_c = λ_a* − λ_b, with
    /// |A_c(s)| ≤ |x_a||x_b||X_ab| (the coordinates only decay), and composite Simpson errs on it by at most
    /// |A_c| τ (|ω_c|·2δ)⁴/2880: the spacing whose fourth moment Σ_c |A_c||ω_c|⁴ (2δ)⁴/2880 keeps every form
    /// within `QUADRATURE` of its total |A|, the integrand's scale at the start, the norm's fall to its
    /// least over the evolution taken in (‖ψ‖² ≥ max(threshold, its end)); the spacings over `remaining`
    /// whole.
    fn weighted(&self,x:&State,remaining:f64,threshold:f64)->Option<f64>{
        let How::Spectral{moments,..}=&self.how else{return None};let n=self.n();
        let start=self.norm(x);let mut end=*x;self.advance(&mut end,0.,remaining,None);let least=self.norm(&end).max(threshold).min(start);
        let allowed=QUADRATURE*least/start;let size:Vec<f64>=x[..n].iter().map(|z|z.abs()).collect();
        // The largest fourth moment relative to its total over the forms.
        let ratio=moments.iter().map(|[total,fourth]|{let(mut t,mut f)=(0.,0.);
            for a in 0..n{let(mut ta,mut fa)=(0.,0.);for b in 0..n{ta+=total[a*n+b]*size[b];fa+=fourth[a*n+b]*size[b];}t+=size[a]*ta;f+=size[a]*fa;}
            if t>0.{f/t}else{0.}}).fold(0.,f64::max);
        if !(ratio>0.){return Some(remaining);}
        let widest=0.5*(2880.*allowed/ratio).powf(0.25);let steps=(remaining/widest).ceil().max(1.);
        Some(remaining/steps)
    }
    /// The state of ψ.
    fn enter(&self,psi:&[Complex])->State{
        let n=self.n();let mut x=[Complex::ZERO;MOST_MIXED];
        match &self.how{How::Spectral{inverse,..}=>{for a in 0..n{x[a]=(0..n).fold(Complex::ZERO,|s,b|s+inverse[a*n+b]*psi[b]);}},_=>x[..n].copy_from_slice(&psi[..n])}
        x
    }
    /// ψ of a state.
    fn leave(&self,x:&State,psi:&mut [Complex]){
        let n=self.n();
        match &self.how{How::Spectral{vectors,..}=>{for a in 0..n{psi[a]=(0..n).fold(Complex::ZERO,|s,b|s+vectors[a*n+b]*x[b]);}},_=>psi[..n].copy_from_slice(&x[..n])}
    }
    /// A state propagated over `len` from the time t; `node`: the grid's node i where it is the whole
    /// spacing from it.
    fn advance(&self,x:&mut State,t:f64,len:f64,node:Option<usize>){
        let n=self.n();
        match(&self.how,node){
            (How::Spectral{phases,..},Some(_))=>{for a in 0..n{x[a]=x[a]*phases[a];}}
            (How::Spectral{values,..},None)=>{for a in 0..n{x[a]=x[a]*rotation(values[a],len);}}
            (How::Taylor{propagator,..},Some(_))=>multiply(propagator,n,x),
            (How::Taylor{..},None)=>propagate_dyn(&self.components.statics.hamiltonian,n,len,x),
            (How::Beating{cache:Some(cache)},Some(i))=>multiply(&cache.propagators[i%cache.propagators.len()],n,x),
            (How::Beating{..},_)=>{
                let(c,w)=cf4();let times=[t+c[0]*len,t+c[1]*len];let mut b=vec![Complex::ZERO;n*n];
                Segment::combine(&self.components,times,w,&mut b);propagate_dyn(&b,n,len,x);
                Segment::combine(&self.components,times,[w[1],w[0]],&mut b);propagate_dyn(&b,n,len,x);
            }
        }
    }
    /// ‖ψ‖² of a state.
    fn norm(&self,x:&State)->f64{match &self.how{How::Spectral{gram,..}=>crate::model::hermitian_form(gram,x,self.n()),_=>norm2(x,self.n())}}
    /// The conditional force ⟨ψ|F|ψ⟩/⟨ψ|ψ⟩ at the time t and ‖ψ‖²; `node`: the grid's node there.
    fn force(&self,x:&State,t:f64,node:Option<usize>)->([f64;3],f64){
        let n=self.n();let norm=self.norm(x);let g=&self.components.statics.gradient;
        let force=match(&self.how,node){
            (How::Spectral{forces,..},_)=>std::array::from_fn(|a|-crate::model::hermitian_form(&forces[a],x,n)/norm),
            (How::Taylor{..},_)=>std::array::from_fn(|a|-crate::model::hermitian_form(&g[a],x,n)/norm),
            (How::Beating{cache:Some(cache)},Some(i))=>{let g=&cache.gradients[i%cache.gradients.len()];std::array::from_fn(|a|-crate::model::hermitian_form(&g[a],x,n)/norm)}
            // G(t) = G₀ + Σ_j e^{iΩ_j t} G_j, the beats in pairs G_{−Ω} = G_Ω†: Re⟨G₀⟩ + 2 Σ_{Ω>0} Re e^{iΩt}⟨G_Ω⟩.
            (How::Beating{..},_)=>{
                let half=self.components.beats.len()/2;let mut f:[f64;3]=std::array::from_fn(|a|crate::model::hermitian_form(&g[a],x,n));
                for beat in &self.components.beats[half..]{let phase=Complex::cis(beat.omega*t);for a in 0..3{f[a]+=2.*(phase*form(&beat.gradient[a],x,n)).re;}}
                f.map(|v|-v/norm)
            }
        };
        (force,norm)
    }
    /// −d‖ψ‖²/dt of a state at the time t: ⟨ψ|i(H − H†)|ψ⟩ = −2 Im⟨ψ|H|ψ⟩.
    fn rate(&self,x:&State,t:f64)->f64{
        let n=self.n();
        match &self.how{
            How::Spectral{rate,..}=>-crate::model::hermitian_form(rate,x,n),
            How::Taylor{decay,..}=>crate::model::hermitian_form(decay,x,n),
            How::Beating{..}=>{
                let mut z=form(&self.components.statics.hamiltonian,x,n);
                for beat in &self.components.beats{z+=Complex::cis(beat.omega*t)*form(&beat.hamiltonian,x,n);}
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
        let mut x=self.enter(psi);let mut t=at;
        // The grid: a spectral segment's from `at`, its spacing weighted by the state's amplitudes (`weighted`)
        // and its phases over it; the others' from the segment's start.
        let weighted=self.weighted(&x,remaining,threshold);
        let(origin,delta)=match weighted{Some(d)=>(at,d),None=>(self.start,self.spacing)};
        let phases:Option<Vec<Complex>>=match(&self.how,weighted){(How::Spectral{values,..},Some(d))=>Some(values.iter().map(|&l|rotation(l,d)).collect()),_=>None};
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
                self.leave(&y,psi);return(t+s-at,true,impulse);
            }
            for a in 0..3{impulse[a]+=width/6.*(fa[a]+4.*fm[a]+fb[a]);}
            x=last;fa=fb;na=nb;t=b;if whole{i+=2;}
        }
        self.leave(&x,psi);
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

/// The fourth order's start from a bare ground state (after a jump, and at t = 0, into a cluster with
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
/// segment's path ("spectral", "taylor", "beating", "cached"), the grid's spacing and nodes, per pair of
/// beats ±Ω its frequency, its bound and whether it was left out (`Segment::resolve`), the share, and
/// the seconds its components and its segment take to build.
pub struct Probe{pub path:&'static str,pub spacing:f64,pub nodes:usize,pub beats:Vec<(f64,f64,bool)>,pub share:f64,pub seconds:[f64;2],pub weighted:Option<usize>}

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

/// The most Newton steps of `tilt`: it converges quadratically where the mean lies inside the recoils' hull, and
/// beyond it each step gathers the weights further on the hull's face (the rest is returned either way).
const MOST_TILT:usize=64;

/// The eigen-decomposition of a symmetric 3 × 3 matrix (Jacobi's rotations to double precision): the values and
/// the unit vectors (rows).
fn symmetric(mut a:[[f64;3];3])->([f64;3],[[f64;3];3]){
    let mut v=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]];
    for _ in 0..64{
        let(mut p,mut q,mut big)=(0,1,0.);for i in 0..3{for j in i+1..3{if a[i][j].abs()>big{big=a[i][j].abs();p=i;q=j;}}}
        if !(big>f64::EPSILON*(a[0][0].abs()+a[1][1].abs()+a[2][2].abs())){break;}
        let theta=0.5*(a[q][q]-a[p][p])/a[p][q];let t=theta.signum()/(theta.abs()+(theta*theta+1.).sqrt());
        let c=1./(t*t+1.).sqrt();let s=t*c;
        for k in 0..3{let(x,y)=(a[k][p],a[k][q]);a[k][p]=c*x-s*y;a[k][q]=s*x+c*y;}
        for k in 0..3{let(x,y)=(a[p][k],a[q][k]);a[p][k]=c*x-s*y;a[q][k]=s*x+c*y;}
        for k in 0..3{let(x,y)=(v[k][p],v[k][q]);v[k][p]=c*x-s*y;v[k][q]=s*x+c*y;}
    }
    ([a[0][0],a[1][1],a[2][2]],std::array::from_fn(|i|[v[0][i],v[1][i],v[2][i]]))
}

/// Base weights s_c over recoils μ_c tilted to the mean b by the least information (# Absorption): q_c ∝ s_c e^{λ·μ_c},
/// λ minimizing the convex Φ(λ) = ln Σ_c s_c e^{λ·(μ_c − b)}, whose gradient is the tilted mean less b and whose Hessian
/// the recoils' covariance under q -- Newton's steps on the covariance's range (its eigenvalues above their round-off),
/// each halved until Φ falls, until it falls no further in double precision. The weights and b less their mean: zero
/// (to round-off) where b lies inside the recoils' hull; beyond it the weights gather on the face toward b and the
/// rest is what no mixture of the recoils reaches.
pub fn tilt(s:&[f64],mu:&[[f64;3]],b:[f64;3])->(Vec<f64>,[f64;3]){
    let n=s.len();
    let at=|l:[f64;3]|->(Vec<f64>,f64){
        let e:Vec<f64>=(0..n).map(|c|(0..3).map(|a|l[a]*(mu[c][a]-b[a])).sum()).collect();
        let top=(0..n).filter(|&c|s[c]>0.).map(|c|e[c]).fold(f64::NEG_INFINITY,f64::max);
        let w:Vec<f64>=(0..n).map(|c|if s[c]>0.{s[c]*(e[c]-top).exp()}else{0.}).collect();let z:f64=w.iter().sum();
        (w.iter().map(|x|x/z).collect(),top+z.ln())
    };
    let mean=|q:&[f64]|->[f64;3]{std::array::from_fn(|a|(0..n).map(|c|q[c]*mu[c][a]).sum())};
    let mut lambda=[0.;3];let(mut q,mut phi)=at(lambda);
    for _ in 0..MOST_TILT{
        let m=mean(&q);let g:[f64;3]=std::array::from_fn(|a|m[a]-b[a]);
        let mut h=[[0.;3];3];for c in 0..n{for i in 0..3{for j in 0..3{h[i][j]+=q[c]*(mu[c][i]-m[i])*(mu[c][j]-m[j]);}}}
        let(values,vectors)=symmetric(h);let floor=4.*f64::EPSILON*values.iter().map(|v|v.abs()).sum::<f64>();
        let mut delta=[0.;3];
        for i in 0..3{if values[i]>floor{let p=-(0..3).map(|a|vectors[i][a]*g[a]).sum::<f64>()/values[i];for a in 0..3{delta[a]+=p*vectors[i][a];}}}
        if !((0..3).map(|a|g[a]*delta[a]).sum::<f64>()<0.){break;}
        let mut step=1.;let mut moved=false;
        while step>f64::EPSILON{
            let next:[f64;3]=std::array::from_fn(|a|lambda[a]+step*delta[a]);let(q2,phi2)=at(next);
            if phi2<phi{lambda=next;q=q2;phi=phi2;moved=true;break;}
            step*=0.5;
        }
        if !moved{break;}
    }
    let m=mean(&q);(q,std::array::from_fn(|a|b[a]-m[a]))
}

/// A photon's absorbed recoil (# Absorption) from the uniform u: its parts -- per near beam its wave vector and
/// folded share; the explicit states' weight and their beams' wave vectors and drive weights --, and its exact
/// conditional mean. The shares are tilted to the mean (`tilt`, the explicit part's recoil mean zero); u picks the
/// part, its leftover the explicit part's beam and then the sign of its mean's shot noise; the kick is the picked
/// beam's recoil (the explicit part's less its drives' mean, ± that mean) and the rest of the mean the tilt leaves.
pub fn absorbed_recoil(beams:&[([f64;3],f64)],explicit:Option<(f64,&[([f64;3],f64)])>,mean:[f64;3],u:f64)->[f64;3]{
    let mut s:Vec<f64>=beams.iter().map(|x|x.1).collect();let mut mu:Vec<[f64;3]>=beams.iter().map(|x|x.0).collect();
    if let Some((w,_))=explicit{s.push(w);mu.push([0.;3]);}
    if !(s.iter().sum::<f64>()>0.){return mean;}
    let(q,rest)=tilt(&s,&mu,mean);
    let mut x=u;let mut pick=(0..q.len()).rev().find(|&c|q[c]>0.).unwrap_or(0);
    for(c,&w)in q.iter().enumerate(){if x<w{pick=c;break;}x-=w;}
    let left=if q[pick]>0.{(x/q[pick]).clamp(0.,1.)}else{0.};
    let mut kick=if pick<beams.len(){beams[pick].0}else{
        let drives=explicit.map_or(&[][..],|x|x.1);let total:f64=drives.iter().map(|d|d.1).sum();
        if !(total>0.){[0.;3]}else{
            let average:[f64;3]=std::array::from_fn(|a|drives.iter().map(|d|d.1*d.0[a]).sum::<f64>()/total);
            let mut y=left*total;let mut l=(0..drives.len()).rev().find(|&i|drives[i].1>0.).unwrap();
            for(i,d)in drives.iter().enumerate(){if y<d.1{l=i;break;}y-=d.1;}
            let sign=if y<0.5*drives[l].1{1.}else{-1.};
            std::array::from_fn(|a|drives[l].0[a]-average[a]+sign*average[a])
        }
    };
    for a in 0..3{kick[a]+=rest[a];}
    kick
}

/// The stream of a trajectory's jump number `jump` (the initial threshold is jump 0's).
pub fn jump_stream(seed:u64,index:u64,jump:u64)->Rng{Rng::new(seed,Purpose::Jump,index,jump)}

/// An atom held at a point (control.rs `Twin`), its generator static there: every cluster's H_eff,
/// per class and photon axis its jump operator L (target × cluster, row-major), and its post-jump
/// transient, computed once.
pub struct Held{pub point:Point,pub locals:Vec<Local>,pub jumps:Vec<Vec<[Vec<Complex>;3]>>}

/// The engine: a model and its settings; for a held atom its generator and jumps there, and each
/// cluster's propagator exp(−i H dt/2) over half the settings' step; the momentum diffusion's hook.
pub struct Engine<'a>{pub model:&'a Model,pub settings:Settings,held:Option<(&'a Held,Vec<[Complex;MOST_STATES*MOST_STATES]>)>,pub diffusion:Option<&'a dyn Diffusion>}

impl<'a> Engine<'a>{
    pub fn new(model:&'a Model,settings:Settings)->Engine<'a>{Engine{model,settings,held:None,diffusion:None}}
    /// The engine of an atom held at a point: no axis free, no loss, the held generator and jumps;
    /// the half step's propagators by the exponentials' own series on the basis states. Its clusters
    /// are of ground states (a twin exists for those alone, control.rs).
    pub fn held(model:&'a Model,settings:&Settings,held:&'a Held)->Engine<'a>{
        let mut s=settings.clone();s.free=[false;3];s.loss=false;
        let half=held.locals.iter().enumerate().map(|(k,local)|{let n=model.dimension(k);let mut u=[Complex::ZERO;MOST_STATES*MOST_STATES];
            for j in 0..n{let mut e=[Complex::ZERO;MOST_STATES];e[j]=Complex::ONE;propagate(&local.hamiltonian,n,0.5*s.dt,&mut e,s.truncation);for i in 0..n{u[i*n+j]=e[i];}}u}).collect();
        Engine{model,settings:s,held:Some((held,half)),diffusion:None}
    }
    /// The trap's harmonic flow for τ: along its principal axes when every lab axis is free, else
    /// along the free lab axes.
    fn flow(&self,t:&mut Trajectory,tau:f64){
        let(s,hm)=(&self.settings,self.model.hbar_over_mass);
        if s.free==[true;3]{
            let(mut r,mut kappa)=([0.;3],[0.;3]);
            for i in 0..3{let(x,q)=flow(dot(s.axes[i],t.r),dot(s.axes[i],t.kappa),s.frequencies[i],hm,tau);for a in 0..3{r[a]+=x*s.axes[i][a];kappa[a]+=q*s.axes[i][a];}}
            (t.r,t.kappa)=(r,kappa);
        }else{for a in 0..3{if s.free[a]{(t.r[a],t.kappa[a])=flow(t.r[a],t.kappa[a],s.lab[a],hm,tau);}}}
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
    /// the start, the middle and the end, the point of the middle, the kick's momentum (the light's
    /// and the trap's anharmonic part) and ψ at the middle. A cluster of ground states (`exact` takes
    /// the others).
    fn advance(&self,t:&mut Trajectory,dt:f64)->(f64,f64,f64,crate::model::Point,[f64;3],[Complex;MOST_MIXED]){
        let(m,s)=(self.model,&self.settings);
        if let Some((h,half))=&self.held{
            // Held: the internal state alone, by the generator there -- a whole step by the half
            // step's propagator, a step's part by the series.
            let n=m.dimension(t.cluster);let local=&h.locals[t.cluster];let n0=norm2(&t.psi,n);let mut psi=t.psi;
            let half_step=|psi:&mut [Complex;MOST_MIXED]|if dt==s.dt{let u=&half[t.cluster];let v=*psi;for i in 0..n{let mut z=Complex::ZERO;for j in 0..n{z+=u[i*n+j]*v[j];}psi[i]=z;}}
                else{propagate(&local.hamiltonian,n,0.5*dt,psi,s.truncation);};
            half_step(&mut psi);let nh=norm2(&psi,n);let middle=psi;half_step(&mut psi);let n1=norm2(&psi,n);
            t.psi=psi;t.time+=dt;return(n0,nh,n1,h.point,[0.;3],middle);
        }
        self.flow(t,0.5*dt);
        let point=m.point(t.r,t.kappa);
        let n=m.dimension(t.cluster);let local=m.local(t.cluster,&point,t.time+0.5*dt);
        let n0=norm2(&t.psi,n);
        let mut psi=t.psi;propagate(&local.hamiltonian,n,0.5*dt,&mut psi,s.truncation);let nh=norm2(&psi,n);let middle=psi;
        let force=m.force(t.cluster,&local,&psi);let harmonic=self.harmonic(t.r);let mut kick=[0.;3];
        for a in 0..3{if s.free[a]{kick[a]=dt*(force[a]-harmonic[a]);t.kappa[a]+=kick[a];}}
        propagate(&local.hamiltonian,n,0.5*dt,&mut psi,s.truncation);let n1=norm2(&psi,n);
        t.psi=psi;
        self.flow(t,0.5*dt);
        t.time+=dt;
        (n0,nh,n1,point,kick,middle)
    }

    /// The exact step of τ (module documentation): the flow for τ/2; at the held point every
    /// cluster's generator at the middle time once, the internal state propagated exactly through
    /// the jumps (`jumps` false: none, the step-doubling check's), the impulse along it; the kick; the
    /// flow for τ/2. The kick's momentum (the light's and the trap's anharmonic part).
    fn exact(&self,t:&mut Trajectory,tau:f64,exact:bool,jumps:bool)->[f64;3]{
        let(m,s)=(self.model,&self.settings);
        self.flow(t,0.5*tau);
        let point=m.point(t.r,t.kappa);let start=t.time;
        let mut segments:Vec<(usize,Segment)>=vec![];let mut impulse=[0.;3];let mut elapsed=0.;
        while elapsed<tau{
            let k=t.cluster;
            if !segments.iter().any(|x|x.0==k){let segment=Segment::new(m.components(k,&point),start,tau,s.truncation,s.beats,s.horizon);t.beats=t.beats.max(segment.dropped/tau);segments.push((k,segment));}
            let segment=&segments.iter().find(|x|x.0==k).unwrap().1;
            let(taken,jumped,part)=segment.evolve(&mut t.psi,start+elapsed,tau-elapsed,if jumps{t.threshold}else{0.});
            for a in 0..3{impulse[a]+=part[a];}elapsed+=taken;
            if !jumped{break;}
            t.time=start+elapsed;self.jump(t,point.r,t.time,exact);
        }
        let harmonic=self.harmonic(t.r);let mut kick=[0.;3];
        for a in 0..3{if s.free[a]{kick[a]=impulse[a]-tau*harmonic[a];t.kappa[a]+=kick[a];}}
        if let Some(d)=self.diffusion{d.step(m,t,&point,tau);}
        self.flow(t,0.5*tau);
        t.time=start+tau;
        kick
    }

    /// The exact step's segment of a cluster at a point over τ from `start` (`Probe`).
    pub fn probe(&self,cluster:usize,point:&Point,start:f64,tau:f64,psi:&[Complex])->Probe{
        let clock=std::time::Instant::now();let c=self.model.components(cluster,point);let built=clock.elapsed().as_secs_f64();
        let(pairs,_)=Segment::bounds(&c,tau,self.settings.horizon);let half=c.beats.len()/2;let omegas:Vec<f64>=c.beats[half..].iter().map(|b|b.omega).collect();
        let clock=std::time::Instant::now();let s=Segment::new(c,start,tau,self.settings.truncation,self.settings.beats,self.settings.horizon);let segment=clock.elapsed().as_secs_f64();
        let kept:Vec<f64>=s.components.beats.iter().map(|b|b.omega).collect();
        let path=match &s.how{How::Spectral{..}=>"spectral",How::Taylor{..}=>"taylor",How::Beating{cache:Some(_)}=>"cached",How::Beating{cache:None}=>"beating"};
        let weighted=s.weighted(&s.enter(psi),tau,0.).map(|d|(tau/d).round() as usize);
        Probe{path,spacing:s.spacing,nodes:(tau/s.spacing).round() as usize,beats:pairs.iter().map(|&(i,x,_)|(omegas[i],x,!kept.contains(&omegas[i]))).collect(),share:self.settings.beats,seconds:[built,segment],weighted}
    }
    /// Whether a cluster takes the exact step: one with explicit excited states or a beating generator
    /// (a held atom's clusters are of ground states and static, control.rs).
    fn takes_exact(&self,cluster:usize)->bool{self.held.is_none()&&(self.model.mixed(cluster)||self.model.beating(cluster))}
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
        }else{let(n0,_,n1,_,kick,_)=self.advance(t,dt);(n0,n1,kick)}
    }
    /// A trajectory's state after `count` jump-free steps of dt (the deterministic flow the step check
    /// compares): flow, kick and the internal state's no-jump evolution, no jump, no loss.
    pub fn coast(&self,t:&Trajectory,dt:f64,count:usize)->Trajectory{let mut x=t.clone();for _ in 0..count{self.free_step(&mut x,dt);}x}
    /// The step-doubling error of one step of dt from a trajectory's state, without jumps: one step
    /// against two of dt/2, as (|Δκ|, |kick|) -- the momentum's difference and the step's
    /// non-harmonic impulse (1/um) -- and (max |Δp|, |Δ ln‖ψ‖²|, |ln‖ψ‖²|) -- the largest difference of
    /// the cluster's state populations, of the norm's decay and the decay itself.
    pub fn doubling(&self,t:&Trajectory,dt:f64)->([f64;2],[f64;3]){
        let n=self.model.dimension(t.cluster);
        let mut one=t.clone();let(n0,n1,kick)=self.free_step(&mut one,dt);
        let mut two=t.clone();self.free_step(&mut two,0.5*dt);let(_,n2,_)=self.free_step(&mut two,0.5*dt);
        let dk=(0..3).map(|a|(one.kappa[a]-two.kappa[a]).powi(2)).sum::<f64>().sqrt();let impulse=kick.iter().map(|x|x*x).sum::<f64>().sqrt();
        let(a,b)=(norm2(&one.psi,n),norm2(&two.psi,n));
        let dp=(0..n).map(|i|(one.psi[i].norm_sqr()/a-two.psi[i].norm_sqr()/b).abs()).fold(0.,f64::max);
        ([dk,impulse],[dp,((n1/n0).ln()-(n2/n0).ln()).abs(),(n1/n0).ln().abs()])
    }

    /// One step of dt: flow, kick and flow (`advance`), or the exact step in a cluster with explicit
    /// excited states. Where ‖ψ‖² crosses the threshold, the step is redone to the crossing, the jump
    /// made there and the rest of the step taken from it in the state the jump left -- exactly when
    /// it lands in a cluster with explicit states: the force and the motion follow the internal state
    /// on either side of the jump. Then the loss.
    pub fn step(&self,t:&mut Trajectory){
        let(m,s)=(self.model,&self.settings);
        // The first order in the Doppler shift where it holds; the exact response beyond.
        let doppler=m.doppler_ratio(t.kappa);let exact=doppler>s.doppler;
        if !exact{t.doppler=t.doppler.max(doppler);}
        // The state the loss is judged at: the step's middle where it went without a jump -- the point
        // the flow's first half reached, κ there with half the kick, ψ after the first half step, the
        // middle's time --, else its end.
        let(point,middle)=if self.takes_exact(t.cluster){self.exact(t,s.dt,exact,true);(m.point(t.r,t.kappa),None)}else{
            let mut left=s.dt;
            loop{
                let start=t.clone();
                let(n0,nh,n1,point,kick,half)=self.advance(t,left);
                if n1>t.threshold{let kappa:[f64;3]=std::array::from_fn(|a|point.kappa[a]+0.5*kick[a]);break(point,Some((kappa,half,t.time-0.5*left)));}
                // The crossing on ln‖ψ‖², linear on the half where it lies; the step redone to it.
                let(ta,tb,la,lb)=if nh<=t.threshold{(0.,0.5*left,n0.ln(),nh.ln())}else{(0.5*left,left,nh.ln(),n1.ln())};
                let at=if lb<la{(ta+(t.threshold.ln()-la)/(lb-la)*(tb-ta)).clamp(ta,tb)}else{tb};
                *t=start;if at>0.{self.advance(t,at);}
                self.jump(t,t.r,t.time,exact);
                left-=at;
                if !(left>0.){break(m.point(t.r,t.kappa),None);}
                if self.takes_exact(t.cluster){self.exact(t,left,exact,true);break(m.point(t.r,t.kappa),None);}
            }
        };
        // The loss: an atom unbound -- its mechanical energy above the trap's far level -- outside the trap's
        // region (`Settings::region`), of that one state.
        if s.loss&&self.harmonic_potential(point.r)>s.region{
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
        let transient:Option<[f64;4]>=(self.held.is_none()&&m.mixed(class.target)).then(||std::array::from_fn(|_|rng.uniform()));
        let(direction,e,f,size)=emit(&mut rng,&amplitude[pick],nt,polarization);
        // The recoils of a moving atom: the photon's, the absorbed beam's and the exact conditional
        // mean's correction.
        if let Some(photons)=&photons{
            let(gradient,parts)=photons.absorption(pick,e);
            self.kick(t,&point,direction,class.wave_number,&f,size,&gradient,&parts,absorbed);
        }
        let mut next=[Complex::ZERO;MOST_MIXED];for i in 0..nt{next[i]=f[i]*(1./size.sqrt());}
        t.cluster=class.target;t.psi=next;t.threshold=threshold;
        if let Some(draws)=transient{self.transient(t,&point,time,draws,&mut rng);}
        if self.held.is_none(){slow_start(m,t,&point,time);}
    }

    /// A moving atom's recoils from a photon (direction n, wave number k, the state it leaves f and
    /// ‖f‖², the gradient of its amplitudes along the polarization ε per row and axis, and its absorbed
    /// light's parts): −k n, and the absorbed light's about the exact conditional mean Im⟨f|∇f⟩/‖f‖²
    /// (`absorbed_recoil`, by the uniform `absorbed`).
    fn kick(&self,t:&mut Trajectory,point:&Point,direction:[f64;3],k:f64,f:&[Complex;MOST_STATES],size:f64,du:&[[Complex;3];MOST_STATES],parts:&Absorption,absorbed:f64){
        let m=self.model;let mut kick=direction.map(|x|-k*x);
        let mean:[f64;3]=std::array::from_fn(|x|{let mut s=0.;for i in 0..MOST_STATES{s+=(f[i].conj()*du[i][x]).im;}s/size});
        let wave=|list:&[(usize,f64)]|->Vec<([f64;3],f64)>{list.iter().map(|&(l,w)|(m.geometry[l].wave_vector(),w)).collect()};
        let(beams,drives)=(wave(&parts.beams),parts.explicit.as_ref().map(|(_,d)|wave(d)));
        let explicit=parts.explicit.as_ref().zip(drives.as_deref()).map(|((w,_),d)|(*w,d));
        let recoil=absorbed_recoil(&beams,explicit,mean,absorbed);
        for a in 0..3{kick[a]+=recoil[a];}
        if let Some(d)=self.diffusion{d.photon(m,t,point,f,&mut kick);}
        for a in 0..3{if self.settings.free[a]{t.kappa[a]+=kick[a];}}
    }

    /// The post-jump transient (module documentation) from the normalized ground state the jump
    /// left: with probability ψ′†Yψ′ one more photon of a piece S_s (by ‖S_s ψ′‖², its kicks a
    /// jump's), else ψ′ ← (1 − ½Y)ψ′. `draws`: whether, which piece, the polarization, the beam.
    fn transient(&self,t:&mut Trajectory,point:&Point,time:f64,draws:[f64;4],rng:&mut Rng){
        let m=self.model;let k=t.cluster;
        let ops=m.transient(k,point,time);let n0=ops.n0;let psi=t.psi;
        let mut y=[Complex::ZERO;MOST_STATES];for a in 0..n0{for b in 0..n0{y[a]+=ops.y[a*n0+b]*psi[b];}}
        let p:f64=(0..n0).map(|a|(psi[a].conj()*y[a]).re).sum();
        if !(p>0.){return;}
        if draws[0]>=p{
            // No photon: ψ′ ← (1 − ½Y)ψ′.
            let mut next=t.psi;for a in 0..n0{next[a]=psi[a]-y[a]*0.5;}let norm=norm2(&next,n0).sqrt();for z in next[..n0].iter_mut(){*z=*z*(1./norm);}
            t.psi=next;return;
        }
        // One more photon: the piece by its weight.
        let amplitudes:Vec<[[Complex;MOST_STATES];3]>=ops.pieces.iter().map(|piece|{let nt=m.ground_dimension(piece.target);
            std::array::from_fn(|a|{let mut u=[Complex::ZERO;MOST_STATES];for i in 0..nt{for b in 0..n0{u[i]+=piece.ops[a][i*n0+b]*psi[b];}}u})}).collect();
        let weights:Vec<f64>=ops.pieces.iter().zip(&amplitudes).map(|(piece,u)|{let nt=m.ground_dimension(piece.target);(0..3).map(|a|norm2(&u[a],nt)).sum()}).collect();
        let total:f64=weights.iter().sum();if !(total>0.){return;}
        let mut x=draws[1]*total;let mut pick=weights.len()-1;for(i,&w)in weights.iter().enumerate(){if x<w{pick=i;break;}x-=w;}
        let piece=&ops.pieces[pick];let nt=m.ground_dimension(piece.target);
        let(direction,e,f,size)=emit(rng,&amplitudes[pick],nt,draws[2]);
        let(_,du,beams)=m.transient_photon(k,point,time,piece,&psi);
        let parts=Absorption{beams:beams.iter().map(|(l,v)|(*l,(0..nt).map(|i|(0..3).fold(Complex::ZERO,|s,a|s+v[a][i]*e[a]).norm_sqr()).sum())).collect(),explicit:None};
        // The piece's gradient along the polarization (`kick`), of the photon's own sign: transient_photon's amplitude is
        // Σ e^{−iε̃t} R ψ′, the photon S ψ′ its negative (X = −Σ e^{−iε̃t} R, model.rs `transient`).
        let along:[[Complex;3];MOST_STATES]=std::array::from_fn(|i|std::array::from_fn(|x|-(0..3).fold(Complex::ZERO,|s,a|s+du[a][x][i]*e[a])));
        self.kick(t,point,direction,piece.wave_number,&f,size,&along,&parts,draws[3]);
        let mut next=[Complex::ZERO;MOST_MIXED];for i in 0..nt{next[i]=f[i]*(1./size.sqrt());}
        t.cluster=piece.target;t.psi=next;
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    use crate::linear::uniform;
    use crate::model::{Beat,Generator};
    use std::f64::consts::TAU;

    /// A kick's mean and second moment per axis over the draw's uniform (midpoints of a fine grid).
    fn over_the_draw(kick:impl Fn(f64)->[f64;3])->([f64;3],[f64;3]){
        let n=1usize<<16;let(mut m,mut s)=([0.;3],[0.;3]);
        for i in 0..n{let k=kick((i as f64+0.5)/n as f64);for a in 0..3{m[a]+=k[a]/n as f64;s[a]+=k[a]*k[a]/n as f64;}}
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
        for u in [0.01,0.4,0.99]{assert!(close(absorbed_recoil(&[(k,0.3)],None,k,u),k,1e-12));}
        let(mean,second)=over_the_draw(|u|absorbed_recoil(&[],Some((0.3,&[(k,0.7)])),[0.;3],u));
        assert!(close(mean,[0.;3],1e-3),"{mean:?}");assert!(close(second,[9.,0.,49.],1e-9),"{second:?}");
    }
    /// The draw over a photon's beams: where the mean is the shares' they are not tilted, and the second moment is the
    /// incoherent one; a mean interference moves off them, within the beams' hull, is reached by the beams' own kicks,
    /// the second moment k_z² on an axis of one |k_z|; beyond the hull the rest is added -- the mean exact in all three.
    #[test]
    fn the_absorbed_recoil_keeps_the_exact_mean_and_the_beams_recoils(){
        let k=pairs();let shares=[0.31,0.05,0.22,0.11,0.2,0.11];let total:f64=shares.iter().sum();
        let beams:Vec<([f64;3],f64)>=k.iter().copied().zip(shares).collect();
        let average:[f64;3]=std::array::from_fn(|a|beams.iter().map(|b|b.1*b.0[a]).sum::<f64>()/total);
        let(q,rest)=tilt(&shares,&k,average);
        assert!(close(rest,[0.;3],1e-12),"{rest:?}");for c in 0..6{assert!((q[c]-shares[c]/total).abs()<=1e-12,"{q:?}");}
        let(mean,second)=over_the_draw(|u|absorbed_recoil(&beams,None,average,u));
        assert!(close(mean,average,2e-3),"{mean:?}");
        for a in 0..3{let incoherent=beams.iter().map(|b|b.1*b.0[a]*b.0[a]).sum::<f64>()/total;assert!((second[a]-incoherent).abs()<=2e-2,"{second:?}");}
        // Inside the hull, off the shares' mean.
        let inside=[0.4,0.3,4.5];let(q,rest)=tilt(&shares,&k,inside);
        assert!(close(rest,[0.;3],1e-6),"{rest:?}");assert!(q.iter().all(|&w|w>0.),"{q:?}");
        let(mean,second)=over_the_draw(|u|{let kick=absorbed_recoil(&beams,None,inside,u);assert!(k.iter().any(|b|close(*b,kick,1e-6)),"{kick:?}");kick});
        assert!(close(mean,inside,2e-3),"{mean:?}");assert!((second[2]-k[0][2]*k[0][2]).abs()<=1e-5,"{second:?}");
        // Beyond it (a superoscillation): the beams toward it, the rest a deterministic kick.
        let beyond=[0.,0.,20.];let(q,rest)=tilt(&shares,&k,beyond);
        assert!((0..3).all(|a|((0..6).map(|c|q[c]*k[c][a]).sum::<f64>()+rest[a]-beyond[a]).abs()<=1e-12),"{q:?} {rest:?}");
        assert!(rest[2]>0.&&(0..6).filter(|&c|k[c][2]<0.).all(|c|q[c]<=1e-6),"{q:?} {rest:?}");
        let(mean,_)=over_the_draw(|u|absorbed_recoil(&beams,None,beyond,u));assert!(close(mean,beyond,2e-3),"{mean:?}");
    }
    /// A photon of folded beams and explicit states: the mean is the folded part's whatever the explicit part weighs (its
    /// kicks have mean zero, the Ehrenfest force carrying its mean); the explicit part's second moment is its drives'.
    #[test]
    fn the_explicit_part_s_kicks_add_no_mean(){
        let k=pairs();let drives:Vec<([f64;3],f64)>=k.iter().copied().zip([0.4,0.1,0.2,0.1,0.15,0.05]).collect();
        let beams=[(k[0],0.02),(k[1],0.01)];let mean=[0.02,0.,0.04];
        let(got,_)=over_the_draw(|u|absorbed_recoil(&beams,Some((1.,&drives)),mean,u));assert!(close(got,mean,3e-3),"{got:?}");
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
    /// decays monotonically), and its gradients, the beats' paired (G_{−Ω} = G_Ω†); with `period`.
    fn random(n:usize,omegas:&[f64],period:Option<f64>,seed:u64)->Components{
        let mut next=uniform(seed);
        let mut h=hermitian(&mut next,n,60.);for i in 0..n{h[i*n+i]+=Complex::new(80.*next(),-12.*(next()+0.5));}
        let gradient=std::array::from_fn(|_|hermitian(&mut next,n,30.));
        let raw:Vec<(Vec<Complex>,[Vec<Complex>;3])>=omegas.iter().map(|_|{let b:Vec<Complex>=(0..n*n).map(|_|Complex::new(next()*20.,next()*20.)).collect();
            (b,std::array::from_fn(|_|(0..n*n).map(|_|Complex::new(next()*10.,next()*10.)).collect()))}).collect();
        let adjoint=|x:&[Complex]|->Vec<Complex>{(0..n*n).map(|k|x[(k%n)*n+k/n].conj()).collect()};
        let mut beats:Vec<Beat>=raw.iter().zip(omegas).rev().map(|((b,g),&w)|Beat{omega:-w,hamiltonian:adjoint(b),gradient:std::array::from_fn(|a|adjoint(&g[a]))}).collect();
        beats.extend(raw.into_iter().zip(omegas).map(|((b,g),&w)|Beat{omega:w,hamiltonian:b,gradient:g}));
        Components{statics:Generator{n,hamiltonian:h,scalar:0.,gradient},beats,period}
    }
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

    /// CF4 is exact for a static generator and of fourth order for a beating one: its error falls 16-fold
    /// with the spacing halved, against RK4 at a far finer step; and at the node rule's spacing it is far
    /// below the impulse's quadrature share.
    #[test]
    fn cf4_is_exact_for_a_static_generator_and_of_fourth_order_with_beats(){
        let n=5;let tau=0.1;let psi=state(n,3);
        let still=random(n,&[],None,7);
        let exact=Segment::new(still.clone(),0.,tau,0.,0.,tau);let mut a=psi;exact.evolve(&mut a,0.,tau,0.);
        let cf4=Segment{components:still,start:0.,spacing:tau/8.,how:How::Beating{cache:None},dropped:0.};let mut b=psi;cf4.evolve(&mut b,0.,tau,0.);
        assert!(distance(&b,&a,n)<=1e-12,"static: {:e}",distance(&b,&a,n));
        let c=random(n,&[150.,300.],None,11);let(target,_)=reference(&c,&psi,0.,tau,20000);
        let error=|steps:usize|{let s=Segment{components:c.clone(),start:0.,spacing:tau/steps as f64,how:How::Beating{cache:None},dropped:0.};let mut x=psi;s.evolve(&mut x,0.,tau,0.);distance(&x,&target,n)};
        let e:Vec<f64>=[32,64,128].iter().map(|&s|error(s)).collect();
        assert!(e[1]/e[2]>12.&&e[1]/e[2]<20.,"errors {e:?}: ratio {}",e[1]/e[2]);
        let rule=Segment::new(c.clone(),0.,tau,0.,0.,tau);let mut x=psi;rule.evolve(&mut x,0.,tau,0.);
        assert!(distance(&x,&target,n)<=1e-6,"at the node rule's spacing {}: {:e}",rule.spacing,distance(&x,&target,n));
    }

    /// Beats of one period: the cache's propagators and node gradients give the uncached grid's state and
    /// impulse, from the step's start and from a time between nodes (after a jump).
    #[test]
    fn a_period_s_cache_gives_the_uncached_grid_s_evolution(){
        let n=4;let w=230.;let c=random(n,&[w,2.*w],Some(TAU/w),5);let(start,tau)=(0.37,0.4);
        let cached=Segment::new(c.clone(),start,tau,1e-9,1e-9,tau);
        assert!(matches!(cached.how,How::Beating{cache:Some(_)}),"the step holds {} spacings",tau/cached.spacing);
        let plain=Segment{components:c,start,spacing:cached.spacing,how:How::Beating{cache:None},dropped:0.};
        for offset in [0.,0.0123]{
            let psi=state(n,9);let(mut a,mut b)=(psi,psi);
            let(_,_,ia)=cached.evolve(&mut a,start+offset,tau-offset,0.);let(_,_,ib)=plain.evolve(&mut b,start+offset,tau-offset,0.);
            assert!(distance(&a,&b,n)<=1e-11,"offset {offset}: {:e}",distance(&a,&b,n));
            let size=ib.iter().map(|x|x.abs()).fold(0.,f64::max);
            for k in 0..3{assert!((ia[k]-ib[k]).abs()<=1e-11*size,"offset {offset}, axis {k}: {} against {}",ia[k],ib[k]);}
        }
    }

    /// A static generator through its eigendecomposition and through its Taylor propagator: the same
    /// root of the norm and state there, and impulses within the quadrature's bound of a fine grid's.
    #[test]
    fn the_spectral_and_the_taylor_segments_agree(){
        let n=6;let tau=0.1;let c=random(n,&[],None,13);let psi=state(n,17);
        let spectral=Segment::new(c.clone(),0.,tau,1e-9,1e-9,tau);let taylor=Segment::new(c.clone(),0.,tau,0.,0.,tau);
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
        let finer=Segment{components:taylor.components.clone(),start:0.,spacing:taylor.spacing/64.,how:How::Taylor{propagator:exponential(&taylor.components.statics.hamiltonian,n,taylor.spacing/64.),decay:vec![]},dropped:0.};
        let mut x=psi;let(_,_,reference)=finer.evolve(&mut x,0.,tau,0.);let mut y=psi;let(_,_,mine)=spectral.evolve(&mut y,0.,tau,0.);
        let size=reference.iter().map(|v|v.abs()).fold(0.,f64::max);
        for k in 0..3{assert!((mine[k]-reference[k]).abs()<=QUADRATURE*size&&(whole[k]-reference[k]).abs()<=QUADRATURE*size,"axis {k}: {} and {} against {}",mine[k],whole[k],reference[k]);}
        let size=ib.iter().map(|v|v.abs()).fold(0.,f64::max);
        for k in 0..3{assert!((ia[k]-ib[k]).abs()<=QUADRATURE*size,"to the root, axis {k}: {} against {}",ia[k],ib[k]);}
    }

    /// A fast weak beat within the exponentials' share is left out (the step then static, through its
    /// eigendecomposition), and the state differs from the beat's resolved evolution within its bound; a
    /// share too small for it keeps it.
    #[test]
    fn a_beat_within_the_share_is_left_out(){
        let n=4;let tau=0.05;let w=4e4;let mut c=random(n,&[w],None,29);
        for b in c.beats.iter_mut(){for z in b.hamiltonian.iter_mut(){*z=*z*1e-5;}for g in b.gradient.iter_mut(){for z in g.iter_mut(){*z=*z*1e-5;}}}
        let size=|h:&[Complex]|column_norm(h,n).max(row_norm(h,n));
        let bound=2.*c.beats.iter().map(|b|size(&b.hamiltonian)).sum::<f64>()*(1.+(size(&c.statics.hamiltonian)+c.beats.iter().map(|b|size(&b.hamiltonian)).sum::<f64>())*tau)/w;
        let left=Segment::new(c.clone(),0.,tau,1e-6,1e-6,tau);assert!(matches!(left.how,How::Spectral{..}),"the beat left out");
        assert!(matches!(Segment::new(c.clone(),0.,tau,1e-12,1e-12,tau).how,How::Beating{..}),"a smaller share keeps it");
        let resolved=Segment{components:c,start:0.,spacing:TAU/w/16.,how:How::Beating{cache:None},dropped:0.};
        let psi=state(n,31);let(mut a,mut b)=(psi,psi);left.evolve(&mut a,0.,tau,0.);resolved.evolve(&mut b,0.,tau,0.);
        assert!(distance(&a,&b,n)<=bound,"{:e} against the bound {bound:e}",distance(&a,&b,n));
    }

    /// A static generator whose fast part the state hardly holds: its spectral grid weighted by the
    /// state's amplitudes is far coarser than the rule's, and the impulse stays within the quadrature's
    /// share of a fine grid's.
    #[test]
    fn a_spectral_grid_follows_the_state_s_amplitudes(){
        let n=6;let tau=0.2;let mut c=random(n,&[],None,37);
        // Two blocks: three slow states near zero, three at −400 rad/us, coupled weakly; the state in the first.
        for a in 0..n{for b in 0..n{let scale=if a<3&&b<3{0.1}else if (a<3)!=(b<3){0.05}else{1.};c.statics.hamiltonian[a*n+b]=c.statics.hamiltonian[a*n+b]*scale;}}
        for a in 3..n{c.statics.hamiltonian[a*n+a]+=Complex::new(-400.,0.);}
        let mut psi=state(n,41);for z in psi[3..n].iter_mut(){*z=*z*0.01;}let norm=norm2(&psi,n).sqrt();for z in psi[..n].iter_mut(){*z=*z*(1./norm);}
        let s=Segment::new(c.clone(),0.,tau,1e-9,1e-9,tau);assert!(matches!(s.how,How::Spectral{..}));
        let weighted=s.weighted(&s.enter(&psi),tau,0.).unwrap();
        // The cross components' weight is about the fast part's amplitude, 1%: (ω·2δ)⁴ may grow about a hundredfold.
        assert!(weighted>=2.*s.spacing,"weighted {weighted:e} against the rule's {:e}",s.spacing);
        let fine=Segment{components:c,start:0.,spacing:s.spacing/16.,how:How::Taylor{propagator:exponential(&s.components.statics.hamiltonian,n,s.spacing/16.),decay:vec![]},dropped:0.};
        let(mut a,mut b)=(psi,psi);let(_,_,mine)=s.evolve(&mut a,0.,tau,0.);let(_,_,reference)=fine.evolve(&mut b,0.,tau,0.);
        assert!(distance(&a,&b,n)<=1e-11,"states {:e}",distance(&a,&b,n));
        let size=reference.iter().map(|v|v.abs()).fold(0.,f64::max);
        for k in 0..3{assert!((mine[k]-reference[k]).abs()<=QUADRATURE*size,"axis {k}: {} against {}",mine[k],reference[k]);}
    }

    /// A beating segment's root of the norm and the impulse up to it: CF4 at the node rule's spacing
    /// against an eight-fold finer one.
    #[test]
    fn a_beating_segment_finds_the_norm_s_root(){
        let n=5;let tau=0.12;let c=random(n,&[170.,410.],None,19);let psi=state(n,23);
        let rule=Segment::new(c.clone(),0.,tau,0.,0.,tau);let fine=Segment{components:c,start:0.,spacing:rule.spacing/8.,how:How::Beating{cache:None},dropped:0.};
        let mut end=psi;rule.evolve(&mut end,0.,tau,0.);let threshold=0.5*(1.+norm2(&end,n));
        let(mut a,mut b)=(psi,psi);let(sa,ja,ia)=rule.evolve(&mut a,0.,tau,threshold);let(sb,jb,ib)=fine.evolve(&mut b,0.,tau,threshold);
        assert!(ja&&jb);
        assert!((sa-sb).abs()<=1e-7*tau,"roots {sa} and {sb}");
        assert!(distance(&a,&b,n)<=1e-6,"states at the root: {:e}",distance(&a,&b,n));
        let size=ib.iter().map(|v|v.abs()).fold(0.,f64::max);
        for k in 0..3{assert!((ia[k]-ib[k]).abs()<=QUADRATURE*size,"axis {k}: {} against {}",ia[k],ib[k]);}
    }
}
