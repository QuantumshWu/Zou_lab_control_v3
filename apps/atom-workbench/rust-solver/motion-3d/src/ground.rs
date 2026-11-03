//! The atom's internal state on clusters of states: the ground manifold and the excited states
//! whose couplings must be kept, every other excited state folded at second order; its generator
//! at any point r of the trap, to first order in the Doppler and recoil shifts of the folded part.
//!
//! # Far components
//! Every beam drives a co-rotating component, V_l e^{−iδ_l t} with δ_l = 2π(ν_l − ν_frame), and a
//! counter-rotating one, R_l e^{+i2π(ν_l + ν_frame)t}, each with its amplitude a_l(r). The beams of
//! one frequency drive one field, Σ_l a_l(r) V_l (and Σ_l a_l(r)* R_l): its components are one
//! component each (`Component`, its size the beams' sum, every beam at its peak in phase),
//! classified by the rule of the population core (`classify`) against the 3D solve's budget
//! (ledger.rs): a component folds when it is far, |ν| ≥ FOLD_DISTANCE ‖L0‖ with its mixing
//! m = size/(|ν| − ‖L0‖) within m² ≤ budget, beyond the largest frequency gap of at least
//! `fold_gap` = 1/√budget above every component that may not fold. A folded component enters as
//! its second-order (van Vleck) shift on the ground and the excited states, with the exact
//! bare-energy denominators -- quadratic in its field, so that its beams interfere: a standing wave,
//! a lattice of crossed beams --; the others are near carriers. Each beam's static shift through the
//! states outside the model O_l (in H0 at the focus, the beams of one frequency summed there) is
//! quadratic in its field too. The beams of one frequency whose fields are parallel (their
//! couplings span one matrix, `Unit`: a retro-reflected tweezer, a beam alone) shift the states by
//! their summed field's intensity s_u(r) = |Σ_l a_l(r) γ_l|²/N_u alone, exactly:
//!
//! ```text
//! H(r) = H0 + Σ_u (s_u(r) − s_u(0)) O_u + Σ_u s_u(r) F_u      (`Ground::hamiltonian_at`)
//! ```
//!
//! O_u and F_u the unit's outside shift and fold at s_u = 1 (a beam alone: s = |a_l|², O_l, F_l). The
//! beams of a frequency whose fields span more matrices (polarizations that differ) are units of one
//! beam each, their interference -- a polarization lattice, which their intensities alone do not
//! fix -- left out, its bound in the ledger (`Ground::incoherent`, model.rs).
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
//! explicit one. Closure: every folded drive from a cluster's ground state, and every folded coupling
//! of an explicit state, is judged again with the cluster's dressing -- the spectral norm of the
//! bounds of its kept couplings and light shifts -- subtracted from its distance; one that fails is
//! kept (promoted), a drive static in the frames between two states of one cluster is kept (it costs
//! nothing), and the rule runs again until nothing changes. When every drive folds, the clusters are
//! the ground states' alone.
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
//! - a pair of channels: its van Vleck shift from the far channel's path (c, b) → (c′, a) → (ĉ, b̂),
//!   (ĉ, b̂) co-resonant with (c, b), W̃ D U with U = −V_c′^f† G V_c^f and W̃ = −V_ĉ^f† G V_c′^f at the
//!   path's energy z (the cluster's centroid in H(r) plus δ_c, G = (K − z)⁻¹) and D = 1/(z − E_a − δ_c′);
//! - a static coupling: its van Vleck shift with the exact bare-energy denominators;
//! - a pair of photons: they are distinguishable, and the jump's channels split by frequency.
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
//! H_eff = Σ_s (E_s − θ_s)|s⟩⟨s| + Σ_kept e^{iΩ̃t} X_ab |a⟩⟨b| + Σ_paths e^{iΩ̃t} W̃ D U
//!       + Σ_kept drives e^{iΩ̃t} V_c[e,b] |e⟩⟨b| + h.c. + Σ_{e,e′∈P} (H_ee′ kept, or its van Vleck shift)
//!       + Σ_{e,e′∈P} ½ Σ_b V_c^f[e,b] V_c^f[e′,b]* (1/(E_e − δ_c − E_b) + 1/(E_e′ − δ_c − E_b)) |e⟩⟨e′|
//!       − (i/2) Σ_φ J_φ†J_φ  (on the explicit states; the folded part is X's)
//! J_{a,φ} = Σ_{(c,b→b′)∈φ} e^{iω̃t} [C_a R_c]_{b′b} |b′⟩⟨b| − Σ_{(e→b′)∈φ} e^{iω̃t} [C_a]_{b′e} |b′⟩⟨e|
//! ```
//!
//! ω̃ = θ_b′ − θ_e for an explicit state's photon, θ_b′ − θ_b − δ_c for a folded one.
//!
//! Readout: the bare ground density is ρ − ½{Y, ρ}, Y = Σ R′†R over the co-resonant channel
//! pairs, and an excited state's population tr(E_i ρ), E_i = Σ R′†|i⟩⟨i|R, plus ρ_ee when explicit.
//!
//! # First order in the Doppler and recoil shifts
//! Absorbing from beam l shifts the excited state by the kinetic energy w_l = (ħ/m)(k_l·κ + k_l²/2);
//! to first order R_l → R_l − w_l R2_l, and `Ground::response` gives the shifted response to all
//! orders. An explicit drive needs neither: the moving atom's phase e^{ik_l·r(t)} is its Doppler shift.
//!
//! # Approximations and their estimates (the solve's ledger)
//! - the far components' fold: Σ m²;
//! - the couplings' fold: Σ m² of the folded pairs of channels and static couplings;
//! - the drives' fold: the largest folded total saturation S_b of a ground state;
//! - the zeroth-order elimination: η², the folded excited population along the trajectories (lib.rs);
//! - first order in the Doppler shift: (kv/σ_min)² (engine.rs);
//! - an emitted photon has its carrier's wave number: ≤ 1e-7 on the D2 line.
use crate::ledger::BUDGET;
use crate::linear::{hermitian_eigenvalues,Dense,Lu};
use crate::{Beam,Complex,Request,STATES};
use atom_population_core::lattice::classification::FOLD_DISTANCE;
use std::f64::consts::TAU;

/// A drive's saturation parameter is 2(g/|ΔE|)²: the relative error of its linear response, the
/// estimate of its fold.
pub const SATURATION:f64=2.;

/// The co-rotating component of the beams of one frequency (`counter` false: V e^{−iνt},
/// ν = δ = 2π(ν_beam − ν_frame)) or their counter-rotating one (R e^{−iνt}, ν = −2π(ν_beam + ν_frame)),
/// each with its Hermitian conjugate; its size Σ_l 2‖V_l‖_F bounds its commutator, every beam at its
/// peak in phase, and `folded` is the rule's verdict.
#[derive(Clone,Debug)]
pub struct Component{pub beams:Vec<usize>,pub counter:bool,pub nu:f64,pub size:f64,pub folded:bool}
/// A unit of the light's static shifts (module documentation, # Far components): its beams, of one
/// frequency, and each one's weight γ_l relative to the first's field -- V_l = γ_l V_first,
/// R_l = γ_l* R_first, O_l = |γ_l|² O_first --, N = (Σ_l |γ_l|)² and its intensity at the focus s(0)
/// = |Σ_l γ_l|²/N. A beam alone: γ = 1, N = 1, s(0) = 1.
#[derive(Clone,Debug)]
pub struct Unit{pub beams:Vec<usize>,pub weights:Vec<Complex>,pub norm:f64,pub focus:f64}
impl Unit{
    /// Its intensity s = |Σ_l a_l γ_l|²/N at the beams' amplitudes a (indexed by beam), in [0, 1].
    pub fn intensity(&self,a:&[Complex])->f64{self.field(a).norm_sqr()/self.norm}
    /// Σ_l a_l γ_l at the beams' amplitudes a (indexed by beam).
    pub fn field(&self,a:&[Complex])->Complex{self.beams.iter().zip(&self.weights).fold(Complex::ZERO,|s,(&l,&w)|s+a[l]*w)}
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
/// A folded pair's van Vleck path from the cluster: channel `from` (in the cluster), the far
/// channel `via`, back into `back` (in the cluster, co-resonant with `from`), its residual beat
/// θ_back − θ_from + δ_back − δ_from.
#[derive(Clone,Copy,Debug)]
pub struct Path{pub from:Channel,pub via:Channel,pub back:Channel,pub residual:f64}
/// A jump channel from a cluster: the photons of one frequency landing in cluster `target` -- the
/// folded drives' (`photons`) and the explicit states' (`direct`: the explicit state's place in the
/// cluster, the ground state it lands in, a position in `Ground::ground_states`) -- each with its
/// residual ω̃ − ω̃₀ (rad/us), and their wave number.
#[derive(Clone,Debug)]
pub struct Class{pub target:usize,pub photons:Vec<(Photon,f64)>,pub direct:Vec<(usize,usize,f64)>,pub wave_number:f64}
/// A cluster: its ground states (positions in `Ground::ground_states`) and explicit excited states
/// (positions in `Ground::excited_states`), its basis in that order, and each one's frame θ (rad/us);
/// its kept couplings between ground states (per carrier pair, and H's static ones: row, column),
/// its van Vleck paths (the folded pairs of channels', and the folded static couplings': row, column
/// and the far ground state, between states of one frame); its kept drives (row, column, carrier,
/// residual θ_e − θ_b − δ_c), H's kept couplings between its explicit states (row, column) and the
/// van Vleck paths of their folded ones (row, column, the far excited state); its jumps' classes and
/// its co-resonant channels (each with its residual, the readout's coherent sums).
#[derive(Clone,Debug)]
pub struct Cluster{
    pub states:Vec<usize>,
    pub excited:Vec<usize>,
    pub frames:Vec<f64>,
    pub excited_frames:Vec<f64>,
    pub groups:Vec<Group>,
    pub statics:Vec<(usize,usize)>,
    pub paths:Vec<Path>,
    pub static_paths:Vec<(usize,usize,usize)>,
    pub drives:Vec<(usize,usize,usize,f64)>,
    pub excited_statics:Vec<(usize,usize)>,
    pub excited_paths:Vec<(usize,usize,usize)>,
    pub classes:Vec<Class>,
    pub coherent:Vec<Vec<(Channel,f64)>>,
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
}

/// A cluster of ground states' fourth-order fold at a point and a time (`Ground::fourth`, the dense
/// reference): H̃ (its anti-Hermitian part its jumps' −(i/2) Σ L̃†L̃), every class's jumps L̃ per axis
/// (target cluster's ground states × cluster), the cluster's post-jump map Π (as a jump's target:
/// the slow part M^{1/2}(1 + YW) of a bare ground state, its weight A = ‖Π f‖²), and the readout to
/// fourth order: M^{−1/2} (the bare ground density M^{−1/2} ρ M^{−1/2}) and per excited state the
/// operator M^{−1/2} Σ_c W_c[i,:]† W_c[i,:] M^{−1/2} of its population; and per excited state i the
/// transient's population after a jump into the cluster, Q_i = Σ_g (κ_g/Γ_g) X_g[i,:]†X_g[i,:] of the
/// bare state f (per coherent group of channels and excited group, κ = 1 − Γ²/(Δ² + Γ²/4): the
/// dressing rings above its mean far from resonance, and rebuilds below it at resonance), with the
/// pieces' trace factors κ (in the order of model.rs `Transient`'s pieces' groups).
#[derive(Clone,Debug)]
pub struct Fourth{pub effective:Dense,pub jumps:Vec<[Dense;3]>,pub slow:Dense,pub root:Dense,pub excited:Vec<Dense>,pub surplus:Vec<(usize,Dense)>,pub kappa:Vec<f64>}

/// A cluster's generator at a point and a time (the dense reference of model.rs): H_eff and the
/// share of it the folded pairs' paths make, every class's jumps per axis (target cluster's ground
/// states × cluster), the readout's Y and E_i (per excited state).
#[derive(Clone,Debug)]
pub struct Local{pub effective:Dense,pub shift:Dense,pub jumps:Vec<[Dense;3]>,pub admixture:Dense,pub excited_readout:Vec<Dense>}

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
    /// 32 × 32, at s_u = 1); the beams of one frequency whose interference the units leave out (their
    /// polarizations differ), each such frequency's beams with the bound of their interference's
    /// shift over the trap (rad/us).
    pub hamiltonian:Dense,
    pub units:Vec<Unit>,
    pub outside:Vec<Dense>,
    pub folds:Vec<Dense>,
    pub incoherent:Vec<(Vec<usize>,f64)>,
    /// Every frequency's two components, classified, and the bare ‖L0‖ they were judged against.
    pub components:Vec<Component>,
    pub l0:f64,
    pub carriers:Vec<Carrier>,
    /// The solve's duration T (us; 0 without one): a mismatch below 1/T is a resonance over the run.
    pub duration:f64,
    /// Per unit, its least intensity within the atom's reach (`Ground::reach`): 1 for a plane wave.
    pub reach:Vec<f64>,
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
    /// The excited states in groups of nearly one energy at the focus (`transient_groups`).
    pub excited_groups:Vec<Vec<usize>>,
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

/// The weights γ_l of beams of one frequency whose fields are parallel (`Unit`): V_l = γ_l V_first,
/// R_l = γ_l* R_first and O_l = |γ_l|² O_first, each to round-off (1e-9 of its norm, as model.rs takes
/// a carrier's span); None where some beam's field is not (its polarization differs, or the first
/// drives nothing).
fn parallel(beams:&[usize],couplings:&[Dense],counters:&[Dense],outside:&[Dense])->Option<Vec<Complex>>{
    if beams.len()==1{return Some(vec![Complex::ONE]);}
    let inner=|a:&Dense,b:&Dense|a.entries.iter().zip(&b.entries).fold(Complex::ZERO,|s,(x,y)|s+x.conj()**y);
    let first=beams[0];let(v,r)=(&couplings[first],&counters[first]);let(nv,nr)=(inner(v,v).re,inner(r,r).re);
    if !(nv>0.&&nr>0.){return None;}
    let close=|a:&Dense,b:&Dense,z:Complex|{let mut d=a.clone();d.add_scaled(-z,b);frobenius(&d)<=1e-9*frobenius(a)};
    beams.iter().map(|&l|{let g=inner(v,&couplings[l])*(1./nv);
        (close(&couplings[l],v,g)&&close(&counters[l],r,g.conj())&&close(&outside[l],&outside[first],Complex::new(g.norm_sqr(),0.))).then_some(g)}).collect()
}

/// The smallest ratio of a fold's frequency gap, far over near: the first order the fold leaves
/// out, (ν_near/ν_far)², within the budget.
pub fn fold_gap()->f64{1./BUDGET.sqrt()}
/// m = size/(|ν| − l0), a coupling's mixing.
pub fn mixing(nu:f64,size:f64,l0:f64)->f64{size/(nu.abs()-l0).max(f64::MIN_POSITIVE)}
/// The rule of the population core (population-core lattice/classification.rs) against the 3D
/// budget: an item of frequency ν and size g may fold when |ν| ≥ FOLD_DISTANCE l0 with
/// m² = (g/(|ν| − l0))² ≤ budget; the significant frequencies are cut at their largest ratio gap of
/// at least `fold_gap` above every one that may not fold, and the items that may fold beyond the
/// cut fold (an item of size zero folds trivially). The verdicts (true: folded) and the cut.
pub fn classify(items:&[(f64,f64)],l0:f64)->(Vec<bool>,f64){
    let foldable=|nu:f64,size:f64|nu.abs()>=FOLD_DISTANCE*l0&&mixing(nu,size,l0).powi(2)<=BUDGET;
    let mut frequencies:Vec<(f64,bool)>=items.iter().filter(|i|i.1>0.).map(|&(nu,size)|(nu.abs(),foldable(nu,size))).collect();
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
    (items.iter().map(|&(nu,size)|size==0.||nu.abs()>=cut&&foldable(nu,size)).collect(),cut)
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

/// The corners of the varying units' intensities (`Unit::varies`) and the levels between them,
/// {0, ¼, ½, ¾, 1} each (the others at their own): H depends on the point through them, affinely.
fn intensity_points(units:&[Unit],beams:&[Beam])->Vec<Vec<f64>>{
    let varying:Vec<usize>=(0..units.len()).filter(|&u|units[u].varies(beams)).collect();
    let levels=[0.,0.25,0.5,0.75,1.];let points=levels.len().pow(varying.len() as u32);
    (0..points).map(|point|{let mut intensities:Vec<f64>=units.iter().map(|u|u.focus).collect();
        for(f,&u)in varying.iter().enumerate(){intensities[u]=levels[(point/levels.len().pow(f as u32))%levels.len()];}intensities}).collect()
}

/// The points an atom's reach is judged at: the thermal ellipsoid's ends along its principal axes
/// (`axes`' rows, half-widths `reach`), and with `inside` a grid of nine per axis within it too, for
/// a pattern on the wavelength's scale (interfering beams).
pub(crate) fn reach_points(axes:&[[f64;3];3],reach:[f64;3],inside:bool)->Vec<[f64;3]>{
    let at=|q:[f64;3]|->[f64;3]{std::array::from_fn(|i|(0..3).map(|a|q[a]*axes[a][i]).sum())};
    let mut out:Vec<[f64;3]>=(0..3).flat_map(|a|[-1.,1.].map(move|s|(a,s))).map(|(a,s)|{let mut q=[0.;3];q[a]=s*reach[a];at(q)}).collect();
    if inside{for i in -4_i32..=4{for j in -4_i32..=4{for k in -4_i32..=4{
        if i*i+j*j+k*k<=16{out.push(at([i as f64*reach[0]/4.,j as f64*reach[1]/4.,k as f64*reach[2]/4.]));}
    }}}}
    out
}

/// Two frames are one: equal to the rounding of the sums of detunings they are made of.
pub(crate) fn same_frame(a:f64,b:f64)->bool{(a-b).abs()<=64.*f64::EPSILON*(a.abs()+b.abs())}

/// The partition the rule leaves for one set of verdicts on the excited couplings: the ground
/// couplings classified, the cut, the clusters (sets of nodes: the ground states, then the excited
/// states at ng + e), each node's cluster, and the frames (NaN off the clusters).
struct Partition{items:Vec<Item>,cut:f64,sets:Vec<Vec<usize>>,set:Vec<Option<usize>>,frames:Vec<f64>}

impl Ground{
    /// The generator of a request with every drive folded (`Ground::with` at an infinite allowance).
    pub fn new(request:&Request)->Result<Ground,String>{Ground::with(request,f64::INFINITY)}
    /// The generator of a request, its drives kept until each ground state's folded rest is within
    /// `allowance` (its total saturation over the light's pattern, the fold's charge), or why it has
    /// none: a counter-rotating component that may not fold, or a static coupling between ground
    /// states that may not.
    pub fn with(request:&Request,allowance:f64)->Result<Ground,String>{
        let mut ground=Ground::bare(request)?;
        ground.cluster_states(allowance)?;
        Ok(ground)
    }
    /// A request's generator before the rule: its states, components, carriers and reach.
    fn bare(request:&Request)->Result<Ground,String>{
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
        for c in &request.emission{let c=Dense::block(c,&all,&excited_states);decay.add_scaled(Complex::ONE,&c.adjoint_product(&c));}
        let emission:[Dense;3]=std::array::from_fn(|a|Dense::block(&request.emission[a],&ground_states,&excited_states));
        let hamiltonian=Dense::block(&request.hamiltonian,&all,&all);
        if (0..STATES).any(|i|(0..STATES).any(|j|i!=j&&hamiltonian[(i,j)]!=Complex::ZERO)){return Err("H0 must be diagonal in the model basis".into());}
        let energies:Vec<f64>=(0..STATES).map(|i|hamiltonian[(i,i)].re).collect();
        // Every component, classified by the population core's rule. ‖L0‖ with H0's diagonal centred:
        // its spread, the decay and the emission.
        let frame=request.frame_frequencies_mhz[excited_states[0]];
        let spread=energies.iter().copied().fold(f64::NEG_INFINITY,f64::max)-energies.iter().copied().fold(f64::INFINITY,f64::min);
        let l0=spread+frobenius(&decay)+emission.iter().map(|c|frobenius(c).powi(2)).sum::<f64>();
        let couplings:Vec<Dense>=request.beams.iter().map(|b|Dense::block(&b.coupling,&excited_states,&ground_states)).collect();
        let counters:Vec<Dense>=request.beams.iter().map(|b|Dense::block(&b.counter_coupling,&excited_states,&ground_states)).collect();
        let outside:Vec<Dense>=request.beams.iter().map(|b|Dense::block(&b.outside_shift,&all,&all)).collect();
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
        if let Some(c)=components.iter().find(|c|c.counter&&!c.folded&&c.size>0.){
            return Err(format!("The counter-rotating coupling of 「{}」 is too strong to fold (population core's rule): 3D motion cannot take it",names(&c.beams)));}
        // The units (`Unit`) and their outside shifts and folds; the interference the units of one beam
        // leave out, bounded over the trap (every beam at most its peak): per pair of beams the van
        // Vleck shift's cross terms C + C† and i(C − C†) of each folded component (polarization), and of
        // the outside shifts, which the request gives per beam, 2 sqrt(‖O_l‖ ‖O_k‖) (exact were they of
        // one sign).
        let fold=|v:&Dense,nu:f64|van_vleck(&energies,v,nu,&ground_states,&excited_states);
        let(mut parts,mut incoherent):(Vec<(Unit,Dense,Dense)>,Vec<(Vec<usize>,f64)>)=(vec![],vec![]);
        for beams in &frequencies{
            let(co,counter)=(components.iter().find(|c|!c.counter&&c.beams==*beams).unwrap(),components.iter().find(|c|c.counter&&c.beams==*beams).unwrap());
            let folds_of=|l:usize|{let mut f=Dense::zeros(STATES,STATES);
                if co.folded{f.add_scaled(Complex::ONE,&fold(&couplings[l],co.nu));}if counter.folded{f.add_scaled(Complex::ONE,&fold(&counters[l],counter.nu));}f};
            match parallel(beams,&couplings,&counters,&outside){
                Some(weights)=>{
                    let norm=weights.iter().map(|w|w.abs()).sum::<f64>().powi(2);let first=beams[0];
                    let focus=weights.iter().fold(Complex::ZERO,|s,&w|s+w).norm_sqr()/norm;
                    parts.push((Unit{beams:beams.clone(),weights,norm,focus},outside[first].scaled(Complex::new(norm,0.)),folds_of(first).scaled(Complex::new(norm,0.))));
                }
                None=>{
                    let mut bound=0.;
                    for(i,&l)in beams.iter().enumerate(){for &k in &beams[i+1..]{
                        for(on,v,nu)in [(co.folded,&couplings,co.nu),(counter.folded,&counters,counter.nu)]{
                            if !on{continue;}
                            let(mut base,mut sum,mut turned)=(fold(&v[l],nu),v[l].clone(),v[l].clone());base.add_scaled(Complex::ONE,&fold(&v[k],nu));
                            sum.add_scaled(Complex::ONE,&v[k]);turned.add_scaled(Complex::new(0.,1.),&v[k]);
                            let(mut x,mut y)=(fold(&sum,nu),fold(&turned,nu));x.add_scaled(-Complex::ONE,&base);y.add_scaled(-Complex::ONE,&base);
                            bound+=x.spectral_norm()+y.spectral_norm();
                        }
                        bound+=2.*(outside[l].spectral_norm()*outside[k].spectral_norm()).sqrt();
                    }}
                    if bound>0.{incoherent.push((beams.clone(),bound));}
                    for &l in beams{parts.push((Unit{beams:vec![l],weights:vec![Complex::ONE],norm:1.,focus:1.},outside[l].clone(),folds_of(l)));}
                }
            }
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
        let duration=request.times_us.last().copied().unwrap_or(0.).max(0.);
        let mut ground=Ground{ground_states,excited_states,levels,decay,emission,beams:request.beams.clone(),folded:couplings.clone(),couplings,hamiltonian,units,outside:unit_outside,folds:unit_folds,incoherent,
            components,l0,carriers,duration,reach:vec![],excited_items:vec![],charge:0.,items:vec![],cut:f64::INFINITY,generator_scale:0.,light:0.,pole:f64::INFINITY,poles:vec![],clusters:vec![],cluster:vec![],
            explicit_of:vec![],excited_groups:vec![],dressed:vec![]};
        ground.reach=ground.reach(request);
        Ok(ground)
    }

    /// Every unit's intensity at a point r (um from the focus).
    pub fn intensities_at(&self,r:[f64;3])->Vec<f64>{let a=self.amplitudes_at(r);self.units.iter().map(|u|u.intensity(&a)).collect()}
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

    /// The elimination for H (32 × 32, rad/us, in the frame): H's ground block and its energies, its
    /// excited block, K, and every near beam's R_l and R2_l of its folded coupling, column by column
    /// -- each column through K decoupled from its cluster's explicit states (`decoupled`).
    pub fn eliminate(&self,hamiltonian:&Dense)->Result<Elimination,String>{
        let(ng,ne)=(self.ground_states.len(),self.excited_states.len());
        let ground=sub(hamiltonian,&self.ground_states,&self.ground_states);
        let energies:Vec<f64>=(0..ng).map(|b|ground[(b,b)].re).collect();
        let excited=sub(hamiltonian,&self.excited_states,&self.excited_states);
        let excited_energies:Vec<f64>=(0..ne).map(|i|excited[(i,i)].re).collect();
        let mut excited_hamiltonian=excited.clone();excited_hamiltonian.add_scaled(Complex::new(0.,-0.5),&self.decay);
        let(mut responses,mut derivatives)=(vec![Dense::zeros(ne,ng);self.beams.len()],vec![Dense::zeros(ne,ng);self.beams.len()]);
        let mut column=vec![Complex::ZERO;ne];
        let decoupled:Vec<Option<Dense>>=self.explicit_of.iter().map(|p|p.iter().any(|&x|x).then(||self.decoupled(&excited_hamiltonian,p))).collect();
        let dressings=self.dressings(&excited,&excited_hamiltonian);
        for(c,carrier)in self.carriers.iter().enumerate(){for b in 0..ng{
            if !self.couples(c,b){continue;}
            let k=decoupled.get(b).and_then(|x|x.as_ref()).unwrap_or(&excited_hamiltonian);
            let lu=denominator(k,-energies[b]-carrier.detuning)?;
            for &l in &carrier.beams{
                self.source(l,b,dressings.get(b).map_or(&[][..],|d|d),&mut column);lu.solve(&mut column);for i in 0..ne{responses[l][(i,b)]=column[i];}
                lu.solve(&mut column);for i in 0..ne{derivatives[l][(i,b)]=column[i];}
            }
        }}
        Ok(Elimination{energies,ground,excited,excited_energies,excited_hamiltonian,responses,derivatives})
    }

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
        for intensities in intensity_points(&self.units,&self.beams){
            let e=self.eliminate(&self.hamiltonian_with(&intensities))?;
            for a in 0..ng{for b in 0..ng{
                out.high[a][b]=out.high[a][b].max(e.energies[a]-e.energies[b]);
                if a!=b{out.statics[a][b]=out.statics[a][b].max(e.ground[(a,b)].abs());}
            }}
            // Per pair of beams the three products; per carrier pair their sums of moduli.
            let mut raman=vec![vec![vec![vec![0.;ng];ng];nc];nc];let(mut light,mut scattering)=(vec![0.;ng],vec![0.;ng]);
            for(c,carrier)in self.carriers.iter().enumerate(){for(d,emitted)in self.carriers.iter().enumerate(){
                for &l in &carrier.beams{for &k in &emitted.beams{
                    let(v,w)=(self.folded[k].adjoint_product(&e.responses[l]),e.responses[k].adjoint_product(&self.folded[l]));
                    let q=e.responses[k].adjoint_product(&self.decay.product(&e.responses[l]));
                    for a in 0..ng{for b in 0..ng{
                        raman[c][d][a][b]+=0.5*(v[(a,b)].abs()+w[(a,b)].abs()+q[(a,b)].abs());
                        if c==d{light[a]+=((v[(a,b)]+w[(a,b)])*0.5).abs();scattering[a]+=q[(a,b)].abs();}
                    }}
                }}
            }}
            for c in 0..nc{for d in 0..nc{for a in 0..ng{for b in 0..ng{out.raman[c][d][a][b]=out.raman[c][d][a][b].max(raman[c][d][a][b]);}}}}
            out.light=out.light.max(light.iter().copied().fold(0.,f64::max));out.scattering=out.scattering.max(scattering.iter().copied().fold(0.,f64::max));
            for(c,carrier)in self.carriers.iter().enumerate(){
                let jumps:Vec<[Dense;3]>=carrier.beams.iter().map(|&l|std::array::from_fn(|x|self.emission[x].product(&e.responses[l]))).collect();
                for b in 0..ng{for a in 0..ng{for x in 0..3{
                    let n:f64=jumps.iter().map(|j|j[x][(a,b)].abs()).sum();out.photons[c][b][a][x]=out.photons[c][b][a][x].max(n);
                }}}
                for b in 0..ng{
                    if !self.couples(c,b){continue;}
                    let lu=denominator(&e.excited_hamiltonian,-e.energies[b]-carrier.detuning)?;
                    out.poles[c]=out.poles[c].min(1./lu.solve_columns(&identity).spectral_norm());
                }
            }
        }
        Ok(out)
    }

    /// Each unit's least intensity within the atom's reach: the region of the trap within
    /// ln(1/budget) k_B T of its bottom, the harmonic trap's thermal ellipsoid holding all but the
    /// budget of a Boltzmann ensemble at the request's temperature (model.rs `choose_profiles` judges
    /// the units' profiles over it too), the trap the most populated initial ground state's at the
    /// focus of the trap beam; at the ellipsoid's ends along its principal axes, and inside it for
    /// interfering beams (`reach_points`). 1 for a plane wave, 0 for every varying unit when the trap
    /// does not confine. Beyond the reach, what an atom meets is measured along its trajectory (the
    /// folded excited population, the ledger's η²).
    fn reach(&self,request:&Request)->Vec<f64>{
        let unconfined=||self.units.iter().map(|u|if u.varies(&self.beams){0.}else{1.}).collect();
        let Some(waist)=request.beams.get(request.trap_beam).and_then(|b|b.waist_um)else{return unconfined()};
        let rho=&request.initial_density;
        let reference=*self.ground_states.iter().max_by(|&&a,&&b|rho[a*STATES+a].re.total_cmp(&rho[b*STATES+b].re)).unwrap();
        // The reference state's potential's Hessian at the focus by central differences over a
        // thousandth of the trap beam's waist (as model.rs `Model::trap`).
        let h=1e-3*waist;let u=|r:[f64;3]|self.hamiltonian_at(r)[(reference,reference)].re;
        let e=|a:usize|{let mut v=[0.;3];v[a]=h;v};let add=|x:[f64;3],y:[f64;3]|->[f64;3]{std::array::from_fn(|i|x[i]+y[i])};let neg=|x:[f64;3]|x.map(|v|-v);
        let u0=u([0.;3]);
        let hessian:[[f64;3];3]=std::array::from_fn(|a|std::array::from_fn(|b|if a==b{(u(e(a))+u(neg(e(a)))-2.*u0)/(h*h)}
            else{(u(add(e(a),e(b)))-u(add(e(a),neg(e(b))))-u(add(neg(e(a)),e(b)))+u(add(neg(e(a)),neg(e(b)))))/(4.*h*h)}));
        let(values,axes)=crate::model::symmetric_eigen(hessian);
        let hm=crate::model::HBAR/request.mass_kg*1e6;let kt=crate::stats::KT_PER_UK*request.temperature_uk;
        let reach:[f64;3]=std::array::from_fn(|a|{let w=(values[a]*hm).max(0.).sqrt();if w>0.{(2.*BUDGET.recip().ln()*kt*hm).sqrt()/w}else{f64::INFINITY}});
        if reach.iter().any(|x|!x.is_finite()){return unconfined();}
        self.units.iter().map(|u|if !u.varies(&self.beams){1.}else{
            reach_points(&axes,reach,u.beams.len()>1).into_iter().map(|r|u.intensity(&self.amplitudes_at(r))).fold(1_f64,f64::min)}).collect()
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
    /// says whether it counts as a change); whether any did.
    fn judge_drives(excited:&mut [ExcitedItem],allowance:f64,kept:&mut dyn FnMut(&mut ExcitedItem)->bool)->bool{
        let grounds:Vec<usize>={let mut g:Vec<usize>=excited.iter().filter_map(|x|match x.kind{ExcitedKind::Drive(d)=>Some(d.ground),_=>None}).collect();g.sort();g.dedup();g};
        let mut changed=false;
        for b in grounds{
            let mut drives:Vec<usize>=(0..excited.len()).filter(|&i|matches!(excited[i].kind,ExcitedKind::Drive(d)if d.ground==b)).collect();
            drives.sort_by(|&i,&j|excited[j].mean.total_cmp(&excited[i].mean));
            let mut total:f64=drives.iter().filter(|&&i|excited[i].folded).map(|&i|excited[i].mean).sum();
            for i in drives{
                if !excited[i].folded{continue;}
                if total<=allowance{break;}
                total-=excited[i].mean;excited[i].folded=false;changed|=kept(&mut excited[i]);
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
        Ground::judge_drives(&mut excited,allowance,&mut |_|false);
        excited
    }
    /// The allowances at which the rule's explicit excited states change (`Ground::with`), from all
    /// folded (infinite) down to every drive kept, each the largest that gives its set of explicit
    /// states before the closure: the totals of a ground state's folded drives as it keeps them from
    /// the most saturated down.
    pub fn allowances(request:&Request)->Result<Vec<f64>,String>{
        let g=Ground::bare(request)?;let ne=g.excited_states.len();
        let excited=g.judged(f64::INFINITY);
        let mut totals:Vec<f64>=vec![];
        for b in 0..g.ground_states.len(){
            let mut means:Vec<f64>=excited.iter().filter_map(|x|match x.kind{ExcitedKind::Drive(d)if d.ground==b=>Some(x.mean),_=>None}).collect();
            means.sort_by(|x,y|y.total_cmp(x));let mut rest:f64=means.iter().sum();
            for m in means{rest-=m;totals.push(rest.max(0.));}
        }
        totals.sort_by(|x,y|y.total_cmp(x));totals.dedup();
        let mut out=vec![f64::INFINITY];let mut last=Ground::explicit_states(&excited,ne);
        for t in totals{
            let explicit=Ground::explicit_states(&g.judged(t),ne);
            if explicit!=last{out.push(t);last=explicit;}
        }
        Ok(out)
    }

    /// The rule over every coupling (module documentation): the excited couplings judged, the
    /// folded drives eliminated, the couplings between ground states judged with them, the clusters
    /// and frames; the closure judging the folded couplings of each cluster again with its dressing,
    /// until nothing changes; then each cluster's machinery.
    fn cluster_states(&mut self,allowance:f64)->Result<(),String>{
        let(ng,ne)=(self.ground_states.len(),self.excited_states.len());
        let floor=if self.duration>0.{1./self.duration}else{0.};
        let mut excited=self.judged(allowance);
        loop{
            let explicit=Ground::explicit_states(&excited,ne);
            // The folded couplings: every kept drive's entry zeroed in each beam of its carrier.
            self.folded=self.couplings.clone();
            for x in &excited{if let(ExcitedKind::Drive(d),false)=(x.kind,x.folded){for &l in &self.carriers[d.carrier].beams{self.folded[l][(d.excited,d.ground)]=Complex::ZERO;}}}
            let b=self.bounds()?;
            let partition=self.partition(&b,&excited,&explicit)?;
            // Each ground state's cluster's explicit states, for the eliminations from here on.
            let explicit_of:Vec<Vec<bool>>=(0..ng).map(|x|{let k=partition.set[x];(0..ne).map(|e|explicit[e]&&partition.set[ng+e]==k).collect()}).collect();
            let moved=explicit_of!=self.explicit_of;self.explicit_of=explicit_of;
            // The closure: each cluster's dressing, the spectral norm of its couplings' bounds.
            let dressing:Vec<f64>=partition.sets.iter().map(|set|{
                let n=set.len();let mut m=Dense::zeros(n,n);
                for(i,&x)in set.iter().enumerate(){for(j,&y)in set.iter().enumerate(){if x<ng&&y<ng{
                    let s:f64=b.raman.iter().flat_map(|r|r.iter()).map(|r|r[x][y].max(r[y][x])).sum();m[(i,j)]=Complex::new(s,0.);}}}
                for item in excited.iter().filter(|x|!x.folded){
                    let(x,y)=match item.kind{ExcitedKind::Drive(d)=>(ng+d.excited,d.ground),ExcitedKind::Static{a,b}=>(ng+a,ng+b)};
                    if let(Some(i),Some(j))=(set.iter().position(|&s|s==x),set.iter().position(|&s|s==y)){m[(i,j)]+=Complex::new(item.size,0.);m[(j,i)]+=Complex::new(item.size,0.);}
                }
                hermitian_eigenvalues(&m).into_iter().fold(0_f64,|a,v|a.max(v.abs()))}).collect();
            let mut changed=false;
            for x in excited.iter_mut().filter(|x|x.folded){
                let reach=match x.kind{ExcitedKind::Drive(d)=>partition.set[d.ground],
                    ExcitedKind::Static{a,b}=>partition.set[ng+a].or(partition.set[ng+b])};
                let Some(k)=reach else{continue};
                let(estimate,mean)=excited_estimate(x,dressing[k],floor);
                // A coupling static in the frames between two states of one cluster costs nothing kept.
                let inside=match x.kind{
                    ExcitedKind::Drive(d)=>partition.set[ng+d.excited]==Some(k)&&same_frame(partition.frames[ng+d.excited],partition.frames[d.ground]+self.carriers[d.carrier].detuning),
                    ExcitedKind::Static{a,b}=>partition.set[ng+a]==Some(k)&&partition.set[ng+b]==Some(k)&&same_frame(partition.frames[ng+a],partition.frames[ng+b])};
                x.estimate=x.estimate.max(estimate);x.mean=x.mean.max(mean);
                if inside||(matches!(x.kind,ExcitedKind::Static{..})&&estimate>BUDGET){x.folded=false;x.promoted=true;changed=true;}
            }
            // The drives with the closure's estimates: each ground state's folded total within the allowance.
            changed|=Ground::judge_drives(&mut excited,allowance,&mut |x|{x.promoted=true;true});
            if !changed&&!moved{self.charge=Ground::charge_of(&excited,ng);return self.assemble(&b,partition,excited);}
        }
    }

    /// The couplings between ground states classified (`classify`) with the folded drives, and the
    /// clusters they make with the kept excited couplings, and the frames.
    fn partition(&mut self,b:&Bounds,excited:&[ExcitedItem],explicit:&[bool])->Result<Partition,String>{
        let(ng,ne,nc)=(self.ground_states.len(),self.excited_states.len(),self.carriers.len());
        self.pole=b.poles.iter().copied().fold(f64::INFINITY,f64::min);self.poles=b.poles.clone();
        // ‖L0‖ of the eliminated generator without the states' energies: its Hermitian part on both
        // sides, its decay and its jumps.
        let l0=2.*b.light+2.*b.scattering;self.generator_scale=l0;self.light=b.light;
        // The worst separation of a pair over the trap with a constant beat added: the separation's
        // range [low, high] of E_a − E_b, shifted.
        let separation=|a:usize,c:usize,beat:f64|->f64{
            let(lo,hi)=(beat-b.high[c][a],beat+b.high[a][c]);if lo<=0.&&hi>=0.{0.}else{lo.abs().min(hi.abs())}
        };
        let mut items:Vec<Item>=vec![];
        for a in 0..ng{for c in a+1..ng{if b.statics[a][c]>0.{
            items.push(Item{kind:Kind::Static{a,b:c},frequency:separation(a,c,0.),size:2.*b.statics[a][c].max(b.statics[c][a]),folded:false});}}}
        let channels:Vec<Channel>=(0..nc).flat_map(|c|(0..ng).map(move|s|Channel{carrier:c,state:s})).filter(|ch|self.couples(ch.carrier,ch.state)).collect();
        for(i,&x)in channels.iter().enumerate(){for &y in &channels[i+1..]{
            // Absorbing x, emitting into y's carrier and landing in y's state: Ω = (E_y + δ_y) − (E_x + δ_x).
            let beat=self.carriers[y.carrier].detuning-self.carriers[x.carrier].detuning;
            let size=2.*b.raman[x.carrier][y.carrier][y.state][x.state].max(b.raman[y.carrier][x.carrier][x.state][y.state]);
            if size>0.{items.push(Item{kind:Kind::Raman{first:x,second:y},frequency:separation(y.state,x.state,beat),size,folded:false});}
        }}
        for from in 0..ng{
            let photons:Vec<Photon>=(0..nc).flat_map(|c|(0..ng).map(move|to|Photon{carrier:c,from,to})).filter(|p|b.photons[p.carrier][from][p.to].iter().any(|&n|n>0.)).collect();
            for(i,&p)in photons.iter().enumerate(){for &q in &photons[i+1..]{
                if p.to==q.to{continue;}
                // Their frequencies (δ_p + E_from − E_p.to) − (δ_q + E_from − E_q.to).
                let beat=self.carriers[p.carrier].detuning-self.carriers[q.carrier].detuning;
                let size:f64=(0..3).map(|x|b.photons[p.carrier][from][p.to][x]*b.photons[q.carrier][from][q.to][x]).sum();
                if size>0.{items.push(Item{kind:Kind::Photons{first:p,second:q},frequency:separation(q.to,p.to,beat),size,folded:false});}
            }}
        }
        // An explicit state's two photons landing in two ground states: the coherence its jumps make
        // between them, |Σ_a C_a C_a*|, at their separation.
        for e in (0..ne).filter(|&e|explicit[e]){for first in 0..ng{for second in first+1..ng{
            let z=(0..3).fold(Complex::ZERO,|s,a|s+self.emission[a][(first,e)]*self.emission[a][(second,e)].conj());
            if z.abs()>0.{items.push(Item{kind:Kind::Direct{excited:e,first,second},frequency:separation(first,second,0.),size:z.abs(),folded:false});}
        }}}
        let(verdicts,cut)=classify(&items.iter().map(|i|(i.frequency,i.size)).collect::<Vec<_>>(),l0);
        for(item,v)in items.iter_mut().zip(verdicts){item.folded=v;}
        if let Some(item)=items.iter().find(|i|!i.folded&&matches!(i.kind,Kind::Static{..})&&{let Kind::Static{a,b}=i.kind else{unreachable!()};self.levels[a]!=self.levels[b]}){
            return Err(format!("光束在 F=1 与 F=2 之间的静态耦合不能忽略（混合 {:.1e}），3D 运动暂不支持",mixing(item.frequency,item.size,l0)));}
        // The clusters: the components of the graph of kept couplings over the ground states (nodes
        // 0..ng) and the excited ones (ng + e).
        let mut union=Union::new(ng+ne);
        let delta=|c:usize|self.carriers[c].detuning;
        // An edge (a, b) with θ_a − θ_b = offset, and its size.
        let mut edges:Vec<(f64,usize,usize,f64)>=vec![];
        for item in items.iter().filter(|i|!i.folded){match item.kind{
            Kind::Static{a,b}=>edges.push((item.size,a,b,0.)),
            Kind::Raman{first,second}=>edges.push((item.size,second.state,first.state,delta(first.carrier)-delta(second.carrier))),
            Kind::Photons{first,second}=>edges.push((item.size,first.to,second.to,delta(first.carrier)-delta(second.carrier))),
            Kind::Direct{first,second,..}=>edges.push((item.size,first,second,0.)),
        }}
        for item in excited.iter().filter(|x|!x.folded){match item.kind{
            ExcitedKind::Drive(d)=>edges.push((item.size,ng+d.excited,d.ground,delta(d.carrier))),
            ExcitedKind::Static{a,b}=>edges.push((item.size,ng+a,ng+b,0.)),
        }}
        for &(_,a,b,_)in &edges{union.join(a,b);}
        // Every set holds a ground state (an excited state is in one through its kept drive); its
        // smallest element is one.
        let sets:Vec<Vec<usize>>=union.sets().into_iter().filter(|s|s[0]<ng&&(s.len()>1||s[0]<ng)).collect();
        let mut set=vec![None;ng+ne];for(k,s)in sets.iter().enumerate(){for &x in s{set[x]=Some(k);}}
        // Each cluster's frames: from its first state along a spanning tree of its kept couplings,
        // the strongest first, each nulling its coupling's beat.
        let focus=self.eliminate(&self.hamiltonian_with(&self.focus_intensities()))?;
        let mut frames=vec![f64::NAN;ng+ne];
        edges.sort_by(|x,y|y.0.total_cmp(&x.0));
        for s in &sets{
            frames[s[0]]=focus.energies[s[0]];
            while let Some(&(_,a,b,offset))=edges.iter().find(|&&(_,a,b,_)|frames[a].is_nan()!=frames[b].is_nan()&&set[a]==set[s[0]]){
                if frames[a].is_nan(){frames[a]=frames[b]+offset;}else{frames[b]=frames[a]-offset;}
            }
        }
        Ok(Partition{items,cut,sets,set,frames})
    }

    /// Each cluster's machinery from the rule's partition: its states and frames, its kept couplings
    /// and drives, its van Vleck paths and shifts, its jump classes and co-resonant channels.
    fn assemble(&mut self,b:&Bounds,partition:Partition,excited:Vec<ExcitedItem>)->Result<(),String>{
        let(ng,ne,nc)=(self.ground_states.len(),self.excited_states.len(),self.carriers.len());
        let Partition{items,cut,sets,set,frames}=partition;
        let delta=|c:usize|self.carriers[c].detuning;
        let theta=|node:usize|frames[node];
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
            Cluster{frames:states.iter().map(|&s|theta(s)).collect(),excited_frames:excited.iter().map(|&e|theta(ng+e)).collect(),states,excited,
                groups:vec![],statics:vec![],paths:vec![],static_paths:vec![],drives:vec![],excited_statics:vec![],excited_paths:vec![],classes:vec![],coherent:vec![]}}).collect();
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
                    if x==y||folded_raman(x,y)==Some(false){entries.push((i,j,theta(t)-theta(s)+delta(d)-delta(c)));}
                }}
                if c==d{for(p,&e)in cl.excited.iter().enumerate(){for(q,&f)in cl.excited.iter().enumerate(){
                    let shared=(0..ng).any(|b|self.carriers[c].beams.iter().any(|&l|self.folded[l][(e,b)]!=Complex::ZERO)&&self.carriers[c].beams.iter().any(|&l|self.folded[l][(f,b)]!=Complex::ZERO));
                    if shared&&same_frame(theta(ng+e),theta(ng+f)){entries.push((n0+p,n0+q,0.));}
                }}}
                if !entries.is_empty(){cl.groups.push(Group{absorbed:c,emitted:d,entries});}
            }}
            let folded_ground_static=|a:usize,b:usize|items.iter().any(|i|i.folded&&i.kind==Kind::Static{a:a.min(b),b:a.max(b)});
            for item in items.iter().filter(|i|!i.folded){if let Kind::Static{a,b}=item.kind{
                if cluster[a]==k{let(i,j)=(cl.position(a).unwrap(),cl.position(b).unwrap());cl.statics.push((i,j));cl.statics.push((j,i));}
            }}
            for i in 0..cl.states.len(){for j in 0..cl.states.len(){if theta(cl.states[i])==theta(cl.states[j]){
                for f in 0..ng{if folded_ground_static(cl.states[i],f)&&folded_ground_static(f,cl.states[j]){cl.static_paths.push((i,j,f));}}
            }}}
            // The folded pairs' paths: from a channel of the cluster through a far one, back into a
            // co-resonant channel of the cluster.
            for &x in &inside{for &y in &channels{
                if folded_raman(x,y)!=Some(true){continue;}
                for &z in &inside{if coresonant(x,z){
                    cl.paths.push(Path{from:x,via:y,back:z,residual:theta(z.state)-theta(x.state)+delta(z.carrier)-delta(x.carrier)});
                }}
            }}
            // The kept drives, H's kept couplings between explicit states, and the van Vleck paths of
            // their folded ones through excited states outside the cluster.
            for(p,&e)in cl.excited.iter().enumerate(){
                for(x,&s)in cl.states.iter().enumerate(){for c in 0..nc{
                    if kept_drive(Drive{carrier:c,excited:e,ground:s}){let w=theta(ng+e)-theta(s)-delta(c);cl.drives.push((n0+p,x,c,if same_frame(theta(ng+e),theta(s)+delta(c)){0.}else{w}));}
                }}
                for(q,&f)in cl.excited.iter().enumerate(){
                    if e!=f&&excited.iter().any(|x|!x.folded&&x.kind==ExcitedKind::Static{a:e.min(f),b:e.max(f)}){cl.excited_statics.push((n0+p,n0+q));}
                    if same_frame(theta(ng+e),theta(ng+f)){for g in (0..ne).filter(|&g|!cl.excited.contains(&g)){
                        if folded_static(e,g)&&folded_static(g,f){cl.excited_paths.push((n0+p,n0+q,g));}
                    }}
                }
            }
            // The co-resonant channels (the readout's coherent sums), each with its residual.
            let mut union=Union::new(inside.len());
            for i in 0..inside.len(){for j in i+1..inside.len(){if coresonant(inside[i],inside[j]){union.join(i,j);}}}
            cl.coherent=union.sets().into_iter().map(|set|{let x0=inside[set[0]];
                set.iter().map(|&i|{let x=inside[i];(x,theta(x.state)+delta(x.carrier)-theta(x0.state)-delta(x0.carrier))}).collect()}).collect();
        }
        // An explicit state's photons' wave number: a carrier's that drives it (the largest drive).
        let wave_number=|e:usize|->f64{excited.iter().filter_map(|x|match x.kind{ExcitedKind::Drive(d)if d.excited==e=>Some((x.size,d.carrier)),_=>None})
            .max_by(|a,b|a.0.total_cmp(&b.0)).map_or(self.carriers.first().map_or(0.,|c|c.wave_number),|(_,c)|self.carriers[c].wave_number)};
        // The jump classes: the photons from a cluster, joined when they land in one cluster with
        // coherent amplitudes -- in one state from co-resonant sources (channels: the same or a kept
        // pair; an explicit state: one label within the cut), in two states as a kept pair of photons
        // of one source (or from two sources, at a residual beat below the cut).
        #[derive(Clone,Copy)]
        enum Source{Folded(Photon),Direct(usize,usize)}
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
            let label=|p:&Source|match *p{Source::Folded(p)=>theta(p.from)+delta(p.carrier),Source::Direct(e,_)=>theta(ng+e)};
            let rate=|p:&Source|theta(to(p))-label(p);
            let mut union=Union::new(photons.len());
            for i in 0..photons.len(){for j in i+1..photons.len(){
                let(p,q)=(photons[i],photons[j]);
                if cluster[to(&p)]!=cluster[to(&q)]{continue;}
                let joined=match(p,q){
                    (Source::Folded(p),Source::Folded(q))=>if p.to==q.to{coresonant(Channel{carrier:p.carrier,state:p.from},Channel{carrier:q.carrier,state:q.from})}
                        else if p.from==q.from{items.iter().any(|it|!it.folded&&matches!(it.kind,Kind::Photons{first,second}if(first==p&&second==q)||(first==q&&second==p)))}
                        else{(rate(&Source::Folded(p))-rate(&Source::Folded(q))).abs()<cut},
                    (Source::Direct(e,a),Source::Direct(f,b))if e==f=>items.iter().any(|it|!it.folded&&it.kind==Kind::Direct{excited:e,first:a.min(b),second:a.max(b)}),
                    _=>if to(&p)==to(&q){(label(&p)-label(&q)).abs()<cut}else{(rate(&p)-rate(&q)).abs()<cut},
                };
                if joined{union.join(i,j);}
            }}
            let classes=union.sets().into_iter().map(|set|{let p0=photons[set[0]];let r0=rate(&p0);
                let residual=|p:&Source|{let r=rate(p);if same_frame(r,r0){0.}else{r-r0}};
                let wave=match p0{Source::Folded(p)=>self.carriers[p.carrier].wave_number,Source::Direct(e,_)=>wave_number(e)};
                let(mut folded,mut direct)=(vec![],vec![]);
                for &i in &set{match photons[i]{Source::Folded(p)=>folded.push((p,match p0{Source::Folded(_)=>rate(&photons[i])-r0,_=>residual(&photons[i])})),
                    Source::Direct(e,t)=>direct.push((clusters[k].excited_position(e).unwrap(),t,residual(&photons[i])))}}
                Class{target:cluster[to(&p0)],photons:folded,direct,wave_number:wave}}).collect();
            clusters[k].classes=classes;
        }
        // The excited states in groups of nearly one energy at the focus: consecutive ones whose
        // separation ΔE has (Γ/ΔE)² beyond the budget ring together after a jump (`transient`).
        let focus=self.eliminate(&self.hamiltonian_with(&self.focus_intensities()))?;
        let mut order:Vec<usize>=(0..ne).collect();order.sort_by(|&a,&b|focus.excited_energies[a].total_cmp(&focus.excited_energies[b]));
        let mut groups:Vec<Vec<usize>>=vec![];
        for e in order{
            let joined=groups.last().is_some_and(|g:&Vec<usize>|{let f=*g.last().unwrap();let width=self.decay[(e,e)].re.max(self.decay[(f,f)].re);
                width*width>BUDGET*(focus.excited_energies[e]-focus.excited_energies[f]).powi(2)});
            if joined{groups.last_mut().unwrap().push(e);}else{groups.push(vec![e]);}
        }
        self.items=items;self.cut=cut;self.clusters=clusters;self.cluster=cluster;self.excited_items=excited;self.excited_groups=groups;self.dressed=dressed;
        Ok(())
    }

    /// Every beam's amplitude at a point r (um from the focus).
    pub fn amplitudes_at(&self,r:[f64;3])->Vec<Complex>{self.beams.iter().map(|b|b.amplitude_at(r)).collect()}

    /// The path energy of a cluster's channels of carrier c at an elimination: the centroid of its
    /// ground states' energies plus δ_c.
    pub fn path_energy(&self,e:&Elimination,cluster:usize,c:usize)->f64{
        let s=&self.clusters[cluster].states;s.iter().map(|&b|e.energies[b]).sum::<f64>()/s.len() as f64+self.carriers[c].detuning
    }

    /// A cluster's generator at a point r and a time t (the dense reference of model.rs, with the
    /// elimination of H(r)): H_eff, the jumps of every class, the readout.
    pub fn local_at(&self,e:&Elimination,k:usize,r:[f64;3],t:f64)->Result<Local,String>{self.local_with(e,k,r,t,&|_|true)}
    /// The same with the van Vleck paths `path` keeps (model.rs tabulates those through a carrier of
    /// one beam).
    pub fn local_with(&self,e:&Elimination,k:usize,r:[f64;3],t:f64,path:&dyn Fn(&Path)->bool)->Result<Local,String>{
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
                // Two explicit states: ½ Σ_b V_c^f[e,b] V_c^f[f,b]* (1/(E_e − δ_c − E_b) + 1/(E_f − δ_c − E_b)).
                let(ex,fx)=(cl.excited[i-n0],cl.excited[j-n0]);let delta=self.carriers[c].detuning;
                for b in 0..self.ground_states.len(){
                    h[(i,j)]+=Complex::cis(w*t)*v[c][(ex,b)]*v[c][(fx,b)].conj()*(0.5*(1./(e.excited_energies[ex]-delta-e.energies[b])+1./(e.excited_energies[fx]-delta-e.energies[b])));
                }
            }
        }
        let mut shift=Dense::zeros(n,n);
        for p in cl.paths.iter().filter(|p|path(p)){
            let z=self.path_energy(e,k,p.from.carrier);let lu=denominator(&e.excited_hamiltonian,-z)?;
            let(c,d,f)=(p.from.carrier,p.via.carrier,p.back.carrier);
            // U = −[V_d† G V_c]_{via, from}, W̃ = −[V_f† G V_d]_{back, via}.
            let mut gc:Vec<Complex>=(0..ne).map(|i|v[c][(i,p.from.state)]).collect();lu.solve(&mut gc);
            let mut gd:Vec<Complex>=(0..ne).map(|i|v[d][(i,p.via.state)]).collect();lu.solve(&mut gd);
            let u=-(0..ne).fold(Complex::ZERO,|s,i|s+v[d][(i,p.via.state)].conj()*gc[i]);
            let w=-(0..ne).fold(Complex::ZERO,|s,i|s+v[f][(i,p.back.state)].conj()*gd[i]);
            let(i,j)=(cl.position(p.back.state).unwrap(),cl.position(p.from.state).unwrap());
            shift[(i,j)]+=Complex::cis(p.residual*t)*w*u*(1./(z-e.energies[p.via.state]-self.carriers[d].detuning));
        }
        h.add_scaled(Complex::ONE,&shift);
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
                let z=z*Complex::cis((cl.excited_frames[p]-cl.frames[j]-self.carriers[c].detuning)*t);h[(n0+p,j)]+=z;h[(j,n0+p)]+=z.conj();
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
        Ok(Local{effective:h,shift,jumps,admixture,excited_readout})
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

    /// Beam l's response R_l to all orders in an excited-energy shift w (rad/us): column b solved
    /// with (K − E_b − δ_c + w)⁻¹, K decoupled from b's cluster's explicit states. R_l − w R2_l is
    /// its first order.
    pub fn response(&self,elimination:&Elimination,beam:usize,shift:f64)->Result<Dense,String>{
        let carrier=self.carriers.iter().find(|c|c.beams.contains(&beam)).unwrap();let coupling=&self.folded[beam];
        let mut out=Dense::zeros(coupling.rows,coupling.cols);let mut column=vec![Complex::ZERO;coupling.rows];
        let dressings=self.dressings(&elimination.excited,&elimination.excited_hamiltonian);
        for b in 0..coupling.cols{
            if (0..coupling.rows).all(|i|coupling[(i,b)]==Complex::ZERO){continue;}
            let decoupled=self.explicit_of.get(b).filter(|p|p.iter().any(|&x|x)).map(|p|self.decoupled(&elimination.excited_hamiltonian,p));
            let lu=denominator(decoupled.as_ref().unwrap_or(&elimination.excited_hamiltonian),shift-elimination.energies[b]-carrier.detuning)?;
            self.source(beam,b,dressings.get(b).map_or(&[][..],|d|d),&mut column);lu.solve(&mut column);for i in 0..coupling.rows{out[(i,b)]=column[i];}
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
    /// The fourth-order fold of a cluster of ground states at r and t (module documentation, # Fourth
    /// order), dense: on the slow invariant subspace Ψ = (x, W x) of the no-jump generator,
    /// W = −R + W3, W3[:,b] = G_b(ω_c) Σ_b′ [Σ_d R_c[:,b′](V_d†R_d)_b′b + Σ_{d≠c} R_d[:,b′](V_d†R_c)_b′b]
    /// for the photons of carrier c (the secular triples), G_b(ω) = (K − E_b − ω)⁻¹; in y = M^{1/2} x,
    /// M = 1 + Σ_c R_c†R_c: H̃ = H_eff + Σ_c V_c†W3^c − ½[R†R, V†R] + ½[R†R, H_g] (its Hermitian part;
    /// the anti-Hermitian part −(i/2) Σ L̃†L̃) and L̃ = C_a (R_c − W3^c − ½ R_c R†R) (the jumps' sign
    /// the module's); after a jump into this cluster the slow part Π = 1 + ½R†R + YW of the bare
    /// state, (YW)_ab = −Σ_c V_c[:,a]† G_a(ω_c) R_c[:,b]. Everything in the frame where each ground
    /// state rotates at its energy, the cluster's frames applied at the end (e^{i(θ_a − θ_b)t}).
    pub fn fourth(&self,e:&Elimination,k:usize,r:[f64;3],t:f64)->Result<Fourth,String>{
        let cl=&self.clusters[k];let(n,ng,nc)=(cl.states.len(),self.ground_states.len(),self.carriers.len());
        if !cl.excited.is_empty(){return Err("The fourth-order fold takes a cluster of ground states".into());}
        let local=self.local_at(e,k,r,t)?;let a=self.amplitudes_at(r);
        let v:Vec<Dense>=self.carriers.iter().map(|c|combine(&c.beams,&a,&self.folded)).collect();
        let rc:Vec<Dense>=self.carriers.iter().map(|c|combine(&c.beams,&a,&e.responses)).collect();
        // R†R and V†R over every ground state, each carrier with itself (the cross terms beat).
        let mut rr=Dense::zeros(ng,ng);let mut vr=Dense::zeros(ng,ng);
        for c in 0..nc{rr.add_scaled(Complex::ONE,&rc[c].adjoint_product(&rc[c]));vr.add_scaled(Complex::ONE,&v[c].adjoint_product(&rc[c]));}
        let resolve=|b:usize,c:usize,column:&mut [Complex]|->Result<(),String>{denominator(&e.excited_hamiltonian,-e.energies[b]-self.carriers[c].detuning)?.solve(column);Ok(())};
        // W3^c per carrier (ne × ng).
        let mut w3:Vec<Dense>=vec![];
        for c in 0..nc{
            let mut u=Dense::zeros(self.excited_states.len(),ng);
            for d in 0..nc{
                u.add_scaled(Complex::ONE,&rc[c].product(&v[d].adjoint_product(&rc[d])));
                if d!=c{u.add_scaled(Complex::ONE,&rc[d].product(&v[d].adjoint_product(&rc[c])));}
            }
            for b in 0..ng{let mut col:Vec<Complex>=(0..u.rows).map(|i|u[(i,b)]).collect();resolve(b,c,&mut col)?;for i in 0..u.rows{u[(i,b)]=col[i];}}
            w3.push(u);
        }
        // H^(4) over the ground states: Σ_c V_c†W3^c − ½[R†R, V†R] + ½[R†R, H_g].
        let mut h4=Dense::zeros(ng,ng);
        for c in 0..nc{h4.add_scaled(Complex::ONE,&v[c].adjoint_product(&w3[c]));}
        let commutator=|x:&Dense,y:&Dense|{let mut m=x.product(y);m.add_scaled(-Complex::ONE,&y.product(x));m};
        h4.add_scaled(Complex::new(-0.5,0.),&commutator(&rr,&vr));h4.add_scaled(Complex::new(0.5,0.),&commutator(&rr,&e.ground));
        // The jumps: each class's photons from the corrected response R̂_c = R_c − W3^c − ½ R_c R†R.
        let corrected:Vec<Dense>=(0..nc).map(|c|{let mut m=rc[c].clone();m.add_scaled(-Complex::ONE,&w3[c]);m.add_scaled(Complex::new(-0.5,0.),&rc[c].product(&rr));m}).collect();
        let jumps:Vec<[Dense;3]>=cl.classes.iter().map(|class|{let target=&self.clusters[class.target];
            std::array::from_fn(|x|{let mut m=Dense::zeros(target.dimension(),n);
                for &(p,w)in &class.photons{let j=self.emission[x].product(&corrected[p.carrier]);
                    m[(target.position(p.to).unwrap(),cl.position(p.from).unwrap())]+=Complex::cis(w*t)*j[(p.to,p.from)];}
                m})}).collect();
        // H̃: the second order's Hermitian part with H^(4)'s, in the cluster's frames, and −(i/2) Σ L̃†L̃.
        let mut effective=Dense::zeros(n,n);
        for i in 0..n{for j in 0..n{
            let(a,b)=(cl.states[i],cl.states[j]);let phase=Complex::cis((cl.frames[i]-cl.frames[j])*t);
            let x=(local.effective[(i,j)]+local.effective[(j,i)].conj())*0.5;
            let y=(h4[(a,b)]+h4[(b,a)].conj())*0.5;
            effective[(i,j)]=x+phase*y;
        }}
        let mut decay=Dense::zeros(n,n);for j in &jumps{for m in j{decay.add_scaled(Complex::ONE,&m.adjoint_product(m));}}
        effective.add_scaled(Complex::new(0.,-0.5),&decay);
        // Π = 1 + ½R†R + YW on the cluster.
        let mut slow=Dense::zeros(n,n);
        for(i,&a)in cl.states.iter().enumerate(){
            slow[(i,i)]+=Complex::ONE;
            for(j,&b)in cl.states.iter().enumerate(){
                let mut z=rr[(a,b)]*0.5;
                for c in 0..nc{let mut col:Vec<Complex>=(0..rc[c].rows).map(|q|rc[c][(q,b)]).collect();resolve(a,c,&mut col)?;
                    z-=(0..col.len()).fold(Complex::ZERO,|acc,q|acc+v[c][(q,a)].conj()*col[q]);}
                slow[(i,j)]+=Complex::cis((cl.frames[i]-cl.frames[j])*t)*z;
            }
        }
        // The readout: M = 1 + Σ_c W_c†W_c on the cluster (W_c = −R_c + W3_c), M^{−1/2} = 1 − ½Δ + ⅜Δ².
        let w:Vec<Dense>=(0..nc).map(|c|{let mut m=w3[c].clone();m.add_scaled(-Complex::ONE,&rc[c]);Dense::from_fn(m.rows,n,|q,j|m[(q,cl.states[j])])}).collect();
        let phases=Dense::from_fn(n,n,|i,j|Complex::cis((cl.frames[i]-cl.frames[j])*t));
        let framed=|m:&Dense|Dense::from_fn(n,n,|i,j|m[(i,j)]*phases[(i,j)]);
        let mut delta=Dense::zeros(n,n);for wc in &w{delta.add_scaled(Complex::ONE,&wc.adjoint_product(wc));}let delta=framed(&delta);
        let mut root=Dense::from_fn(n,n,|i,j|if i==j{Complex::ONE}else{Complex::ZERO});root.add_scaled(Complex::new(-0.5,0.),&delta);root.add_scaled(Complex::new(0.375,0.),&delta.product(&delta));
        let excited=(0..self.excited_states.len()).map(|i|{let mut m=Dense::zeros(n,n);
            for wc in &w{let row=Dense::from_fn(1,n,|_,j|wc[(i,j)]);m.add_scaled(Complex::ONE,&row.adjoint_product(&row));}
            root.product(&framed(&m)).product(&root)}).collect();
        // The transient after a jump into the cluster: per coherent group X (ne × n) and per excited group
        // its trace factor κ and its excited states' surplus.
        let ne=self.excited_states.len();let mut surplus:Vec<(usize,Dense)>=(0..ne).map(|i|(i,Dense::zeros(n,n))).collect();let mut kappa=vec![];
        for group in &cl.coherent{
            let mut x=Dense::zeros(ne,n);
            for &(ch,w)in group{let col=cl.position(ch.state).unwrap();let phase=Complex::cis(-w*t);for i in 0..ne{x[(i,col)]+=phase*rc[ch.carrier][(i,ch.state)];}}
            let(ch,_)=group[0];
            for levels in &self.excited_groups{
                let i0=levels[0];let gamma=self.decay[(i0,i0)].re;let delta=e.excited_energies[i0]-e.energies[ch.state]-self.carriers[ch.carrier].detuning;
                let k=1.-gamma*gamma/(delta*delta+0.25*gamma*gamma);kappa.push(k);
                for &i in levels{let row=Dense::from_fn(1,n,|_,j|x[(i,j)]);surplus[i].1.add_scaled(Complex::new(k/gamma,0.),&row.adjoint_product(&row));}
            }
        }
        Ok(Fourth{effective,jumps,slow,root,excited,surplus,kappa})
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
                let e=g.eliminate(&g.hamiltonian_at(x)).unwrap();
                for k in 0..g.clusters.len(){
                    let local=g.local_at(&e,k,x,0.37*p as f64).unwrap();let n=g.clusters[k].dimension();
                    let mut sum=Dense::zeros(n,n);for j in &local.jumps{for m in j{sum.add_scaled(Complex::ONE,&m.adjoint_product(m));}}
                    // The folded pairs' paths change the decay at second order, outside the jumps (the
                    // budget's); the rest is the jumps' exactly.
                    let mut h=local.effective.clone();h.add_scaled(-Complex::ONE,&local.shift);
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
                let e=g.eliminate(&g.hamiltonian_at(x)).unwrap();
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
        let r=perturbed(1);let g=Ground::new(&r).unwrap();let e=g.eliminate(&g.hamiltonian_at([0.;3])).unwrap();let pole=g.pole;
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
        let e=g.eliminate(&g.hamiltonian_at([0.;3])).unwrap();
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
