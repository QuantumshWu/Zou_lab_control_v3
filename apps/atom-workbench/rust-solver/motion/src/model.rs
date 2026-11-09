//! The atom's internal model at a point of its motion (SPEC §1.2–§1.8): the far light H_far, the one potential V_L, the trap and
//! its analytic Hessian, the reach; the units and their existence gates; the clusters and their frames; the elimination
//! (Reiter–Sørensen with Brillouin–Wigner dressing, first-order Doppler); the one generator assembly (H, G = ∇H_h, jump operators);
//! the readout, the hazard, the photons and their recoil.
//!
//! # Notation (SPEC §0 units)
//! States s: the 32 of the model basis, the ground states first (`GROUND` of them), H0 diagonal in it (energies E_s, rad/µs, in
//! the frame where the excited states rotate at ν_frame). The emission operators C_a (a = x, y, z, ground ← excited, 1/√µs),
//! D = Σ_a C_a†C_a on the excited states. Each frequency ω of the light is one coherent field E_ω(r) = Σ_{l∈ω} a_l(r) E_l (`field`)
//! and drives two components, each an excited ← ground coupling linear in that field: the co-rotating V_ω(r) = Σ_k E_ω,k(r) U_ω,k
//! at ν = δ_ω and the counter-rotating W_ω(r) = Σ_k E_ω,k(r)* U′_ω,k at ν = −(δ_ω + 2ν_frame) (`Component`). Frequencies are exact
//! integer coordinates on the request's frequency axes and ν_frame (`Beat`).
//!
//! # What is exact and what is folded
//! Every coupling is kept (it enters the cluster's generator as it is and the propagator resolves it) or folded at second order;
//! which is decided per unit (`Unit`, SPEC §5.2) by `solve`, within the existence gates (`Gate`, SPEC §1.5) this module evaluates.
//! A folded far component enters through its van Vleck shift and its Reiter–Sørensen jumps (`far`), linear in T_ω = E_ω E_ω† and
//! exact at the atom; a folded drive of a near carrier through its dressed response (`eliminate`), whose nonlinear dependence on
//! the far light and the other carriers is tabulated over the field's chart (`table`).
//!
//! # A generator along a step's path (propagation.md §1.2–§1.3)
//! Every term is a field's value at the point times plane-wave phases: absorbing beam l and emitting beam l′ it holds
//! e^{i(k_l − k_l′)·r}. On the step's straight path r_h + v(t − t_m) a cluster's generator is Σ_j M_j e^{iω_j(t − t_m)},
//! ω_j = Ω_j + q_j·v (`Part`): Ω_j the term's beat in the cluster's frames, q_j its wave vector less the Doppler frames'.
//! The envelopes and the tables are held at the point.
pub mod far;
pub mod potential;
pub mod reach;
pub mod units;
pub mod cluster;
pub mod eliminate;
pub mod assemble;
pub mod readout;
pub mod photon;

use crate::field::Light;
use crate::linalg::{C64, Mat};
use crate::request::Request;
pub use crate::request::{GROUND, STATES};

/// A frequency, exactly: integer coordinates on the request's frequency axes, then on the excited states' frame frequency ν_frame;
/// its value is 2π Σ_k n_k f_k (rad/µs, `Model::omega`). Equal frequencies have equal coordinates: a frame or a beat that is zero
/// in exact arithmetic is zero here (SPEC §1.4 frames), whatever the floating point makes of its value.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct Beat(pub Vec<i64>);

/// One component of the light: frequency `frequency` (an index into the field's frequencies), its co-rotating coupling
/// (`counter` false: V_ω e^{−iνt}, ν = δ_ω) or its counter-rotating one (W_ω e^{−iνt}, ν = −(δ_ω + 2ν_frame)).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Component { pub frequency: usize, pub counter: bool }

/// The lights on, per frequency of the field: L in V_L (SPEC §1.3). `Model::running` is every light, `Model::prepared` the trap
/// beam's frequency alone (before t = 0).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Lights(pub Vec<bool>);

/// A point of a trajectory: position r (µm), momentum κ = p/ħ (1/µm), time t (µs).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point { pub r: [f64; 3], pub kappa: [f64; 3], pub t: f64 }

/// A scalar's second-order expansion at a point, analytic: value, gradient, Hessian (rad/µs, rad/µs/µm, rad/µs/µm²).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quadratic { pub value: f64, pub gradient: [f64; 3], pub hessian: [[f64; 3]; 3] }

/// The trap (SPEC §1.3): the local minimum of the prepared potential the descent from the focus reaches, and its analytic
/// Hessian's eigenvectors and frequencies there.
#[derive(Clone, Debug, PartialEq)]
pub struct Trap {
    /// The trap's centre (µm from the focus).
    pub centre: [f64; 3],
    /// V_prep at the centre (rad/µs): V_min, the reach's floor.
    pub minimum: f64,
    /// The principal axes: unit rows, the Hessian's eigenvectors.
    pub axes: [[f64; 3]; 3],
    /// The angular frequencies along them (rad/µs): ω_i = √(λ_i ħ/m), λ_i > 0 the Hessian's eigenvalues (rad/µs/µm²).
    pub frequencies: [f64; 3],
    /// The angular frequencies along the lab axes x, y, z (rad/µs): √(∂²V/∂x_a² ħ/m) at the centre, the curvature along each
    /// lab axis (the T windows' and n̄'s periods, the classical flag).
    pub lab: [f64; 3],
}

/// A proven envelope of the prepared potential (the initial ensemble's exact rejection sampling): with the trap light's
/// static form a†Q_b a in its beams' amplitudes a_l (far.rs: V_prep,b = Σ_l'l a_l'* a_l [P_l'l]_bb), −V_prep(r; p) ≤
/// Σ_l'l |Q_l'l| √(A_l'(r) A_l(r)) ≤ (Σ_l'l |Q_l'l|) max_l A_l(r), since |a†Qa| ≤ Σ|Q_l'l||a_l'||a_l| and a paraxial beam's
/// |a_l(r)|² ≤ A_l(r) = 1/(1 + u_l(r)), u_l = 2ρ²/w₀² + ζ²/z_R² (|a|² = e^{−t/(1+s)}/(1 + s), s = ζ²/z_R², t = 2ρ²/w₀², and
/// e^{−x} ≤ 1/(1 + x) give |a|² ≤ 1/(1 + s + t); a plane wave's A = 1).
#[derive(Clone, Debug, PartialEq)]
pub struct Envelope {
    /// The trap light's beams (indices into the request's beams).
    pub beams: Vec<usize>,
    /// |Q_l'l| (rad/µs), row-major over `beams`.
    pub weights: Vec<f64>,
}

/// The model's counted work at trajectory states (SPEC §5.1, counted by algorithm, not calibrated): the assembly's complex
/// multiply-adds per call, and what the propagation's count (step's) takes, the cluster's dimension and its parts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Work { pub assembly: f64, pub states: f64, pub parts: f64 }

/// The atoms' reach (SPEC §1.8): the sublevel set {r : V_prep(r) − V_min ≤ ln(1/s) k_B T_r} of the prepared potential, with the
/// momenta of that kinetic energy. Its extrema are provable bounds: interval subdivision of `cell` with Lipschitz remainders.
#[derive(Clone, Debug, PartialEq)]
pub struct Reach {
    /// T_r (µK): the temperature the atoms reach (never lowered; the solve's trials read it).
    pub temperature_uk: f64,
    /// s: the unsampled share (SPEC §3).
    pub share: f64,
    /// ln(1/s) k_B T_r (rad/µs): the energy above V_min the reach holds, in position and in kinetic energy.
    pub energy: f64,
    /// V_min (rad/µs).
    pub minimum: f64,
    /// A box (µm, [low, high] per lab axis) that provably holds the sublevel set: where the subdivisions start.
    pub cell: [[f64; 2]; 3],
    /// The largest speed |v| (µm/µs) in the reach: √(2 (ħ/m) energy).
    pub speed: f64,
    /// The ground populations p of the prepared potential it is the sublevel set of.
    pub populations: [f64; GROUND],
}

/// A unit (SPEC §5.2): one coupling the solve keeps or folds. The more exact level of every fold is keeping it.
#[derive(Clone, Debug, PartialEq)]
pub enum Unit {
    /// A drive of near carrier `carrier` (a co-rotating component) from ground state `ground` into excited state `excited`.
    /// Kept: the excited state is explicit and the drive enters the generator as it is. Folded into a state another kept drive
    /// makes explicit, it is a pair of channels through that state, and its error is that pair's (units.rs).
    Drive { carrier: Component, excited: usize, ground: usize },
    /// A coupling between ground states `a` ← `b` at one beat: the second-order term absorbing carrier `from` and emitting
    /// carrier `to` (a Raman or a Zeeman coupling of the light, one state's beat of two carriers), or with both None a static
    /// coupling of the far light. Between two clusters, kept joins them; within one, kept resolves its beat (a far beat).
    Term { a: usize, b: usize, from: Option<Component>, to: Option<Component> },
    /// A far component's fold (its van Vleck shift and Reiter–Sørensen jumps). Kept: a co-rotating component is a near carrier;
    /// a counter-rotating one cannot be kept (its fold must exist, else the request is refused).
    Far { component: Component },
    /// The beat T^{ωω′} e^{−i(ν − ν′)t} of two far components' fields in the far light's shifts. Kept: a beat of the generator;
    /// folded: dropped.
    FarBeat { first: Component, second: Component },
    /// A static coupling of an explicit excited state `excited` to another excited state `other`. Kept: `other` is explicit too.
    Excited { excited: usize, other: usize },
    /// The coherence of the photons landing in ground state `landing` at two frame frequencies `first` and `second` (in the
    /// landing cluster's frames). Kept: one class, their interference with its beat; folded: two classes.
    Coherence { landing: usize, first: Beat, second: Beat },
}

/// A unit's existence gate over the reach (SPEC §1.5): provable upper bounds over the reach in phase space. The fold exists only
/// where both are below ½ (`Gate::exists`); else the unit is kept.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gate {
    /// The static Stewart bound: the coherent peak's upper bound of S = 2m² (a drive's: its ground state's total S_b over its
    /// folded drives).
    pub saturation: f64,
    /// The motional bound: the upper bound of a = |dΔE/dt|/|ΔE|², the sweep v·∇ΔE + k·v̇.
    pub sweep: f64,
}

/// A folded unit's errors on the outputs at one trajectory state (SPEC §1.5 → §5.2): its small parameter measured there and mapped
/// by the leading-order relations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UnitError {
    /// The small parameter measured: S (a drive's ground state), m² (a coupling's mixing), a (the sweep), (w/σ)² (the Doppler order).
    pub parameter: f64,
    /// The absolute error it makes on the display rows' populations: ≤ S_b; m²(p_a + p_b); m²√(p_i p_j); m².
    pub population: f64,
    /// The relative error it makes on T: ≈ √2 S; a; (w/σ)².
    pub temperature: f64,
}

/// A trajectory state the units' errors are measured at (the solve's trial states).
#[derive(Clone, Debug, PartialEq)]
pub struct Sample { pub cluster: usize, pub at: Point, pub psi: Vec<C64> }

/// A cluster (SPEC §1.4): the states the kept couplings join, its ground states first and then its explicit excited states;
/// a trajectory's state space between two jumps.
#[derive(Clone, Debug, PartialEq)]
pub struct Cluster {
    /// Its states (indices of the model basis): the ground states, then the explicit excited states.
    pub states: Vec<usize>,
    /// How many of them are ground states.
    pub ground: usize,
    /// Each state's frame θ_s, exactly: ψ_s = e^{−iθ_s t} x_s, so that x_s holds E_s − θ_s on the diagonal and a term a ← b of
    /// beat Ω (∝ e^{iΩt}) is at Ω + θ_a − θ_b.
    pub frames: Vec<Beat>,
    /// Each state's Doppler frame K_s (1/µm): on a path r_h + v(t − t_m) the state is ψ_s = e^{iK_s·v(t − t_m)} x_s, so that x_s
    /// holds K_s·v on the diagonal and a term a ← b of wave vector q is at q − (K_a − K_b). Exact where every pair of its states
    /// holds one wave vector; the rest are path beats.
    pub waves: Vec<[f64; 3]>,
    /// The Doppler frames exactly (`Wave`).
    pub(crate) waves_exact: Vec<Beat>,
}

/// One part of a cluster's generator at a point (SPEC §1.4): on the step's path H(t) = Σ_j M_j e^{iω_j(t − t_m)},
/// ω_j = Ω_j + q_j·v, and the gradient of its Hermitian part G(t) = Σ_j G_j e^{iω_j(t − t_m)}. Parts come in pairs (Ω, q),
/// (−Ω, −q) (one part where both vanish), their Hermitian parts paired, G_{−j} = G_j†.
#[derive(Clone, Debug, PartialEq)]
pub struct Part {
    /// Its beat in the cluster's frames, exactly.
    pub beat: Beat,
    /// Ω (rad/µs): the beat's value.
    pub omega: f64,
    /// q (1/µm): its wave vector less the Doppler frames' K_a − K_b.
    pub wave: [f64; 3],
    /// M_j at the point, n × n on the cluster's states (rad/µs), its phase e^{i(Ω t + q·r)} there: the Hermitian part of that
    /// beat with its decay −(i/2) Σ J†J.
    pub generator: Mat,
    /// G_j, the gradient of its Hermitian part at the point (rad/µs/µm), per lab axis.
    pub force: [Mat; 3],
}

/// A cluster's generator at a point with some lights on: its parts, from the one assembly (`Model::assemble`).
#[derive(Clone, Debug, PartialEq)]
pub struct Generator { pub cluster: usize, pub at: Point, pub parts: Vec<Part> }

/// One class of photons (SPEC §1.6) of a cluster's state at a point: the photons of one frame frequency landing in one cluster,
/// folded and explicit amplitudes added. (A class of one frequency landing in two clusters is two of these: the joint law of
/// direction, polarization and landing is the same either way, ∝ ‖P_C Σ_a ε_a u_a‖².)
#[derive(Clone, Debug, PartialEq)]
pub struct Photon {
    /// The cluster the jump lands the atom in.
    pub landing: usize,
    /// |k_φ| (1/µm): the emission recoil is −k_φ n.
    pub wave_number: f64,
    /// u_a = J_{a,φ} ψ (a = x, y, z) on the landing cluster's states: the class's probability ∝ Σ_a ‖u_a‖², the direction's
    /// density ∝ tr G − n·Re(G)·n (G_ab = ⟨u_a|u_b⟩), the polarization ε across n by ‖Σ_a ε_a u_a‖², the state after f = Σ_a ε_a u_a.
    pub amplitudes: [Vec<C64>; 3],
    /// ∇u_a of the folded part, whose absorption's mean the jump carries (`Photon::recoil`): gradients[i][a], i the lab axis.
    pub gradients: [[Vec<C64>; 3]; 3],
    /// The explicit states' absorption sources, whose mean the Ehrenfest force carries: per source its amplitudes (u_a's part
    /// the source holds, with the square root of its share) and its log-gradient ∇ln σ (1/µm, complex per lab axis).
    pub explicit: Vec<([Vec<C64>; 3], [C64; 3])>,
}

/// The absorption recoil's first two moments of a jump into f = Σ_a ε_a u_a (SPEC §1.6, one rule): the mean the jump carries
/// (the folded part's, Im⟨f|∇f_F⟩/‖f‖²) and the total second moment M (1/µm²), the local second Kramers–Moyal coefficient
/// (the folded part's Re⟨∇f_F|∇f_F⟩/‖f‖² and the explicit sources' Re[∇ln σ*⊗∇ln σ] by their shares). The kick is the mean plus
/// a zero-mean fluctuation of covariance M − mean meanᵀ, positive semidefinite by Cauchy–Schwarz.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Recoil { pub mean: [f64; 3], pub second: [[f64; 3]; 3] }

/// The model: a request's states and light with a set of units kept (SPEC §1).
pub struct Model {
    pub(crate) light: Light,
    pub(crate) axes: far::Axes,
    /// The light's distinct wave vectors (1/µm): the coordinates of every exact wave vector (`Wave`); per request beam its own.
    pub(crate) waves: Vec<[f64; 3]>,
    pub(crate) beam_wave: Vec<usize>,
    pub(crate) hbar_over_mass: f64,
    /// The bare energies E_s (H0's diagonal).
    pub(crate) energies: Vec<f64>,
    /// H0 less every frequency's outside form at the focus: the states with every light off.
    pub(crate) base: Mat,
    pub(crate) emission: [Mat; 3],
    /// D on the excited states.
    pub(crate) decay: Mat,
    /// The far light per frequency (far.rs).
    pub(crate) tones: Vec<far::Tone>,
    /// The units kept.
    pub(crate) kept: Vec<Unit>,
    pub(crate) clusters: Vec<Cluster>,
    /// Per ground state its cluster.
    pub(crate) membership: [usize; GROUND],
    /// The near carriers (eliminate.rs).
    pub(crate) carriers: Vec<eliminate::Carrier>,
    /// The photons' sources (photon.rs) and per cluster its classes.
    pub(crate) sources: Vec<photon::Source>,
    pub(crate) classes: Vec<Vec<photon::Class>>,
    /// The states' labels and the beams' names (the ledger's and the method's words).
    pub(crate) labels: Vec<String>,
    pub(crate) names: Vec<String>,
}

/// An exact wave vector: integer coordinates on the light's distinct wave vectors (`Model::waves`), with a `Beat`'s integer
/// arithmetic, its value Σ_w n_w k_w. Equal wave vectors have equal coordinates: a Doppler frame exact in exact arithmetic leaves
/// a residual of exactly zero.
pub(crate) type Wave = Beat;

impl Model {
    /// The model of a request with the units `kept` (SPEC §1.4): the far light as its far units say, the near carriers'
    /// eliminations, the clusters the kept couplings join with their frames. Refused where a fold cannot be made: a
    /// counter-rotating component kept, or whose fold does not exist (units.rs `Model::far_gate`).
    pub fn new(request: &Request, light: &Light, kept: &[Unit]) -> Result<Model, String> {
        let axes = far::Axes::new(request)?;
        let mut waves: Vec<[f64; 3]> = vec![];
        let beam_wave = light
            .beams
            .iter()
            .map(|b| {
                let k = b.wave_vector();
                match waves.iter().position(|w| *w == k) {
                    Some(at) => at,
                    None => {
                        waves.push(k);
                        waves.len() - 1
                    }
                }
            })
            .collect();
        let near = |c: Component| kept.contains(&Unit::Far { component: c });
        let mut tones = vec![];
        for frequency in 0..light.frequencies.len() {
            if near(Component { frequency, counter: true }) {
                return Err("反旋分量不能作为近共振光保留：其折叠必须存在".into());
            }
            tones.push(far::Tone::new(request, light, &axes, frequency, [!near(Component { frequency, counter: false }), true])?);
        }
        let mut base = request.hamiltonian.clone();
        for t in &tones {
            base.add_scaled(-C64::ONE, &t.focus);
        }
        let mut model = Model {
            light: light.clone(),
            axes,
            waves,
            beam_wave,
            hbar_over_mass: crate::units::hbar_over_mass(request.mass_kg),
            energies: far::energies(request),
            base,
            emission: request.emission.clone(),
            decay: far::decay(request),
            tones,
            kept: kept.to_vec(),
            clusters: vec![],
            membership: [0; GROUND],
            carriers: vec![],
            sources: vec![],
            classes: vec![],
            labels: request.states.clone(),
            names: request.beams.iter().map(|b| b.name.clone()).collect(),
        };
        for component in model.components().into_iter().filter(|c| c.counter) {
            let gate = model.far_gate(component);
            if !gate.exists() {
                return Err(format!("反旋分量的折叠不存在（S = {:.2}）：请求超出模型的适用范围", gate.saturation));
            }
        }
        model.carriers = eliminate::Carrier::all(&model, request)?;
        let edges: Vec<cluster::Edge> = kept.iter().filter_map(|u| model.edge(u)).collect();
        let (mut clusters, membership) = cluster::clusters(STATES, &edges, &model.axes.zero(), &model.no_wave());
        for c in clusters.iter_mut() {
            c.waves = c.waves_exact.iter().map(|w| model.wave_value(w)).collect();
        }
        model.clusters = clusters;
        model.membership = membership;
        for ci in 0..model.carriers.len() {
            for l in 0..model.carriers[ci].beams.len() {
                for e in GROUND..STATES {
                    for b in 0..GROUND {
                        if model.carriers[ci].folded[l][(e, b)] != C64::ZERO && model.drive_kept(ci, e, b) {
                            model.carriers[ci].folded[l][(e, b)] = C64::ZERO;
                        }
                    }
                }
            }
        }
        model.sources = model.far_sources();
        model.sources.extend(model.explicit_sources());
        model.sources.extend(model.near_sources());
        model.classes = (0..model.clusters.len()).map(|c| model.classes(c)).collect();
        Ok(model)
    }

    /// Every light: L_run.
    pub fn running(&self) -> Lights {
        Lights(vec![true; self.light.frequencies.len()])
    }
    /// The trap beam's frequency alone: L_prep.
    pub fn prepared(&self) -> Lights {
        Lights((0..self.light.frequencies.len()).map(|w| w == self.light.trap).collect())
    }
    /// The clusters.
    pub fn clusters(&self) -> &[Cluster] {
        &self.clusters
    }
    /// The cluster a ground state belongs to.
    pub fn cluster_of(&self, ground: usize) -> usize {
        self.membership[ground]
    }
    /// ħ/m (µm²/µs).
    pub fn hbar_over_mass(&self) -> f64 {
        self.hbar_over_mass
    }
    /// A frequency's value (rad/µs).
    pub fn omega(&self, f: &Beat) -> f64 {
        self.axes.omega(f)
    }
    /// An exact wave vector's value (1/µm), summed from its coordinates in their order.
    pub(crate) fn wave_value(&self, w: &Wave) -> [f64; 3] {
        let mut out = [0.0; 3];
        for (&n, k) in w.0.iter().zip(&self.waves) {
            if n != 0 {
                for i in 0..3 {
                    out[i] += n as f64 * k[i];
                }
            }
        }
        out
    }
    /// A request beam's wave vector, exactly.
    pub(crate) fn beam_wave(&self, beam: usize) -> Wave {
        let mut w = vec![0; self.waves.len()];
        w[self.beam_wave[beam]] = 1;
        Beat(w)
    }
    /// The zero wave vector.
    pub(crate) fn no_wave(&self) -> Wave {
        Beat(vec![0; self.waves.len()])
    }
    /// The light's components: per frequency its co- and counter-rotating one.
    pub(crate) fn components(&self) -> Vec<Component> {
        (0..self.light.frequencies.len()).flat_map(|frequency| [false, true].map(|counter| Component { frequency, counter })).collect()
    }
    /// The loss energy E_s = (ħ/m)κ²/2 + V_run(r; p(ψ)) (rad/µs).
    pub fn energy(&self, cluster: usize, at: &Point, psi: &[C64]) -> f64 {
        let kinetic = 0.5 * self.hbar_over_mass * at.kappa.iter().map(|k| k * k).sum::<f64>();
        kinetic + self.potential(at.r, &self.ground_populations(cluster, psi), &self.running())
    }
    /// The loss test (SPEC §1.7): outside the trap's region and E_s above V_run's far value.
    pub fn lost(&self, cluster: usize, at: &Point, psi: &[C64]) -> bool {
        self.light.outside(&self.light.amplitudes(at.r))
            && self.energy(cluster, at, psi) > self.asymptote(&self.ground_populations(cluster, psi), &self.running())
    }
    /// A cluster state's ground populations, normalized on the ground states (zero elsewhere).
    pub(crate) fn ground_populations(&self, cluster: usize, psi: &[C64]) -> [f64; GROUND] {
        let c = &self.clusters[cluster];
        let total: f64 = psi[..c.ground].iter().map(|z| z.norm_sqr()).sum();
        let mut p = [0.0; GROUND];
        for (i, &s) in c.states[..c.ground].iter().enumerate() {
            p[s] = psi[i].norm_sqr() / total;
        }
        p
    }
}

/// The model as one run's integrator sees it (`step::Dynamics`): the model, its running and prepared lights, and the reference
/// state's ground populations.
pub struct Run<'a> {
    pub model: &'a Model,
    pub reference: [f64; GROUND],
    running: Lights,
    prepared: Lights,
}

impl<'a> Run<'a> {
    pub fn new(model: &'a Model, reference: [f64; GROUND]) -> Run<'a> {
        Run { model, reference, running: model.running(), prepared: model.prepared() }
    }
}

impl crate::step::Dynamics for Run<'_> {
    fn hbar_over_mass(&self) -> f64 {
        self.model.hbar_over_mass
    }
    fn reference(&self, r: [f64; 3]) -> Quadratic {
        self.model.expansion(r, &self.reference, &self.prepared)
    }
    fn cluster(&self, cluster: usize) -> &Cluster {
        &self.model.clusters[cluster]
    }
    fn cluster_of(&self, ground: usize) -> usize {
        self.model.membership[ground]
    }
    fn generator(&self, cluster: usize, at: &Point) -> Generator {
        self.model.assemble(cluster, at, &self.running)
    }
    fn photons(&self, cluster: usize, at: &Point, psi: &[C64]) -> Vec<Photon> {
        self.model.photons(cluster, at, psi)
    }
    fn recoil(&self, photon: &Photon, polarization: &[f64; 3]) -> Recoil {
        photon.recoil(polarization)
    }
    fn energy(&self, cluster: usize, at: &Point, psi: &[C64]) -> f64 {
        self.model.energy(cluster, at, psi)
    }
    fn lost(&self, cluster: usize, at: &Point, psi: &[C64]) -> bool {
        self.model.lost(cluster, at, psi)
    }
    fn readout(&self, cluster: usize, at: &Point, psi: &[C64]) -> [f64; STATES] {
        self.model.readout(cluster, at, psi)
    }
}


pub use photon::hazard;
