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
use crate::model::rs::fold::{Grouping, Local, Out, Plan};
use crate::model::rs::{Making, Rs};
use crate::model::classify::{Account, Judgment, Scales, Tier};
use crate::model::{Beat, Cluster, Moments, Point, Quadratic, Scattering};
use crate::request::{GROUND, STATES};
use crate::scene::{Key, Scene};
use crate::select::Selection;
use crate::step::{Emitted, Internal, Optical};
use crate::units::{hbar_over_mass, rounding, RAD_PER_US_PER_MHZ};

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
    /// The run's own number (the thread's buffers and pooled nodes are its).
    id: usize,
    /// The run's judgment of its node's beats and blocks (model/classify.rs, `judge`).
    judgment: Judgment,
    /// The exact blocks of P (module documentation, # Exact blocks): places in P, each sorted (its ground states first);
    /// and each place's block.
    blocks: Vec<Vec<usize>>,
    block_of: Vec<usize>,
    beats: RwLock<Beats>,
    /// The candidate tones and their keys (module documentation, # The generator by beats): every node holds the
    /// layout's beats, so they are the run's, made once (`Selected::tones`).
    candidates: Vec<Candidate>,
    keys: Vec<Vec<i64>>,
    /// The lift's groups in the order of their keys, each with its elements' fast index (an index among the run's
    /// fast indices of the lift) and the value of its slow beat (`Selected::rows`).
    lifts: Vec<(usize, Vec<(usize, f64)>)>,
}

/// Which beats a model drops (`Selected::made`): those its judgment drops (model/classify.rs, # The drop tier), or (a
/// check's) exactly the candidate pairs of the listed |Ω| (rad/µs, to 10⁻⁹ of it), none of an empty list.
#[derive(Clone, Debug)]
pub(crate) enum Drops {
    Judged,
    #[cfg(test)]
    Only(Vec<f64>),
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
/// momentum it was made at; its counted work (real multiply-adds of putting the model in the frame, P × P's matrices
/// counted whole; `Rs::at`'s own is not counted, its count not given); and at a middle the lift by the lab's groups
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
            template: Node { kappa: [0.0; 3], at: [0.0; 3], generator: vec![(0, Mat::zeros(n, n))], gradient: Vec::new(), classes: Vec::new(), kicked: false, whole: true, work: 0.0, lift: Vec::new() },
            tables: Tables::default(),
            id: RUNS.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            judgment: Judgment { tiers: Vec::new(), blocks: vec![(0..n).collect()], dropped: 0.0, pairs: Vec::new() },
            blocks: vec![(0..n).collect()],
            block_of: vec![0; n],
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
        run.tones();
        run.lift_tables();
        Ok(run)
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
        let mut template = Node { kappa: [0.0; 3], at: [0.0; 3], generator: vec![(0, Mat::zeros(n, n))], gradient: Vec::new(), classes: Vec::new(), kicked: false, whole: true, work: 0.0, lift: Vec::new() };
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
        self.judgment = judgment;
        self.decay_terms();
        // The pooled nodes were made on the layout before.
        POOL.with(|p| p.borrow_mut().retain(|x| x.0 != self.id));
    }

    /// The run's judgment of its node's beats (model/classify.rs): per slot of the structure's node, before the beats
    /// left out were taken from the layout, its beat and tier; the blocks; what the dropped beats cost.
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
        work.out.contract(self.rs.plan(), &self.lab, &work.local, None, Making::node(false));
        let nodes = 3.0 * (work.local.work + work.out.work);
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
        // The account.
        let request = &self.scene.request;
        let shortest = (0..STATES)
            .flat_map(|i| (0..STATES).map(move |j| (i, j)))
            .filter(|&(i, j)| request.emission.iter().any(|c| c[(i, j)] != C64::ZERO))
            .map(|(i, j)| self.scene.jumps().wave_number[(i, j)].re.abs())
            .fold(f64::INFINITY, smaller);
        // The model's own error is measured after the pilot, on the run's paths (select.rs): at the model's making it is
        // not known, and the drop is weighed at none of it -- the drop's D is booked on top of whatever it is.
        let error = [0.0_f64; 2];
        let scales = crate::estimate::Scales { hbar_over_mass: self.hbar_over_mass, frequencies: [0.0; 3] };
        let account = Account {
            populations: error[0],
            temperature: error[1],
            duration: request.times_us.last().copied().unwrap_or(0.0),
            escape: crate::estimate::Escape::new(depth, self.hbar_over_mass),
            floor: crate::estimate::semiclassical([1.0, shortest * shortest], &scales),
        };
        (pairs, account, nodes)
    }

    /// A step's counted work with the beats `kept` of the generator other than the static one (model/classify.rs, # The
    /// drop tier; complex multiply-adds): the three nodes' (`nodes`), and the propagation's part the tones change
    /// (spectral.rs, # The second order by tones) -- per tone member its envelope to the eigenbasis, 6n³; per product
    /// of two first-order classes beyond the static group's with itself 2n³ at each of three nodes; per beating class
    /// followed three products of n³; the middle's solve where there are tones, n³ + n³/3 + n³ --, the tones T the
    /// kept beats with their opposites, the classes the distinct sums of two of their exact beats and the static one.
    fn counted(&self, kept: &[usize], nodes: f64) -> f64 {
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
        let cube = (self.rs.keep.len() as f64).powi(3);
        let first = (1 + 2 * tones) as f64;
        let solve = if tones > 0 { 7.0 / 3.0 } else { 0.0 };
        nodes + cube * (12.0 * tones as f64 + 6.0 * (first * first - 1.0) + 3.0 * (classes.len() as f64 - 1.0) + solve)
    }

    /// The model at a point in the frame, by beats (`Node`; module documentation, # The generator by beats), with the
    /// channels' photon forms where `kicked` (a jump's node): the fold at the point (`Rs::evaluate`), its contraction by
    /// the lab's frequencies, put in the node's structure (`layout`), all in the thread's buffers.
    pub fn node(&self, at: V3, kappa: V3, kicked: bool) -> Node {
        let mut node = self.template.clone();
        self.node_into(&mut node, at, kappa, (kicked, true), false);
        node
    }

    /// A jump's photon's node at a point (`SelectedMiddle::photon`, the tests'): the photon classes alone.
    #[cfg(test)]
    pub(crate) fn photon_node(&self, at: V3, kappa: V3) -> Node {
        let mut node = self.template.clone();
        self.node_into(&mut node, at, kappa, (true, false), false);
        node
    }

    /// The node at a point (`node`) into one of the run's layout: `kicked` with the photon forms, `whole` with the
    /// generator (else the photon classes alone, a jump's photon's: `Making::PHOTONS`); with the lift and its gradient
    /// where `lifted` (a middle's, `Node::lift`: copied from the contraction, which makes them for the model's gradient
    /// anyway).
    fn node_into(&self, node: &mut Node, at: V3, kappa: V3, (kicked, whole): (bool, bool), lifted: bool) {
        NODES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let key = self.id;
        WORK.with(|w| {
            let mut w = w.borrow_mut();
            if w.as_ref().is_none_or(|x| x.0 != key) {
                *w = Some((key, Work::default()));
            }
            let work = &mut w.as_mut().expect("the thread's buffers").1;
            self.scene.point_into(at, kappa, &mut work.point);
            let making = if whole { Making { lift: lifted, ..Making::node(kicked) } } else { Making { lift: lifted, ..Making::PHOTONS } };
            self.rs.evaluate(&self.scene, &work.point, &mut work.local, making);
            work.out.contract(self.rs.plan(), &self.lab, &work.local, None, making);
            self.fill(node, &work.out, (kicked, whole), &mut work.jumps);
            node.work = crate::decide::COMPLEX * (work.local.work + work.out.work);
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
    fn pooled(&self, at: V3, kappa: V3, made: (bool, bool), lifted: bool) -> Rc<Node> {
        let key = self.id;
        let mut node = POOL.with(|p| {
            let mut p = p.borrow_mut();
            p.iter().position(|x| x.0 == key).map(|i| p.swap_remove(i).1)
        })
        .unwrap_or_else(|| Rc::new(self.template.clone()));
        self.node_into(Rc::get_mut(&mut node).expect("a pooled node is no one else's"), at, kappa, made, lifted);
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
    fn fill(&self, node: &mut Node, out: &Out, (kicked, whole): (bool, bool), jumps: &mut Vec<C64>) {
        let n = self.rs.keep.len();
        for c in node.classes.iter_mut() {
            for axis in c.axes.iter_mut() {
                for j in axis.iter_mut() {
                    for e in j.elements.iter_mut() {
                        (e.value, e.gradient, e.kick) = (C64::ZERO, [C64::ZERO; 3], C64::ZERO);
                    }
                }
            }
        }
        for ((list, c), (_, _, which, _, _)) in self.tables.channels.iter().zip(&out.channels).zip(&self.lab.channel_outputs) {
            if *which == 1 && !kicked {
                continue;
            }
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
            return;
        }
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
            }
        }
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
        let node = self.model.pooled(at, path.kappa, (false, true), s == 0.0);
        self.made.set(self.made.get() + node.work);
        #[cfg(test)]
        MADE.with(|m| m.borrow_mut()[0] += 1);
        self.nodes.borrow_mut().push(node.clone());
        node
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

    /// tr(Ĝ_a σ) per axis of H_eff's gradient components at a node, each beat by its factor (none: left out).
    fn contract(node: &Node, each: impl Fn(usize) -> Option<C64>, rho: &[C64]) -> [C64; 3] {
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
            let jumps: usize = node.classes.iter().map(|c| c.axes.iter().map(Vec::len).sum::<usize>()).sum();
            let wave_number = p.wave_number.expect("a jump's node holds the photon forms");
            Some(Emitted { landing: 0, wave_number, amplitudes, kick: p.moments.kick, work: crate::decide::COMPLEX * (5 * jumps * n * n) as f64 })
        });
        self.model.release(node);
        emitted
    }

    /// A jump's photon's node at the time s from the middle along the path (module documentation, # The photons): the
    /// photon classes alone (`Making::PHOTONS`: the channels with their gradients and photon forms, no H_eff and no
    /// generator), made for the jump, its work counted, and not kept -- nothing else reads the model at a jump's point.
    fn photon(&self, path: &Path, s: f64) -> Rc<Node> {
        let at: V3 = std::array::from_fn(|i| self.r[i] + path.velocity[i] * s);
        let node = self.model.pooled(at, path.kappa, (true, false), false);
        self.made.set(self.made.get() + node.work);
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

/// A class's photons of the state ψ at the time t (`photons`): where `whole`, u_a, ∇u_a, the photon forms' K u_a where
/// the node holds them, its moments over ‖ψ‖² = `norm` and its wave number; else u_a alone. Returns its rate W/‖ψ‖²,
/// the moments' weight (`Moments::of`, then `over`: the same operations either way).
fn class_photons(class: &Class, kicked: bool, (values, t): (&[f64], f64), psi: &[C64], norm: f64, p: &mut Photons, whole: bool) -> f64 {
    let n = psi.len();
    let Photons { u, du, ku, .. } = &mut *p;
    for v in u.iter_mut().chain(ku.iter_mut()).chain(du.iter_mut().flatten()) {
        v.clear();
        v.resize(n, C64::ZERO);
    }
    for (a, axis) in class.axes.iter().enumerate() {
        for jump in axis {
            let f = if jump.beat == 0 { C64::ONE } else { C64::cis(-values[jump.beat] * t) };
            // out += f M ψ of the component's elements' M, row by row, each row's sum in column order.
            let apply = |m: &dyn Fn(&Element) -> C64, out: &mut Vec<C64>| {
                for row in jump.elements.chunk_by(|x, y| x.at / n == y.at / n) {
                    let z: C64 = row.iter().map(|e| m(e) * psi[e.at % n]).sum();
                    out[row[0].at / n] += f * z;
                }
            };
            apply(&|e| e.value, &mut u[a]);
            if !whole {
                continue;
            }
            if kicked {
                apply(&|e| e.kick, &mut ku[a]);
            }
            for (i, d) in du.iter_mut().enumerate() {
                apply(&|e| e.gradient[i], &mut d[a]);
            }
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
        let node = self.node(path, 0.0);
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
        (scattering, crate::decide::COMPLEX * (12 * n * n * node.classes.len()) as f64)
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
        SelectedMiddle { model: self, r, t, nodes: RefCell::new(nodes), made: Cell::new(0.0) }
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
}

#[cfg(test)]
mod tests;
