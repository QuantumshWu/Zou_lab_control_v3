//! The trajectory's model (RUSTPLAN §3.2 and its corrections of 2026-11-19, "模型只定一次", "一个折叠公式"): the P the
//! two-point check selected (select.rs) and its RS fold (model/rs.rs `Rs`) as the onefold integrator's
//! `step::Dynamics` / `step::Middle`. Nothing here decides or folds: P, the fold's formula and its coefficients are the
//! checkpoint's, `Rs::at` gives the model at a point, and this module puts it in the integrator's frame and terms.
//!
//! # One cluster, one frame
//! The trajectory's states are P, all of them one cluster (the ground states first: `Rs::keep` is sorted), in the frame
//! θ of floquet/frame.rs chosen at the checkpoint's centre at rest: ψ_a = e^{−iθ_a t} x_a with θ_a an exact vector over
//! the generators (the beams, the carrier) and α. No Doppler frame (K_s = 0, as onefold): at the step's outer nodes the
//! model is taken at the path's own points, so the beams' plane waves turn with the path there and the classical
//! centre of mass gives each key its K·v; its time phases are each beam's frequency at rest with its recoil,
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
//! (`step::Middle::beats`) are the node's beats other than zero with their opposites, each pair one candidate at ν = |Ω|, its key a
//! unit vector over the middle's candidates (no two candidates share a beat, so a sum of two keys is zero iff the
//! members are a candidate's two: the one relation the propagation's second order reads), in the order of their exact
//! vectors. The static group and which beats are tones are the propagation's choice (spectral.rs).
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
//! Im⟨u_a|∇u_a⟩ and 2D = Σ_c [Σ_a Re⟨∇u_a|⊗|∇u_a⟩ − s_c² W_c b_c b_cᵀ] (model/photon.rs, # The absorbed light's
//! momentum: the one rule). The photon's wave number is the class's rate-weighted mean of the elements' k_eg, read on the
//! channels' photon forms (`Channel::kicks`, made at a jump's node alone): Σ_a Re⟨u_a|K_aψ⟩/W.
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
//! # What is not here
//! No landing transient, no decision, no representation per trajectory: the integrator's plain landing L_qψ/‖·‖ is the
//! model's (D5), the trajectory holds nothing of the model.
//!
//! # The propagation's interface (FIXSTEP)
//! What a fixed-model propagation needs of the model is a `Node` at a point and momentum: the generator's components by
//! exact beat with their values (`Selected::components`), H_eff's gradient's, and the photon classes. A propagation that
//! takes the model once a step (STEPBENCH L2) asks for the middle's alone; the onefold integrator asks for three.

use crate::linalg::{larger, largest, smaller};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::{Arc, RwLock};

use crate::field::Light;
use crate::floquet::frame::{exact_sum, FrameKey, Ratio};
use crate::floquet::{Frame, Hilbert};
use crate::linalg::{inner, norm_sqr, C64, Mat, V3};
use crate::model::rs::fold::{Grouping, Local, Out};
use crate::model::rs::{model_of, Making, ModelAt, Rs};
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
    /// The fold's grouping by the lab's frequencies (`Selected::at`), and the node's structure on it (`layout`).
    lab: Grouping,
    template: Node,
    tables: Tables,
    /// The run's own number (the thread's buffers and pooled nodes are its).
    id: usize,
    /// The beats the rounding rule kept and dropped (`prune`).
    kept: Vec<(usize, f64, f64, bool)>,
    beats: RwLock<Beats>,
}

/// The runs' numbers (`Selected::id`).
static RUNS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

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

/// A thread's buffers for the nodes of one run (`Selected::node`): the point, the fold's numbers, the contraction.
#[derive(Default)]
struct Work {
    point: crate::scene::Point,
    local: Local,
    out: Out,
    /// The jump components on the ground rows (`GROUND` × n each), and one product (n × n).
    jumps: Vec<C64>,
    product: Vec<C64>,
}

#[cfg(test)]
thread_local! {
    /// The nodes the middles made, without and with the photon forms (a check's count).
    pub(crate) static MADE: RefCell<[usize; 2]> = const { RefCell::new([0; 2]) };
}

thread_local! {
    /// The thread's buffers, for the run whose number it holds.
    static WORK: RefCell<Option<(usize, Work)>> = const { RefCell::new(None) };
    /// Nodes no middle holds any more, by their run's number: a node is made in one of them (`Selected::node`).
    static POOL: RefCell<Vec<(usize, Node)>> = const { RefCell::new(Vec::new()) };
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
/// counted whole; `Rs::at`'s own is not counted, its count not given).
#[derive(Clone, Debug)]
pub struct Node {
    pub kappa: V3,
    pub at: V3,
    pub generator: Vec<(usize, Mat)>,
    pub gradient: Vec<(usize, [Mat; 3])>,
    pub classes: Vec<Class>,
    pub kicked: bool,
    pub work: f64,
}

/// A candidate tone (`step::Middle::beats`): its beat ν > 0 and the beats of its members at e^{+iνu} and e^{−iνu}.
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
    /// The step's half-width h, as the step gives it (`Middle::nodes`): its outer nodes' offset, the chord of
    /// `gradient`'s slope. A jump's node at another offset does not change it.
    half: Cell<Option<f64>>,
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
        let cluster = Cluster { ground: GROUND, frames: vec![Beat::whole(vec![]); n], waves: vec![[0.0; 3]; n], waves_exact: vec![Beat::whole(vec![]); n], states: rs.keep.clone() };
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
        let spherical = rs.spherical();
        let spherical = Grouping::new(&spherical, &label(spherical.keys()));
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
            lab,
            template: Node { kappa: [0.0; 3], at: [0.0; 3], generator: vec![(0, Mat::zeros(n, n))], gradient: Vec::new(), classes: Vec::new(), kicked: false, work: 0.0 },
            tables: Tables::default(),
            id: RUNS.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            kept: Vec::new(),
            beats: RwLock::new(beats),
            rs,
            report,
            scene,
        };
        run.layout(&spherical);
        run.prune();
        Ok(run)
    }

    /// The selection's report (stderr): P by levels and state by state, the states left out, the selection's notes (a
    /// point not judged and why: `Selection::notes`), each candidate's errors at the gate's three points (centre, edge,
    /// middle: `Candidate::points`) -- "不判" at a point not judged, never its zero --, the time.
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
        let mut out = format!("三点选态：P = {} 态 {:?}，用时 {:.2} s
", self.rs.keep.len(), levels, self.report.seconds);
        out += &format!("  P 的态：{}
", names(&self.rs.keep));
        out += &format!("  删去 {} 态：{}
", self.rs.folded.len(), if self.rs.folded.is_empty() { "无".to_string() } else { names(&self.rs.folded) });
        for note in self.report.notes() {
            out += &format!("  注：{note}
");
        }
        let judged: Vec<bool> = self.report.reference.iter().map(Option::is_some).collect();
        for c in &self.report.candidates {
            let e = |p: &crate::select::PointError, at: usize| {
                if judged[at] {
                    format!("布居 {:.2e} R {:.2e} F {:.2e}", p.populations, p.rate, p.force)
                } else {
                    "不判".to_string()
                }
            };
            let [centre, edge, middle] = c.points();
            out += &format!("  候选 {} 态：中心 {}；边缘 {}；中间 {}{}
", c.keep.len(), e(centre, 0), e(edge, 1), e(middle, 2), if c.passes() { "（过）" } else { "" });
        }
        out
    }

    /// The model's error as the run books it (solve.rs `Booking`), measured at the gate's points on the chosen P
    /// (`Candidate::points`: the centre, the edge, the middle): on the populations the largest of its population errors
    /// (absolute), on T the largest of its rates' and forces' (relative) -- a rate or a force wrong by x moves T by x. A
    /// point not judged reads zero there (`Selection::notes`). No number where the selection holds no measurement of P.
    pub fn error(&self) -> [f64; 2] {
        self.report.candidates.last().filter(|c| c.keep == self.rs.keep).map_or([f64::NAN; 2], |c| {
            let points = c.points();
            [largest(0.0, points.iter().map(|p| p.populations)), largest(0.0, points.iter().flat_map(|p| [p.rate, p.force]))]
        })
    }

    /// The run's light (the region, the beams' geometries).
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

    /// A beat's value (rad/µs): n·(φ, Φ_tw, Φ_ν) + Σ_i p_i Φ_i, its own vector's, correctly rounded.
    fn value(&self, (n, p): &Exact) -> f64 {
        let fast = n.iter().zip(&self.frame.fast).filter(|(k, _)| **k != 0).map(|(&k, f)| k as f64 * f);
        exact_sum(fast.chain(p.iter().zip(&self.axes).filter(|(c, _)| c.num != 0).map(|(c, x)| c.num as f64 / c.den as f64 * x)))
    }

    /// The handle of a beat, made where first met.
    fn intern(&self, beats: &mut Beats, exact: Exact) -> usize {
        if let Some(&id) = beats.ids.get(&exact) {
            return id;
        }
        let id = beats.exact.len();
        beats.value.push(self.value(&exact));
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
    /// asks for.
    fn at(&self, r: V3, kappa: V3, making: Making) -> ModelAt {
        let point = self.scene.point_at(r, kappa);
        let mut local = Local::default();
        self.rs.evaluate(&self.scene, &point, &mut local);
        let mut out = Out::default();
        out.contract(self.rs.plan(), &self.lab, &local, None, making);
        model_of(self.rs.plan(), &self.lab, &out, &local.values.frequencies, making)
    }

    /// The node's structure (module documentation, # The nodes), made once from the fold's lab grouping: the beats of
    /// H_eff's and its gradient's structural elements, each output element's place, the photon classes with their
    /// channels' components by beat and their elements, and the decay's products between them.
    fn layout(&mut self, spherical: &Grouping) {
        let n = self.rs.keep.len();
        let lab = &self.lab;
        let mut template = Node { kappa: [0.0; 3], at: [0.0; 3], generator: vec![(0, Mat::zeros(n, n))], gradient: Vec::new(), classes: Vec::new(), kicked: false, work: 0.0 };
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
    }

    /// The node's beats by the rounding rule (coordinator 2026-11-20): a beat is the model's where its coefficients at
    /// unit amplitude -- the node at the trap's centre at rest, every beam's amplitude one, the tweezer's full -- reach
    /// ε times the largest of all its components' (H_eff's with the decay, θ left out; and the same of the gradient):
    /// below it, in value and in gradient, the beat is the rounding of the products and no part of the model (no
    /// candidate tone). The beats left and their two ratios are kept for the report (`Selected::beats_kept`).
    fn prune(&mut self) {
        let n = self.rs.keep.len();
        let node = self.node([0.0; 3], [0.0; 3], false);
        let mut largest = vec![0.0_f64; node.generator.len()];
        for (k, (_, m)) in node.generator.iter().enumerate() {
            for a in 0..n {
                for b in 0..n {
                    let x = if k == 0 && a == b { m[(a, b)] + C64::real(self.theta[a]) } else { m[(a, b)] };
                    largest[k] = larger(largest[k], x.abs());
                }
            }
        }
        let gradients: Vec<(usize, f64)> = node.gradient.iter().map(|(beat, g)| (*beat, g.iter().map(Mat::max_abs).fold(0.0, larger))).collect();
        let (top, gtop) = (largest.iter().copied().fold(0.0, larger), gradients.iter().map(|g| g.1).fold(0.0, larger));
        // A NaN coefficient would read every ratio as NaN and drop every beat but the zero one (a comparison with NaN is
        // false): the node at unit amplitude is numbers, or the run is refused here.
        assert!(top.is_finite() && gtop.is_finite(), "the node's coefficients at unit amplitude are not finite");
        let epsilon = 2.0 * crate::units::UNIT_ROUNDOFF;
        let mut keep = vec![true; node.generator.len()];
        self.kept = Vec::new();
        for (k, &(beat, _)) in node.generator.iter().enumerate() {
            let value = largest[k] / larger(top, f64::MIN_POSITIVE);
            let slope = gradients.iter().find(|g| g.0 == beat).map_or(0.0, |g| g.1 / larger(gtop, f64::MIN_POSITIVE));
            keep[k] = k == 0 || value >= epsilon || slope >= epsilon;
            self.kept.push((beat, value, slope, keep[k]));
        }
        // The slots left, and every place of the layout renumbered (a place at a beat dropped dropped with it).
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
            // A place at a beat dropped keeps its entry (the outputs' order), marked: nothing is written there.
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
        // The pooled nodes were made on the layout before.
        POOL.with(|p| p.borrow_mut().retain(|x| x.0 != self.id));
    }

    /// The node's beats with their coefficients' ratios at unit amplitude (`prune`): (beat, value's, gradient's,
    /// kept).
    pub fn beats_kept(&self) -> &[(usize, f64, f64, bool)] {
        &self.kept
    }

    /// The model at a point in the frame, by beats (`Node`; module documentation, # The generator by beats), with the
    /// channels' photon forms where `kicked` (a jump's node): the fold at the point (`Rs::evaluate`), its contraction by
    /// the lab's frequencies, put in the node's structure (`layout`), all in the thread's buffers.
    pub fn node(&self, at: V3, kappa: V3, kicked: bool) -> Node {
        let key = self.id;
        let mut node = POOL.with(|p| {
            let mut p = p.borrow_mut();
            p.iter().position(|x| x.0 == key).map(|i| p.swap_remove(i).1)
        }).unwrap_or_else(|| self.template.clone());
        WORK.with(|w| {
            let mut w = w.borrow_mut();
            if w.as_ref().is_none_or(|x| x.0 != key) {
                *w = Some((key, Work::default()));
            }
            let work = &mut w.as_mut().expect("the thread's buffers").1;
            self.scene.point_into(at, kappa, &mut work.point);
            self.rs.evaluate(&self.scene, &work.point, &mut work.local);
            work.out.contract(self.rs.plan(), &self.lab, &work.local, None, Making { model: true, kicks: kicked, lift: false });
            self.fill(&mut node, &work.out, kicked, &mut work.jumps, &mut work.product);
            node.work = crate::decide::COMPLEX * (work.local.work + work.out.work);
        });
        node.kappa = kappa;
        node.at = at;
        node.kicked = kicked;
        node
    }

    /// A node's numbers from a contraction by the lab's frequencies (`layout`'s places).
    fn fill(&self, node: &mut Node, out: &Out, kicked: bool, jumps: &mut Vec<C64>, product: &mut Vec<C64>) {
        let n = self.rs.keep.len();
        for (_, m) in node.generator.iter_mut() {
            m.data.fill(C64::ZERO);
        }
        for (_, g) in node.gradient.iter_mut() {
            for m in g.iter_mut() {
                m.data.fill(C64::ZERO);
            }
        }
        for c in node.classes.iter_mut() {
            for axis in c.axes.iter_mut() {
                for j in axis.iter_mut() {
                    for e in j.elements.iter_mut() {
                        (e.value, e.gradient, e.kick) = (C64::ZERO, [C64::ZERO; 3], C64::ZERO);
                    }
                }
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
        let half = C64::new(0.0, -0.5);
        for d in &self.tables.decay {
            let (cl, cr) = (&self.tables.jumps[d.left].cols, &self.tables.jumps[d.right].cols);
            let (sl, sr) = (self.tables.starts[d.left], self.tables.starts[d.right]);
            product.clear();
            product.resize(cl.len() * cr.len(), C64::ZERO);
            crate::linalg::adjoint_product(&jumps[sl..sl + GROUND * cl.len()], &jumps[sr..sr + GROUND * cr.len()], (cl.len(), GROUND, cr.len()), product);
            {
                let forward = &mut node.generator[d.forward].1.data;
                for (i, &a) in cl.iter().enumerate() {
                    for (j, &b) in cr.iter().enumerate() {
                        forward[a * n + b] += half * product[i * cr.len() + j];
                    }
                }
            }
            if d.left != d.right {
                let back = &mut node.generator[d.back].1.data;
                for (i, &a) in cl.iter().enumerate() {
                    for (j, &b) in cr.iter().enumerate() {
                        back[b * n + a] += half * product[i * cr.len() + j].conj();
                    }
                }
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
    /// beat's phase; over their sum.
    fn rows(&self, at: &Point, psi: &[C64]) -> [f64; STATES] {
        let model = self.at(at.r, at.kappa, Making { model: false, kicks: false, lift: true });
        let n = model.states.len();
        let mut rho = vec![C64::ZERO; n * n];
        crate::step::density(psi, &mut rho);
        let mut out = [0.0; STATES];
        for (a, &s) in model.states.iter().enumerate() {
            out[s] = rho[a * n + a].re;
        }
        let q = model.folded.len();
        let mut by_fast: Vec<([i64; 3], Mat)> = Vec::new();
        for part in &model.lift {
            let table = self.elements(&part.key, Of::Lift);
            let beats: Vec<([i64; 3], f64)> = self.read(|b| table.iter().map(|&i| (b.exact[i].0, b.value[i])).collect());
            for (x, &(fast, value)) in beats.iter().enumerate() {
                let z = part.value.data[x];
                if z == C64::ZERO {
                    continue;
                }
                // Within a fast index the slow beat's phase at the time (the fast part's is the index's, common).
                let slow = value - self.value(&(fast, vec![]));
                let i = match by_fast.iter().position(|f| f.0 == fast) {
                    Some(i) => i,
                    None => {
                        by_fast.push((fast, Mat::zeros(q, n)));
                        by_fast.len() - 1
                    }
                };
                by_fast[i].1.data[x] += z * C64::cis(-slow * at.t);
            }
        }
        for (_, m) in by_fast {
            let lifted = crate::model::rs::lifted(&model.states, &model.folded, &m, &rho);
            for &s in &model.folded {
                out[s] += lifted[s];
            }
        }
        let total: f64 = out.iter().sum();
        out.map(|x| x / total)
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
        let key = self.model.id;
        let nodes = std::mem::take(&mut *self.nodes.borrow_mut());
        POOL.with(|p| {
            let mut p = p.borrow_mut();
            for node in nodes {
                if let Ok(node) = Rc::try_unwrap(node) {
                    p.push((key, node));
                }
            }
        });
    }
}

impl SelectedMiddle<'_> {
    /// The node at the time s from the middle along the path, made where first asked for.
    fn node(&self, path: &Path, s: f64, kicked: bool) -> Rc<Node> {
        let at: V3 = std::array::from_fn(|i| self.r[i] + path.velocity[i] * s);
        let made = self.nodes.borrow().iter().position(|x| x.kappa == path.kappa && x.at == at);
        if let Some(i) = made {
            let node = self.nodes.borrow()[i].clone();
            if node.kicked || !kicked {
                return node;
            }
            // A jump at a node made without the photon forms: the node again with them, in its place, its work counted
            // once.
            let node = Rc::new(self.model.node(at, path.kappa, true));
            self.nodes.borrow_mut()[i] = node.clone();
            return node;
        }
        let node = Rc::new(self.model.node(at, path.kappa, kicked));
        self.made.set(self.made.get() + node.work);
        #[cfg(test)]
        MADE.with(|m| m.borrow_mut()[usize::from(kicked)] += 1);
        self.nodes.borrow_mut().push(node.clone());
        node
    }

    /// e^{−iΩ (t_m + s)} of a beat.
    fn phase(&self, values: &[f64], beat: usize, s: f64) -> C64 {
        if beat == 0 {
            C64::ONE
        } else {
            C64::cis(-values[beat] * (self.t + s))
        }
    }

    /// The candidates of the path (module documentation, # The generator by beats): the nonzero beats of the nodes made
    /// at its momentum, each with its opposite, in the order of their exact vectors.
    fn candidates(&self, path: &Path) -> Vec<Candidate> {
        let mut met: Vec<usize> = Vec::new();
        for node in self.nodes.borrow().iter().filter(|x| x.kappa == path.kappa) {
            for &(beat, _) in &node.generator {
                if beat != 0 && !met.contains(&beat) {
                    met.push(beat);
                }
            }
        }
        self.model.read(|b| {
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
            out.into_iter().map(|c| c.1).collect()
        })
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
        let node = self.node(path, s, true);
        let photons = self.model.read(|b| photons(&node, &b.value, self.t + s, psi));
        let rates: Vec<f64> = photons.iter().map(|p| p.moments.weight).collect();
        let k = draw(&rates)?;
        let p = &photons[k];
        let n = psi.len();
        let amplitudes = p.u.each_ref().map(|u| u.iter().copied().collect::<Internal>());
        let jumps: usize = node.classes.iter().map(|c| c.axes.iter().map(Vec::len).sum::<usize>()).sum();
        let wave_number = p.wave_number.expect("a jump's node holds the photon forms");
        Some(Emitted { landing: 0, wave_number, amplitudes, kick: p.moments.kick, work: crate::decide::COMPLEX * (5 * jumps * n * n) as f64 })
    }
}

/// A class's photons of a state (module documentation, # The photons): u_a, its moments over ‖ψ‖² (model.rs `Moments`,
/// the one rule: the rate W, Σ Im⟨u|∇u⟩, Σ Re⟨∇u|⊗|∇u⟩, the share s of b a jump carries and s b), and the photon's wave
/// number where the node holds the photon forms (`Node::kicked`).
struct Photons {
    u: [Vec<C64>; 3],
    moments: Moments,
    wave_number: Option<f64>,
}

/// Every class's photons of the state ψ at the time t on a node (module documentation, # The photons).
fn photons(node: &Node, values: &[f64], t: f64, psi: &[C64]) -> Vec<Photons> {
    let n = psi.len();
    let norm = norm_sqr(psi);
    node.classes
        .iter()
        .map(|class| {
            let mut u: [Vec<C64>; 3] = std::array::from_fn(|_| vec![C64::ZERO; n]);
            let mut du: [[Vec<C64>; 3]; 3] = std::array::from_fn(|_| std::array::from_fn(|_| vec![C64::ZERO; n]));
            let mut ku: [Vec<C64>; 3] = std::array::from_fn(|_| vec![C64::ZERO; n]);
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
                    if node.kicked {
                        apply(&|e| e.kick, &mut ku[a]);
                    }
                    for (i, d) in du.iter_mut().enumerate() {
                        apply(&|e| e.gradient[i], &mut d[a]);
                    }
                }
            }
            let moments = Moments::of(&u, &du, class.absorbed);
            let weight = moments.weight;
            let wave_number = node.kicked.then(|| if weight > 0.0 { (0..3).map(|a| inner(&u[a], &ku[a]).re).sum::<f64>() / weight } else { 0.0 });
            Photons { u, moments: moments.over(norm), wave_number }
        })
        .collect()
}

#[cfg(test)]
impl SelectedMiddle<'_> {
    /// Every class's photon moments of the state ψ at the middle (model.rs `Moments`, over ‖ψ‖²; the tests').
    pub(crate) fn moments(&self, path: &Path, psi: &[C64]) -> Vec<Moments> {
        let node = self.node(path, 0.0, false);
        self.model.read(|b| photons(&node, &b.value, self.t, psi)).into_iter().map(|p| p.moments).collect()
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

    fn path(&self, kappa: [f64; 3], velocity: [f64; 3]) -> Path {
        Path { kappa, velocity }
    }

    fn gauge(&self, _: &Path, rates: &mut [f64]) {
        rates.fill(0.0);
    }

    /// Ĥ(s) = Σ_k G_k(s) e^{−iΩ_k(t_m + s)} (module documentation, # The generator by beats).
    fn generator(&self, path: &Path, s: f64, h: &mut Mat) {
        let node = self.node(path, s, false);
        h.data.fill(C64::ZERO);
        self.model.read(|b| {
            for (beat, g) in &node.generator {
                let f = self.phase(&b.value, *beat, s);
                for (z, x) in h.data.iter_mut().zip(&g.data) {
                    *z += f * *x;
                }
            }
        });
    }

    fn beats(&self, path: &Path, beats: &mut Vec<f64>, keys: &mut Vec<Vec<i64>>) {
        let candidates = self.candidates(path);
        let count = candidates.len();
        for (m, c) in candidates.iter().enumerate() {
            beats.push(c.nu);
            keys.push((0..count).map(|i| i64::from(i == m)).collect());
        }
    }

    /// Candidate m's members at s: G_plus(s) e^{iν t_m} and G_minus(s) e^{−iν t_m} (their parts of `generator`).
    fn envelopes(&self, path: &Path, m: usize, s: f64, plus: &mut Mat, minus: &mut Mat) {
        let c = &self.candidates(path)[m];
        let node = self.node(path, s, false);
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
            }
        });
    }

    /// Σ_j Re tr(Ĝ⁺(s_j) σ_j + Ĝ⁻(s_j) σ_j†) of candidate m's members' H_eff gradients at the phases of `envelopes`.
    fn tone_force(&self, path: &Path, m: usize, sums: &[(f64, &[C64], &[C64])]) -> [f64; 3] {
        let c = &self.candidates(path)[m];
        let n = self.states();
        let mut out = [0.0; 3];
        for &(s, re, im) in sums {
            let node = self.node(path, s, false);
            let sigma: Vec<C64> = re.iter().zip(im).map(|(r, i)| *r + C64::I * *i).collect();
            let dagger: Vec<C64> = (0..n * n).map(|x| sigma[(x % n) * n + x / n].conj()).collect();
            let forms = self.model.read(|b| {
                let mut f = [C64::ZERO; 3];
                for (member, rho) in [(c.plus, &sigma), (c.minus, &dagger)] {
                    let Some(k) = member else { continue };
                    let phase = self.phase(&b.value, k, 0.0);
                    let x = SelectedMiddle::contract(&node, |beat| (beat == k).then_some(phase), rho);
                    for a in 0..3 {
                        f[a] += x[a];
                    }
                }
                f
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

    /// Re tr(Ĝ(s) ρ): H_eff's gradient at the time s, every beat at its phase.
    fn force(&self, path: &Path, s: f64, rho: &[C64]) -> [f64; 3] {
        let node = self.node(path, s, false);
        let f = self.model.read(|b| SelectedMiddle::contract(&node, |beat| Some(self.phase(&b.value, beat, s)), rho));
        f.map(|z| z.re)
    }

    /// The gradient at the middle by its beats: Ω_m = −Ω_k with Re tr(∂H_k e^{−iΩ_k t_m} ρ).
    fn tones(&self, path: &Path, rho: &[C64], each: &mut dyn FnMut(f64, [f64; 3])) {
        let node = self.node(path, 0.0, false);
        let values = self.model.read(|b| b.value.clone());
        for (beat, _) in &node.gradient {
            let f = SelectedMiddle::contract(&node, |k| (k == *beat).then(|| self.phase(&values, k, 0.0)), rho);
            each(if *beat == 0 { 0.0 } else { -values[*beat] }, f.map(|z| z.re));
        }
    }

    /// The light the state scatters at the middle beside the Ehrenfest force and the jumps (module documentation,
    /// # The photons).
    fn scattering(&self, path: &Path, psi: &[C64]) -> (Scattering, f64) {
        let node = self.node(path, 0.0, false);
        let photons = self.model.read(|b| photons(&node, &b.value, self.t, psi));
        let n = self.states();
        (Scattering::of_moments(&photons.iter().map(|p| p.moments).collect::<Vec<_>>()), crate::decide::COMPLEX * (12 * n * n * node.classes.len()) as f64)
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

    /// Ĝ_a(s) as matrices, and where asked their slope along the path at the middle: the envelopes' chord through the
    /// step's outer nodes and each beat's phase's derivative −iΩ.
    fn gradient(&self, path: &Path, s: f64, g: &mut [Vec<C64>; 3], slope: Option<&mut [Vec<C64>; 3]>) -> f64 {
        assert!(s == 0.0 || slope.is_none(), "the gradient's slope at the middle (the step's optical sector's)");
        let n = self.states();
        let values = self.model.read(|b| b.value.clone());
        let at = |node: &Node, s: f64, turned: bool, out: &mut [Vec<C64>; 3]| {
            for (a, o) in out.iter_mut().enumerate() {
                o.clear();
                o.resize(n * n, C64::ZERO);
                for (beat, d) in &node.gradient {
                    let mut f = self.phase(&values, *beat, s);
                    if turned {
                        f = f * C64::new(0.0, if *beat == 0 { 0.0 } else { -values[*beat] });
                    }
                    for (z, x) in o.iter_mut().zip(&d[a].data) {
                        *z += f * *x;
                    }
                }
            }
        };
        let node = self.node(path, s, false);
        at(&node, s, false, g);
        let mut counted = 3.0 * node.gradient.len() as f64 * (n * n) as f64;
        if let Some(slope) = slope {
            let h = self.half.get().expect("the step's outer nodes: its picture's ends");
            let (before, after) = (self.node(path, -h, false), self.node(path, h, false));
            let mut minus: [Vec<C64>; 3] = Default::default();
            // The envelopes' chord at the middle's phases, and the phases' turn on the middle's envelopes.
            at(&before, 0.0, false, &mut minus);
            at(&after, 0.0, false, slope);
            for (p, m) in slope.iter_mut().zip(&minus) {
                p.iter_mut().zip(m).for_each(|(p, m)| *p = (*p - *m) / (2.0 * h));
            }
            let mut turned: [Vec<C64>; 3] = Default::default();
            at(&node, 0.0, true, &mut turned);
            for (o, t) in slope.iter_mut().zip(&turned) {
                o.iter_mut().zip(t).for_each(|(o, t)| *o += *t);
            }
            counted *= 4.0;
        }
        counted
    }

    /// P's explicit excited states' optical sector (step.rs, # The optical sector's noise): its ground states, γ_Q, and
    /// the light pattern's fastest beat along the path over the non-tweezer beams.
    /// The step's three nodes' forms (`step::Middle::nodes`), the step's half-width h kept for `gradient`'s slope.
    fn nodes(&self, path: &Path, h: f64, rho: [&[C64]; 3], held: Option<(&[C64], &dyn Fn(f64) -> f64)>, end: Option<(&[C64], &Path)>) -> ([[f64; 3]; 3], Option<crate::step::Forms>) {
        self.half.set(Some(h));
        let forms = std::array::from_fn(|j| match held {
            Some((held, weight)) if j == 1 => self.centre(path, held, weight, rho[j]),
            _ => self.force(path, [-h, 0.0, h][j], rho[j]),
        });
        (forms, end.map(|(rho, fixed)| crate::step::forms_at(self, fixed, rho)))
    }

    fn optical(&self, path: &Path) -> Option<Optical> {
        let decay = self.model.optical?;
        let (most, least) = self.model.waves.iter().map(|k| (0..3).map(|a| k[a] * path.velocity[a]).sum::<f64>()).fold((f64::NEG_INFINITY, f64::INFINITY), |(most, least), x| (larger(most, x), smaller(least, x)));
        Some(Optical { ground: GROUND, decay, beat: if most >= least { most - least } else { 0.0 } })
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
        SelectedMiddle { model: self, r, t, nodes: RefCell::new(Vec::new()), half: Cell::new(None), made: Cell::new(0.0) }
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

    /// Outside the trap's region with the loss energy above its far value, zero.
    fn lost(&self, cluster: usize, at: &Point, psi: &[C64]) -> bool {
        self.light.outside(at.r) && self.energy(cluster, at, psi) > 0.0
    }

    fn readout(&self, _cluster: usize, at: &Point, psi: &[C64]) -> [f64; STATES] {
        self.rows(at, psi)
    }
}

#[cfg(test)]
mod tests;
