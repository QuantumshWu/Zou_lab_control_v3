//! The trajectory's model (RUSTPLAN §3.2 and its corrections of 2026-11-19, "模型只定一次", "一个折叠公式"): the P the
//! two-point check selected (select.rs) and its RS fold (model/rs.rs `Rs`) as the onefold integrator's
//! `step::Dynamics` / `step::Middle`. Nothing here decides or folds: P, the fold's formula and its coefficients are the
//! checkpoint's, `Rs::at` gives the model at a point, and this module puts it in the integrator's frame and terms.
//!
//! # One cluster, one frame
//! The trajectory's states are P, all of them one cluster (the ground states first: `Rs::keep` is sorted), in the frame
//! θ of floquet/frame.rs chosen at the checkpoint's centre at rest: ψ_a = e^{−iθ_a t} x_a with θ_a an exact vector over
//! the generators (the beams, the carrier) and α. No Doppler frame (K_s = 0, as onefold): at the step's outer nodes the
//! model is taken at the path's own points, so the beams' plane waves turn with the path there, and its time phases
//! are the lab's, ω_c = Σ_g c_g ω_g of a component's key c. An element a ← b of a component of key c then holds
//! e^{−iΩt}, Ω = (c − θ_a + θ_b)·(ω, α) (floquet/assemble.rs `to_frame`, the same vector), and x_a holds −θ_a on the
//! diagonal. The fold keeps its slow elements alone, so every Ω is the slow remainder of that vector
//! (`Frame::split`): a beat of the lights against the frame -- a tilt's, a lattice's of two frequencies --, exact.
//!
//! # The generator by beats
//! Each node's model is grouped by its exact beats (`Beats`): Ĥ(t) = Σ_k G_k e^{−iΩ_k t}, G_k the H_eff elements of the
//! beat k with −θ on the static one's diagonal, and the decay −(i/2) Σ_ch L_ch(t)†L_ch(t) of the jump channels as they
//! are in the frame: a channel's elements at their slow beats -- its common fast index (n_φ, n_tw) is a phase of the
//! channel as a whole, its photon's frequency, no part of |L ψ|² --, each pair of its beats at their difference. So
//! i(Ĥ − Ĥ†) = Σ L†L at every time by construction, the photons' operators the decay's. The integrator's candidate tones
//! (`step::Middle::beats`) are the nonzero beats with their opposites, each pair one candidate at ν = |Ω|, its key a
//! unit vector over the middle's candidates (no two candidates share a beat, so a sum of two keys is zero iff the
//! members are a candidate's two: the one relation the propagation's second order reads), in the order of their exact
//! vectors. The static group and which beats are tones are the propagation's choice (spectral.rs).
//!
//! # The nodes
//! A middle evaluates `Rs::at` where the step asks for the model -- its middle and its two outer nodes, at the path's
//! momentum κ_m (the fold's denominators at their Doppler frequencies, D2) --, once each, and every form of the step
//! is read off those three (`Node`). A jump's photon is the model at the jump's own place and time.
//!
//! # The photons
//! A class is a fast index of the channels (n_φ, n_tw): its photons land on P's ground states, its three Cartesian
//! axes the polarization's amplitudes u_a = L_(a,n)(t) ψ. A jump of the class carries its conditional mean b =
//! Σ_a Im⟨u_a|∇u_a⟩/W of the absorbed momentum as far as one absorbed photon's goes, s b with s = min(1, k_c/|b|), k_c
//! the largest wave number of the beams its channels' keys hold; the step carries the rest, Σ_c (1 − s_c) Σ_a
//! Im⟨u_a|∇u_a⟩ and 2D = Σ_c [Σ_a Re⟨∇u_a|⊗|∇u_a⟩ − s_c² W_c b_c b_cᵀ] (model/photon.rs, # The absorbed light's
//! momentum: the one rule). The photon's wave number is the class's rate-weighted mean of the elements' k_eg, read on the
//! channels' photon forms (`Channel::kicks`): Σ_a Re⟨u_a|K_aψ⟩/W.
//!
//! # The rest of the model
//! The splitting's reference, the prepared wells, the loss energy and the reported trap are one potential
//! (`Scene::trap_potential`, RUSTPLAN §2.3): per ground state the tweezer's intensity times its unit fold's ground shift
//! and its own outside form's, weighed by ground populations -- the reference state's, an initial component's, or the
//! state's own (an explicit excited state's by its decay's branching). The unit fold's ground shift is the one fold's
//! (`Tweezer::reference` of the tweezer alone at unit amplitude). The displayed rows are P's populations with Q's lifted
//! through M_c (`ModelAt::lift`), each fast index of the lift apart (its terms do not interfere over time), over their
//! sum.
//!
//! # What is not here
//! No landing transient, no decision, no representation per trajectory: the integrator's plain landing L_qψ/‖·‖ is the
//! model's (D5), the trajectory holds nothing of the model.
//!
//! # The propagation's interface (FIXSTEP)
//! What a fixed-model propagation needs of the model is a `Node` at a point and momentum: the generator's components by
//! exact beat with their values (`Selected::components`), H_eff's gradient's, and the photon classes. A propagation that
//! takes the model once a step (STEPBENCH L2) asks for the middle's alone; the onefold integrator asks for three.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::{Arc, RwLock};

use crate::field::Light;
use crate::floquet::frame::{exact_sum, FrameKey, Ratio};
use crate::floquet::{Frame, Hilbert};
use crate::linalg::{inner, norm_sqr, C64, Mat, V3};
use crate::model::rs::{ModelAt, ModelPart, Rs, Tweezer};
use crate::model::{Beat, Cluster, Point, Quadratic, Scattering};
use crate::request::{GROUND, STATES};
use crate::scene::{Component, Components, Key, Scene};
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
    /// The request's frequency axes Φ_i (rad/µs): the primitives of the frame at rest, a beat's value Σ p_i Φ_i.
    axes: Vec<f64>,
    /// θ_a of P's states (rad/µs).
    theta: Vec<f64>,
    cluster: Cluster,
    light: Light,
    hbar_over_mass: f64,
    /// The tweezer's unit fold's ground shift (rad/µs; `Scene::trap_potential`'s depth).
    depth: [f64; GROUND],
    /// The reference state's ground populations (V_ref's weights).
    reference: [f64; GROUND],
    /// Per state of P its decay's branching onto the ground states (a ground state's its own).
    branching: Vec<[f64; GROUND]>,
    /// γ_Q, the least decay over two of P's explicit excited states; None where P holds none.
    optical: Option<f64>,
    /// The non-tweezer beams' wave vectors (the light pattern's beats along a path).
    waves: Vec<V3>,
    beats: RwLock<Beats>,
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
}

/// Which elements' beats (`Selected::elements`): H_eff's (P × P), the jumps' slow remainders (P × P), the lift's (Q × P).
#[derive(Clone, Copy, PartialEq)]
enum Of {
    Generator,
    Jump,
    Lift,
}

/// One jump component of a class's axis: its beat, L, ∇L and the photon form K = k∘L.
pub struct Jump {
    pub beat: usize,
    pub value: Mat,
    pub gradient: [Mat; 3],
    pub kick: Mat,
}

/// A class of photons (module documentation, # The photons): its fast index, k_c, and per Cartesian axis its
/// components.
pub struct Class {
    pub fast: [i64; 2],
    pub absorbed: f64,
    pub axes: [Vec<Jump>; 3],
}

/// The model at one point of a path in the frame (module documentation, # The nodes): the generator's components by
/// beat, H_eff's gradient's, the photon classes; the point and momentum it was made at; its counted work (real
/// multiply-adds of putting the model in the frame; `Rs::at`'s own is not counted, its count not given).
pub struct Node {
    pub kappa: V3,
    pub at: V3,
    pub generator: Vec<(usize, Mat)>,
    pub gradient: Vec<(usize, [Mat; 3])>,
    pub classes: Vec<Class>,
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
    /// The outer nodes' offset last made: the step's half (its picture's ends; `gradient`'s slope).
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
        let zero = vec![0.0; scene.request.beams.len()];
        let centre = scene.components([0.0; 3], &zero, 1.0);
        let frame = Frame::new(&scene, &Hilbert::full(&scene, &centre), &zero);
        let rs = Rs::new(&scene, &report.keep, frame.sectors());
        let request = &scene.request;
        if rs.keep.iter().filter(|&&s| s < GROUND).count() != GROUND {
            return Err("选出的 P 必须含全部基态（跳跃落到任一基态）".into());
        }
        let axes = request.frequencies_mhz.iter().map(|f| RAD_PER_US_PER_MHZ * f).collect();
        let theta = rs.keep.iter().map(|&s| frame.theta[s].iter().zip(&frame.values).map(|(&k, w)| k as f64 * w).sum()).collect();
        let n = rs.keep.len();
        let cluster = Cluster { ground: GROUND, frames: vec![Beat::whole(vec![]); n], waves: vec![[0.0; 3]; n], waves_exact: vec![Beat::whole(vec![]); n], states: rs.keep.clone() };
        // The tweezer's unit fold alone: its static part's ground diagonal (`Scene::trap_potential`'s depth).
        let length = scene.generators.len();
        let unit = Components {
            parts: vec![Component { key: vec![0; length], frequency: 0.0, value: Mat::zeros(STATES, STATES), gradient: zeros(STATES, STATES) }],
            tweezer: (C64::ONE, [C64::ZERO; 3]),
        };
        let folded = Tweezer::new(&scene).reference(&scene, &unit).0;
        let depth = std::array::from_fn(|g| folded.still().value[(g, g)].re);
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
        let optical = rs.keep.iter().filter(|&&s| s >= GROUND).map(|&e| 0.5 * (0..GROUND).map(|g| decay(e, g)).sum::<f64>()).reduce(f64::min);
        let waves = scene.geometry.iter().enumerate().filter(|(l, _)| *l != scene.tweezer()).map(|(_, g)| g.wave_vector()).collect();
        let mut beats = Beats::default();
        let still: Exact = ([0; 3], vec![Ratio::ZERO; request.frequencies_mhz.len()]);
        beats.ids.insert(still.clone(), 0);
        beats.exact.push(still);
        beats.value.push(0.0);
        Ok(Selected {
            light: Light::new(request),
            hbar_over_mass: hbar_over_mass(request.mass_kg),
            frame,
            axes,
            theta,
            cluster,
            depth,
            reference,
            branching,
            optical,
            waves,
            beats: RwLock::new(beats),
            rs,
            report,
            scene,
        })
    }

    /// The selection's report (stderr): P by levels, each candidate's errors at the two points, the time.
    pub fn report(&self) -> String {
        let labels = &self.scene.request.states;
        let mut levels: Vec<String> = Vec::new();
        for &s in &self.rs.keep {
            let label = labels[s].rsplit_once('-').map_or(labels[s].as_str(), |(level, _)| level).to_string();
            if !levels.contains(&label) {
                levels.push(label);
            }
        }
        let mut out = format!("两点选态：P = {} 态 {:?}，用时 {:.2} s\n", self.rs.keep.len(), levels, self.report.seconds);
        for c in &self.report.candidates {
            let e = |p: &crate::select::PointError| format!("布居 {:.2e} R {:.2e} F {:.2e}", p.populations, p.rate, p.force);
            out += &format!("  候选 {} 态：中心 {}；边缘 {}{}\n", c.keep.len(), e(&c.centre), e(&c.edge), if c.passes() { "（过）" } else { "" });
        }
        out
    }

    /// The model's error as the run books it (solve.rs `Booking`), measured at the two points on the chosen P: on the
    /// populations the larger of its population errors (absolute), on T the largest of its rate's and force's
    /// (relative) -- a rate or a force wrong by x moves T by x. No number where the selection holds no measurement of P.
    pub fn error(&self) -> [f64; 2] {
        self.report.candidates.last().filter(|c| c.keep == self.rs.keep).map_or([f64::NAN; 2], |c| {
            [c.centre.populations.max(c.edge.populations), c.centre.rate.max(c.centre.force).max(c.edge.rate).max(c.edge.force)]
        })
    }

    /// The run's light (the region, the beams' geometries).
    pub fn light(&self) -> &Light {
        &self.light
    }

    /// The prepared potential of the ground populations p (module documentation, # The rest of the model).
    pub fn prepared(&self, p: &[f64; GROUND]) -> Potential<'_> {
        Potential { model: self, weights: *p }
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

    /// What `then` reads of the beats as they stand.
    fn read<R>(&self, then: impl FnOnce(&Beats) -> R) -> R {
        then(&self.beats.read().unwrap_or_else(|e| e.into_inner()))
    }

    /// The beats' values (rad/µs; a beat's handle its place) and exact vectors as they stand.
    pub fn components(&self) -> (Vec<f64>, Vec<Exact>) {
        self.read(|b| (b.value.clone(), b.exact.clone()))
    }

    /// The model at a point for the momentum κ (`Rs::at`).
    fn at(&self, r: V3, kappa: V3) -> ModelAt {
        self.rs.at(&self.scene, &self.scene.components_at(r, kappa))
    }

    /// The model at a point in the frame, by beats (`Node`; module documentation, # The generator by beats).
    pub fn node(&self, at: V3, kappa: V3) -> Node {
        let model = self.at(at, kappa);
        let n = model.states.len();
        let mut work = 0;
        let mut statics = Mat::zeros(n, n);
        for (a, t) in self.theta.iter().enumerate() {
            statics[(a, a)] = C64::real(-t);
        }
        let mut generator: Vec<(usize, Mat)> = vec![(0, statics)];
        let mut gradient: Vec<(usize, [Mat; 3])> = Vec::new();
        for part in &model.effective {
            let table = self.elements(&part.key, Of::Generator);
            for (x, &beat) in table.iter().enumerate() {
                let (z, dz) = (part.value.data[x], part.gradient.each_ref().map(|d| d.data[x]));
                if z != C64::ZERO {
                    let i = slot(&mut generator, beat, || Mat::zeros(n, n));
                    generator[i].1.data[x] += z;
                }
                if dz.iter().any(|d| *d != C64::ZERO) {
                    let i = slot(&mut gradient, beat, || zeros(n, n));
                    for (g, d) in gradient[i].1.iter_mut().zip(dz) {
                        g.data[x] += d;
                    }
                }
            }
            work += 4 * n * n;
        }
        // The photon classes: each channel's components by their slow beats, on its axis of its class.
        let mut classes: Vec<Class> = Vec::new();
        for channel in &model.jumps {
            let place = match classes.iter().position(|c| c.fast == channel.fast) {
                Some(i) => i,
                None => {
                    classes.push(Class { fast: channel.fast, absorbed: 0.0, axes: Default::default() });
                    classes.len() - 1
                }
            };
            let class = &mut classes[place];
            for part in &channel.parts {
                let kick = channel.kicks.iter().find(|k| k.key == part.key);
                let beams = part.key.iter().enumerate().filter(|(g, k)| **k != 0 && *g < self.scene.geometry.len()).map(|(g, _)| self.scene.geometry[g].wave_number);
                class.absorbed = beams.fold(class.absorbed, f64::max);
                let table = self.elements(&part.key, Of::Jump);
                // The part's elements by their beats (a part's elements may beat apart: two beams' frequencies).
                let axis = &mut class.axes[channel.emission];
                for (x, &beat) in table.iter().enumerate() {
                    let (z, dz, k) = (part.value.data[x], part.gradient.each_ref().map(|d| d.data[x]), kick.map_or(C64::ZERO, |k| k.value.data[x]));
                    if z == C64::ZERO && k == C64::ZERO && dz.iter().all(|d| *d == C64::ZERO) {
                        continue;
                    }
                    let i = match axis.iter().position(|j| j.beat == beat) {
                        Some(i) => i,
                        None => {
                            axis.push(Jump { beat, value: Mat::zeros(n, n), gradient: zeros(n, n), kick: Mat::zeros(n, n) });
                            axis.len() - 1
                        }
                    };
                    let jump = &mut axis[i];
                    jump.value.data[x] += z;
                    jump.kick.data[x] += k;
                    for (g, d) in jump.gradient.iter_mut().zip(dz) {
                        g.data[x] += d;
                    }
                }
                work += 5 * n * n;
            }
        }
        // The decay −(i/2) Σ_ch L(t)†L(t) by beats: L_k'†L_k e^{−i(Ω_k − Ω_k')t} at the beat k − k'.
        for class in &classes {
            for axis in &class.axes {
                for left in axis {
                    for right in axis {
                        let beat = if left.beat == right.beat {
                            0
                        } else {
                            let difference = self.read(|b| minus(&b.exact[right.beat].1, &b.exact[left.beat].1));
                            let mut beats = self.beats.write().unwrap_or_else(|e| e.into_inner());
                            self.intern(&mut beats, ([0; 3], difference))
                        };
                        let i = slot(&mut generator, beat, || Mat::zeros(n, n));
                        generator[i].1.add_scaled(C64::new(0.0, -0.5), &left.value.adjoint_product(&right.value));
                        work += n * n * n;
                    }
                }
            }
        }
        Node { kappa, at, generator, gradient, classes, work: crate::decide::COMPLEX * work as f64 }
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
        let model = self.at(at.r, at.kappa);
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
            let one = ModelAt {
                states: model.states.clone(),
                folded: model.folded.clone(),
                groups: model.groups.clone(),
                effective: vec![],
                jumps: vec![],
                lift: vec![ModelPart { key: vec![], frequency: 0.0, value: m, gradient: zeros(q, n) }],
            };
            let lifted = one.lift(&rho);
            for &s in &model.folded {
                out[s] += lifted[s];
            }
        }
        let total: f64 = out.iter().sum();
        out.map(|x| x / total)
    }
}

/// A potential of the run (module documentation, # The rest of the model): Σ_g p_g V_g(r) of `Scene::trap_potential`.
pub struct Potential<'a> {
    model: &'a Selected,
    weights: [f64; GROUND],
}

impl Potential<'_> {
    /// Its second-order expansion at r.
    pub fn at(&self, r: V3) -> Quadratic {
        self.rounded(r).0
    }

    /// Its expansion at r with the rounding of its value: `rounding` of its ground terms times their moduli
    /// (model/potential.rs `descend`).
    pub fn rounded(&self, r: V3) -> (Quadratic, f64) {
        let wells = self.model.scene.trap_potential(r, &self.model.depth);
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

impl SelectedMiddle<'_> {
    /// The node at the time s from the middle along the path, made where first asked for.
    fn node(&self, path: &Path, s: f64) -> Rc<Node> {
        let at: V3 = std::array::from_fn(|i| self.r[i] + path.velocity[i] * s);
        if let Some(node) = self.nodes.borrow().iter().find(|x| x.kappa == path.kappa && x.at == at) {
            return node.clone();
        }
        let node = Rc::new(self.model.node(at, path.kappa));
        self.made.set(self.made.get() + node.work);
        if s != 0.0 {
            self.half.set(Some(s.abs()));
        }
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
        let node = self.node(path, s);
        let photons = self.model.read(|b| photons(&node, &b.value, self.t + s, psi));
        let rates: Vec<f64> = photons.iter().map(|p| p.weight).collect();
        let k = draw(&rates)?;
        let p = &photons[k];
        let n = psi.len();
        let amplitudes = p.u.each_ref().map(|u| u.iter().copied().collect::<Internal>());
        let jumps: usize = node.classes.iter().map(|c| c.axes.iter().map(Vec::len).sum::<usize>()).sum();
        Some(Emitted { landing: 0, wave_number: p.wave_number, amplitudes, kick: p.kick, work: crate::decide::COMPLEX * (5 * jumps * n * n) as f64 })
    }
}

/// A class's photons of a state (module documentation, # The photons): u_a, its rate W, the moments Σ Im⟨u|∇u⟩ and
/// Σ Re⟨∇u|⊗|∇u⟩ (these three over ‖ψ‖²), the share s of b a jump carries and s b, and the photon's wave number.
struct Photons {
    u: [Vec<C64>; 3],
    weight: f64,
    mean: [f64; 3],
    second: [[f64; 3]; 3],
    share: f64,
    kick: [f64; 3],
    wave_number: f64,
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
                    let apply = |m: &Mat, out: &mut Vec<C64>| {
                        for (row, o) in out.iter_mut().enumerate() {
                            let z: C64 = (0..n).map(|col| m.data[row * n + col] * psi[col]).sum();
                            *o += f * z;
                        }
                    };
                    apply(&jump.value, &mut u[a]);
                    apply(&jump.kick, &mut ku[a]);
                    for (i, d) in jump.gradient.iter().enumerate() {
                        apply(d, &mut du[i][a]);
                    }
                }
            }
            let weight: f64 = (0..3).map(|a| norm_sqr(&u[a])).sum();
            let mean: [f64; 3] = std::array::from_fn(|i| (0..3).map(|a| inner(&u[a], &du[i][a]).im).sum());
            let second: [[f64; 3]; 3] = std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|a| inner(&du[i][a], &du[j][a]).re).sum()));
            let b: [f64; 3] = mean.map(|x| if weight > 0.0 { x / weight } else { 0.0 });
            let size = b.iter().map(|x| x * x).sum::<f64>().sqrt();
            let share = if size > class.absorbed { class.absorbed / size } else { 1.0 };
            let wave_number = if weight > 0.0 { (0..3).map(|a| inner(&u[a], &ku[a]).re).sum::<f64>() / weight } else { 0.0 };
            Photons { u, weight: weight / norm, mean: mean.map(|x| x / norm), second: second.map(|r| r.map(|x| x / norm)), share, kick: b.map(|x| share * x), wave_number }
        })
        .collect()
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
            }
        });
    }

    /// Σ_j Re tr(Ĝ⁺(s_j) σ_j + Ĝ⁻(s_j) σ_j†) of candidate m's members' H_eff gradients at the phases of `envelopes`.
    fn tone_force(&self, path: &Path, m: usize, sums: &[(f64, &[C64], &[C64])]) -> [f64; 3] {
        let c = &self.candidates(path)[m];
        let n = self.states();
        let mut out = [0.0; 3];
        for &(s, re, im) in sums {
            let node = self.node(path, s);
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
        let node = self.node(path, s);
        let f = self.model.read(|b| SelectedMiddle::contract(&node, |beat| Some(self.phase(&b.value, beat, s)), rho));
        f.map(|z| z.re)
    }

    /// The gradient at the middle by its beats: Ω_m = −Ω_k with Re tr(∂H_k e^{−iΩ_k t_m} ρ).
    fn tones(&self, path: &Path, rho: &[C64], each: &mut dyn FnMut(f64, [f64; 3])) {
        let node = self.node(path, 0.0);
        let values = self.model.read(|b| b.value.clone());
        for (beat, _) in &node.gradient {
            let f = SelectedMiddle::contract(&node, |k| (k == *beat).then(|| self.phase(&values, k, 0.0)), rho);
            each(if *beat == 0 { 0.0 } else { -values[*beat] }, f.map(|z| z.re));
        }
    }

    /// The light the state scatters at the middle beside the Ehrenfest force and the jumps (module documentation,
    /// # The photons).
    fn scattering(&self, path: &Path, psi: &[C64]) -> (Scattering, f64) {
        let node = self.node(path, 0.0);
        let photons = self.model.read(|b| photons(&node, &b.value, self.t, psi));
        let mut force = [0.0; 3];
        let mut diffusion = [[0.0; 3]; 3];
        let mut size = 0.0_f64;
        for p in &photons {
            let b: [f64; 3] = p.mean.map(|x| if p.weight > 0.0 { x / p.weight } else { 0.0 });
            for i in 0..3 {
                force[i] += (1.0 - p.share) * p.mean[i];
                for j in 0..3 {
                    diffusion[i][j] += p.second[i][j] - p.share * p.share * p.weight * b[i] * b[j];
                }
                size = size.max(p.second[i][i].abs() + 2.0 * p.weight * b[i] * b[i]);
            }
        }
        let n = self.states();
        let links = n * (3 * node.classes.len() + 1);
        (Scattering { force, diffusion, size, links }, crate::decide::COMPLEX * (12 * n * n * node.classes.len()) as f64)
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
        let node = self.node(path, s);
        at(&node, s, false, g);
        let mut counted = 3.0 * node.gradient.len() as f64 * (n * n) as f64;
        if let Some(slope) = slope {
            let h = self.half.get().expect("the step's outer nodes: its picture's ends");
            let (before, after) = (self.node(path, -h), self.node(path, h));
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
    fn optical(&self, path: &Path) -> Option<Optical> {
        let decay = self.model.optical?;
        let (most, least) = self.model.waves.iter().map(|k| (0..3).map(|a| k[a] * path.velocity[a]).sum::<f64>()).fold((f64::NEG_INFINITY, f64::INFINITY), |(most, least), x| (most.max(x), least.min(x)));
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

/// 临时入口（`select::choose` 落地前用；合并时去掉）: a request's run on a given P, its selection's report the two
/// points' errors given -- (populations, R, F) at the centre and at the edge, PYSEL's (pysel summary.md) -- or none
/// (its error then the state's whole).
#[cfg(test)]
pub(crate) fn keeping(request: &crate::request::Request, mut keep: Vec<usize>, errors: Option<[[f64; 3]; 2]>, reference: [f64; GROUND]) -> Selected {
    use crate::krylov::Route;
    use crate::select::{Candidate, PointError};
    keep.sort_unstable();
    keep.dedup();
    let point = |e: [f64; 3]| PointError { populations: e[0], rate: e[1], force: e[2], route: Route::Krylov };
    let candidates = errors.map(|[c, e]| vec![Candidate { keep: keep.clone(), centre: point(c), edge: point(e) }]).unwrap_or_default();
    let none = || (crate::floquet::Readouts::default(), Route::Krylov);
    let report = Selection { keep, candidates, reference: [none(), none()], seconds: 0.0 };
    Selected::new(Scene::new(request).unwrap(), report, reference).unwrap()
}

#[cfg(test)]
mod tests;
