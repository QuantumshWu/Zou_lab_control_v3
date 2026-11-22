//! The path errors of candidate models (PIPELINE_V3_FINAL §二 A.4, B.7–8; foldstudy ENGFOLD, PATHGAP): along atom
//! paths -- trajectories' positions and momenta in time, the pilot's (solve.rs `walks`) --, each candidate model's
//! internal density matrix and the reference's (the engine's own 32-state model) are advanced on the same path by the
//! same segments, and their readings compared: the displayed rows, the levels' sums, the photon rates by class, the
//! force. The difference on a path is the model's error of what it reads along that path; its window means over the
//! paths, with their spread, are the direct part of the model's error (DESIGN2 A, the path-measured part). Nothing here
//! selects or books: the module and its tests' calls (stage 3 takes its interface).
//!
//! # The sector
//! A model's state is a density matrix on P. P falls into its exact blocks (model/selected.rs, # Exact blocks): every
//! matrix of the generator is block-diagonal, and a jump lands on the ground states. Only the blocks that hold a ground
//! state are ever fed -- the initial state is a ground state's, every jump lands on ground states --, and of their
//! coherences only those a jump makes: the sector is the closure of the pairs (X, X) of the ground-holding blocks under
//! the jumps, a pair (X, Y) wherever a jump takes a pair (A, B) of the sector to ground states of X and Y (on every
//! scene so far every jump maps a block into one block, ENGFOLD: the sector is the diagonal pairs, Σ_X b_X², 576 for
//! PGC3D's 32 states against 1024). Its vector holds each pair's b_X × b_Y matrix row by row, pair after pair.
//!
//! # The generator
//! On the sector dρ/dt = L(t)ρ = −i(H(t)ρ − ρH(t)†) + Σ_{c,a} J_{c,a}(t) ρ J_{c,a}(t)†, H the node's generator (H_eff in
//! the frame with its decay −(i/2)ΣL†L: i(H − H†) = Σ J†J by construction, so the trace is kept) by its exact beats,
//! H(t) = Σ_k G_k e^{−iΩ_k t}, and J_{c,a}(t) a photon class's axis, its components at their beats likewise
//! (model/selected.rs, # The generator by beats, # The photons). The beats are taken exactly at every time; the
//! amplitudes G_k, J are the model's at the path's place and momentum (`Selected::node`). The recycling is one map R(t)
//! from the sector to its ground pairs' entries, made at every time from the jumps' elements.
//!
//! # The segments
//! Between two times the generator is advanced by the commutator-free fourth-order Magnus step of Blanes and Moan
//! (2006) -- exp(h(α₁L₁ + α₂L₂)) exp(h(α₂L₁ + α₁L₂)), L₁, L₂ at the segment's two Gauss points --, every exponential's
//! action exact to the double's rounding (`Op::exp`: the truncated Taylor series with scaling of Al-Mohy and Higham
//! (2011), its degree and scaling from a bound of the operator's 1-norm made from its structure: the series is summed
//! to the rounding whatever the bound). The internal coherences turn at up to the block's spectral width ω (the static
//! beat's Hermitian part's eigenvalues' span, widened by every other beat's norm: `Form::amps`): where hω ≫ 1
//! the step's error does not fall with h (PATHGAP item 3: R4's readings stand on one plateau from h = 1/8 to 1/64 µs,
//! hω ≈ 200 … 25, and fall only below; a comparison of h and h/2 on the plateau reads "converged"). So a segment is
//! no longer than one turn of the fastest coherence, hω ≤ 2π (`Scheme::Sparse`'s `turn`), off the plateau on every
//! scene measured (it begins at hω ≈ 15–25); there the error falls at the step's order and a comparison of h and h/2
//! reads it (`check`: valid only where the coarser of the two keeps hω ≤ 2π, refused elsewhere).
//!
//! # The nodes
//! The model is made at few nodes and interpolated between them on the blocks (PATHGAP item 3: "segments dense, nodes
//! sparse"): the path's knots are its step ends -- the trajectory's path is C¹ there, its acceleration jumps --, each
//! knot interval cut into `nodes` equal parts, the model made at every part's ends (`Selected::node`), and the amplitudes
//! at a Gauss point the cubic Lagrange interpolant of the four nearest nodes of its own knot interval (never across a
//! knot); the beats' phases exact at the point's time. The interpolation's error falls as the parts' length to the
//! fourth; it is read by the same comparison at twice the nodes (`check`). H is interpolated by its beats, R by its
//! beats' differences (`Amps`): each such amplitude is smooth along the path, the phases are not. Readouts are made at
//! knots, on the exact node there. (`Scheme::Exact` is ENGFOLD's: uniform segments, the model made at both Gauss
//! points of each -- the prototype's numbers on its own paths, to the reproduction's test.)
//!
//! # The readings
//! At a readout, on the node there: the displayed rows -- P's populations with Q's lifted (`Dynamics::readout`, the
//! engine's one readout: it takes a state vector, so ρ is taken through its eigenvectors, the unnormalized rows being
//! linear in ρ), over their sum; the photon rate of each class, Σ_a tr(J_{c,a} ρ J_{c,a}†); the force,
//! −tr(∇H_h ρ) + Σ_{c,a} Im tr(ρ J_{c,a}†∇J_{c,a}) (the Ehrenfest force and the scattered light's absorbed momentum).
//! The rates and the force are the trajectory's: over tr ρ_P, the state's norm on P, as the integrator takes them; the
//! rows over their sum as shown. (ENGFOLD's prototype took the rates and the force over the rows' sum too; the two
//! differ by the lifted Q population. `Reading::total` holds that sum.)
//!
//! # The memory
//! The path's length is set by the generator's slowest nonzero decay rate λ (`memory`): the sector Liouvillian's
//! eigenvalues at a point, a rounding zero |μ| ≤ κ_μ γ_d ‖L‖₁ (its first-order perturbation bound, κ_μ its condition)
//! the stationary states'; τ_mem = 1/λ, and a path's length L = min(T_run, 5τ_mem) (`Memory::length`).
//!
//! # The budget
//! Selection needs the model's error A to a small share of the target, not to the rounding: a scheme's discretization
//! error on the fold error only has to lie well within that share, and is booked as a part of A. `Sparse` is the
//! converged scheme (its check reads 1e-7 … 1e-5 of the fold error on every scene); its cost is about 4‖L‖₁ products a
//! µs whatever the segment (a polynomial exponential cannot resolve the coherences turning at ω with fewer than about
//! ω T products). The cheap schemes take the path's own steps: `Steps` (one exponential a step: the products' count
//! still about 4‖L‖₁ T), and `Lawson`, which takes H's fast turning exactly in its eigenbasis and leaves to the
//! substeps only what turns at the decay's rate and at the beats -- its cost set by the substeps, not by ‖L‖. Their
//! errors on the fold error are read against `Sparse` on the same paths (tests, `paths_replay`, `py/scheme.py`); the
//! selection reads Lawson's against its own refinements (select.rs `settle`: `Sparse` follows the fastest coherence,
//! and on scenes whose 32 states hold levels far off the light's frame it does not end).
//!
//! What a Lawson piece needs before its substeps reads no state -- its node and amplitudes, H at its middle and the
//! blocks' eigendecompositions, the substep's exponentials, R at every substep's middle, the node a reading takes at
//! its end (`Frozen`) --, so it is made on the pool, a chunk of pieces ahead of the propagation, which alone runs in
//! order; the readings of a chunk are made on the pool after it (`Form::along_from`).
//!
//! # The errors
//! Per path the candidate's readings less the reference's at the same times, averaged over a time window; over the
//! paths their mean and its standard error (the paths' spread over √P), and per path its worst row (`compare`,
//! `Window`). The windows are octaves of τ_mem from the start's transient to L (`windows`); an output time takes its
//! window's error, past L the last window's (`window_of`): the path has forgotten its start there. The paths are the
//! pilot's trajectories at the run's step (solve.rs `walks`), each from its own initial state.
use crate::linalg::{fused_product, fused_product_onto, hermitian_eigen, larger, mul_add, C64, Eigen, Mat, V3};
use crate::floquet::frame::Ratio;
use crate::model::selected::{Exact, Node, Selected};
use crate::model::Point;
use crate::request::{GROUND, STATES};
use crate::step::Dynamics;
use crate::units::rounding;
use rayon::prelude::*;

/// An atom's path (module documentation): a trajectory's place and momentum at its step ends from its start, and its
/// internal state at its start by the request's states.
#[derive(Clone, Debug, Default)]
pub struct Walk {
    pub index: u64,
    /// The internal state at the first end: (state, amplitude), normalized.
    pub start: Vec<(usize, C64)>,
    /// (t, r, κ) at the step ends (µs, µm, 1/µm), the times increasing.
    pub ends: Vec<[f64; 7]>,
}

impl Walk {
    /// The path at the time t: r, and κ = ṙ/(ħ/m), of the cubic Hermite interpolant of r between the step ends with
    /// ṙ = (ħ/m)κ there (ENGFOLD's `path_at`; outside the ends the first or last piece's; a path of one end stands).
    pub fn at(&self, hm: f64, t: f64) -> (V3, V3) {
        let s = &self.ends;
        if s.len() == 1 {
            return ([s[0][1], s[0][2], s[0][3]], [s[0][4], s[0][5], s[0][6]]);
        }
        let i = s.partition_point(|x| x[0] <= t).clamp(1, s.len() - 1) - 1;
        let (a, b) = (&s[i], &s[i + 1]);
        let dt = b[0] - a[0];
        let x = (t - a[0]) / dt;
        let (h00, h10, h01, h11) = (2. * x * x * x - 3. * x * x + 1., x * x * x - 2. * x * x + x, -2. * x * x * x + 3. * x * x, x * x * x - x * x);
        let (d00, d10, d01, d11) = (6. * x * x - 6. * x, 3. * x * x - 4. * x + 1., -6. * x * x + 6. * x, 3. * x * x - 2. * x);
        let r = std::array::from_fn(|k| h00 * a[1 + k] + h10 * dt * hm * a[4 + k] + h01 * b[1 + k] + h11 * dt * hm * b[4 + k]);
        let kappa = std::array::from_fn(|k| (d00 * a[1 + k] + d10 * dt * hm * a[4 + k] + d01 * b[1 + k] + d11 * dt * hm * b[4 + k]) / dt / hm);
        (r, kappa)
    }

    /// The path's last time.
    pub fn end(&self) -> f64 {
        self.ends.last().map_or(0.0, |e| e[0])
    }

    /// The knots on [from, to]: from, the step ends strictly inside, to.
    fn knots(&self, from: f64, to: f64) -> Vec<f64> {
        let mut k = vec![from];
        k.extend(self.ends.iter().map(|e| e[0]).filter(|&t| t > from && t < to));
        k.push(to);
        k
    }
}

/// How a path is cut and where the model is made along it (module documentation, # The segments, # The nodes).
#[derive(Clone, Copy, Debug)]
pub enum Scheme {
    /// Knots at the path's step ends, each knot interval in `nodes` (≥ 3) equal parts with the model made at their ends
    /// and interpolated between; segments of hω ≤ `turn` (rad); a readout at every `read`-th knot and at the end.
    Sparse { nodes: usize, turn: f64, read: usize },
    /// ENGFOLD's: segments of the length h from the start, the model made at both Gauss points of each; a readout at
    /// every `read`-th segment's end and at the start.
    Exact { h: f64, read: usize },
    /// One segment a step of the path (its knot interval), the model made where the segment takes it -- at its middle,
    /// one exponential (`gauss` false: the midpoint rule, second order), or at its two Gauss points (CF4) --, each
    /// exponential's series summed to the relative tolerance `tol` (the double's rounding where it is no larger); a
    /// readout at every `read`-th knot and at the end. Its error is not the rounding's: it is read against `Sparse` on
    /// a piece of the paths and booked (module documentation, # The budget).
    Steps { gauss: bool, read: usize, tol: f64 },
    /// Each step of the path in `pieces` equal pieces, the model made at each piece's middle: H there integrated exactly
    /// in its eigenbasis (block by block), the rest -- the recycling Σ J ρ J†, H's beats -- by Lawson's RK4 in `per_us`
    /// substeps a µs; a readout at every `read`-th knot and at the end. The substeps are set by the decay, not by ‖L‖.
    /// (H's change along a piece is the pieces' to follow, not the substeps': in the remainder its couplings would turn
    /// at the detunings.) The scheme's error is read against `Sparse` and booked (module documentation, # The budget).
    Lawson { per_us: f64, pieces: usize, read: usize },
}

impl Scheme {
    /// The scheme of the path errors (module documentation): a segment within one turn of the fastest coherence, the
    /// model at `nodes` parts of each step.
    pub fn sparse(nodes: usize) -> Scheme {
        Scheme::Sparse { nodes, turn: std::f64::consts::TAU, read: 1 }
    }
}

/// The work of advancing models along paths: nodes made, exponentials and their matrix–vector products, segments,
/// readouts (each one's readout calls), and the seconds of each part (one thread's wall time).
#[derive(Clone, Copy, Debug, Default)]
pub struct Counts {
    pub nodes: u64,
    pub exponentials: u64,
    pub products: u64,
    pub segments: u64,
    pub readouts: u64,
    pub readout_calls: u64,
    pub node_seconds: f64,
    pub propagation_seconds: f64,
    pub readout_seconds: f64,
    /// The path's time advanced (µs).
    pub time: f64,
    /// The largest condition of an eigenvector matrix the scheme took (`Scheme::Lawson`).
    pub condition: f64,
}

impl Counts {
    pub fn add(&mut self, o: &Counts) {
        self.nodes += o.nodes;
        self.exponentials += o.exponentials;
        self.products += o.products;
        self.segments += o.segments;
        self.readouts += o.readouts;
        self.readout_calls += o.readout_calls;
        self.node_seconds += o.node_seconds;
        self.propagation_seconds += o.propagation_seconds;
        self.readout_seconds += o.readout_seconds;
        self.time += o.time;
        self.condition = larger(self.condition, o.condition);
    }
    pub fn seconds(&self) -> f64 {
        self.node_seconds + self.propagation_seconds + self.readout_seconds
    }
}

/// A model's readings at a time of a path (module documentation, # The readings).
#[derive(Clone, Debug)]
pub struct Reading {
    pub t: f64,
    pub r: V3,
    pub kappa: V3,
    /// The rows as shown: over their sum.
    pub rows: [f64; STATES],
    /// Per photon class (`Series::classes`) its rate (1/µs), over tr ρ_P.
    pub rates: Vec<f64>,
    /// The force (rad/µs/µm, dκ/dt), over tr ρ_P.
    pub force: V3,
    /// The rows' sum before their normalization (P's trace with Q's lifted populations), and tr ρ_P.
    pub total: f64,
    pub trace: f64,
}

/// A model's readings along one path, its photon classes' fast indices, the work they took, and its state at the end
/// (the sector vector: the path goes on from there, `Form::along_from`).
#[derive(Clone, Debug, Default)]
pub struct Series {
    pub classes: Vec<[i64; 2]>,
    pub readings: Vec<Reading>,
    pub counts: Counts,
    pub state: Vec<C64>,
}

/// A model on its sector (module documentation, # The sector, # The generator): the structure every node shares, made
/// once from one node.
pub struct Form<'a> {
    pub model: &'a Selected,
    n: usize,
    /// The ground-holding blocks (places in P), and each place's (block, position in it); usize::MAX off the sector.
    blocks: Vec<Vec<usize>>,
    block_of: Vec<(usize, usize)>,
    /// The sector's block pairs and where each one's matrix starts; the vector's length.
    pairs: Vec<(usize, usize)>,
    offsets: Vec<usize>,
    dim: usize,
    /// H's slots' beats; where each block's entries start within a slot, and a slot's length.
    h_beats: Vec<usize>,
    h_block: Vec<usize>,
    h_slot: usize,
    /// The jumps: their elements' count (in the node's order); the recycling's groups by
    /// the beat Δ = Ω_b − Ω_b′ of two components of one channel (rad/µs; one exact difference one group, its value
    /// correctly rounded -- tilt2's 17 differences met as 51 numbers to the bit, each group R's whole size --: every
    /// (b, b) one, at Δ = 0), at e^{−iΔt}, and its terms -- (group, R's place, two elements): R_group[place] +=
    /// J_e J_e′* --, in the order of their places.
    elements: usize,
    beats_r: Vec<f64>,
    terms: Vec<(u32, u32, u32, u32)>,
    /// R (# The generator) as R^T: its columns' and rows' sector places, column-major over rows.
    cols: Vec<usize>,
    rows: Vec<usize>,
    /// The beats' values (rad/µs; `Selected::components`).
    values: Vec<f64>,
    /// The states of P; whether the model folds any state (its rows then lift Q's).
    states: Vec<usize>,
    folds: bool,
    classes: Vec<[i64; 2]>,
    hbar_over_mass: f64,
}

/// A node's amplitudes on the sector (# The nodes): H's slots' blocks, the recycling's R per group of its beats (each
/// cols × rows, R^T), and the blocks' spectral width (rad/µs) at the node -- the interpolated quantities: H and R are
/// linear in the Liouvillian, each beat's amplitude a smooth function of the time along the path.
#[derive(Clone, Debug)]
struct Amps {
    h: Vec<C64>,
    r: Vec<C64>,
    /// Per R group and column Σ_rows |R| (the 1-norm's bound of a combination: Σ |w| of these).
    columns: Vec<f64>,
    width: f64,
}

/// Al-Mohy and Higham's θ_m for the double's rounding (2011, Table 3.1; scipy's `_expm_multiply._theta`): the largest
/// ‖A‖₁ at which m terms of the Taylor series of e^A reach it.
const THETA: [(usize, f64); 35] = [
    (1, 2.29e-16), (2, 2.58e-8), (3, 1.39e-5), (4, 3.4e-4), (5, 2.4e-3), (6, 9.07e-3), (7, 2.38e-2), (8, 5.0e-2), (9, 8.96e-2), (10, 1.44e-1),
    (11, 2.14e-1), (12, 3.0e-1), (13, 4.0e-1), (14, 5.14e-1), (15, 6.41e-1), (16, 7.81e-1), (17, 9.31e-1), (18, 1.09), (19, 1.26), (20, 1.44),
    (21, 1.62), (22, 1.82), (23, 2.01), (24, 2.22), (25, 2.43), (26, 2.64), (27, 2.86), (28, 3.08), (29, 3.31), (30, 3.54),
    (35, 4.7), (40, 6.0), (45, 7.2), (50, 8.5), (55, 9.9),
];

/// The Taylor degree m and the scaling s for an operator of the 1-norm `norm` (Al-Mohy and Higham's fragment 3.1 for a
/// known norm): m·⌈norm/θ_m⌉ least.
fn degree(norm: f64) -> (usize, usize) {
    if norm == 0.0 {
        return (0, 1);
    }
    let mut best = (usize::MAX, 1, 1);
    for &(m, theta) in &THETA {
        let s = (norm / theta).ceil().max(1.0) as usize;
        if m * s < best.0 {
            best = (m * s, m, s);
        }
    }
    (best.1, best.2)
}

/// The commutator-free Magnus step's weights (Blanes and Moan 2006): α₁ = (3 − 2√3)/12, α₂ = (3 + 2√3)/12, and the
/// Gauss points ½ ∓ √3/6.
const SQRT3: f64 = 1.732_050_807_568_877_2;
const ALPHA: [f64; 2] = [(3.0 - 2.0 * SQRT3) / 12.0, (3.0 + 2.0 * SQRT3) / 12.0];
const GAUSS: [f64; 2] = [0.5 - SQRT3 / 6.0, 0.5 + SQRT3 / 6.0];

impl<'a> Form<'a> {
    /// The model's form (module documentation): its sector from its blocks and its jumps' structure, read on the node at
    /// the point `at` at rest (every node holds the layout's beats and elements, whatever their values there).
    pub fn new(model: &'a Selected, at: V3) -> Form<'a> {
        let node = model.node(at, [0.0; 3], false);
        let states = model.rs.keep.clone();
        let n = states.len();
        let ground: Vec<bool> = states.iter().map(|&s| s < GROUND).collect();
        let blocks: Vec<Vec<usize>> = model.blocks().iter().filter(|b| b.iter().any(|&x| ground[x])).cloned().collect();
        let mut block_of = vec![(usize::MAX, usize::MAX); n];
        for (x, b) in blocks.iter().enumerate() {
            for (i, &p) in b.iter().enumerate() {
                block_of[p] = (x, i);
            }
        }
        // The jumps: per channel (class, axis) its components, each its beat and its elements (flat index, ground row,
        // source column: places in P).
        let mut channels: Vec<Vec<(usize, Vec<(usize, usize, usize)>)>> = Vec::new();
        let mut elements = 0;
        let mut classes = Vec::new();
        for class in &node.classes {
            classes.push(class.fast);
            for axis in &class.axes {
                let mut components = Vec::new();
                for jump in axis {
                    let at: Vec<(usize, usize, usize)> = jump.elements.iter().enumerate().map(|(k, e)| (elements + k, e.at / n, e.at % n)).collect();
                    elements += jump.elements.len();
                    components.push((jump.beat, at));
                }
                channels.push(components);
            }
        }
        // The sector: the closure of the ground-holding blocks' own pairs under the jumps.
        let nb = blocks.len();
        let mut held = vec![false; nb * nb];
        for x in 0..nb {
            held[x * nb + x] = true;
        }
        let of = |p: usize| block_of[p].0;
        loop {
            let mut grown = false;
            for ch in &channels {
                let entries: Vec<(usize, usize)> = ch.iter().flat_map(|c| c.1.iter().map(|e| (e.1, e.2))).collect();
                for &(g, s) in &entries {
                    for &(g2, s2) in &entries {
                        let (bs, bs2, bg, bg2) = (of(s), of(s2), of(g), of(g2));
                        if bs == usize::MAX || bs2 == usize::MAX || bg == usize::MAX || bg2 == usize::MAX {
                            continue;
                        }
                        if held[bs * nb + bs2] && !held[bg * nb + bg2] {
                            held[bg * nb + bg2] = true;
                            grown = true;
                        }
                    }
                }
            }
            if !grown {
                break;
            }
        }
        let mut pairs: Vec<(usize, usize)> = (0..nb).map(|x| (x, x)).collect();
        for x in 0..nb {
            for y in 0..nb {
                if x != y && held[x * nb + y] {
                    pairs.push((x, y));
                }
            }
        }
        let mut offsets = Vec::with_capacity(pairs.len() + 1);
        let mut dim = 0;
        for &(x, y) in &pairs {
            offsets.push(dim);
            dim += blocks[x].len() * blocks[y].len();
        }
        offsets.push(dim);
        let mut pair_of = vec![usize::MAX; nb * nb];
        for (k, &(x, y)) in pairs.iter().enumerate() {
            pair_of[x * nb + y] = k;
        }
        let place = |a: usize, b: usize| -> Option<usize> {
            let ((x, i), (y, j)) = (block_of[a], block_of[b]);
            if x == usize::MAX || y == usize::MAX || pair_of[x * nb + y] == usize::MAX {
                return None;
            }
            Some(offsets[pair_of[x * nb + y]] + i * blocks[y].len() + j)
        };
        // R's rows: the ground pairs of the sector; its columns: the source pairs a channel's two elements take; its
        // terms per pair of components of a channel.
        let mut row_of = vec![u32::MAX; n * n];
        let mut rows = Vec::new();
        let mut col_of = vec![u32::MAX; n * n];
        let mut cols = Vec::new();
        let mut raw: Vec<(usize, usize, usize, usize, usize)> = Vec::new();
        let (values, exact) = model.components();
        let mut beats_r: Vec<f64> = Vec::new();
        let mut keys: Vec<Exact> = Vec::new();
        for ch in &channels {
            for (b1, c1) in ch {
                for (b2, c2) in ch {
                    let key = if b1 == b2 { ([0; 3], Vec::new()) } else { difference(&exact[*b1], &exact[*b2]) };
                    let bp = keys.iter().position(|k| *k == key).unwrap_or_else(|| {
                        beats_r.push(if key.0 == [0; 3] && key.1.is_empty() { 0.0 } else { model.value_of(&key) });
                        keys.push(key);
                        beats_r.len() - 1
                    });
                    for &(e1, g, s) in c1 {
                        for &(e2, g2, s2) in c2 {
                            if let (Some(r), Some(c)) = (place(g, g2), place(s, s2)) {
                                if row_of[g * n + g2] == u32::MAX {
                                    row_of[g * n + g2] = rows.len() as u32;
                                    rows.push(r);
                                }
                                if col_of[s * n + s2] == u32::MAX {
                                    col_of[s * n + s2] = cols.len() as u32;
                                    cols.push(c);
                                }
                                raw.push((bp, g * n + g2, s * n + s2, e1, e2));
                            }
                        }
                    }
                }
            }
        }
        let nr = rows.len();
        let mut terms: Vec<(u32, u32, u32, u32)> = raw.iter().map(|&(bp, gg, ss, e1, e2)| (bp as u32, (col_of[ss] as usize * nr + row_of[gg] as usize) as u32, e1 as u32, e2 as u32)).collect();
        terms.sort_unstable();
        let mut h_block = Vec::with_capacity(nb);
        let mut h_slot = 0;
        for b in &blocks {
            h_block.push(h_slot);
            h_slot += b.len() * b.len();
        }
        let form = Form {
            model,
            n,
            blocks,
            block_of,
            pairs,
            offsets,
            dim,
            h_beats: node.generator.iter().map(|g| g.0).collect(),
            h_block,
            h_slot,
            elements,
            beats_r,
            terms,
            cols,
            rows,
            values,
            folds: !model.rs.folded.is_empty(),
            states,
            classes,
            hbar_over_mass: model.hbar_over_mass(),
        };
        assert!(form.beats_known(&node), "a node's beat outside the model's values");
        form
    }

    /// The form's sizes (the measurements' line): P's states, the blocks, the sector, H's slots, R and its groups, the
    /// terms, the jumps' elements, the photon classes.
    #[cfg(test)]
    pub(crate) fn shape(&self) -> String {
        format!(
            "n {}, blocks {:?}, pairs {}, dim {}, H beats {} (nonzero {}), H slot {}, R {} x {} by {} beat groups, terms {}, elements {}, classes {}",
            self.n, self.blocks.iter().map(|b| b.len()).collect::<Vec<_>>(), self.pairs.len(), self.dim, self.h_beats.len(), self.h_beats.iter().filter(|&&b| self.values[b] != 0.0).count(), self.h_slot,
            self.cols.len(), self.rows.len(), self.beats_r.len(), self.terms.len(), self.elements, self.classes.len()
        )
    }

    /// The beats' values (rad/µs): H's nonzero ones, and R's groups' (the measurements' line).
    #[cfg(test)]
    pub(crate) fn beat_values(&self) -> (Vec<f64>, Vec<f64>) {
        (self.h_beats.iter().map(|&b| self.values[b]).filter(|&v| v != 0.0).collect(), self.beats_r.clone())
    }

    /// The sector vector's length (module documentation, # The sector).
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The ground-holding blocks' sizes and the sector's pairs.
    pub fn blocks(&self) -> (Vec<usize>, usize) {
        (self.blocks.iter().map(Vec::len).collect(), self.pairs.len())
    }

    /// Whether every beat a node holds has its value.
    fn beats_known(&self, node: &Node) -> bool {
        let v = self.values.len();
        node.generator.iter().all(|g| g.0 < v)
            && node.gradient.iter().all(|g| g.0 < v)
            && node.classes.iter().all(|c| c.axes.iter().all(|a| a.iter().all(|j| j.beat < v)))
    }

    /// A beat's phase e^{−iΩt} at the time t.
    fn phase(&self, beat: usize, t: f64) -> C64 {
        let v = self.values[beat];
        if v == 0.0 {
            C64::ONE
        } else {
            C64::cis(-v * t)
        }
    }

    /// The model's node at the path's place and momentum at the time t.
    fn node_at(&self, walk: &Walk, t: f64) -> Node {
        let (r, kappa) = walk.at(self.hbar_over_mass, t);
        self.model.node(r, kappa, false)
    }

    /// A node's amplitudes on the sector (`Amps`): H's slots' blocks; R per group of beats from the jumps' elements; and the
    /// width -- per block the static beat's Hermitian part's eigenvalues' span, widened on each side by every other
    /// beat's Frobenius norm (Weyl: e^{−iΩt}G_k and its Hermitian part move no eigenvalue by more), the widest of the
    /// sector's pairs' spans (a coherence between two blocks turns at their levels' differences).
    fn amps(&self, node: &Node) -> Amps {
        self.amps_with(node, true)
    }

    /// `amps`, the width and R's columns' bounds made only where `width` (the segments' rule and the exponential's norm
    /// need them; Lawson's substeps do not: NaN and none there).
    fn amps_with(&self, node: &Node, width: bool) -> Amps {
        let n = self.n;
        let mut h = vec![C64::ZERO; self.h_beats.len() * self.h_slot];
        for (k, (_, m)) in node.generator.iter().enumerate() {
            for (x, b) in self.blocks.iter().enumerate() {
                let at = k * self.h_slot + self.h_block[x];
                let w = b.len();
                for (i, &p) in b.iter().enumerate() {
                    for (j, &q) in b.iter().enumerate() {
                        h[at + i * w + j] = m.data[p * n + q];
                    }
                }
            }
        }
        let mut j = Vec::with_capacity(self.elements);
        for class in &node.classes {
            for axis in &class.axes {
                for jump in axis {
                    j.extend(jump.elements.iter().map(|e| e.value));
                }
            }
        }
        let size = self.cols.len() * self.rows.len();
        let mut r = vec![C64::ZERO; self.beats_r.len() * size];
        for &(bp, place, e1, e2) in &self.terms {
            r[bp as usize * size + place as usize] += j[e1 as usize] * j[e2 as usize].conj();
        }
        if !width {
            return Amps { h, r, columns: Vec::new(), width: f64::NAN };
        }
        let ends: Vec<(f64, f64)> = self
            .blocks
            .iter()
            .enumerate()
            .map(|(x, b)| {
                let w = b.len();
                let at = |k: usize| k * self.h_slot + self.h_block[x];
                let (mut lo, mut hi, mut moved) = (0.0, 0.0, 0.0);
                for (k, &beat) in self.h_beats.iter().enumerate() {
                    let block = &h[at(k)..at(k) + w * w];
                    if self.values[beat] == 0.0 {
                        let e = hermitian_eigen(&Mat { rows: w, cols: w, data: block.to_vec() });
                        (lo, hi) = (e.values[0], e.values[w - 1]);
                    } else {
                        moved += block.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
                    }
                }
                (lo - moved, hi + moved)
            })
            .collect();
        let width = self.pairs.iter().map(|&(x, y)| larger(ends[x].1 - ends[y].0, ends[y].1 - ends[x].0)).fold(0.0, larger);
        let nr = self.rows.len();
        let columns = r.chunks(nr.max(1)).map(|c| c.iter().map(|z| z.abs()).sum::<f64>()).collect();
        Amps { h, r, columns, width }
    }

    /// The operator h·L′ of one exponential into `op` (`Op`): its generator Σ (weight × the amplitudes at their phases
    /// at the time) over the list `of` of (node's amplitudes, weight, time) -- a CF4 exponential's c₁, c₂ times each
    /// Gauss point's interpolation weights --, each node's weights merged first, so that its H and R are read once:
    /// H = Σ_k e^{−iΩ_k t} H_k, R = Σ_g e^{−iΔ_g t} R_g. L′ = L − 2 Im(c)·1 for the shift c of H by its mean diagonal
    /// over the sector (−i(cρ − ρc*) = 2 Im(c) ρ, a scalar factor of the exponential); with its 1-norm's bound, column
    /// by column: the off-diagonal moduli of H's two columns, the diagonal's difference, R's column bounded by its
    /// nodes' (the Taylor degree and scaling from a bound: the series is still summed to the rounding).
    fn op_into(&self, of: &[(&Amps, f64, f64)], h: f64, op: &mut Op) {
        op.fit(self);
        let (slots, groups) = (self.h_beats.len(), self.beats_r.len());
        let mut nodes: Vec<&Amps> = Vec::with_capacity(of.len());
        op.wh.clear();
        op.wr.clear();
        for &(a, w, t) in of {
            let i = nodes.iter().position(|x| std::ptr::eq(*x, a)).unwrap_or_else(|| {
                nodes.push(a);
                op.wh.resize(nodes.len() * slots, C64::ZERO);
                op.wr.resize(nodes.len() * groups, C64::ZERO);
                nodes.len() - 1
            });
            for (k, &beat) in self.h_beats.iter().enumerate() {
                op.wh[i * slots + k] += self.phase(beat, t) * w;
            }
            for (g, &delta) in self.beats_r.iter().enumerate() {
                op.wr[i * groups + g] += if delta == 0.0 { C64::real(w) } else { C64::cis(-delta * t) * w };
            }
        }
        op.hc.fill(C64::ZERO);
        for (i, a) in nodes.iter().enumerate() {
            for k in 0..slots {
                mul_add(false, &mut op.hc, op.wh[i * slots + k], &a.h[k * self.h_slot..(k + 1) * self.h_slot]);
            }
        }
        let mut sum = C64::ZERO;
        let mut count = 0;
        for (x, b) in self.blocks.iter().enumerate() {
            let w = b.len();
            for i in 0..w {
                sum += op.hc[self.h_block[x] + i * w + i];
                count += 1;
            }
        }
        let shift = sum * (1.0 / count as f64);
        for (x, b) in self.blocks.iter().enumerate() {
            let w = b.len();
            let at = self.h_block[x];
            let (left, right, diag, off) = (&mut op.a1[x], &mut op.a2[x], &mut op.diag[x], &mut op.off[x]);
            off.fill(0.0);
            for i in 0..w {
                for j in 0..w {
                    let raw = op.hc[at + i * w + j];
                    let z = if i == j { raw - shift } else { raw };
                    left[i * w + j] = C64::new(0.0, -h) * z;
                    // (H − c)† at (j, i), times i h: the right factor of ρ(H − c)†.
                    right[j * w + i] = C64::new(0.0, h) * z.conj();
                    if i == j {
                        diag[i] = z;
                    } else {
                        off[j] += z.abs();
                    }
                }
            }
        }
        // R and its columns' bound: Σ_j Σ_g |h w_jg| Σ_rows |R_jg| ≥ Σ_rows |R's combination|.
        let size = op.rt.len();
        let ncols = self.cols.len();
        op.rt.fill(C64::ZERO);
        op.colsum.fill(0.0);
        for (i, a) in nodes.iter().enumerate() {
            for g in 0..groups {
                let c = op.wr[i * groups + g] * h;
                mul_add(false, &mut op.rt, c, &a.r[g * size..(g + 1) * size]);
                let m = c.abs();
                for (ci, &place) in self.cols.iter().enumerate() {
                    op.colsum[place] += m * a.columns[g * ncols + ci];
                }
            }
        }
        let mut norm = 0.0_f64;
        for (p, &(x, y)) in self.pairs.iter().enumerate() {
            let wy = self.blocks[y].len();
            for k in 0..self.blocks[x].len() {
                for l in 0..wy {
                    let s = h * (op.off[x][k] + op.off[y][l] + (op.diag[x][k] - op.diag[y][l].conj()).abs()) + op.colsum[self.offsets[p] + k * wy + l];
                    norm = larger(norm, s);
                }
            }
        }
        op.eta = (2.0 * shift.im * h).exp();
        op.norm = norm;
    }

    /// The exponential's action e^{hL} v on the sector vector of the generator of `of` (`op_into`), in place
    /// (`Op::exp`).
    fn exp(&self, of: &[(&Amps, f64, f64)], h: f64, v: &mut [C64], work: &mut Work, counts: &mut Counts) {
        let mut op = std::mem::take(&mut work.op);
        self.op_into(of, h, &mut op);
        counts.products += op.exp(self, v, work) as u64;
        counts.exponentials += 1;
        work.op = op;
    }

    /// One segment of the length h (# The segments): CF4 on the generators at its Gauss points, each given as its
    /// time and its nodes' amplitudes with their interpolation weights.
    fn segment(&self, at: [(f64, &[(&Amps, f64)]); 2], h: f64, v: &mut [C64], work: &mut Work, counts: &mut Counts) {
        let mut of: Vec<(&Amps, f64, f64)> = Vec::with_capacity(at[0].1.len() + at[1].1.len());
        for c in [[ALPHA[1], ALPHA[0]], [ALPHA[0], ALPHA[1]]] {
            of.clear();
            for (k, (t, nodes)) in at.iter().enumerate() {
                of.extend(nodes.iter().map(|&(a, w)| (a, c[k] * w, *t)));
            }
            self.exp(&of, h, v, work, counts);
        }
        counts.segments += 1;
    }

    /// The sector vector of a state (normalized), by the request's states; Err where a state of it is not in the sector.
    fn initial(&self, start: &[(usize, C64)]) -> Result<Vec<C64>, String> {
        let mut psi = vec![C64::ZERO; self.n];
        for &(s, z) in start {
            let p = self.states.iter().position(|&x| x == s).ok_or_else(|| format!("state {s} is not in P"))?;
            if self.block_of[p].0 == usize::MAX {
                return Err(format!("state {s} lies in no ground-holding block"));
            }
            psi[p] = z;
        }
        let norm: f64 = psi.iter().map(|z| z.norm_sqr()).sum();
        let mut v = vec![C64::ZERO; self.dim];
        for a in 0..self.n {
            for b in 0..self.n {
                if let Some(p) = self.place(a, b) {
                    v[p] = psi[a] * psi[b].conj() * (1.0 / norm);
                }
            }
        }
        Ok(v)
    }

    /// The sector place of the element (a, b) of ρ (places in P), None off the sector.
    fn place(&self, a: usize, b: usize) -> Option<usize> {
        let ((x, i), (y, j)) = (self.block_of[a], self.block_of[b]);
        if x == usize::MAX || y == usize::MAX {
            return None;
        }
        let k = self.pairs.iter().position(|&p| p == (x, y))?;
        Some(self.offsets[k] + i * self.blocks[y].len() + j)
    }

    /// ρ on P (n × n, row by row) from the sector vector: zero off the sector.
    fn density(&self, v: &[C64]) -> Vec<C64> {
        let n = self.n;
        let mut rho = vec![C64::ZERO; n * n];
        for (k, &(x, y)) in self.pairs.iter().enumerate() {
            let (bx, by) = (&self.blocks[x], &self.blocks[y]);
            for (i, &a) in bx.iter().enumerate() {
                for (j, &b) in by.iter().enumerate() {
                    rho[a * n + b] = v[self.offsets[k] + i * by.len() + j];
                }
            }
        }
        rho
    }

    /// The readings of the sector state v on the node at the time t (module documentation, # The readings).
    fn read(&self, node: &Node, walk: &Walk, t: f64, v: &[C64], counts: &mut Counts) -> Reading {
        let clock = std::time::Instant::now();
        let n = self.n;
        let rho = self.density(v);
        let trace: f64 = (0..n).map(|a| rho[a * n + a].re).sum();
        let (r, kappa) = walk.at(self.hbar_over_mass, t);
        // The rows: the diagonal where nothing is folded; else through ρ's eigenvectors on the sector's places, each
        // one's rows as shown un-normalized by its own sum (P's part over ‖ψ‖²).
        let mut rows = [0.0; STATES];
        if !self.folds {
            for (a, &s) in self.states.iter().enumerate() {
                rows[s] = rho[a * n + a].re;
            }
        } else {
            let support: Vec<usize> = (0..n).filter(|&a| self.block_of[a].0 != usize::MAX).collect();
            let m = support.len();
            let sub = Mat::from_fn(m, m, |i, j| rho[support[i] * n + support[j]]);
            let eigen = hermitian_eigen(&sub);
            let point = Point { r, kappa, t };
            let mut psi = vec![C64::ZERO; n];
            for k in 0..m {
                let lambda = eigen.values[k];
                psi.fill(C64::ZERO);
                for (i, &a) in support.iter().enumerate() {
                    psi[a] = eigen.vectors[(i, k)];
                }
                let shown = self.model.readout(0, &point, &psi);
                counts.readout_calls += 1;
                let on_p: f64 = self.states.iter().map(|&s| shown[s]).sum();
                for (x, y) in rows.iter_mut().zip(shown) {
                    *x += lambda * y / on_p;
                }
            }
        }
        let total: f64 = rows.iter().sum();
        let rows = rows.map(|x| x / total);
        // The rates and the force on the node at t: J(t), ∇J(t) by class and axis, ∇H(t).
        let mut rates = vec![0.0; self.classes.len()];
        let mut force = [0.0; 3];
        for (beat, g) in &node.gradient {
            let f = self.phase(*beat, t);
            for (axis, m) in g.iter().enumerate() {
                let mut s = C64::ZERO;
                for x in 0..n {
                    for y in 0..n {
                        s += m.data[x * n + y] * rho[y * n + x];
                    }
                }
                force[axis] -= (f * s).re;
            }
        }
        let mut entries: Vec<(usize, usize, C64, [C64; 3])> = Vec::new();
        for (c, class) in node.classes.iter().enumerate() {
            for axis in &class.axes {
                entries.clear();
                for jump in axis {
                    let f = self.phase(jump.beat, t);
                    for e in &jump.elements {
                        let (g, s) = (e.at / n, e.at % n);
                        let (z, d) = (f * e.value, e.gradient.map(|x| f * x));
                        match entries.iter_mut().find(|x| x.0 == g && x.1 == s) {
                            Some(x) => {
                                x.2 += z;
                                for a in 0..3 {
                                    x.3[a] += d[a];
                                }
                            }
                            None => entries.push((g, s, z, d)),
                        }
                    }
                }
                // tr(J ρ J†) = Σ J[g,u] ρ[u,s] conj(J[g,s]); Im tr(ρ J†∇J) = Im Σ conj(J[g,u]) ∇J[g,s] ρ[s,u].
                let mut rate = C64::ZERO;
                let mut absorbed = [C64::ZERO; 3];
                for &(g, u, z, _) in entries.iter() {
                    for &(g2, s, w, d) in entries.iter() {
                        if g2 != g {
                            continue;
                        }
                        rate += z * rho[u * n + s] * w.conj();
                        for a in 0..3 {
                            absorbed[a] += z.conj() * d[a] * rho[s * n + u];
                        }
                    }
                }
                rates[c] += rate.re / trace;
                for a in 0..3 {
                    force[a] += absorbed[a].im;
                }
            }
        }
        let force = force.map(|f| f / trace);
        counts.readouts += 1;
        counts.readout_seconds += clock.elapsed().as_secs_f64();
        Reading { t, r, kappa, rows, rates, force, total, trace }
    }

    /// The model's readings along a path from `from` to `to` from the state `start` (module documentation): its
    /// series, and Err where the state is not the model's.
    pub fn along(&self, walk: &Walk, start: &[(usize, C64)], from: f64, to: f64, scheme: Scheme) -> Result<Series, String> {
        self.along_from(walk, self.initial(start)?, from, to, scheme)
    }

    /// `along` from a sector state at `from` (a series' end state: the same path gone on, its knots and nodes the
    /// same as in one stretch where `from` is a knot).
    pub fn along_from(&self, walk: &Walk, state: Vec<C64>, from: f64, to: f64, scheme: Scheme) -> Result<Series, String> {
        let mut v = state;
        if v.len() != self.dim {
            return Err(format!("a state of {} numbers for a sector of {}", v.len(), self.dim));
        }
        if !(to > from) {
            return Err(format!("an empty stretch of the path: [{from}, {to}]"));
        }
        let mut counts = Counts::default();
        let mut work = Work::new(self);
        let mut readings = Vec::new();
        let made = |t: f64, counts: &mut Counts| -> Node {
            let clock = std::time::Instant::now();
            let node = self.node_at(walk, t);
            counts.nodes += 1;
            counts.node_seconds += clock.elapsed().as_secs_f64();
            node
        };
        match scheme {
            Scheme::Steps { gauss, read, tol } => {
                work.tol = tol;
                let knots = walk.knots(from, to);
                let node = made(from, &mut counts);
                readings.push(self.read(&node, walk, from, &v, &mut counts));
                for (i, w) in knots.windows(2).enumerate() {
                    let (a, b) = (w[0], w[1]);
                    let h = b - a;
                    let times: Vec<f64> = if gauss { GAUSS.iter().map(|c| a + c * h).collect() } else { vec![a + 0.5 * h] };
                    let amps: Vec<Amps> = times
                        .iter()
                        .map(|&t| {
                            let node = made(t, &mut counts);
                            let clock = std::time::Instant::now();
                            let x = self.amps(&node);
                            counts.node_seconds += clock.elapsed().as_secs_f64();
                            x
                        })
                        .collect();
                    let clock = std::time::Instant::now();
                    if gauss {
                        self.segment([(times[0], &[(&amps[0], 1.0)]), (times[1], &[(&amps[1], 1.0)])], h, &mut v, &mut work, &mut counts);
                    } else {
                        self.exp(&[(&amps[0], 1.0, times[0])], h, &mut v, &mut work, &mut counts);
                        counts.segments += 1;
                    }
                    counts.propagation_seconds += clock.elapsed().as_secs_f64();
                    if (i + 1) % read == 0 || i + 2 == knots.len() {
                        let node = made(b, &mut counts);
                        readings.push(self.read(&node, walk, b, &v, &mut counts));
                    }
                }
            }
            Scheme::Lawson { per_us, pieces, read } => {
                assert!(pieces >= 1, "a step in one piece at least");
                let knots = walk.knots(from, to);
                let node = made(from, &mut counts);
                readings.push(self.read(&node, walk, from, &v, &mut counts));
                // The pieces in order, each with the time of the reading at its end where one is made there.
                let mut plan: Vec<Piece> = Vec::new();
                for (i, w) in knots.windows(2).enumerate() {
                    let (a, b) = (w[0], w[1]);
                    let piece = (b - a) / pieces as f64;
                    // The substeps a piece: a product's rounding up never adds one (a rate of whole substeps a piece).
                    let n = (piece * per_us * (1.0 - 4.0 * f64::EPSILON)).ceil().max(1.0) as usize;
                    for j in 0..pieces {
                        let p0 = a + j as f64 * piece;
                        let read = (j + 1 == pieces && ((i + 1) % read == 0 || i + 2 == knots.len())).then_some(b);
                        plan.push(Piece { tm: p0 + 0.5 * piece, a: p0, d: piece / n as f64, n, read });
                    }
                }
                // A chunk of pieces frozen on the pool while the one before is propagated; its readings on the pool
                // after it. The chunk: what `FROZEN_BYTES` holds of a piece's numbers, at most four per thread.
                let most = plan.iter().map(|p| p.n).max().unwrap_or(1);
                let size = 16 * (self.h_beats.len() * self.h_slot + most * self.cols.len() * self.rows.len() + 5 * self.h_slot + 2 * self.dim) + 1;
                let chunk = (FROZEN_BYTES / size).clamp(1, 4 * rayon::current_num_threads());
                let freeze = |range: std::ops::Range<usize>| -> Vec<Result<Frozen, String>> { plan[range].par_iter().map(|p| self.freeze(walk, p)).collect() };
                let mut lw = Lawson::new(self);
                let mut ahead = freeze(0..chunk.min(plan.len()));
                let mut start = 0;
                while start < plan.len() {
                    let current: Vec<Frozen> = std::mem::take(&mut ahead).into_iter().collect::<Result<_, _>>()?;
                    let end = start + current.len();
                    let (next, made) = rayon::join(
                        || freeze(end..(end + chunk).min(plan.len())),
                        || {
                            let clock = std::time::Instant::now();
                            let mut made: Vec<(f64, &Node, Vec<C64>)> = Vec::new();
                            for f in &current {
                                self.lawson_run(f, &mut v, &mut lw, &mut counts);
                                if let Some((t, node)) = &f.read {
                                    made.push((*t, node, v.clone()));
                                }
                            }
                            counts.propagation_seconds += clock.elapsed().as_secs_f64();
                            made
                        },
                    );
                    for f in &current {
                        counts.nodes += 1 + u64::from(f.read.is_some());
                        counts.node_seconds += f.seconds;
                        counts.condition = larger(counts.condition, f.condition);
                    }
                    let read: Vec<(Reading, Counts)> = made
                        .par_iter()
                        .map(|(t, node, state)| {
                            let mut c = Counts::default();
                            let r = self.read(node, walk, *t, state, &mut c);
                            (r, c)
                        })
                        .collect();
                    for (r, c) in read {
                        readings.push(r);
                        counts.add(&Counts { time: 0.0, ..c });
                    }
                    (ahead, start) = (next, end);
                }
            }
            Scheme::Exact { h, read } => {
                let segments = ((to - from) / h).round() as usize;
                for k in 0..=segments {
                    let t = from + k as f64 * h;
                    if k % read == 0 {
                        let node = made(t, &mut counts);
                        readings.push(self.read(&node, walk, t, &v, &mut counts));
                    }
                    if k < segments {
                        let times = GAUSS.map(|c| from + (k as f64 + c) * h);
                        let amps = times.map(|tg| {
                            let node = made(tg, &mut counts);
                            let clock = std::time::Instant::now();
                            let a = self.amps(&node);
                            counts.node_seconds += clock.elapsed().as_secs_f64();
                            a
                        });
                        let clock = std::time::Instant::now();
                        self.segment([(times[0], &[(&amps[0], 1.0)]), (times[1], &[(&amps[1], 1.0)])], h, &mut v, &mut work, &mut counts);
                        counts.propagation_seconds += clock.elapsed().as_secs_f64();
                    }
                }
            }
            Scheme::Sparse { nodes, turn, read } => {
                assert!(nodes >= 3, "a cubic interpolant takes four nodes a knot interval");
                let knots = walk.knots(from, to);
                let mut first = made(from, &mut counts);
                readings.push(self.read(&first, walk, from, &v, &mut counts));
                let mut grid: Vec<Amps> = vec![self.amps(&first)];
                for (i, w) in knots.windows(2).enumerate() {
                    let (a, b) = (w[0], w[1]);
                    let part = (b - a) / nodes as f64;
                    grid.truncate(1);
                    for j in 1..=nodes {
                        let t = if j == nodes { b } else { a + j as f64 * part };
                        let node = made(t, &mut counts);
                        let clock = std::time::Instant::now();
                        grid.push(self.amps(&node));
                        counts.node_seconds += clock.elapsed().as_secs_f64();
                        if j == nodes {
                            first = node;
                        }
                    }
                    let width = grid.iter().map(|x| x.width).fold(0.0, larger);
                    let count = ((b - a) * width / turn).ceil().max(1.0) as usize;
                    let h = (b - a) / count as f64;
                    let clock = std::time::Instant::now();
                    for s in 0..count {
                        let t0 = a + s as f64 * h;
                        let times = GAUSS.map(|c| t0 + c * h);
                        let of = times.map(|tg| {
                            let (start, weights) = stencil(nodes, (tg - a) / part);
                            let of: [(&Amps, f64); 4] = std::array::from_fn(|q| (&grid[start + q], weights[q]));
                            of
                        });
                        self.segment([(times[0], &of[0]), (times[1], &of[1])], h, &mut v, &mut work, &mut counts);
                    }
                    counts.propagation_seconds += clock.elapsed().as_secs_f64();
                    grid.swap(0, nodes);
                    if (i + 1) % read == 0 || i + 2 == knots.len() {
                        readings.push(self.read(&first, walk, b, &v, &mut counts));
                    }
                }
            }
        }
        counts.time = to - from;
        Ok(Series { classes: self.classes.clone(), readings, counts, state: v })
    }

    /// The sector's Liouvillian at the point and time, dense (d × d), column by column through the operator.
    fn liouvillian(&self, r: V3, kappa: V3, t: f64) -> Mat {
        let node = self.model.node(r, kappa, false);
        let amps = self.amps(&node);
        let mut op = Op::default();
        self.op_into(&[(&amps, 1.0, t)], 1.0, &mut op);
        let d = self.dim;
        let mut work = Work::new(self);
        let mut l = Mat::zeros(d, d);
        let mut e = vec![C64::ZERO; d];
        let mut out = vec![C64::ZERO; d];
        for j in 0..d {
            e.fill(C64::ZERO);
            e[j] = C64::ONE;
            op.apply(self, &e, &mut out, &mut work);
            for i in 0..d {
                l[(i, j)] = out[i];
            }
            // The shift's scalar back: L = L′ + 2 Im(c)·1.
            l[(j, j)] += C64::real(op.eta.ln());
        }
        l
    }
}

/// The cubic Lagrange stencil at u (in parts of the knot interval, 0 … m, m ≥ 3 parts): the first of the four nearest
/// nodes of the interval's own and their weights.
fn stencil(m: usize, u: f64) -> (usize, [f64; 4]) {
    let j = (u.floor().max(0.0) as usize).min(m - 1);
    let start = j.saturating_sub(1).min(m - 3);
    let x: [f64; 4] = std::array::from_fn(|k| (start + k) as f64);
    (start, std::array::from_fn(|k| (0..4).filter(|&l| l != k).map(|l| (u - x[l]) / (x[k] - x[l])).product()))
}

/// The exact vector of a beat difference Ω_a − Ω_b (model/selected.rs `Exact`): the fast integers' and the slow
/// ratios' differences, the ratios to the longer's length and its trailing zeros cut (one difference, one vector).
fn difference(a: &Exact, b: &Exact) -> Exact {
    let n = std::array::from_fn(|k| a.0[k] - b.0[k]);
    let at = |p: &Vec<Ratio>, k: usize| p.get(k).copied().unwrap_or(Ratio::ZERO);
    let mut p: Vec<Ratio> = (0..a.1.len().max(b.1.len()))
        .map(|k| {
            let (x, y) = (at(&a.1, k), at(&b.1, k));
            Ratio::new(x.num * y.den - y.num * x.den, x.den * y.den)
        })
        .collect();
    while p.last().is_some_and(|r| r.num == 0) {
        p.pop();
    }
    (n, p)
}

/// The scratch of the operator's products: the gathered columns and R's product, the Taylor series' vectors, the
/// operator.
struct Work {
    /// The series' relative tolerance (`Scheme::Steps`; the double's rounding elsewhere).
    tol: f64,
    gathered: Vec<C64>,
    product: Vec<C64>,
    b: Vec<C64>,
    f: Vec<C64>,
    next: Vec<C64>,
    op: Op,
}

impl Work {
    fn new(form: &Form) -> Work {
        let mut op = Op::default();
        op.fit(form);
        Work {
            tol: 0.0,
            gathered: vec![C64::ZERO; form.cols.len()],
            product: vec![C64::ZERO; form.rows.len()],
            b: vec![C64::ZERO; form.dim],
            f: vec![C64::ZERO; form.dim],
            next: vec![C64::ZERO; form.dim],
            op,
        }
    }
}

/// The operator h·L′ on the sector (`Form::op_into`): per block −ih(H − c) and ih(H − c)† (the right factor), h R^T,
/// the shift's factor e^{2 Im(c) h} and the 1-norm's bound; and its making's scratch: the nodes' merged weights per H
/// slot and R group, H's combination, the norm's per-block sums.
#[derive(Default)]
struct Op {
    a1: Vec<Vec<C64>>,
    a2: Vec<Vec<C64>>,
    rt: Vec<C64>,
    eta: f64,
    norm: f64,
    wh: Vec<C64>,
    wr: Vec<C64>,
    hc: Vec<C64>,
    diag: Vec<Vec<C64>>,
    off: Vec<Vec<f64>>,
    colsum: Vec<f64>,
}

impl Op {
    /// Its buffers for the form's blocks, made where they are not yet.
    fn fit(&mut self, form: &Form) {
        if self.a1.len() == form.blocks.len() {
            return;
        }
        self.a1 = form.blocks.iter().map(|b| vec![C64::ZERO; b.len() * b.len()]).collect();
        self.a2 = self.a1.clone();
        self.diag = form.blocks.iter().map(|b| vec![C64::ZERO; b.len()]).collect();
        self.off = form.blocks.iter().map(|b| vec![0.0; b.len()]).collect();
        self.rt = vec![C64::ZERO; form.cols.len() * form.rows.len()];
        self.hc = vec![C64::ZERO; form.h_slot];
        self.colsum = vec![0.0; form.dim];
    }

    /// out = h L′ z: per pair (X, Y) −ih(H_X − c) Z + Z ih(H_Y − c)†, and R's recycling into the ground pairs.
    fn apply(&self, form: &Form, z: &[C64], out: &mut [C64], work: &mut Work) {
        for (k, &(x, y)) in form.pairs.iter().enumerate() {
            let (wx, wy) = (form.blocks[x].len(), form.blocks[y].len());
            let (o, e) = (form.offsets[k], form.offsets[k + 1]);
            fused_product(&self.a1[x], &z[o..e], (wx, wx, wy), &mut out[o..e]);
            fused_product_onto(&z[o..e], &self.a2[y], (wx, wy, wy), &mut out[o..e]);
        }
        let nr = form.rows.len();
        if nr > 0 {
            for (g, &c) in work.gathered.iter_mut().zip(&form.cols) {
                *g = z[c];
            }
            fused_product(&work.gathered, &self.rt, (1, form.cols.len(), nr), &mut work.product);
            for (p, &r) in work.product.iter().zip(&form.rows) {
                out[r] += *p;
            }
        }
    }

    /// e^{hL} v in place (Al-Mohy and Higham 2011, Algorithm 3.2, for the norm's bound): s stages of the Taylor series
    /// of e^{hL′/s} to the double's rounding, each times e^{2 Im(c) h/s}; the count of products made.
    fn exp(&self, form: &Form, v: &mut [C64], work: &mut Work) -> usize {
        let (m, s) = degree(self.norm);
        let vmax = |z: &[C64]| z.iter().map(|x| larger(x.re.abs(), x.im.abs())).fold(0.0, larger);
        let tol = larger(work.tol, 2f64.powi(-53) / 2f64.sqrt());
        let eta = self.eta.powf(1.0 / s as f64);
        let mut count = 0;
        let mut b = std::mem::take(&mut work.b);
        let mut f = std::mem::take(&mut work.f);
        let mut next = std::mem::take(&mut work.next);
        f.copy_from_slice(v);
        b.copy_from_slice(v);
        for _ in 0..s {
            let mut c1 = vmax(&b);
            for j in 0..m {
                self.apply(form, &b, &mut next, work);
                count += 1;
                let scale = 1.0 / (s * (j + 1)) as f64;
                for (x, y) in b.iter_mut().zip(&next) {
                    *x = *y * scale;
                }
                let c2 = vmax(&b);
                for (x, y) in f.iter_mut().zip(&b) {
                    *x += *y;
                }
                if c1 + c2 <= tol * vmax(&f) {
                    break;
                }
                c1 = c2;
            }
            for x in f.iter_mut() {
                *x *= eta;
            }
            b.copy_from_slice(&f);
        }
        v.copy_from_slice(&f);
        (work.b, work.f, work.next) = (b, f, next);
        count
    }
}

/// The memory a path's chunk of frozen pieces may hold (bytes; `Form::along_from`): 64 MB, a few chunks' at once on
/// the pool as the paths run side by side.
const FROZEN_BYTES: usize = 64 << 20;

/// A piece of a Lawson step (`Scheme::Lawson`): its middle t_m and start (µs), its substeps' length and count, and the
/// time of the reading at its end where the path is read there.
struct Piece {
    tm: f64,
    a: f64,
    d: f64,
    n: usize,
    read: Option<f64>,
}

/// A piece frozen (`Form::freeze`; module documentation, # The budget): its plan, its node's H amplitudes, H(t_m) and
/// per block its eigenvectors V and V⁻¹ with their adjoints, per sector entry the substep's half and whole exponentials,
/// R^T at each substep's middle (one where it holds no beat: constant over the piece), the reading's node at its end;
/// the largest eigenvector condition, and the seconds it took. It reads no state.
struct Frozen {
    a: f64,
    d: f64,
    n: usize,
    amps: Amps,
    hm: Vec<C64>,
    v: Vec<Vec<C64>>,
    vinv: Vec<Vec<C64>>,
    vadj: Vec<Vec<C64>>,
    vinvadj: Vec<Vec<C64>>,
    half: Vec<C64>,
    full: Vec<C64>,
    rs: Vec<Vec<C64>>,
    read: Option<(f64, Node)>,
    condition: f64,
    seconds: f64,
}

/// A path's Lawson scratch (`Scheme::Lawson`; `Form::lawson_run`): H(t), the beats' change −iΔH(t) and its right
/// factor per block, the stages' vectors.
struct Lawson {
    now: Vec<C64>,
    dh: Vec<Vec<C64>>,
    dha: Vec<Vec<C64>>,
    rho: Vec<C64>,
    tmp: Vec<C64>,
    bare: Vec<C64>,
    k: [Vec<C64>; 4],
    s: [Vec<C64>; 3],
    gathered: Vec<C64>,
    product: Vec<C64>,
}

impl Lawson {
    fn new(form: &Form) -> Lawson {
        let d = form.dim;
        let sq = |b: &Vec<usize>| vec![C64::ZERO; b.len() * b.len()];
        Lawson {
            now: vec![C64::ZERO; form.h_slot],
            dh: form.blocks.iter().map(sq).collect(),
            dha: form.blocks.iter().map(sq).collect(),
            rho: vec![C64::ZERO; d],
            tmp: vec![C64::ZERO; d],
            bare: vec![C64::ZERO; d],
            k: std::array::from_fn(|_| vec![C64::ZERO; d]),
            s: std::array::from_fn(|_| vec![C64::ZERO; d]),
            gathered: vec![C64::ZERO; form.cols.len()],
            product: vec![C64::ZERO; form.rows.len()],
        }
    }
}

impl Form<'_> {
    /// x_XY ← A_X x_XY B_Y for every sector pair (A, B per block, b × b row by row), through `tmp`.
    fn sandwich(&self, a: &[Vec<C64>], b: &[Vec<C64>], x: &[C64], out: &mut [C64], tmp: &mut [C64]) {
        for (k, &(px, py)) in self.pairs.iter().enumerate() {
            let (wx, wy) = (self.blocks[px].len(), self.blocks[py].len());
            let (o, e) = (self.offsets[k], self.offsets[k + 1]);
            fused_product(&a[px], &x[o..e], (wx, wx, wy), &mut tmp[o..e]);
            fused_product(&tmp[o..e], &b[py], (wx, wy, wy), &mut out[o..e]);
        }
    }

    /// H(t) on the blocks of the amplitudes Σ_j w_j a_j at their phases at t, into `out` (Σ b² entries).
    fn h_at(&self, of: &[(&Amps, f64)], t: f64, out: &mut [C64]) {
        out.fill(C64::ZERO);
        for (k, &beat) in self.h_beats.iter().enumerate() {
            let f = self.phase(beat, t);
            for &(a, w) in of {
                mul_add(false, out, f * w, &a.h[k * self.h_slot..(k + 1) * self.h_slot]);
            }
        }
    }

    /// R(t)^T of the amplitudes Σ_j w_j a_j, its groups at their phases at t, into `out`.
    fn r_at(&self, of: &[(&Amps, f64)], t: f64, out: &mut [C64]) {
        let size = out.len();
        out.fill(C64::ZERO);
        for (g, &delta) in self.beats_r.iter().enumerate() {
            let f = if delta == 0.0 { C64::ONE } else { C64::cis(-delta * t) };
            for &(a, w) in of {
                mul_add(false, out, f * w, &a.r[g * size..(g + 1) * size]);
            }
        }
    }

    /// The remainder N = L(t) − L₀ on an eigenbasis vector σ, into `out` (eigenbasis), of the frozen piece `f`, ΔH(t) =
    /// H(t) − H(t_m) (in `lw.dh` as −iΔH and `lw.dha` as its right factor iΔH†; none where `moving` is false) and R(t)
    /// (`rt`): ρ = VσV†, then −i(ΔHρ − ρΔH†) + Rρ, back by V⁻¹ · V⁻†.
    fn lawson_n(&self, f: &Frozen, rt: &[C64], lw: &mut Lawson, moving: bool, sigma: &[C64], out: &mut [C64]) {
        let (mut rho, mut tmp, mut bare) = (std::mem::take(&mut lw.rho), std::mem::take(&mut lw.tmp), std::mem::take(&mut lw.bare));
        self.sandwich(&f.v, &f.vadj, sigma, &mut rho, &mut tmp);
        if moving {
            for (k, &(px, py)) in self.pairs.iter().enumerate() {
                let (wx, wy) = (self.blocks[px].len(), self.blocks[py].len());
                let (o, e) = (self.offsets[k], self.offsets[k + 1]);
                fused_product(&lw.dh[px], &rho[o..e], (wx, wx, wy), &mut bare[o..e]);
                fused_product_onto(&rho[o..e], &lw.dha[py], (wx, wy, wy), &mut bare[o..e]);
            }
        } else {
            bare.fill(C64::ZERO);
        }
        let nr = self.rows.len();
        if nr > 0 {
            for (g, &c) in lw.gathered.iter_mut().zip(&self.cols) {
                *g = rho[c];
            }
            fused_product(&lw.gathered, rt, (1, self.cols.len(), nr), &mut lw.product);
            for (p, &r) in lw.product.iter().zip(&self.rows) {
                bare[r] += *p;
            }
        }
        self.sandwich(&f.vinv, &f.vinvadj, &bare, out, &mut tmp);
        (lw.rho, lw.tmp, lw.bare) = (rho, tmp, bare);
    }

    /// ΔH(t) = H(t) − H(t_m) per block into `lw.dh` (−iΔH) and `lw.dha` (iΔH†, the right factor), H(t) of `of`, H(t_m)
    /// `hm`.
    fn lawson_dh(&self, lw: &mut Lawson, hm: &[C64], of: &[(&Amps, f64)], t: f64) {
        let mut now = std::mem::take(&mut lw.now);
        self.h_at(of, t, &mut now);
        for (x, b) in self.blocks.iter().enumerate() {
            let w = b.len();
            let at = self.h_block[x];
            for i in 0..w {
                for j in 0..w {
                    let z = now[at + i * w + j] - hm[at + i * w + j];
                    lw.dh[x][i * w + j] = C64::new(0.0, -1.0) * z;
                    lw.dha[x][j * w + i] = C64::new(0.0, 1.0) * z.conj();
                }
            }
        }
        lw.now = now;
    }

    /// A piece frozen (`Frozen`): its node at t_m and its amplitudes, H(t_m) and its eigendecomposition per block (ρ̃ =
    /// V⁻¹ρV⁻†, e^{L₀c} diagonal there), the substep's exponentials, R^T at its substeps' middles, and the reading's
    /// node at its end. Err where an eigendecomposition fails.
    fn freeze(&self, walk: &Walk, p: &Piece) -> Result<Frozen, String> {
        let clock = std::time::Instant::now();
        let node = self.node_at(walk, p.tm);
        #[cfg(test)]
        traced(0, clock);
        #[cfg(test)]
        let part = std::time::Instant::now();
        let mut amps = self.amps_with(&node, false);
        drop(node);
        #[cfg(test)]
        traced(1, part);
        #[cfg(test)]
        let part = std::time::Instant::now();
        let mut hm = vec![C64::ZERO; self.h_slot];
        self.h_at(&[(&amps, 1.0)], p.tm, &mut hm);
        let nb = self.blocks.len();
        let (mut v, mut vinv, mut vadj, mut vinvadj) = (Vec::with_capacity(nb), Vec::with_capacity(nb), Vec::with_capacity(nb), Vec::with_capacity(nb));
        let mut values: Vec<Vec<C64>> = Vec::with_capacity(nb);
        let mut condition: f64 = 0.0;
        for (x, b) in self.blocks.iter().enumerate() {
            let w = b.len();
            let m = Mat { rows: w, cols: w, data: hm[self.h_block[x]..self.h_block[x] + w * w].to_vec() };
            let mut eig = Eigen::default();
            if !eig.decompose(&m) {
                return Err(format!("H's block {x} at {} µs has no eigendecomposition", p.tm));
            }
            condition = larger(condition, eig.condition);
            let (mut a, mut ai, mut aa, mut aia) = (vec![C64::ZERO; w * w], vec![C64::ZERO; w * w], vec![C64::ZERO; w * w], vec![C64::ZERO; w * w]);
            for i in 0..w {
                for j in 0..w {
                    let (p, q) = (eig.vectors[(i, j)], eig.inverse[(i, j)]);
                    a[i * w + j] = p;
                    ai[i * w + j] = q;
                    aa[j * w + i] = p.conj();
                    aia[j * w + i] = q.conj();
                }
            }
            v.push(a);
            vinv.push(ai);
            vadj.push(aa);
            vinvadj.push(aia);
            values.push(eig.values.clone());
        }
        #[cfg(test)]
        traced(2, part);
        #[cfg(test)]
        let part = std::time::Instant::now();
        // μ = −i(λ_a − conj λ_b) per sector entry, and the substep's exponentials.
        let (mut half, mut full) = (vec![C64::ZERO; self.dim], vec![C64::ZERO; self.dim]);
        for (k, &(px, py)) in self.pairs.iter().enumerate() {
            let wy = self.blocks[py].len();
            for (i, la) in values[px].iter().enumerate() {
                for (j, lb) in values[py].iter().enumerate() {
                    let at = self.offsets[k] + i * wy + j;
                    let mu = C64::new(0.0, -1.0) * (*la - lb.conj());
                    half[at] = (mu * (0.5 * p.d)).exp();
                    full[at] = (mu * p.d).exp();
                }
            }
        }
        // R^T at every substep's middle -- once where it holds no beat (constant over the piece) --, its amplitudes
        // dropped then.
        let still = !(self.beats_r.len() > 1 || self.beats_r.first().is_some_and(|x| *x != 0.0));
        let rs = (0..if still { 1 } else { p.n })
            .map(|j| {
                let mut rt = vec![C64::ZERO; self.cols.len() * self.rows.len()];
                self.r_at(&[(&amps, 1.0)], if still { p.tm } else { p.a + (j as f64 + 0.5) * p.d }, &mut rt);
                rt
            })
            .collect();
        amps.r = Vec::new();
        #[cfg(test)]
        traced(3, part);
        #[cfg(test)]
        let part = std::time::Instant::now();
        let read = p.read.map(|t| (t, self.node_at(walk, t)));
        #[cfg(test)]
        traced(4, part);
        Ok(Frozen { a: p.a, d: p.d, n: p.n, amps, hm, v, vinv, vadj, vinvadj, half, full, rs, read, condition, seconds: clock.elapsed().as_secs_f64() })
    }

    /// A frozen piece's substeps (`Scheme::Lawson`): the remainder N(t) -- the recycling R(t) (the piece's, at each
    /// substep's middle) and H's beats from t_m, of the piece's amplitudes at their phases -- by Lawson's RK4 in its n
    /// substeps of d, on the state `v`.
    fn lawson_run(&self, f: &Frozen, v: &mut [C64], lw: &mut Lawson, counts: &mut Counts) {
        let of = [(&f.amps, 1.0)];
        let d = f.d;
        let changing = self.h_beats.iter().any(|&b| self.values[b] != 0.0);
        // σ = V⁻¹ ρ V⁻†.
        let mut sigma = std::mem::take(&mut lw.s[0]);
        let mut tmp = std::mem::take(&mut lw.tmp);
        self.sandwich(&f.vinv, &f.vinvadj, v, &mut sigma, &mut tmp);
        lw.tmp = tmp;
        let mut k = std::mem::take(&mut lw.k);
        let [mut x1, mut x2] = [std::mem::take(&mut lw.s[1]), std::mem::take(&mut lw.s[2])];
        for step in 0..f.n {
            let t = f.a + step as f64 * d;
            let rt: &[C64] = &f.rs[if f.rs.len() == 1 { 0 } else { step }];
            // k1 = N(t)σ; x1 = E½(σ + d/2 k1); k2 = N(t + d/2) x1.
            if changing {
                self.lawson_dh(lw, &f.hm, &of, t);
            }
            self.lawson_n(f, rt, lw, changing, &sigma, &mut k[0]);
            for i in 0..sigma.len() {
                x1[i] = f.half[i] * (sigma[i] + k[0][i] * (0.5 * d));
            }
            if changing {
                self.lawson_dh(lw, &f.hm, &of, t + 0.5 * d);
            }
            self.lawson_n(f, rt, lw, changing, &x1, &mut k[1]);
            // x2 = E½σ + d/2 k2; k3 = N(t + d/2) x2.
            for i in 0..sigma.len() {
                x2[i] = f.half[i] * sigma[i] + k[1][i] * (0.5 * d);
            }
            self.lawson_n(f, rt, lw, changing, &x2, &mut k[2]);
            // x1 = E σ + d E½ k3; k4 = N(t + d) x1.
            for i in 0..sigma.len() {
                x1[i] = f.full[i] * sigma[i] + f.half[i] * k[2][i] * d;
            }
            if changing {
                self.lawson_dh(lw, &f.hm, &of, t + d);
            }
            self.lawson_n(f, rt, lw, changing, &x1, &mut k[3]);
            // σ ← E(σ + d/6 k1) + d/3 E½(k2 + k3) + d/6 k4.
            for i in 0..sigma.len() {
                sigma[i] = f.full[i] * (sigma[i] + k[0][i] * (d / 6.0)) + f.half[i] * (k[1][i] + k[2][i]) * (d / 3.0) + k[3][i] * (d / 6.0);
            }
            counts.products += 4;
            counts.segments += 1;
        }
        // ρ = V σ V†.
        let mut tmp = std::mem::take(&mut lw.tmp);
        self.sandwich(&f.v, &f.vadj, &sigma, v, &mut tmp);
        lw.tmp = tmp;
        lw.k = k;
        (lw.s[0], lw.s[1], lw.s[2]) = (sigma, x1, x2);
        counts.exponentials += 1;
    }
}

/// The generator's memory at a point (module documentation, # The memory): its slowest nonzero decay rate (1/µs), the
/// stationary modes' count (the rounding zeros), and the memory time 1/λ.
#[derive(Clone, Copy, Debug)]
pub struct Memory {
    pub rate: f64,
    pub zeros: usize,
    pub time: f64,
    /// The next three nonzero decay rates, increasing (a complex mode and its partner each listed): the fast memory.
    pub rates: [f64; 3],
}

impl Memory {
    /// A path's length for a run of the length `run` (µs): min(T_run, 5 τ_mem).
    pub fn length(&self, run: f64) -> f64 {
        run.min(5.0 * self.time)
    }
}

/// The memory of a model at the point r, κ and the time t (`Memory`): None where the sector's Liouvillian does not
/// decompose.
pub fn memory(form: &Form, r: V3, kappa: V3, t: f64) -> Option<Memory> {
    let l = form.liouvillian(r, kappa, t);
    let d = l.rows;
    let mut eigen = Eigen::default();
    if !eigen.decompose(&l) {
        return None;
    }
    let scale = rounding(d) * l.norm1();
    let (mut zeros, mut rate) = (0, f64::INFINITY);
    let mut all: Vec<f64> = Vec::new();
    for (i, mu) in eigen.values.iter().enumerate() {
        // κ_μ = ‖r_i‖₂ ‖l_i‖₂ of its right vector (R's column) and left one (R⁻¹'s row), l_i r_i = 1.
        let right: f64 = (0..d).map(|x| eigen.vectors[(x, i)].norm_sqr()).sum::<f64>().sqrt();
        let left: f64 = (0..d).map(|x| eigen.inverse[(i, x)].norm_sqr()).sum::<f64>().sqrt();
        if mu.abs() <= right * left * scale {
            zeros += 1;
        } else {
            rate = rate.min(-mu.re);
            all.push(-mu.re);
        }
    }
    all.sort_by(f64::total_cmp);
    // The next three nonzero rates (a complex mode's partner holds its rate: L(ρ†) = L(ρ)†).
    let rates = std::array::from_fn(|k| all.get(k + 1).copied().unwrap_or(f64::NAN));
    Some(Memory { rate, zeros, time: 1.0 / rate, rates })
}

/// A path mean and its standard error over the paths (the paths' spread over √P; NaN of one path).
#[derive(Clone, Copy, Debug, Default)]
pub struct Mean {
    pub value: f64,
    pub error: f64,
}

impl Mean {
    fn of(x: &[f64]) -> Mean {
        let p = x.len() as f64;
        let value = x.iter().sum::<f64>() / p;
        let var = x.iter().map(|y| (y - value) * (y - value)).sum::<f64>() / (p - 1.0);
        Mean { value, error: (var / p).sqrt() }
    }
}

/// A candidate's direct error over one time window (module documentation): per quantity its window mean's difference
/// to the reference on each path, and those over the paths (`Mean`).
#[derive(Clone, Debug, Default)]
pub struct Window {
    pub from: f64,
    pub to: f64,
    pub paths: usize,
    /// Every displayed row; each level's sum (its states' labels without the last part); the worst row of the means.
    pub rows: Vec<Mean>,
    pub levels: Vec<(String, Mean)>,
    pub worst: (usize, Mean),
    /// Per path its worst row's window mean |Δ| (for the distribution and the account).
    pub worst_per_path: Vec<f64>,
    /// Per photon class (the reference's, by fast index) its rate's difference (1/µs), the reference's mean rate, and
    /// the total rate's relative difference.
    pub rates: Vec<([i64; 2], Mean, f64)>,
    pub rate: Mean,
    /// The force's difference per lab axis (rad/µs/µm), the rms difference over the rms force per path, and
    /// ⅓ Σ_a ⟨|δF_a||v_a|⟩ (rad/µs², the feedback's drive: PATHGAP item 4).
    pub force: [Mean; 3],
    pub force_rms: Mean,
    pub power: Mean,
}

/// The windows of a path of the length `length` (µs): (0, e₁], (e₁, 2e₁], … doubling up to the length, e₁ the
/// memory time halved down to the first below four times the path's `step` -- the start's transient resolved where it
/// changes fastest, the error's approach to its stationary value (the memory's scale) in octaves; the last window ends
/// at the length.
pub fn windows(length: f64, memory: f64, step: f64) -> Vec<(f64, f64)> {
    let mut e = memory;
    while e > 4.0 * step {
        e *= 0.5;
    }
    let mut edges = vec![0.0];
    while e < length {
        edges.push(e);
        e *= 2.0;
    }
    edges.push(length);
    edges.windows(2).map(|x| (x[0], x[1])).collect()
}

/// The window an output time takes its error from (PIPELINE_V3_FINAL B.8: "the time window it falls in"): the window
/// (from, to] holding t; past the last window's end the last window -- a path of L = 5τ_mem has forgotten its start
/// there, the error stationary --; None at or before the first window's start (every model starts in the same state).
pub fn window_of(windows: &[(f64, f64)], t: f64) -> Option<usize> {
    if windows.is_empty() || t <= windows[0].0 {
        return None;
    }
    Some(windows.iter().position(|w| t <= w.1).unwrap_or(windows.len() - 1))
}

/// The candidate's direct errors against the reference on the same paths (`Window` per window `(from, to]`):
/// `candidate[p]` and `reference[p]` read along path p at the same times; `labels` the request's states. A window's
/// statistics are over the paths read in it.
pub fn compare(candidate: &[Series], reference: &[Series], windows: &[(f64, f64)], labels: &[String], hbar_over_mass: f64) -> Vec<Window> {
    let mut levels: Vec<(String, Vec<usize>)> = Vec::new();
    for (s, l) in labels.iter().enumerate() {
        let level = l.rsplit_once('-').map_or(l.as_str(), |x| x.0).trim_end_matches('-').to_string();
        match levels.iter_mut().find(|x| x.0 == level) {
            Some(x) => x.1.push(s),
            None => levels.push((level, vec![s])),
        }
    }
    // The photon classes of the first path that reads any (a path that ended before the windows reads none).
    let classes = &reference.iter().find(|s| !s.readings.is_empty()).unwrap_or(&reference[0]).classes;
    windows
        .iter()
        .map(|&(from, to)| {
            let p = candidate.len();
            let mut rows = vec![vec![0.0; p]; STATES];
            let mut lv = vec![vec![0.0; p]; levels.len()];
            let mut rates = vec![vec![0.0; p]; classes.len()];
            let mut rate_ref = vec![0.0; classes.len()];
            let mut rate = vec![0.0; p];
            let mut force = vec![[0.0; 3]; p];
            let mut force_rms = vec![0.0; p];
            let mut power = vec![0.0; p];
            let mut read = vec![false; p];
            for (k, (c, r)) in candidate.iter().zip(reference).enumerate() {
                let pairs: Vec<(&Reading, &Reading)> = c.readings.iter().zip(&r.readings).filter(|(x, _)| x.t > from && x.t <= to).collect();
                assert!(pairs.iter().all(|(x, y)| x.t == y.t), "the two series read at the same times");
                let q = pairs.len() as f64;
                read[k] = !pairs.is_empty();
                let (mut df2, mut f2, mut total_m, mut total_r) = (0.0, 0.0, 0.0, 0.0);
                for (x, y) in &pairs {
                    for s in 0..STATES {
                        rows[s][k] += (x.rows[s] - y.rows[s]) / q;
                    }
                    for (l, (_, states)) in levels.iter().enumerate() {
                        lv[l][k] += states.iter().map(|&s| x.rows[s] - y.rows[s]).sum::<f64>() / q;
                    }
                    for (ci, fast) in classes.iter().enumerate() {
                        let mine = c.classes.iter().position(|f| f == fast).map_or(0.0, |i| x.rates[i]);
                        rates[ci][k] += (mine - y.rates[ci]) / q;
                        rate_ref[ci] += y.rates[ci] / q / p as f64;
                    }
                    total_m += x.rates.iter().sum::<f64>() / q;
                    total_r += y.rates.iter().sum::<f64>() / q;
                    for a in 0..3 {
                        let d = x.force[a] - y.force[a];
                        force[k][a] += d / q;
                        df2 += d * d / q;
                        f2 += y.force[a] * y.force[a] / q;
                        power[k] += d.abs() * (hbar_over_mass * y.kappa[a]).abs() / (3.0 * q);
                    }
                }
                rate[k] = total_m / total_r - 1.0;
                force_rms[k] = (df2 / f2).sqrt();
            }
            // The paths read in the window (a lost atom's path ends early).
            let held: Vec<usize> = (0..p).filter(|&k| read[k]).collect();
            let of = |x: &[f64]| Mean::of(&held.iter().map(|&k| x[k]).collect::<Vec<_>>());
            let row_means: Vec<Mean> = rows.iter().map(|x| of(x)).collect();
            let worst_row = (0..STATES).max_by(|&a, &b| row_means[a].value.abs().total_cmp(&row_means[b].value.abs())).unwrap();
            let worst_per_path = held.iter().map(|&k| rows.iter().map(|x| x[k].abs()).fold(0.0, larger)).collect();
            Window {
                from,
                to,
                paths: held.len(),
                worst: (worst_row, row_means[worst_row]),
                rows: row_means,
                levels: levels.iter().zip(&lv).map(|((name, _), x)| (name.clone(), of(x))).collect(),
                worst_per_path,
                rates: classes.iter().zip(&rates).zip(&rate_ref).map(|((f, x), r)| (*f, of(x), *r * p as f64 / held.len() as f64)).collect(),
                rate: of(&rate),
                force: std::array::from_fn(|a| of(&force.iter().map(|f| f[a]).collect::<Vec<_>>())),
                force_rms: of(&force_rms),
                power: of(&power),
            }
        })
        .collect()
}

/// The direct errors of candidate models along the same paths (module documentation; PIPELINE_V3_FINAL B.8): every
/// model -- the reference first -- advanced along every path from its start state on [from, to] by one scheme, on the
/// pool (the dearest jobs first); per candidate its windows (`compare`) and per model its series.
pub struct PathErrors {
    pub series: Vec<Vec<Series>>,
    pub windows: Vec<Vec<Window>>,
}

/// The path errors (`PathErrors`) of `candidates` against `reference` on the walks, each from `start(walk)` over
/// [from, to]; Err where a model does not hold a walk's start.
#[allow(clippy::too_many_arguments)]
pub fn path_errors(reference: &Form, candidates: &[&Form], walks: &[Walk], start: &(dyn Fn(&Walk) -> Vec<(usize, C64)> + Sync), from: f64, to: f64, windows: &[(f64, f64)], scheme: Scheme) -> Result<PathErrors, String> {
    let models: Vec<&Form> = std::iter::once(reference).chain(candidates.iter().copied()).collect();
    let mut jobs: Vec<(usize, usize)> = (0..models.len()).flat_map(|m| (0..walks.len()).map(move |w| (m, w))).collect();
    jobs.sort_by(|a, b| models[b.0].dim.cmp(&models[a.0].dim));
    let done: Vec<((usize, usize), Result<Series, String>)> = jobs.par_iter().map(|&(m, w)| ((m, w), models[m].along(&walks[w], &start(&walks[w]), from, to.min(walks[w].end()), scheme))).collect();
    let mut series: Vec<Vec<Series>> = vec![vec![Series::default(); walks.len()]; models.len()];
    for ((m, w), s) in done {
        series[m][w] = s?;
    }
    let labels = &reference.model.scene.request.states;
    let hm = reference.hbar_over_mass;
    let windows = (1..models.len()).map(|m| compare(&series[m], &series[0], windows, labels, hm)).collect();
    Ok(PathErrors { series, windows })
}

/// The h-against-h/2 check of a scheme on one path piece (module documentation, # The segments, # The budget): the
/// candidate's difference to the reference read at the scheme and at its halves -- `Sparse`: twice the nodes and half
/// the turn; `Lawson`: twice the pieces and the substeps --, the largest change of a row's, a rate's (relative to the
/// reference's) and the force's (relative to the reference's rms) over the readings; None for `Sparse` where the coarser
/// scheme's segments may leave the plateau's side (turn > 2π) -- the comparison reads nothing there -- and for the
/// fixed schemes.
pub fn check(candidate: &Form, reference: &Form, walk: &Walk, start: &[(usize, C64)], from: f64, to: f64, scheme: Scheme) -> Option<Result<[f64; 3], String>> {
    let halves = match scheme {
        Scheme::Sparse { nodes, turn, read } if turn <= std::f64::consts::TAU => Scheme::Sparse { nodes: 2 * nodes, turn: 0.5 * turn, read },
        Scheme::Lawson { per_us, pieces, read } => Scheme::Lawson { per_us: 2.0 * per_us, pieces: 2 * pieces, read },
        _ => return None,
    };
    let run = |s: Scheme| -> Result<(Series, Series), String> { Ok((candidate.along(walk, start, from, to, s)?, reference.along(walk, start, from, to, s)?)) };
    Some((|| {
        let (c1, r1) = run(scheme)?;
        let (c2, r2) = run(halves)?;
        let mut most = [0.0_f64; 3];
        let f2: f64 = r2.readings.iter().map(|x| x.force.iter().map(|f| f * f).sum::<f64>()).sum::<f64>() / r2.readings.len() as f64;
        for (((a, b), c), d) in c1.readings.iter().zip(&r1.readings).zip(&c2.readings).zip(&r2.readings) {
            for s in 0..STATES {
                most[0] = most[0].max(((a.rows[s] - b.rows[s]) - (c.rows[s] - d.rows[s])).abs());
            }
            let (ra, rb, rc, rd) = (a.rates.iter().sum::<f64>(), b.rates.iter().sum::<f64>(), c.rates.iter().sum::<f64>(), d.rates.iter().sum::<f64>());
            most[1] = most[1].max(((ra - rb) - (rc - rd)).abs() / rd);
            let df: f64 = (0..3).map(|x| ((a.force[x] - b.force[x]) - (c.force[x] - d.force[x])).powi(2)).sum::<f64>();
            most[2] = most[2].max((df / f2).sqrt());
        }
        Ok(most)
    })())
}

/// The freezes' parts' nanoseconds over every thread (the measurements' trace, select.rs `traced`): the node, its
/// amplitudes, the eigendecompositions, the exponentials and R, the reading's node.
#[cfg(test)]
pub(crate) static FREEZE: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];

/// A freeze's part `k` (`FREEZE`) since `from`.
#[cfg(test)]
fn traced(k: usize, from: std::time::Instant) {
    FREEZE[k].fetch_add(from.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
}

/// The freezes' parts (`FREEZE`) in seconds, and zeroed.
#[cfg(test)]
pub(crate) fn freeze_parts() -> [f64; 5] {
    std::array::from_fn(|k| FREEZE[k].swap(0, std::sync::atomic::Ordering::Relaxed) as f64 * 1e-9)
}

#[cfg(test)]
thread_local! {
    /// What a test asks of the pilot's paths (solve.rs `walks`: how many, to when) and what it was handed.
    pub(crate) static WALKS: std::cell::RefCell<Option<(usize, f64, Vec<Walk>)>> = const { std::cell::RefCell::new(None) };
}

/// The pilot's paths to a test that asks for them (`WALKS`; solve.rs `attempt`).
#[cfg(test)]
pub(crate) fn hand_out(walks: &dyn Fn(usize, f64) -> Vec<Walk>) {
    WALKS.with(|w| {
        if let Some((count, until, out)) = w.borrow_mut().as_mut() {
            *out = walks(*count, *until);
        }
    });
}

#[cfg(test)]
mod tests;
