//! The one elimination (ELIMINATION.md §2–§6 with its v2 and v3; SPEC §1.4–§1.5): the excited states the light does not
//! need explicit, folded onto the slow states at a point -- exactly in the static part, to first order in the frequency
//! offsets the motion and the explicit beats bring --, the numbers the two criteria and the model's error read, and the
//! frames.
//!
//! # The system
//! The ground states G (at rest: their frame is the lab's) and the excited states E (in the frame where they turn at
//! ν_frame, request.rs); H0 has no G–E entry and neither has any light: a dipole couples G and E alone. The light is a set
//! of classes (`Class`): one optical frequency, co- or counter-rotating, at ν_c in the excited states' frame; its beams are
//! one coherent field at the point (`Beam`: each beam's coupling E ← G at its amplitude there, with its wave vector).
//! Raman couplings of two ground states, light shifts, the photons' coherences: products of the elimination, none a
//! unit of its own.
//!
//! # P, Q and the frames
//! P is G and the explicit excited states E_x (`Slow`), each state s with its frame θ_s: ψ_s = e^{−iθ_s t} x_s. An entry
//! a ← b that turns as e^{−iΩt} in the lab turns as e^{−i(Ω + θ_b − θ_a)t} in the frames -- its residual beat; it is
//! static where that is zero. A P state's amplitude drives the excited states at a frequency of its own: a ground state s
//! through class c at f = ν_c + θ_s, an explicit state through the excited block's own static couplings at f = θ_s. Each
//! such frequency is a replica (`Replicas`): the excited states' amplitude at it is one Floquet copy, Q_f = E less the
//! explicit states of frame f (those are P at that frequency). A drive of class c into an explicit state of another frame
//! is folded at its replica like any other: the explicit state is a Q state at that frequency.
//!
//! # Groups (gauge invariance)
//! The frames are a gauge: a constant added to the frames of every state of one component of the kept links (a group,
//! `Slow::groups`; one cluster of the run) changes no physics, and no decision may change with it. So the static part is
//! a group's alone: the direct entries of two states of one group and one frame, the explicit drives of one group at
//! residual zero, and the second-order entries through a replica of one group -- a replica is a frequency of one group
//! (`Replicas::of`), its Q the excited states less that group's explicit states of that frame. What couples two groups
//! -- through replicas of the two whatever their frames, or a direct entry -- is a P-layer component at its lab beat,
//! judged by criterion two on the actual spectrum like any other (δ = Ω + λ_p − λ_q, the same in every gauge: a
//! group's constant moves its replicas' frequencies and its eigenvalues oppositely). Its eigenstates are each one
//! group's. Before, two groups at equal frames shared replicas, so that which of their couplings were static -- and
//! never judged -- depended on the frames' gauge (the tilted 1D PGC's F = 2 Raman pair of mixing 0.228, kept in one
//! gauge and hidden in another), and an explicit block took drives from both ground hyperfine levels, 6.8 GHz apart
//! (S3: one cluster of 20 states, its static generator spanning 43 411 rad/µs).
//!
//! # The fold (Bloch: exact in the static part)
//! The static generator on P in the frames is H = A + Σ_f W'_f T_f, A the direct static P entries (H0's static entries,
//! the explicit drives at residual zero, −θ on the diagonal, the explicit states' −(i/2)D), W_f (Q_f ← P) the couplings
//! into replica f and W'_f (P ← Q_f) back, and T_f (Q_f ← P) the response: the solution of the Sylvester equation
//!   T_f H − (K_f − f) T_f = W_f,   K_f = (H_E − (i/2)D) on Q_f,
//! that is, in H's eigenbasis, T_f r_p = (λ_p + f − K_f)⁻¹ W_f r_p: each eigenstate of P at its own eigenvalue, and the
//! eigenvalues self-consistent (`fold`). With it (x, T x) is an invariant subspace of the static Floquet generator and H
//! its generator (the Bloch equation; Löwdin: the λ_p are exact eigenvalues of the static generator) -- whatever the
//! saturation, where the fold exists (the iteration contracts). The bare levels' fold and the fold at the atom's levels
//! are this one function taken at a coarser point; there is no other form.
//!
//! # The definition and the route it is evaluated on (`Fold::series`)
//! The self-consistent fold above is the definition, and the reference the criteria and the errors are read against. A
//! run evaluates the same function by its series about a reference z₀: the resolvent's Neumann series
//! R(z) = Σ_k (z₀ − z)^k R₀^{k+1}, R₀ = (z₀ + f − K)⁻¹, with z the P generator H as a right factor, gives
//! T_f = Σ_k (−1)^k R₀^{k+1} W_f (H − z₀)^k and H = A + Σ_k (−1)^k N_k (H − z₀)^k, N_k = Σ_f W′_f R₀^{k+1} W_f -- one
//! function, not a second form. Its radius is the distance d of z₀ + f from the excited block's spectrum; its remainder
//! after order K is within q^{K+1}/(1 − q), q = ‖H − z₀‖₂/d (`Fold::series_ratio`), and the order is the least whose
//! remainder is within the share (criterion one's ρ books it). The reference is the P levels with the far light alone
//! and one constant c of the whole run, the middle of the near light's scalar shift over the prior's atoms: no field in
//! it, the N_k are bilinear in the near fields with coefficients of the far light's coordinates alone -- a table's
//! dimensions are the far light's. (PGC3D's F = 2 at its cooling light's coherent peak: about the far levels alone
//! q = 0.548; with the centre c, q = 0.0056, the series' order 0 within 7.9e-5 of the fold, order 1 within 4.0e-7.)
//! Where no order reaches the share (q near one) the block is made explicit by criterion one.
//!
//! # First order in the frequency offsets
//! A P amplitude whose frequency is λ_p + Δ -- the Doppler shift of a beam, k·v, an explicit beat -- drives
//! (λ_p + Δ + f − K)⁻¹ W r_p = T r_p − Δ T2 r_p + Δ² R² R(λ + Δ) W r_p, T2 the solution of T2 H − (K − f) T2 = T
//! (−∂T/∂f). The first two terms are the model's; the third is the remainder whose relative size ρ(Δ) is the first
//! criterion (`Fold::remainder`).
//!
//! # The criteria's numbers (`Fold::components`, `Fold::photons`)
//! - The remainder ρ_p(Δ) = ‖Δ² R(λ_p)² R(λ_p + Δ) W r_p‖ / ‖R(λ_p) W r_p‖ of each P eigenstate, the replicas added in
//!   squares (different frequencies), with each excited state's part of it: the excited block that leaves the most is
//!   the one a partition makes explicit (criterion one).
//! - The P-layer components: two replicas f ≠ f' make the entries W'_{f'} T_f of P, turning at f − f' less the
//!   two beams' Doppler beat (k_l − k_l')·v; in H's eigenbasis an entry of size a across the distance
//!   δ = Ω + λ_p − λ_p' mixes a/|δ| (criterion two, v5: its error averaged is 2(a/δ)² -- the population's time mean after the light
//!   is switched on, 4(a/δ)² sin²(δt/2) --, explicit where that is above the share, else averaged into its static shift
//!   |a|²/δ).
//! - The photons' cross terms: the jumps C_a T_f of two replicas land in the same ground states at two photon
//!   frequencies; their interference turns at the difference, and of a rate γ it moves the populations by γ/Ω (the
//!   secular approximation, first order: kept above the share).
use crate::linalg::{schur, Eigen, Mat, C64};
use crate::units::rounding;

/// The atom's levels at a point (rad/µs, 1/√µs): the ground block in the lab frame, the excited block in the excited
/// states' frame, the decay D = Σ_a C_a†C_a on the excited states, and the emission C_a (ground ← excited).
#[derive(Clone, Debug)]
pub(crate) struct Levels {
    pub ground: Mat,
    pub excited: Mat,
    pub decay: Mat,
    pub emission: [Mat; 3],
}

/// One beam of a class at the point: its coupling (excited ← ground, its amplitude there in it) and its wave vector
/// (1/µm; a counter-rotating class's beam takes −k).
#[derive(Clone, Debug)]
pub(crate) struct Beam {
    pub coupling: Mat,
    pub wave: [f64; 3],
}

/// One class of the light: an optical frequency, co- or counter-rotating, ν (rad/µs, in the excited states' frame), its
/// beams one coherent field.
#[derive(Clone, Debug)]
pub(crate) struct Class {
    pub nu: f64,
    pub beams: Vec<Beam>,
}

/// P beside the ground states: the explicit excited states (indices into E), the frame θ_s of every P state (rad/µs;
/// the ground states first, then the explicit ones in their order), and its group: the component of the kept links it
/// lies in (module documentation, # Groups; one cluster's states are one group).
#[derive(Clone, Debug)]
pub(crate) struct Slow {
    pub explicit: Vec<usize>,
    pub frames: Vec<f64>,
    pub groups: Vec<usize>,
}

impl Slow {
    /// The P of one cluster: all its states one group.
    pub(crate) fn single(explicit: Vec<usize>, frames: Vec<f64>) -> Slow {
        let groups = vec![0; frames.len()];
        Slow { explicit, frames, groups }
    }
}

/// The replicas (module documentation, # P, Q and the frames; # Groups): their frequencies f and groups, the replica
/// each class drives each ground state's amplitude into, and the one each explicit state's static couplings drive. Made
/// of exact frequencies by the caller; `Replicas::of` groups equal values of one group.
#[derive(Clone, Debug)]
pub(crate) struct Replicas {
    pub frequencies: Vec<f64>,
    pub groups: Vec<usize>,
    /// [class][ground state].
    pub driven: Vec<Vec<usize>>,
    /// [explicit state].
    pub held: Vec<usize>,
}

/// Whether two frequencies are one: equal within the rounding of the sums that make them -- a class's frequency and a
/// frame, itself a sum of at most a component per state along the frames' paths (`frames`): γ_STATES of their sizes.
/// Two physically different frequencies differ by a beat of the request's frequency axes, many orders more.
pub(crate) fn same(a: f64, b: f64) -> bool {
    (a - b).abs() <= rounding(crate::request::STATES) * (a.abs() + b.abs())
}

impl Replicas {
    /// The replicas of frequencies that are one (`same`) within one group.
    pub(crate) fn of(classes: &[Class], slow: &Slow, ground: usize) -> Replicas {
        let (mut frequencies, mut groups): (Vec<f64>, Vec<usize>) = (vec![], vec![]);
        let mut place = |f: f64, group: usize| match frequencies.iter().zip(&groups).position(|(&x, &g)| g == group && same(x, f)) {
            Some(at) => at,
            None => {
                frequencies.push(f);
                groups.push(group);
                frequencies.len() - 1
            }
        };
        let driven = classes.iter().map(|c| (0..ground).map(|g| place(c.nu + slow.frames[g], slow.groups[g])).collect()).collect();
        let held = (0..slow.explicit.len()).map(|i| place(slow.frames[ground + i], slow.groups[ground + i])).collect();
        Replicas { frequencies, groups, driven, held }
    }
}

/// What drives one replica: a class's beam from the ground states (`class` Some), or the excited block's static
/// couplings from the explicit states of its frame (`class` None, its wave vector zero); its coupling W (Q_f ← P), the
/// coupling back W' (P ← Q_f), and its coupling back from every excited state.
#[derive(Clone, Debug)]
pub(crate) struct Feed {
    pub class: Option<usize>,
    pub wave: [f64; 3],
    pub w: Mat,
    pub back: Mat,
    /// The coupling back from every excited state (P ← E): what the feed's light does to an excited amplitude at another
    /// replica (a P-layer component's emitted leg).
    pub emitted: Mat,
}

/// One replica folded: its frequency, Q_f, the sum of its feeds' W and W', its response T (module documentation).
#[derive(Clone, Debug)]
pub(crate) struct Folded {
    pub frequency: f64,
    /// Its group (an index into the fold's `members`, # Groups): its feeds' places, where its equation is solved.
    pub group: usize,
    pub q: Vec<usize>,
    pub feeds: Vec<Feed>,
    pub w: Mat,
    pub back: Mat,
    pub t: Mat,
    /// K_f = (H_E − (i/2)D) on Q_f and its Schur form.
    kq: Mat,
    k: Schur,
}

/// The fold at a point (module documentation): P's static generator H in the frames, the replicas, and H's
/// eigendecomposition (the eigenvalues λ_p, the right eigenvectors r_p as unit columns, the left ones as the inverse's
/// rows). `exists` is whether the self-consistent iteration contracted to its rounding: where it does not, the fold does
/// not exist at this point (a resonance the partition has to make explicit).
#[derive(Clone, Debug)]
pub(crate) struct Fold {
    pub ground: usize,
    pub explicit: Vec<usize>,
    pub frames: Vec<f64>,
    /// A: the direct static entries.
    pub direct: Mat,
    pub generator: Mat,
    pub replicas: Vec<Folded>,
    pub eigen: Eigen,
    pub exists: bool,
    pub iterations: usize,
    /// The reference z₀ where the fold is its series' order 0 about it (`fold`'s `reference`), none where it is the
    /// self-consistent fold.
    pub reference: Option<f64>,
    /// The direct entries of P that turn in the frames: (row, column, residual beat Ω, value), the entry ∝ e^{−iΩt}.
    pub turning: Vec<(usize, usize, f64, C64)>,
    /// Per class and ground state the replica (an index into `replicas`) its drive feeds.
    driven: Vec<Vec<Option<usize>>>,
    /// The groups' places (# Groups), each eigenstate's group, and per group the Schur form of the generator's block.
    pub members: Vec<Vec<usize>>,
    pub groups: Vec<usize>,
    schurs: Vec<Schur>,
}

/// A Schur form A = Q S Q†, block by block (`Schur::of`): per block of the states A's entries join, its states, its
/// place in the form's order and its own S (upper triangular, row by row) and Q (column by column, on its states); the
/// whole S row by row in the form's order (block-diagonal).
#[derive(Clone, Debug)]
struct Schur {
    n: usize,
    blocks: Vec<SchurBlock>,
    s: Vec<C64>,
}

#[derive(Clone, Debug)]
struct SchurBlock {
    states: Vec<usize>,
    start: usize,
    s: Vec<C64>,
    q: Vec<C64>,
}

impl Schur {
    /// The Schur form of a square matrix, block by block: the states its entries join (the connected components of its
    /// pattern, a ↔ b where either entry is not zero) are decomposed apart (linalg.rs `schur`) and laid one after another
    /// -- block-diagonal and each block upper triangular is upper triangular, Q block-diagonal on the blocks' states is
    /// unitary --: exactly the whole one's form, its work the blocks' (the excited block's hyperfine and magnetic
    /// sublevels that nothing couples), and so every product with it.
    fn of(a: &Mat) -> Option<Schur> {
        let n = a.rows;
        let mut block: Vec<usize> = (0..n).collect();
        for i in 0..n {
            for j in 0..n {
                if a[(i, j)] != C64::ZERO {
                    let (x, y) = (block[i], block[j]);
                    if x != y {
                        block.iter_mut().filter(|z| **z == y).for_each(|z| *z = x);
                    }
                }
            }
        }
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&i| (block[i], i));
        let (mut blocks, mut dense) = (vec![], vec![C64::ZERO; n * n]);
        let mut at = 0;
        while at < n {
            let end = (at..n).find(|&k| block[order[k]] != block[order[at]]).unwrap_or(n);
            let states = order[at..end].to_vec();
            let m = states.len();
            let mut h: Vec<C64> = (0..m * m).map(|x| a[(states[x / m], states[x % m])]).collect();
            let (mut q, mut v, mut sums) = (vec![C64::ZERO; m * m], vec![C64::ZERO; m], vec![C64::ZERO; m]);
            schur(&mut h, &mut q, &mut v, &mut sums, m)?;
            for x in 0..m {
                for y in 0..m {
                    dense[(at + x) * n + at + y] = h[x * m + y];
                }
            }
            blocks.push(SchurBlock { states, start: at, s: h, q });
            at = end;
        }
        Some(Schur { n, blocks, s: dense })
    }
    /// The Schur form of the adjoint from this one: A† = Q S† Q†, S† lower triangular, so with each block's columns of Q
    /// in reversed order it is upper triangular -- no decomposition.
    fn adjoint(&self) -> Schur {
        let n = self.n;
        let mut dense = vec![C64::ZERO; n * n];
        let blocks = self
            .blocks
            .iter()
            .map(|b| {
                let m = b.states.len();
                let s: Vec<C64> = (0..m * m).map(|x| b.s[(m - 1 - x % m) * m + (m - 1 - x / m)].conj()).collect();
                let q: Vec<C64> = (0..m * m).map(|x| b.q[(m - 1 - x / m) * m + x % m]).collect();
                for x in 0..m {
                    for y in 0..m {
                        dense[(b.start + x) * n + b.start + y] = s[x * m + y];
                    }
                }
                SchurBlock { states: b.states.clone(), start: b.start, s, q }
            })
            .collect();
        Schur { n, blocks, s: dense }
    }
    /// Q† x (x with n rows), into the form's order.
    fn left_in(&self, x: &Mat) -> Mat {
        let mut out = Mat::zeros(self.n, x.cols);
        for b in &self.blocks {
            let m = b.states.len();
            for k in 0..m {
                for (i, &state) in b.states.iter().enumerate() {
                    let c = b.q[k * m + i].conj();
                    if c != C64::ZERO {
                        for j in 0..x.cols {
                            let v = x.data[state * x.cols + j];
                            out.data[(b.start + k) * x.cols + j] += c * v;
                        }
                    }
                }
            }
        }
        out
    }
    /// Q y.
    fn left_out(&self, y: &Mat) -> Mat {
        let mut out = Mat::zeros(self.n, y.cols);
        for b in &self.blocks {
            let m = b.states.len();
            for (i, &state) in b.states.iter().enumerate() {
                for k in 0..m {
                    let c = b.q[k * m + i];
                    if c != C64::ZERO {
                        for j in 0..y.cols {
                            let v = y.data[(b.start + k) * y.cols + j];
                            out.data[state * y.cols + j] += c * v;
                        }
                    }
                }
            }
        }
        out
    }
    /// x Q (x with n columns), into the form's order.
    fn right_in(&self, x: &Mat) -> Mat {
        let mut out = Mat::zeros(x.rows, self.n);
        for b in &self.blocks {
            let m = b.states.len();
            for r in 0..x.rows {
                for k in 0..m {
                    out.data[r * self.n + b.start + k] = b.states.iter().enumerate().map(|(i, &state)| x.data[r * x.cols + state] * b.q[k * m + i]).sum();
                }
            }
        }
        out
    }
    /// y Q†.
    fn right_out(&self, y: &Mat) -> Mat {
        let mut out = Mat::zeros(y.rows, self.n);
        for b in &self.blocks {
            let m = b.states.len();
            for r in 0..y.rows {
                for (i, &state) in b.states.iter().enumerate() {
                    out.data[r * self.n + state] = (0..m).map(|k| y.data[r * y.cols + b.start + k] * b.q[k * m + i].conj()).sum();
                }
            }
        }
        out
    }
    /// (σ − S)⁻¹ y in place (back substitution, block by block).
    fn shifted_solve(&self, sigma: C64, y: &mut [C64]) {
        for b in &self.blocks {
            let m = b.states.len();
            for row in (0..m).rev() {
                let mut acc = y[b.start + row];
                for k in row + 1..m {
                    acc += b.s[row * m + k] * y[b.start + k];
                }
                y[b.start + row] = acc / (sigma - b.s[row * m + row]);
            }
        }
    }
    /// (σ − A)⁻¹ v.
    fn resolvent(&self, sigma: C64, v: &[C64]) -> Vec<C64> {
        let column = Mat { rows: self.n, cols: 1, data: v.to_vec() };
        let mut y = self.left_in(&column).data;
        self.shifted_solve(sigma, &mut y);
        self.left_out(&Mat { rows: self.n, cols: 1, data: y }).data
    }
}

/// T (H + r) − (L + l) T = W for the Schur forms L = Q_L S_L Q_L† and H = Q_H S_H Q_H† and scalars l, r
/// (Bartels–Stewart): Y = Q_L† T Q_H solves Y (S_H + r) − (S_L + l) Y = Q_L† W Q_H column by column,
/// ((S_H)_jj + r − l − S_L) Y_j = F_j − Σ_{i<j} (S_H)_ij Y_i.
fn solve_sylvester(left: &Schur, l: C64, right: &Schur, r: C64, w: &Mat) -> Mat {
    if left.n == 0 {
        return Mat::zeros(0, w.cols);
    }
    let fq = right.right_in(&left.left_in(w));
    let n = right.n;
    let mut y = Mat::zeros(left.n, n);
    let mut column = vec![C64::ZERO; left.n];
    for j in 0..n {
        for m in 0..left.n {
            let mut acc = fq.data[m * n + j];
            for i in 0..j {
                let c = right.s[i * n + j];
                if c != C64::ZERO {
                    acc -= c * y.data[m * n + i];
                }
            }
            column[m] = acc;
        }
        left.shifted_solve(right.s[j * n + j] + r - l, &mut column);
        for m in 0..left.n {
            y.data[m * n + j] = column[m];
        }
    }
    left.left_out(&right.right_out(&y))
}

/// A replica's Sylvester equation T H − (K − f) T = W (module documentation, # The fold) on its own group's places
/// `places` (# Groups): H's block there
/// (its Schur form `h`), W's columns there; T is nothing on the other groups' places -- no coupling of the replica
/// reaches them (by construction: a replica is a group's), so they are not solved for.
fn sylvester_in(k: &Schur, h: &Schur, f: f64, w: &Mat, places: &[usize]) -> Mat {
    solve_in(k, C64::real(-f), h, C64::ZERO, w, places)
}

/// `solve_sylvester` on a group's places `places` (`sylvester_in`): the right factor's Schur form `right` the group's
/// block, W's columns there, the solution nothing elsewhere.
fn solve_in(left: &Schur, l: C64, right: &Schur, r: C64, w: &Mat, places: &[usize]) -> Mat {
    let rows: Vec<usize> = (0..w.rows).collect();
    let part = solve_sylvester(left, l, right, r, &w.block(&rows, places));
    let mut out = Mat::zeros(w.rows, w.cols);
    for i in 0..w.rows {
        for (j, &x) in places.iter().enumerate() {
            out[(i, x)] = part[(i, j)];
        }
    }
    out
}

/// Per group (`members`) the Schur form of a matrix's block on its places.
fn schurs_of(m: &Mat, members: &[Vec<usize>]) -> Option<Vec<Schur>> {
    members.iter().map(|places| Schur::of(&m.block(places, places))).collect()
}

fn difference(a: &Mat, b: &Mat) -> f64 {
    a.data.iter().zip(&b.data).map(|(x, y)| (*x - *y).norm_sqr()).sum::<f64>().sqrt()
}

/// The fixed point of x ← step(x) from x₀ (a step gives its value and the scale of its terms), and whether it exists.
/// It exists where the iteration's first contraction -- the change of the second step over the first's -- is at most a
/// quarter: the iteration's factor at the start is the derivative of the fold in its argument at the unperturbed
/// spectra, W'R²W, the excited share m² of the states it folds (for one line exactly |T|²), and Stewart's theorem
/// has the invariant subspace, and the iteration contracting to it, where m = ‖W‖/sep < ½ (ELIMINATION.md §3).
/// Beyond it (false) the fold is not this point's -- a slow and a fast mode come together (a two-level atom on
/// resonance at s = ½, its exceptional point, starts at exactly a quarter) -- and the partition makes the excited
/// block explicit. Where it exists the iteration goes on to its floor: until a change is within the rounding of a
/// step's operations (`terms`) times the scale, or no longer falls. Its iterations with it.
fn settle(start: &Mat, mut step: impl FnMut(&Mat) -> (Mat, f64), terms: usize) -> (Mat, usize, bool) {
    let (mut x, mut last, mut iterations, mut first) = (start.clone(), f64::INFINITY, 0, None);
    loop {
        iterations += 1;
        let (next, scale) = step(&x);
        let change = difference(&next, &x);
        x = next;
        if change <= rounding(terms) * scale {
            return (x, iterations, true);
        }
        match first {
            None => first = Some(change),
            Some(f) if iterations == 2 && change > 0.25 * f => return (x, iterations, false),
            _ => {}
        }
        if !(change < last) {
            return (x, iterations, true);
        }
        last = change;
    }
}

/// The fold at a point (module documentation, # The fold): the replicas' couplings, then the self-consistent
/// iteration H ← A + Σ_f W'_f T_f(H) from H = A (`settle`: `Fold::exists` false where it does not contract within the
/// fold's existence bound) -- or, about a `reference` z₀, its series' order 0: one step from H = z₀, T_f = R₀ W_f and
/// H = A + Σ_f W'_f R₀ W_f (module documentation, # The definition and the route it is evaluated on; the run's own
/// evaluation, what its states are read on).
pub(crate) fn fold(levels: &Levels, classes: &[Class], slow: &Slow, replicas: &Replicas) -> Fold {
    fold_about(levels, classes, slow, replicas, None)
}

/// `fold`, about a reference where one is given.
pub(crate) fn fold_about(levels: &Levels, classes: &[Class], slow: &Slow, replicas: &Replicas, reference: Option<f64>) -> Fold {
    let ground = levels.ground.rows;
    let nx = slow.explicit.len();
    let n = ground + nx;
    let excited = levels.excited.rows;
    let k_full = {
        let mut k = levels.excited.clone();
        k.add_scaled(C64::new(0.0, -0.5), &levels.decay);
        k
    };
    let frame = |s: usize| slow.frames[s];
    let together = |x: usize, y: usize| slow.groups[x] == slow.groups[y];
    // The groups' places, in the order they first come.
    let mut members: Vec<Vec<usize>> = vec![];
    for s in 0..n {
        match members.iter_mut().find(|m| slow.groups[m[0]] == slow.groups[s]) {
            Some(m) => m.push(s),
            None => members.push(vec![s]),
        }
    }
    let group_of = |label: usize| members.iter().position(|m| slow.groups[m[0]] == label).expect("a group's places");
    // A: the direct static entries of a group; the turning ones and those of two groups apart (# Groups).
    let mut a = Mat::zeros(n, n);
    let mut turning = vec![];
    let mut put = |a: &mut Mat, row: usize, column: usize, value: C64| {
        if value == C64::ZERO {
            return;
        }
        if together(row, column) && same(frame(column), frame(row)) {
            a[(row, column)] += value;
        } else {
            turning.push((row, column, frame(column) - frame(row), value));
        }
    };
    for r in 0..ground {
        for c in 0..ground {
            put(&mut a, r, c, levels.ground[(r, c)]);
        }
    }
    for (i, &x) in slow.explicit.iter().enumerate() {
        for (j, &y) in slow.explicit.iter().enumerate() {
            put(&mut a, ground + i, ground + j, k_full[(x, y)]);
        }
    }
    for s in 0..n {
        a[(s, s)] -= C64::real(frame(s));
    }
    // The explicit drives of a group at residual zero are A's; the others are folded at their replica (an explicit
    // state of another frame or another group is Q there).
    for class in classes {
        for beam in &class.beams {
            for (i, &x) in slow.explicit.iter().enumerate() {
                for g in 0..ground {
                    let v = beam.coupling[(x, g)];
                    if v != C64::ZERO && together(g, ground + i) && same(class.nu + frame(g), frame(ground + i)) {
                        a[(ground + i, g)] += v;
                        a[(g, ground + i)] += v.conj();
                    }
                }
            }
        }
    }
    // The replicas: Q_f and the feeds; one Schur form per distinct Q (K_f is K on Q_f, whatever f).
    let mut forms: Vec<(Vec<usize>, Mat, Schur)> = vec![];
    let mut folded: Vec<Folded> = replicas
        .frequencies
        .iter()
        .zip(&replicas.groups)
        .map(|(&f, &group)| {
            let q: Vec<usize> = (0..excited).filter(|&e| !slow.explicit.iter().enumerate().any(|(i, &x)| x == e && slow.groups[ground + i] == group && same(frame(ground + i), f))).collect();
            if !forms.iter().any(|x| x.0 == q) {
                let kq = k_full.block(&q, &q);
                let k = Schur::of(&kq).expect("the Schur form of the excited block");
                forms.push((q.clone(), kq, k));
            }
            let (_, kq, k) = forms.iter().find(|x| x.0 == q).expect("its form");
            Folded { frequency: f, group: group_of(group), q, feeds: vec![], w: Mat::zeros(0, 0), back: Mat::zeros(0, 0), t: Mat::zeros(0, 0), kq: kq.clone(), k: k.clone() }
        })
        .collect();
    for (c, class) in classes.iter().enumerate() {
        for beam in &class.beams {
            for (r, rep) in folded.iter_mut().enumerate() {
                let columns: Vec<usize> = (0..ground).filter(|&g| replicas.driven[c][g] == r).collect();
                if columns.is_empty() {
                    continue;
                }
                let w = Mat::from_fn(rep.q.len(), n, |i, s| if columns.contains(&s) { beam.coupling[(rep.q[i], s)] } else { C64::ZERO });
                if w.max_abs() == 0.0 {
                    continue;
                }
                let back = w.adjoint();
                let emitted = Mat::from_fn(n, excited, |s, e| if columns.contains(&s) { beam.coupling[(e, s)].conj() } else { C64::ZERO });
                rep.feeds.push(Feed { class: Some(c), wave: beam.wave, w, back, emitted });
            }
        }
    }
    for (r, rep) in folded.iter_mut().enumerate() {
        let columns: Vec<usize> = (0..nx).filter(|&i| replicas.held[i] == r).collect();
        if columns.is_empty() {
            continue;
        }
        let w = Mat::from_fn(rep.q.len(), n, |i, s| if s >= ground && columns.contains(&(s - ground)) { k_full[(rep.q[i], slow.explicit[s - ground])] } else { C64::ZERO });
        let back = Mat::from_fn(n, rep.q.len(), |s, i| if s >= ground && columns.contains(&(s - ground)) { k_full[(slow.explicit[s - ground], rep.q[i])] } else { C64::ZERO });
        if w.max_abs() == 0.0 && back.max_abs() == 0.0 {
            continue;
        }
        let emitted = Mat::from_fn(n, excited, |s, e| if s >= ground && columns.contains(&(s - ground)) { k_full[(slow.explicit[s - ground], e)] } else { C64::ZERO });
        rep.feeds.push(Feed { class: None, wave: [0.0; 3], w, back, emitted });
    }
    let kept: Vec<usize> = (0..folded.len()).filter(|&r| !folded[r].feeds.is_empty()).collect();
    folded.retain(|rep| !rep.feeds.is_empty());
    let driven: Vec<Vec<Option<usize>>> = replicas.driven.iter().map(|row| row.iter().map(|&r| kept.iter().position(|&k| k == r)).collect()).collect();
    for rep in folded.iter_mut() {
        let (mut w, mut back) = (Mat::zeros(rep.q.len(), n), Mat::zeros(n, rep.q.len()));
        for feed in &rep.feeds {
            w.add_scaled(C64::ONE, &feed.w);
            back.add_scaled(C64::ONE, &feed.back);
        }
        rep.w = w;
        rep.back = back;
    }
    // The self-consistent iteration: H ← A + Σ_f W'_f T_f(H), to the rounding of a Sylvester solution's entries (sums of
    // n_P n_Q products) times the terms' scale; the fold exists where it gets there, contracting.
    let mut step = |h: &Mat| {
        let sh = schurs_of(h, &members).expect("the Schur form of P's generator");
        let (mut next, mut scale) = (a.clone(), a.frobenius());
        for rep in folded.iter_mut() {
            rep.t = sylvester_in(&rep.k, &sh[rep.group], rep.frequency, &rep.w, &members[rep.group]);
            next.add_scaled(C64::ONE, &rep.back.product(&rep.t));
            scale += rep.back.frobenius() * rep.t.frobenius();
        }
        (next, scale)
    };
    let (h, iterations, exists) = match reference {
        None => settle(&a, step, (n + excited) * (n + excited)),
        Some(z0) => (step(&Mat::from_fn(n, n, |i, j| if i == j { C64::real(z0) } else { C64::ZERO })).0, 1, true),
    };
    // The responses at the generator they made it of, at its final form (about a reference: R₀ W, as made).
    let sh = schurs_of(&h, &members).expect("the Schur form of P's generator");
    if reference.is_none() {
        for rep in folded.iter_mut() {
            rep.t = sylvester_in(&rep.k, &sh[rep.group], rep.frequency, &rep.w, &members[rep.group]);
        }
    }
    // H is a group's alone (# Groups): its eigenstates are each one group's, decomposed group by group -- two groups'
    // equal eigenvalues mix no eigenvector of one with the other's.
    let mut eigen = Eigen::default();
    let mut groups = vec![0; n];
    if members.len() == 1 {
        eigen.decompose(&h);
    } else {
        let (mut values, mut vectors, mut inverse) = (vec![C64::ZERO; n], Mat::zeros(n, n), Mat::zeros(n, n));
        let (mut condition, mut work) = (0.0_f64, 0.0);
        let mut next = 0;
        for m in &members {
            let mut part = Eigen::default();
            part.decompose(&h.block(m, m));
            for (k, _) in m.iter().enumerate() {
                values[next + k] = part.values[k];
                groups[next + k] = members.iter().position(|x| std::ptr::eq(x, m)).expect("its group");
                for (i, &x) in m.iter().enumerate() {
                    vectors[(x, next + k)] = part.vectors[(i, k)];
                    inverse[(next + k, x)] = part.inverse[(k, i)];
                }
            }
            (condition, work, next) = (condition.max(part.condition), work + part.work, next + m.len());
        }
        (eigen.values, eigen.vectors, eigen.inverse, eigen.condition, eigen.work) = (values, vectors, inverse, condition, work);
    }
    Fold { ground, explicit: slow.explicit.clone(), frames: slow.frames.clone(), direct: a, generator: h, replicas: folded, eigen, exists, iterations, reference, turning, driven, members, groups, schurs: sh }
}

/// One P-layer component's reading (criterion two): its size a in H's eigenbasis, its distance δ = Ω + λ_p − λ_p',
/// its residual beat Ω, the eigenstates (p' ← p) and what turns it (two classes, or a turning direct entry).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Component {
    pub row: usize,
    pub column: usize,
    pub size: f64,
    pub distance: C64,
    pub beat: f64,
    pub classes: (Option<usize>, Option<usize>),
}

impl Component {
    /// Its first-order mixing a/|δ|.
    pub(crate) fn mixing(&self) -> f64 {
        if self.size == 0.0 {
            0.0
        } else {
            self.size / self.distance.abs()
        }
    }
}

/// A cross term of two photon frequencies (criterion two for the jumps): from eigenstate `from`, landing in eigenstates
/// `landing` (a, b), its size γ, the beat Ω between the two photons, and the two replicas.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Cross {
    pub from: usize,
    pub landing: (usize, usize),
    pub size: f64,
    pub beat: f64,
    pub classes: (Option<usize>, Option<usize>),
}

impl Fold {
    /// P's size.
    pub(crate) fn size(&self) -> usize {
        self.generator.rows
    }
    /// The right eigenvector r_p (unit) and the left one l_p (l_p† r_q = δ_pq) of H.
    pub(crate) fn right(&self, p: usize) -> Vec<C64> {
        (0..self.size()).map(|i| self.eigen.vectors[(i, p)]).collect()
    }
    pub(crate) fn left(&self, p: usize) -> Vec<C64> {
        (0..self.size()).map(|i| self.eigen.inverse[(p, i)].conj()).collect()
    }
    /// Its ground weight ‖P_G r_p‖².
    pub(crate) fn ground_weight(&self, p: usize) -> f64 {
        (0..self.ground).map(|i| self.eigen.vectors[(i, p)].norm_sqr()).sum()
    }
    /// The P state an eigenstate holds most of.
    pub(crate) fn dominant(&self, p: usize) -> usize {
        (0..self.size()).max_by(|&a, &b| self.eigen.vectors[(a, p)].norm_sqr().total_cmp(&self.eigen.vectors[(b, p)].norm_sqr())).unwrap_or(0)
    }

    /// The remainder of the first-order expansion of eigenstate p's response across a shift Δ of its frequency
    /// (module documentation, # First order): ρ_p(Δ), and per replica and excited state (E's index) its part of the
    /// remainder's square over the response's square -- the parts sum to ρ_p².
    pub(crate) fn remainder(&self, p: usize, shift: f64, excited: usize) -> (f64, Vec<Vec<f64>>) {
        let r = self.right(p);
        let lambda = self.eigen.values[p];
        let mut parts = vec![vec![0.0; excited]; self.replicas.len()];
        let (mut num, mut den) = (0.0, 0.0);
        for (j, rep) in self.replicas.iter().enumerate().filter(|(_, rep)| rep.group == self.groups[p]) {
            let wr = rep.w.apply(&r);
            let sigma = lambda + C64::real(rep.frequency);
            let x0 = rep.k.resolvent(sigma, &wr);
            let shifted = rep.k.resolvent(sigma + C64::real(shift), &wr);
            let second = rep.k.resolvent(sigma, &x0);
            for (i, &e) in rep.q.iter().enumerate() {
                let d = shifted[i] - x0[i] + second[i] * shift;
                parts[j][e] += d.norm_sqr();
                num += d.norm_sqr();
                den += x0[i].norm_sqr();
            }
        }
        if den == 0.0 {
            return (0.0, vec![vec![0.0; excited]; self.replicas.len()]);
        }
        parts.iter_mut().flatten().for_each(|x| *x /= den);
        ((num / den).sqrt(), parts)
    }

    /// Replica j's response to a P state of the eigen-coefficients `coefficients` (each eigenstate at its own eigenvalue,
    /// T x = Σ_p c_p R(λ_p + f) W r_p), feed by feed: per feed its class (None: the explicit states' own couplings) and
    /// its part R(λ_p + f) W_feed r_p summed on Q_f.
    pub(crate) fn feed_responses(&self, j: usize, coefficients: &[C64]) -> Vec<(Option<usize>, Vec<C64>)> {
        let rep = &self.replicas[j];
        rep.feeds
            .iter()
            .map(|feed| {
                let mut v = vec![C64::ZERO; rep.q.len()];
                for (p, &c) in coefficients.iter().enumerate().filter(|&(p, _)| self.groups[p] == rep.group) {
                    let x = rep.k.resolvent(self.eigen.values[p] + C64::real(rep.frequency), &feed.w.apply(&self.right(p)));
                    v.iter_mut().zip(&x).for_each(|(a, b)| *a += c * *b);
                }
                (feed.class, v)
            })
            .collect()
    }

    /// The P-layer components of two different replicas at the velocity v (µm/µs), per pair of feeds, in H's
    /// eigenbasis (module documentation, # The criteria's numbers), and the turning direct entries; the entries no
    /// larger than the rounding of their sums left out.
    pub(crate) fn components(&self, velocity: [f64; 3]) -> Vec<Component> {
        let n = self.size();
        let mut out = vec![];
        let lambda = &self.eigen.values;
        let doppler = |k: [f64; 3]| k[0] * velocity[0] + k[1] * velocity[1] + k[2] * velocity[2];
        let eigenbasis = |m: &Mat| self.eigen.inverse.product(&m.product(&self.eigen.vectors));
        for (i, rep) in self.replicas.iter().enumerate() {
            for feed in &rep.feeds {
                let response = sylvester_in(&rep.k, &self.schurs[rep.group], rep.frequency, &feed.w, &self.members[rep.group]);
                for (j, other) in self.replicas.iter().enumerate() {
                    if i == j {
                        continue;
                    }
                    for back in &other.feeds {
                        // The other replica's light emitted from this one's excited amplitude: its coupling back from Q_f.
                        let m = Mat::from_fn(n, n, |r, c| rep.q.iter().enumerate().map(|(y, &e)| back.emitted[(r, e)] * response[(y, c)]).sum());
                        if m.max_abs() == 0.0 {
                            continue;
                        }
                        let beat = rep.frequency - other.frequency - (doppler(feed.wave) - doppler(back.wave));
                        let e = eigenbasis(&m);
                        let floor = rounding(n + rep.q.len()) * back.emitted.frobenius() * response.frobenius();
                        for p in 0..n {
                            for q in 0..n {
                                let size = e[(q, p)].abs();
                                if size <= floor {
                                    continue;
                                }
                                out.push(Component { row: q, column: p, size, distance: C64::real(beat) + lambda[p] - lambda[q], beat, classes: (back.class, feed.class) });
                            }
                        }
                    }
                }
            }
        }
        for &(row, column, beat, value) in &self.turning {
            let m = Mat::from_fn(n, n, |r, c| if r == row && c == column { value } else { C64::ZERO });
            let e = eigenbasis(&m);
            for p in 0..n {
                for q in 0..n {
                    let size = e[(q, p)].abs();
                    if size > 0.0 {
                        out.push(Component { row: q, column: p, size, distance: C64::real(beat) + lambda[p] - lambda[q], beat, classes: (None, None) });
                    }
                }
            }
        }
        out
    }

    /// The photons of eigenstate p per replica (and of the explicit states' own decay, at their frames): per photon
    /// frequency the vectors u_a = C_a T_f r_p on the ground states.
    pub(crate) fn photon_vectors(&self, levels: &Levels, p: usize) -> Vec<(f64, Option<usize>, [Vec<C64>; 3])> {
        let r = self.right(p);
        let mut out: Vec<(f64, Option<usize>, [Vec<C64>; 3])> = vec![];
        for rep in &self.replicas {
            let x = rep.t.apply(&r);
            let u: [Vec<C64>; 3] = std::array::from_fn(|a| (0..self.ground).map(|g| rep.q.iter().enumerate().map(|(i, &e)| levels.emission[a][(g, e)] * x[i]).sum()).collect());
            out.push((rep.frequency, rep.feeds.iter().find_map(|f| f.class), u));
        }
        for (i, &e) in self.explicit.iter().enumerate() {
            let f = self.frames[self.ground + i];
            let u: [Vec<C64>; 3] = std::array::from_fn(|a| (0..self.ground).map(|g| levels.emission[a][(g, e)] * r[self.ground + i]).collect());
            match out.iter_mut().find(|x| same(x.0, f)) {
                Some(x) => (0..3).for_each(|a| x.2[a].iter_mut().zip(&u[a]).for_each(|(s, v)| *s += *v)),
                None => out.push((f, None, u)),
            }
        }
        out
    }

    /// The cross terms of two photon frequencies from every eigenstate (module documentation): landing in eigenstates
    /// a and b (their left vectors' ground parts), γ = |Σ_a ⟨l_a|u_{f,a}⟩ ⟨l_b|u_{f',a}⟩*|, Ω = f − f' − (λ_a − λ_b).
    pub(crate) fn photons(&self, levels: &Levels) -> Vec<Cross> {
        let n = self.size();
        let mut out = vec![];
        let landing: Vec<Vec<C64>> = (0..n).map(|q| self.left(q)[..self.ground].to_vec()).collect();
        for p in 0..n {
            let vectors = self.photon_vectors(levels, p);
            for (i, (f, first, u)) in vectors.iter().enumerate() {
                for (j, (g, second, v)) in vectors.iter().enumerate() {
                    if i == j {
                        continue;
                    }
                    for a in 0..n {
                        for b in 0..n {
                            let s: C64 = (0..3).map(|k| crate::linalg::inner(&landing[a], &u[k]) * crate::linalg::inner(&landing[b], &v[k]).conj()).sum();
                            if s == C64::ZERO {
                                continue;
                            }
                            let beat = f - g - (self.eigen.values[a].re - self.eigen.values[b].re);
                            out.push(Cross { from: p, landing: (a, b), size: s.abs(), beat, classes: (*first, *second) });
                        }
                    }
                }
            }
        }
        out
    }

    /// The normalization of P's coordinates (ELIMINATION.md §2: the first order's renormalization, here whole): the
    /// slow state (x, T x) has the norm x†(1 + X)x, so its normalized coordinates are y = S x, S = (1 + X)^{1/2}; returns
    /// S and S⁻¹ (X Hermitian and positive: its eigenbasis).
    pub(crate) fn normalization(&self) -> (Mat, Mat) {
        let x = crate::linalg::hermitian_eigen(&self.metric());
        let n = self.size();
        let make = |f: &dyn Fn(f64) -> f64| Mat::from_fn(n, n, |i, j| (0..n).map(|k| x.vectors[(i, k)] * f(x.values[k].max(0.0)) * x.vectors[(j, k)].conj()).sum());
        (make(&|v| (1.0 + v).sqrt()), make(&|v| 1.0 / (1.0 + v).sqrt()))
    }

    /// The landing map in the normalized coordinates: a bare P state f lands at y = S x, x = (1 + Σ_f S_f T_f)⁻¹ f
    /// (`landed`): the matrix S (1 + Σ S_f T_f)⁻¹. None where the left iteration does not settle.
    pub(crate) fn landing(&self, s: &Mat) -> Option<Mat> {
        let left = self.left_operators()?;
        let n = self.size();
        let mut out = Mat::zeros(n, n);
        for j in 0..n {
            let f: Vec<C64> = (0..n).map(|i| if i == j { C64::ONE } else { C64::ZERO }).collect();
            let x = self.landed_with(&left, &f)?;
            let y = s.apply(&x);
            for i in 0..n {
                out[(i, j)] = y[i];
            }
        }
        Some(out)
    }

    /// The fold's series about a reference z₀ (the engine's evaluation route; the self-consistent fold is the
    /// definition): the resolvent's Neumann series R(z) = Σ_k (z₀ − z)^k R₀^{k+1}, R₀ = (z₀ + f − K)⁻¹, with z the P
    /// generator H itself (a right factor, as in the Sylvester equation) gives the wave operator
    /// T_f = Σ_k (−1)^k R₀^{k+1} W_f (H − z₀)^k and the generator H = A + Σ_k (−1)^k N_k (H − z₀)^k,
    /// N_k = Σ_f W′_f R₀^{k+1} W_f. The N_k (and the jumps' C R₀^{k+1} W) are bilinear in the fields with coefficients
    /// of the reference levels alone: what a table holds without the near light's coordinates. Returns N_0..N_K.
    pub(crate) fn series(&self, z0: C64, orders: usize) -> Vec<Mat> {
        let n = self.size();
        let mut out = vec![Mat::zeros(n, n); orders + 1];
        for rep in &self.replicas {
            let sigma = z0 + C64::real(rep.frequency);
            let mut x = rep.w.clone();
            for form in out.iter_mut() {
                for &j in &self.members[rep.group] {
                    let column: Vec<C64> = (0..x.rows).map(|i| x[(i, j)]).collect();
                    let solved = rep.k.resolvent(sigma, &column);
                    for i in 0..x.rows {
                        x[(i, j)] = solved[i];
                    }
                }
                form.add_scaled(C64::ONE, &rep.back.product(&x));
            }
        }
        out
    }

    /// The generator of the series to order K (`series`): the fixed point of H ← A + Σ_k (−1)^k N_k (H − z₀)^k from A
    /// (`settle`, on n × n matrices alone); None where it does not settle.
    pub(crate) fn series_generator(&self, forms: &[Mat], z0: C64) -> Option<Mat> {
        let n = self.size();
        let (h, _, exists) = settle(
            &self.direct,
            |h| {
                let mut e = h.clone();
                for i in 0..n {
                    e[(i, i)] -= z0;
                }
                let (mut next, mut power, mut scale) = (self.direct.clone(), Mat::identity(n), self.direct.frobenius());
                for (k, form) in forms.iter().enumerate() {
                    let term = form.product(&power);
                    scale += term.frobenius();
                    next.add_scaled(C64::real(if k % 2 == 0 { 1.0 } else { -1.0 }), &term);
                    power = power.product(&e);
                }
                (next, scale)
            },
            n * n * forms.len(),
        );
        exists.then_some(h)
    }

    /// The series' ratio q = ‖H − z₀‖₂ / d (the spectral norm, the largest singular value; d the least distance of
    /// z₀ + f from the excited block's spectrum over the replicas, 1/‖R₀‖ where the excited block is normal): its
    /// remainder after order K is within q^{K+1}/(1 − q) of the fold, relatively, where q < 1 (the Neumann series'
    /// radius) -- the order an evaluation takes is the least whose remainder is within the share (criterion one's ρ
    /// books it).
    pub(crate) fn series_ratio(&self, z0: C64) -> f64 {
        self.series_norm(z0) / self.series_distance(z0)
    }
    /// ‖H − z₀‖₂ (`series_ratio`).
    pub(crate) fn series_norm(&self, z0: C64) -> f64 {
        let mut e = self.generator.clone();
        for i in 0..self.size() {
            e[(i, i)] -= z0;
        }
        crate::linalg::hermitian_eigen(&e.adjoint_product(&e)).values.iter().fold(0.0_f64, |m, &x| m.max(x)).sqrt()
    }
    /// The least distance of z₀ + f from the excited block's spectrum over the replicas (`series_ratio`).
    pub(crate) fn series_distance(&self, z0: C64) -> f64 {
        self.replicas.iter().flat_map(|rep| (0..rep.k.n).map(move |i| (z0 + C64::real(rep.frequency) - rep.k.s[i * rep.k.n + i]).abs())).fold(f64::INFINITY, f64::min)
    }

    /// The excited share X = Σ_f T_f†T_f (the metric of P's coordinates: the slow state (x, T x) has the norm
    /// x†(1 + X)x).
    pub(crate) fn metric(&self) -> Mat {
        let n = self.size();
        let mut x = Mat::zeros(n, n);
        for rep in &self.replicas {
            x.add_scaled(C64::ONE, &rep.t.adjoint_product(&rep.t));
        }
        x
    }

    /// The left wave operators S_f (P ← Q_f): the rows (y, y S) span the slow left invariant subspace,
    /// (y, yS)[[A, W'], [W, K − f]] = λ(y, yS) gives M S_f − S_f (K_f − f) = W'_f with M = A + Σ_f S_f W_f -- the right
    /// one's Sylvester for the adjoint (S† M† − (K† − f) S† = W'†), settled the same way (`settle`). Without decay S = T†.
    /// With them the fast invariant subspace is (−S y, y), y on Q. None where the left iteration does not settle.
    pub(crate) fn left_operators(&self) -> Option<Vec<Mat>> {
        let n = self.size();
        let adjoints: Vec<Schur> = self.replicas.iter().map(|rep| rep.k.adjoint()).collect();
        let mut s: Vec<Mat> = self.replicas.iter().map(|rep| Mat::zeros(n, rep.q.len())).collect();
        let excited = self.replicas.iter().map(|rep| rep.q.len()).max().unwrap_or(1);
        // About a reference, the series' order 0: S_f = W′_f R₀, one step from M = z₀.
        if let Some(z0) = self.reference {
            let sm = schurs_of(&Mat::from_fn(n, n, |i, j| if i == j { C64::real(z0) } else { C64::ZERO }), &self.members).expect("a scalar's Schur form");
            for ((rep, k), si) in self.replicas.iter().zip(&adjoints).zip(s.iter_mut()) {
                *si = sylvester_in(k, &sm[rep.group], rep.frequency, &rep.back.adjoint(), &self.members[rep.group]).adjoint();
            }
            return Some(s);
        }
        // From the right generator, which the left one is similar to, M = (1 + ST) H (1 + ST)⁻¹: within the excited share
        // X of it.
        let (_, _, settled) = settle(
            &self.generator,
            |m| {
                let sm = schurs_of(&m.adjoint(), &self.members).expect("the Schur form of P's left generator");
                let (mut next, mut scale) = (self.direct.clone(), self.direct.frobenius());
                for ((rep, k), si) in self.replicas.iter().zip(&adjoints).zip(s.iter_mut()) {
                    *si = sylvester_in(k, &sm[rep.group], rep.frequency, &rep.back.adjoint(), &self.members[rep.group]).adjoint();
                    next.add_scaled(C64::ONE, &si.product(&rep.w));
                    scale += si.frobenius() * rep.w.frobenius();
                }
                (next, scale)
            },
            (n + excited) * (n + excited),
        );
        settled.then_some(s)
    }

    /// The slow part of a bare P state f (its Q amplitudes zero: a jump's landing, the state when the light is switched
    /// on): the spectral projection of (f, 0) on the slow invariant subspace (x, T x) along the fast one (−S y, y):
    /// x − S y = f, y = −T x, so x = (1 + Σ_f S_f T_f)⁻¹ f (Π = (1; T)(1 + ST)⁻¹(1, S) is the projector, Π² = Π).
    pub(crate) fn landed(&self, f: &[C64]) -> Option<Vec<C64>> {
        let s = self.left_operators()?;
        self.landed_with(&s, f)
    }
    fn landed_with(&self, s: &[Mat], f: &[C64]) -> Option<Vec<C64>> {
        let mut overlap = Mat::identity(self.size());
        for (si, rep) in s.iter().zip(&self.replicas) {
            overlap.add_scaled(C64::ONE, &si.product(&rep.t));
        }
        let lu = crate::linalg::Lu::new(&overlap)?;
        let mut x = f.to_vec();
        lu.solve(&mut x);
        Some(x)
    }

    /// The transient of a bare state f (ELIMINATION v3 3 and 4, one term): what the waiting time for the next photon
    /// from f holds beyond its slow part's, E_exact − ∫‖slow‖² dt. The bare state is its slow part (x₀, T x₀) and its
    /// fast part (−S y₀, y₀), y₀ = −T x₀ (`landed`), the fast part the homogeneous solution of the excited block,
    /// e^{−i(K − f)t} -- it decays at the excited states' Γ/2 within 1/Γ, and it is the same term whether the bare state
    /// is a jump's landing or the light being switched on. Its survival adds ∫‖fast‖² + 2 Re∫⟨fast|slow⟩; to the
    /// leading order in the fold (the fast generator K − f and the fast part's P amplitude −S y₀ against the slow
    /// one's), per replica,
    ///   τ = (T x₀)† 𝓛 (T x₀) − 2 Re (T x₀)† 𝒳 x₀,
    /// 𝓛 = ∫ e^{iK†t} e^{−iKt} dt (𝓛 K − K† 𝓛 = −i; 𝓛 = 1/Γ on a line of one width) and 𝒳 = ∫ e^{i(K−f)†t} (T − S†)
    /// e^{−iHt} dt (𝒳 H − (K† − f) 𝒳 = −i(T − S†)): the norm the fast part holds over its life, and its interference
    /// with the slow part's excited and ground amplitudes. Without decay S† = T and only the first term is left. On one
    /// line it is the whole difference between the mean scattering rate after a jump and the optical Bloch steady state
    /// beside the threshold's clamp (tests). Returns (x₀, τ).
    pub(crate) fn transient(&self, s: &[Mat], f: &[C64]) -> Option<(Vec<C64>, f64)> {
        self.transients(s, &[f.to_vec()])?.pop()
    }

    /// `transient` of several bare states: each replica's 𝓛 and 𝒳 and the landing's overlap made once.
    pub(crate) fn transients(&self, s: &[Mat], fs: &[Vec<C64>]) -> Option<Vec<(Vec<C64>, f64)>> {
        let mut overlap = Mat::identity(self.size());
        for (si, rep) in s.iter().zip(&self.replicas) {
            overlap.add_scaled(C64::ONE, &si.product(&rep.t));
        }
        let lu = crate::linalg::Lu::new(&overlap)?;
        let h = schurs_of(&self.generator, &self.members)?;
        let solved: Vec<Option<(Mat, Mat)>> = self
            .replicas
            .iter()
            .zip(s)
            .map(|(rep, si)| {
                let nq = rep.q.len();
                if nq == 0 {
                    return None;
                }
                let k = &rep.k;
                let kh = k.adjoint();
                let identity = Mat::from_fn(nq, nq, |i, j| if i == j { C64::new(0.0, -1.0) } else { C64::ZERO });
                let lyapunov = solve_sylvester(&kh, C64::ZERO, k, C64::ZERO, &identity);
                let mut b = rep.t.clone();
                b.add_scaled(-C64::ONE, &si.adjoint());
                let cross = solve_in(&kh, C64::real(-rep.frequency), &h[rep.group], C64::ZERO, &b.scaled(C64::new(0.0, -1.0)), &self.members[rep.group]);
                Some((lyapunov, cross))
            })
            .collect();
        Some(
            fs.iter()
                .map(|f| {
                    let mut x0 = f.clone();
                    lu.solve(&mut x0);
                    let mut tau = 0.0;
                    for (rep, made) in self.replicas.iter().zip(&solved) {
                        let Some((lyapunov, cross)) = made else { continue };
                        let y = rep.t.apply(&x0);
                        tau += crate::linalg::inner(&y, &lyapunov.apply(&y)).re - 2.0 * crate::linalg::inner(&y, &cross.apply(&x0)).re;
                    }
                    (x0, tau)
                })
                .collect(),
        )
    }
}

/// The fold's series about a reference as the run evaluates it (module documentation, # The definition and the route it
/// is evaluated on): per replica its frequency f, Q_f and the Schur form of K_f there -- K the excited block of the
/// levels the reference is taken at (the far light's) --, which replica each class drives each ground state's amplitude
/// into, and the reference z₀. The responses R₀^{j+1} W (`respond`) are linear in a class's coupling W with
/// coefficients of these levels alone: what a table holds without the near light's coordinates.
pub(crate) struct Reference {
    replicas: Vec<(f64, Vec<usize>, Schur)>,
    driven: Vec<Vec<usize>>,
    ground: usize,
    excited: usize,
    z0: f64,
}

impl Reference {
    /// The reference of the levels `levels` (their excited block and decay; the ground block is the caller's, in z₀),
    /// the classes' frequencies `nus`, the slow states and their frames, at z₀ (rad/µs, in the frames).
    pub(crate) fn new(levels: &Levels, nus: &[f64], slow: &Slow, z0: f64) -> Reference {
        let ground = levels.ground.rows;
        let excited = levels.excited.rows;
        let classes: Vec<Class> = nus.iter().map(|&nu| Class { nu, beams: vec![] }).collect();
        let replicas = Replicas::of(&classes, slow, ground);
        let mut k = levels.excited.clone();
        k.add_scaled(C64::new(0.0, -0.5), &levels.decay);
        let mut forms: Vec<(Vec<usize>, Schur)> = vec![];
        let out = replicas
            .frequencies
            .iter()
            .map(|&f| {
                let q: Vec<usize> = (0..excited).filter(|&e| !slow.explicit.iter().enumerate().any(|(i, &x)| x == e && same(slow.frames[ground + i], f))).collect();
                if !forms.iter().any(|x| x.0 == q) {
                    forms.push((q.clone(), Schur::of(&k.block(&q, &q)).expect("the Schur form of the excited block")));
                }
                let schur = forms.iter().find(|x| x.0 == q).expect("its form").1.clone();
                (f, q, schur)
            })
            .collect();
        Reference { replicas: out, driven: replicas.driven, ground, excited, z0 }
    }

    /// The responses of class c's coupling (E × the ground states) to the orders 0..=`orders`: R₀^{j+1} W on every
    /// excited state (zero off each replica's Q) and every slow state (E × n, n the slow states: the explicit states'
    /// columns are zero, a class couples E ← G alone), R₀ = (z₀ + f − K_f)⁻¹ of the replica each ground state's
    /// amplitude drives.
    pub(crate) fn respond(&self, class: usize, coupling: &Mat, n: usize, orders: usize) -> Vec<Mat> {
        let mut out = vec![Mat::zeros(self.excited, n); orders + 1];
        for (r, (f, q, k)) in self.replicas.iter().enumerate() {
            let sigma = C64::real(self.z0 + f);
            for g in (0..self.ground).filter(|&g| self.driven[class][g] == r) {
                let mut x: Vec<C64> = q.iter().map(|&e| coupling[(e, g)]).collect();
                if x.iter().all(|z| *z == C64::ZERO) {
                    continue;
                }
                for order in out.iter_mut() {
                    x = k.resolvent(sigma, &x);
                    for (i, &e) in q.iter().enumerate() {
                        order[(e, g)] = x[i];
                    }
                }
            }
        }
        out
    }

    /// The complex multiply-adds of one coupling's responses to `orders` (`respond`): per driven column and order a
    /// resolvent's two changes of basis and its back substitution on Q.
    pub(crate) fn work(&self, orders: usize) -> f64 {
        let q = self.replicas.iter().map(|r| r.1.len()).max().unwrap_or(0) as f64;
        (self.ground * (orders + 1)) as f64 * 2.5 * q * q
    }
}

/// The frames that make the explicit components turn least (v3 2; the coordinator's 2026-11-13 rule): every component
/// a ← b at lab beat Ω (∝ e^{−iΩt}) turns at Ω + θ_b − θ_a in the frames; the frames θ minimize max |Ω + θ_b − θ_a| over
/// the components. By linear-programming duality that least largest residual t* is the largest |Σ_{e∈C} ±Ω_e|/|C| over
/// the cycles C of the components' graph, the minimum mean cycle of the directed graph with an arc b → a of weight Ω
/// and a → b of −Ω per component (Karp 1978); the frames are then the shortest paths from a source joined to every state
/// at no cost, the arcs at weight + t* (no negative cycle there): θ_a ≤ θ_b + Ω + t* and θ_b ≤ θ_a − Ω + t*. Without a
/// cycle t* = 0 and every component is static. Returns the frames (states that no component touches: 0) and t*.
pub(crate) fn frames(states: usize, components: &[(usize, usize, f64)]) -> (Vec<f64>, f64) {
    let arcs: Vec<(usize, usize, f64)> = components.iter().flat_map(|&(a, b, omega)| [(b, a, omega), (a, b, -omega)]).collect();
    // Karp: D_k(v) the least weight of a walk of k arcs from the source (the source's arcs counted as the first).
    let n = states;
    let mut d = vec![vec![f64::INFINITY; n]; n + 1];
    d[0].iter_mut().for_each(|x| *x = 0.0);
    for k in 1..=n {
        for &(from, to, w) in &arcs {
            if d[k - 1][from].is_finite() {
                let x = d[k - 1][from] + w;
                if x < d[k][to] {
                    d[k][to] = x;
                }
            }
        }
    }
    let mut least = f64::INFINITY;
    for v in 0..n {
        if !d[n][v].is_finite() {
            continue;
        }
        let worst = (0..n).filter(|&k| d[k][v].is_finite()).map(|k| (d[n][v] - d[k][v]) / (n - k) as f64).fold(f64::NEG_INFINITY, f64::max);
        least = least.min(worst);
    }
    // The least mean cycle's weight is −t* (every cycle holds both ways round, so its mean is ≤ 0).
    let residual = if least.is_finite() { (-least).max(0.0) } else { 0.0 };
    // Bellman–Ford at weight + t*.
    let mut theta = vec![0.0; n];
    for _ in 0..n {
        let mut moved = false;
        for &(from, to, w) in &arcs {
            let x = theta[from] + w + residual;
            if x < theta[to] {
                theta[to] = x;
                moved = true;
            }
        }
        if !moved {
            break;
        }
    }
    (theta, residual)
}

// ---------------------------------------------------------------------------------------------------------------------
// The system at a point, and the point's closure (criteria one and two, the frames).

/// The model's share of a target (ELIMINATION v2 4; SPEC §5.2): what the step's share s leaves the model, ε√(1 − s²) --
/// 0.0447 of the populations' 0.05 (absolute), twice that of T's 0.10 (relative).
pub(crate) fn share() -> f64 {
    crate::units::POPULATION_TARGET * (1.0 - crate::units::STEP_SHARE * crate::units::STEP_SHARE).sqrt()
}

/// A request's levels and light at a point r (µm): H0 and every frequency's outside form at the point's field
/// (request.rs `OutsideForm`, its value at the focus taken out of H0), the decay and the emission, and the classes --
/// per frequency of the light its co-rotating part at δ_ω and its counter-rotating part at −(δ_ω + 2ν_frame), each
/// beam's coupling at its amplitude there (the counter-rotating one at its conjugate, its wave vector −k).
pub(crate) fn system(request: &crate::request::Request, light: &crate::field::Light, r: [f64; 3]) -> (Levels, Vec<Class>, Vec<(usize, bool)>) {
    use crate::request::{GROUND, STATES};
    let amplitudes = light.amplitudes(r);
    let focus = light.amplitudes([0.0; 3]);
    let mut h = request.hamiltonian.clone();
    for f in &light.frequencies {
        let (e, e0) = (f.field(&amplitudes).value, f.field(&focus).value);
        let form = super::far::outside_form(&request.outside_forms[f.form].matrices);
        for p in 0..3 {
            for q in 0..3 {
                let weight = e[p].conj() * e[q] - e0[p].conj() * e0[q];
                if weight != C64::ZERO {
                    h.add_scaled(weight, &form[p][q]);
                }
            }
        }
    }
    let (ground, excited): (Vec<usize>, Vec<usize>) = ((0..GROUND).collect(), (GROUND..STATES).collect());
    let emission: [Mat; 3] = std::array::from_fn(|a| request.emission[a].block(&ground, &excited));
    let mut decay = Mat::zeros(excited.len(), excited.len());
    for c in &emission {
        decay.add_scaled(C64::ONE, &c.adjoint_product(c));
    }
    let levels = Levels { ground: h.block(&ground, &ground), excited: h.block(&excited, &excited), decay, emission };
    let omega = |index: &[i32]| std::f64::consts::TAU * index.iter().zip(&request.frequencies_mhz).map(|(&n, f)| n as f64 * f).sum::<f64>();
    let frame = std::f64::consts::TAU * request.frame_frequencies_mhz[GROUND];
    let (mut classes, mut sources) = (vec![], vec![]);
    for (w, f) in light.frequencies.iter().enumerate() {
        let delta = omega(&f.index);
        for counter in [false, true] {
            let beams = f
                .beams
                .iter()
                .map(|&l| {
                    let b = &request.beams[l];
                    let a = amplitudes[l].value;
                    let (coupling, k) = if counter { (b.counter_coupling.block(&excited, &ground).scaled(a.conj()), light.beams[l].wave_vector().map(|x| -x)) } else { (b.coupling.block(&excited, &ground).scaled(a), light.beams[l].wave_vector()) };
                    Beam { coupling, wave: k }
                })
                .filter(|b| b.coupling.max_abs() > 0.0)
                .collect::<Vec<_>>();
            if !beams.is_empty() {
                classes.push(Class { nu: if counter { -(delta + 2.0 * frame) } else { delta }, beams });
                sources.push((w, counter));
            }
        }
    }
    (levels, classes, sources)
}

/// Per class the remainder of its fold taken at the bare levels instead of the point's (ELIMINATION.md §2: the bare
/// levels' fold is this function at a coarser point): of each ground-like eigenstate p, its dominant ground state g at
/// its bare energy E_g (H0's, `bare`, in the frames), ‖[R(E_g + f) − R(λ_p + f)] W_c r_p‖ / ‖R(λ_p + f) W_c r_p‖ over
/// the class's replicas, the largest -- where it is within what no output resolves the class's fold is its bare
/// levels' constant (far.rs), at no cost of its own; else at the point.
pub(crate) fn bare_remainders(fold: &Fold, classes: usize, bare: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0_f64; classes];
    for p in (0..fold.size()).filter(|&p| fold.ground_weight(p) >= 0.5) {
        let r = fold.right(p);
        let g = fold.dominant(p);
        if g >= fold.ground {
            continue;
        }
        let lambda = fold.eigen.values[p];
        let level = C64::real(bare[g] - fold.frames[g]);
        for (c, row) in fold.driven.iter().enumerate() {
            let (mut num, mut den) = (0.0, 0.0);
            for (j, rep) in fold.replicas.iter().enumerate().filter(|(_, rep)| rep.group == fold.groups[p]) {
                let columns: Vec<usize> = (0..fold.ground).filter(|&s| row[s] == Some(j)).collect();
                if columns.is_empty() {
                    continue;
                }
                let feeds: Vec<&Feed> = rep.feeds.iter().filter(|f| f.class == Some(c)).collect();
                let mut wr = vec![C64::ZERO; rep.q.len()];
                for f in feeds {
                    let v = f.w.apply(&r);
                    wr.iter_mut().zip(&v).for_each(|(a, b)| *a += *b);
                }
                let f = C64::real(rep.frequency);
                let (at, bared) = (rep.k.resolvent(lambda + f, &wr), rep.k.resolvent(level + f, &wr));
                num += at.iter().zip(&bared).map(|(a, b)| (*a - *b).norm_sqr()).sum::<f64>();
                den += at.iter().map(|a| a.norm_sqr()).sum::<f64>();
            }
            if den > 0.0 {
                out[c] = out[c].max((num / den).sqrt());
            }
        }
    }
    out
}

/// A model's series constants (module documentation, # The definition and the route it is evaluated on; ELIMINATION v5):
/// per cluster the centre c of its near light's scalar shift over the atoms read (the reference z₀ is the cluster's far
/// levels' mean and c), the series' largest ratio q over them, and the order K every evaluation takes -- the least
/// whose remainder q^{K+1}/(1 − q) is within the model's share (`share`).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Series {
    pub centre: Vec<f64>,
    pub ratio: f64,
    pub order: usize,
}

impl Series {
    /// The least order whose remainder at the ratio q is within the share; None where the series does not converge
    /// (q ≥ 1: the fold's existence is criterion one's to restore, a block made explicit).
    pub(crate) fn order_of(q: f64) -> Option<usize> {
        if !(q < 1.0) {
            return None;
        }
        (0..).find(|&k| q.powi(k as i32 + 1) / (1.0 - q) <= share())
    }
}

impl super::Model {
    /// The fold of a model's cluster at a point, as the run's series stands for it (`Series`): the levels with the far
    /// light alone (eliminate.rs `Model::far_hamiltonian`), every carrier a class of its coherent field's couplings
    /// folded at the atom and explicit, the cluster's explicit states and frames; with the far levels' mean in the
    /// frames and the norm of what the far levels leave out of the excited block -- the carriers' own static forms there,
    /// bilinear in the near field (eliminate.rs `level_pair`) -- in the Frobenius norm, a bound of the spectral one.
    pub(crate) fn cluster_fold(&self, cluster: usize, amplitudes: &[crate::field::Amplitude], about: bool) -> Option<(Fold, Levels, f64, f64)> {
        use crate::request::{GROUND, STATES};
        let running = self.running();
        let h = &self.far_hamiltonian(amplitudes, &running, 0, &[])[0];
        let c = &self.clusters[cluster];
        let g = c.ground;
        let grounds: Vec<usize> = c.states[..g].to_vec();
        let excited: Vec<usize> = (GROUND..STATES).collect();
        let classes: Vec<Class> = self
            .carriers
            .iter()
            .map(|cc| {
                let mut coupling = Mat::zeros(excited.len(), g);
                for (l, (folded, explicit)) in cc.folded.iter().zip(&cc.explicit).enumerate() {
                    let a = amplitudes[cc.beams[l]].value;
                    coupling.add_scaled(a, &folded.block(&excited, &grounds));
                    coupling.add_scaled(a, &explicit.block(&excited, &grounds));
                }
                Class { nu: cc.detuning, beams: vec![Beam { coupling, wave: [0.0; 3] }] }
            })
            .collect();
        if classes.iter().all(|class| class.beams[0].coupling.max_abs() == 0.0) {
            return None;
        }
        let levels = Levels {
            ground: h.block(&grounds, &grounds),
            excited: h.block(&excited, &excited),
            decay: self.decay.clone(),
            emission: std::array::from_fn(|x| self.emission[x].block(&grounds, &excited)),
        };
        let frames: Vec<f64> = c.frames.iter().map(|f| self.omega(f)).collect();
        let far = (0..g).map(|y| h[(grounds[y], grounds[y])].re - frames[y]).sum::<f64>() / g as f64;
        let slow = Slow::single(c.states[g..].iter().map(|&s| s - GROUND).collect(), frames);
        let replicas = Replicas::of(&classes, &slow, g);
        let fold = fold_about(&levels, &classes, &slow, &replicas, about.then(|| far + self.series.centre[cluster]));
        // The carriers' static forms on the excited block.
        let mut left = Mat::zeros(excited.len(), excited.len());
        for cc in &self.carriers {
            let f = &self.light.frequencies[cc.frequency];
            let n = f.beams.len();
            for (x, form) in self.tones[cc.frequency].pairs.iter().enumerate() {
                let z = super::far::pair(&amplitudes[f.beams[x / n]], &amplitudes[f.beams[x % n]]).value;
                if z != C64::ZERO {
                    left.add_scaled(z, &form.block(&excited, &excited));
                }
            }
        }
        Some((fold, levels, far, left.frobenius()))
    }

    /// The model's series constants at the atoms `points` (`Series`): per cluster the centre of the near light's scalar
    /// shift -- the mean of the fold's ground-like eigenvalues less the far levels' mean, over the atoms its range's
    /// middle --, then at each atom the ratio of the series about z₀ = far mean + c, q = (‖H − z₀‖₂ + ‖δK‖)/d (the
    /// Neumann series of the resolvent in both its arguments: `Fold::series_ratio`, and the excited block the reference
    /// leaves out), the largest; and the order it asks for. An atom where a fold does not exist (criterion one's: its
    /// closure makes a block explicit) is read as the series' end: q = ∞.
    pub(crate) fn series_at(&self, points: &[([f64; 3], [f64; 3])]) -> Series {
        use rayon::prelude::*;
        let clusters = self.clusters.len();
        let read: Vec<Vec<Option<(f64, Fold, f64, f64)>>> = points
            .par_iter()
            .map(|&(r, _)| {
                let amplitudes = self.light.amplitudes(r);
                (0..clusters)
                    .map(|cluster| {
                        let (fold, _, far, left) = self.cluster_fold(cluster, &amplitudes, false)?;
                        let like: Vec<usize> = (0..fold.size()).filter(|&p| fold.ground_weight(p) >= 0.5).collect();
                        let shift = like.iter().map(|&p| fold.eigen.values[p].re).sum::<f64>() / like.len().max(1) as f64 - far;
                        Some((shift, fold, far, left))
                    })
                    .collect()
            })
            .collect();
        let centre: Vec<f64> = (0..clusters)
            .map(|cluster| {
                let (low, high) = read.iter().filter_map(|at| at[cluster].as_ref()).fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), x| (l.min(x.0), h.max(x.0)));
                if low.is_finite() { 0.5 * (low + high) } else { 0.0 }
            })
            .collect();
        let ratio = read
            .iter()
            .flat_map(|at| at.iter().enumerate())
            .filter_map(|(cluster, x)| x.as_ref().map(|x| (cluster, x)))
            .map(|(cluster, (_, fold, far, left))| {
                if !fold.exists {
                    return f64::INFINITY;
                }
                let z0 = C64::real(far + centre[cluster]);
                (fold.series_norm(z0) + left) / fold.series_distance(z0)
            })
            .fold(0.0, f64::max);
        Series { centre, ratio, order: Series::order_of(ratio).unwrap_or(usize::MAX) }
    }

    /// The excited hyperfine blocks F′ of the model's states (`blocks`), from its labels.
    pub(crate) fn blocks(&self) -> Vec<Vec<usize>> {
        excited_blocks(&self.labels)
    }

    /// The keep rule's objects of this model (ELIMINATION.md §6: its two kinds, and the photons' cross terms) as the
    /// model's units: every excited block a carrier drives that is not explicit (`Unit::Block`); every P-layer link of
    /// two states of the model's clusters at a beat of two classes (a carrier's co-rotating component or none: lab
    /// beats ν_d − ν_c, ±ν_c and zero) that its generator does not hold as it is (`Unit::Term`); every pair of photon
    /// classes landing in one cluster (`Unit::Coherence`, photon.rs `Model::coherence_units`). The kept ones left out.
    pub(crate) fn candidates(&self) -> Vec<super::Unit> {
        use crate::request::GROUND;
        let mut out: Vec<super::Unit> = vec![];
        for cluster in &self.clusters {
            let grounds = &cluster.states[..cluster.ground];
            for (b, block) in self.blocks().iter().enumerate() {
                if block.iter().any(|&e| cluster.states.contains(&(GROUND + e))) {
                    continue;
                }
                for cc in self.carriers.iter().filter(|cc| cc.folded.iter().any(|m| block.iter().any(|&e| grounds.iter().any(|&g| m[(GROUND + e, g)] != C64::ZERO)))) {
                    out.push(super::Unit::Block { block: b, component: super::Component { frequency: cc.frequency, counter: false }, grounds: ground_set(grounds) });
                }
            }
        }
        let classes: Vec<super::Beat> = std::iter::once(self.axes.zero()).chain(self.carriers.iter().map(|cc| cc.beat.clone())).collect();
        let mut beats: Vec<super::Beat> = vec![];
        for x in &classes {
            for y in &classes {
                let beat = x.plus(-1, y);
                if !beats.contains(&beat) {
                    beats.push(beat);
                }
            }
        }
        let states: Vec<usize> = self.clusters.iter().flat_map(|c| c.states.iter().copied()).collect();
        for beat in &beats {
            for &a in &states {
                for &b in &states {
                    if (a == b && beat.is_zero()) || (a >= GROUND && b >= GROUND) {
                        continue;
                    }
                    let unit = super::eliminate::term_unit(a, b, beat);
                    if !out.contains(&unit) && !self.in_generator(a, b, beat) {
                        out.push(unit);
                    }
                }
            }
        }
        out.extend(self.coherence_units());
        out.retain(|u| !self.kept.contains(u));
        out
    }

    /// The candidates' errors at a trajectory state (`candidates`; ELIMINATION.md §6: the prior's function at a run's
    /// state): the fold of the sample's cluster at its point (`cluster_fold`, the definition the run's series stands for)
    /// read by `errors` at the state, the atom's velocity and the frequency range its amplitude spans -- the cluster's
    /// residual beats and the pattern its moving atom sees, Δ = max|Ω| + 2 k_max v (`closure`) --, each part on its
    /// object's unit: a block's on `Unit::Block`, a link's on the `Unit::Term` of its two states at its lab beat. The
    /// photons' cross terms on the run's photon classes, which land in every cluster: of each pair landing in one, the
    /// rate of their interference at the state γ = |Σ_a ⟨u_a|u′_a⟩| over the state's norm, against their beat Ω -- γ/|Ω|
    /// (v3 1: the secular approximation's first order), on `Unit::Coherence`. A relative rate error x moves the
    /// populations by at most x (absolute) and T by x (relative).
    pub(crate) fn readings(&self, units: &[super::Unit], sample: &super::Sample, tables: Option<&super::tabulate::Tables>) -> Vec<super::UnitError> {
        use crate::request::GROUND;
        use super::{Unit, UnitError};
        let mut out = vec![UnitError::NONE; units.len()];
        let book = |unit: &Unit, x: f64, out: &mut Vec<UnitError>| {
            if let Some(i) = units.iter().position(|u| u == unit) {
                let e = &mut out[i];
                (e.parameter, e.population, e.temperature) = (e.parameter.max(x), e.population + x, e.temperature + x);
            }
        };
        let c = &self.clusters[sample.cluster];
        let g = c.ground;
        let amplitudes = self.light.amplitudes(sample.at.r);
        let velocity = sample.at.kappa.map(|k| k * self.hbar_over_mass);
        let speed = crate::linalg::norm3(velocity);
        if let Some((fold, levels, _, _)) = self.cluster_fold(sample.cluster, &amplitudes, true) {
            let plan = self.plan(sample.cluster, &self.running());
            let fastest = self.light.beams.iter().map(|b| b.wave_number).fold(0.0, f64::max);
            let range = plan.sampled.iter().map(|t| t[0]).fold(0.0, f64::max) + 2.0 * fastest * speed;
            let state = |label: usize| if label < g { c.states[label] } else { GROUND + label - g };
            let class = |k: Option<usize>| k.map_or(self.axes.zero(), |k| self.carriers[k].beat.clone());
            let term = |l: &Link| super::eliminate::term_unit(state(l.a), state(l.b), &class(l.classes.0).plus(-1, &class(l.classes.1)));
            let linked = |l: &Link| {
                let Unit::Term { a, b, beat } = term(l) else { unreachable!("a link's unit is a term") };
                self.in_generator(a, b, &beat)
            };
            let blocks = self.blocks();
            let errors = errors(&fold, &levels, &blocks, &linked, None, &sample.psi, velocity, range);
            for (object, x) in &errors.parts {
                match object {
                    Object::Block(b, Some(k)) => book(&Unit::Block { block: *b, component: super::Component { frequency: self.carriers[*k].frequency, counter: false }, grounds: ground_set(&c.states[..g]) }, *x, &mut out),
                    Object::Block(_, None) => {}
                    Object::Link(l) => book(&term(l), *x, &mut out),
                    Object::Photons(..) => {}
                }
            }
        }
        // The photons' cross terms.
        let lights = self.running();
        let plan = self.plan(sample.cluster, &lights);
        let table = tables.and_then(|t| t.of(sample.cluster, &lights));
        let (photons, _) = self.photons_of(&plan, table, &sample.at, &sample.psi, super::photon::Gradients::None);
        let norm = crate::linalg::norm_sqr(&sample.psi);
        for unit in units.iter().filter(|u| matches!(u, Unit::Coherence { .. })) {
            let Unit::Coherence { first, second, .. } = unit else { continue };
            for (cluster, i, j) in self.coherence_classes(unit) {
                if cluster != sample.cluster {
                    continue;
                }
                let (Some(u), Some(v)) = (photons.iter().find(|p| p.class == i), photons.iter().find(|p| p.class == j)) else { continue };
                let gamma = (0..3).map(|a| crate::linalg::inner(&u.amplitudes[a], &v.amplitudes[a])).sum::<C64>().abs() / norm;
                book(unit, gamma / self.omega(&second.plus(-1, first)).abs(), &mut out);
            }
        }
        out
    }

    /// The model's prior (ELIMINATION.md §3–§4, the brief's "先验"): at the reach's own atoms `points` (position,
    /// velocity) the point's closure (`closure`) from what is kept already, the atoms together until no atom adds
    /// anything; and per class its fold's route, at the point or at its bare levels (`bare_remainders`, within the share
    /// no output resolves at every atom: `estimate::unseen`). What it keeps, as the model's units: every drive of a
    /// class folded at the point (`Unit::Bare`: its fold taken at the levels there), the explicit drives into the
    /// blocks (`Unit::Drive`), the P-layer links of two ground states (`Unit::Term`, their lab beat exactly).
    pub(crate) fn prior(&self, request: &crate::request::Request, points: &[([f64; 3], [f64; 3])]) -> Vec<super::Unit> {
        use rayon::prelude::*;
        let blocks = blocks(request);
        let bare = super::far::energies(request);
        let unseen = crate::estimate::unseen(crate::units::MOST_TRAJECTORIES as f64);
        let mut kept = Kept::default();
        let mut routes: Vec<f64> = vec![];
        let mut sources: Vec<(usize, bool)> = vec![];
        loop {
            let read: Vec<(Kept, Vec<f64>, Vec<(usize, bool)>)> = points
                .par_iter()
                .map(|&(r, v)| {
                    let (levels, classes, sources) = system(request, &self.light, r);
                    let closed = closure(&levels, &classes, &blocks, v, &kept);
                    let routes = bare_remainders(&closed.fold, classes.len(), &bare);
                    (closed.kept, routes, sources)
                })
                .collect();
            let before = (kept.blocks.len(), kept.links.len());
            for (k, r, s) in read {
                for b in k.blocks {
                    if !kept.blocks.contains(&b) {
                        kept.blocks.push(b);
                    }
                }
                for l in k.links {
                    if !kept.links.contains(&l) {
                        kept.links.push(l);
                    }
                }
                if routes.len() < r.len() {
                    routes.resize(r.len(), 0.0);
                }
                routes.iter_mut().zip(&r).for_each(|(a, b)| *a = a.max(*b));
                sources = s;
            }
            if (kept.blocks.len(), kept.links.len()) == before {
                break;
            }
        }
        let mut units = vec![];
        let component = |c: usize| super::Component { frequency: sources[c].0, counter: sources[c].1 };
        for (c, &rho) in routes.iter().enumerate() {
            if sources[c].1 || rho <= unseen {
                continue;
            }
            for &(comp, e, g) in self.driven.iter().filter(|d| d.0 == component(c)) {
                units.push(super::Unit::Bare { component: comp, excited: e, ground: g });
            }
        }
        for l in &kept.links {
            match l.classes {
                (None, Some(c)) if l.a >= crate::request::GROUND => units.push(super::Unit::Drive { component: component(c), excited: l.a, ground: l.b }),
                (emitted, absorbed) if l.a < crate::request::GROUND && l.b < crate::request::GROUND => {
                    let beat = |c: Option<usize>| c.map_or(self.axes.zero(), |c| self.frequency(component(c)));
                    units.push(super::Unit::Term { a: l.a, b: l.b, beat: beat(emitted).plus(-1, &beat(absorbed)) });
                }
                _ => {}
            }
        }
        // Two links can be one unit (a static coupling and one class's own at the lab beat zero).
        let mut distinct: Vec<super::Unit> = vec![];
        for unit in units {
            if !distinct.contains(&unit) {
                distinct.push(unit);
            }
        }
        distinct
    }
}

/// A set of ground states as `Unit::Block` holds it: bit g for ground state g.
pub(crate) fn ground_set(grounds: &[usize]) -> u32 {
    grounds.iter().fold(0, |set, &g| set | 1 << g)
}

/// A request's excited hyperfine blocks F' (indices into E): the states of one manifold and F, the unit a partition
/// moves -- the Zeeman shifts and the light's tensor shifts couple a block's sublevels, so splitting one breaks the
/// block structure (ELIMINATION.md §3).
pub(crate) fn blocks(request: &crate::request::Request) -> Vec<Vec<usize>> {
    excited_blocks(&request.states)
}

/// The excited blocks of the states' labels (`blocks`).
pub(crate) fn excited_blocks(states: &[String]) -> Vec<Vec<usize>> {
    use crate::request::{Level, GROUND};
    let mut out: Vec<(String, Vec<usize>)> = vec![];
    for (e, label) in states[GROUND..].iter().enumerate() {
        let l = Level::parse(label).expect("a request's state label");
        let name = format!("{:?}{}", l.manifold, l.f);
        match out.iter_mut().find(|b| b.0 == name) {
            Some(b) => b.1.push(e),
            None => out.push((name, vec![e])),
        }
    }
    out.into_iter().map(|b| b.1).collect()
}

/// A P-layer link the model keeps (criterion two's explicit components, the explicit drives among them): an entry
/// a ← b of P (a state: a ground state's index, an excited one's GROUND + its index) at its lab beat, made by the
/// emitted and the absorbed class (None: a direct entry, H0's or the excited block's).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Link {
    pub a: usize,
    pub b: usize,
    pub classes: (Option<usize>, Option<usize>),
}

impl Link {
    /// Its lab beat Ω (∝ e^{−iΩt}): the absorbed class's frequency less the emitted one's.
    pub(crate) fn beat(&self, classes: &[Class]) -> f64 {
        self.classes.1.map_or(0.0, |c| classes[c].nu) - self.classes.0.map_or(0.0, |c| classes[c].nu)
    }
}

/// What a model keeps explicit (ELIMINATION.md §6: the keep rule's two kinds of objects): excited blocks, and P-layer
/// links -- the explicit components, the explicit drives into the blocks with them -- and the photons' cross terms kept
/// (pairs of photon frequencies, by the classes that make them; None the explicit states' own).
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Kept {
    pub blocks: Vec<usize>,
    pub links: Vec<Link>,
    pub photons: Vec<(Option<usize>, Option<usize>)>,
}

/// A point's closure (criteria one and two at one atom, ELIMINATION.md §3–§4 with v2 and v3): the kept set from
/// `start` on, made larger until the point needs nothing more, with its frames and fold.
#[derive(Clone, Debug)]
pub(crate) struct Closed {
    pub kept: Kept,
    pub frames: Vec<f64>,
    /// The frames' least largest residual t* (`frames`), and the step's requirement t* + |K| v (rad/µs).
    pub residual: f64,
    pub requirement: f64,
    pub fold: Fold,
    /// The worst remainder ρ over the ground-like eigenstates, and the largest mixing of an averaged component.
    pub remainder: f64,
    pub mixing: f64,
}

/// The P states (labels as in `Link`) of a kept set, the frames from its links (`frames`) and its groups (`components`
/// of the links, each explicit block's states one group with them: a block is moved whole, its sublevels coupled by
/// the Zeeman and tensor shifts): the ground states, then the kept blocks' states.
fn slow_of(blocks: &[Vec<usize>], kept: &Kept, classes: &[Class], ground: usize) -> (Slow, f64) {
    let explicit: Vec<usize> = kept.blocks.iter().flat_map(|&b| blocks[b].iter().copied()).collect();
    let place = |s: usize| if s < ground { Some(s) } else { explicit.iter().position(|&e| e == s - ground).map(|i| ground + i) };
    let edges: Vec<(usize, usize, f64)> = kept.links.iter().filter_map(|l| Some((place(l.a)?, place(l.b)?, l.beat(classes)))).collect();
    let n = ground + explicit.len();
    let (frames, residual) = frames(n, &edges);
    let within = kept.blocks.iter().flat_map(|&b| blocks[b].windows(2).filter_map(|w| Some((place(ground + w[0])?, place(ground + w[1])?))));
    let groups = components(n, edges.iter().map(|e| (e.0, e.1)).chain(within));
    (Slow { explicit, frames, groups }, residual)
}

/// The closure at one point (`Closed`): from `start`, at the atom's velocity v (µm/µs) --
/// 1. the fold in the frames of the kept links; where it does not exist (`settle`), the excited block with the largest
///    part of the ground-like eigenstates' excited share -- over all of them -- is made explicit;
/// 2. criterion one: every ground-like eigenstate's remainder ρ(±Δ), Δ = t* + 2 k_max v the frequency range its
///    amplitude spans (the frames' residual, and the Doppler shifts and the pattern a moving atom sees: at most
///    k_max v and |k_l − k_l'| v ≤ 2 k_max v), within the share; beyond it the blocks with the largest parts are made
///    explicit until the rest is within it;
/// - a block made explicit for an eigenstate is explicit at the replica that holds its part and in the P component the
///   eigenstate holds (`components`): the drives into it at that replica from that component's ground states are kept
///   (its frame). A drive from another component -- the other ground hyperfine level, 6.8 GHz away -- turns at that
///   component's own frequency, where the block is a Q state folded like any other (ELIMINATION §2–§3, v2 5); a block
///   explicit in one component is joined by another's drives only where that component's own part asks for it;
/// 3. criterion two: every component of two replicas whose averaged error 2(a/δ)² is beyond the share (v5: the
///    population's time mean after a sudden switch-on) is kept (a link of its two dominant states and classes), and every photon cross term whose γ/|Ω| is
///    above the share;
/// until nothing more is kept. Each round only adds.
pub(crate) fn closure(levels: &Levels, classes: &[Class], blocks: &[Vec<usize>], velocity: [f64; 3], start: &Kept) -> Closed {
    let ground = levels.ground.rows;
    let share = share();
    let speed = (velocity[0] * velocity[0] + velocity[1] * velocity[1] + velocity[2] * velocity[2]).sqrt();
    let fastest = classes.iter().flat_map(|c| c.beams.iter()).map(|b| (b.wave[0] * b.wave[0] + b.wave[1] * b.wave[1] + b.wave[2] * b.wave[2]).sqrt()).fold(0.0, f64::max);
    let block_of = |e: usize| blocks.iter().position(|b| b.contains(&e));
    let mut kept = start.clone();
    loop {
        let (slow, residual) = slow_of(blocks, &kept, classes, ground);
        // An explicit block's drives at its frame from every ground state of its group, as the group grows: they are
        // static in the frames (A's), and the run's cluster holds them as it is.
        let grown: Vec<Link> = slow
            .explicit
            .iter()
            .enumerate()
            .flat_map(|(i, &e)| (0..classes.len()).flat_map(move |c| (0..ground).map(move |g| (i, e, c, g))))
            .filter(|&(i, e, c, g)| slow.groups[g] == slow.groups[ground + i] && same(classes[c].nu + slow.frames[g], slow.frames[ground + i]) && classes[c].beams.iter().any(|beam| beam.coupling[(e, g)] != C64::ZERO))
            .map(|(_, e, c, g)| Link { a: ground + e, b: g, classes: (None, Some(c)) })
            .filter(|l| !kept.links.contains(l))
            .collect();
        if !grown.is_empty() {
            kept.links.extend(grown);
            continue;
        }
        let replicas = Replicas::of(classes, &slow, ground);
        let fold = fold(levels, classes, &slow, &replicas);
        let n = fold.size();
        let ground_like: Vec<usize> = (0..n).filter(|&p| fold.ground_weight(p) >= 0.5).collect();
        // The parts of a ground-like eigenstate's response (or remainder) per block and replica, each split among the
        // replica's classes (`class_shares`): (block, replica, class, part).
        let parts_of = |p: usize, shift: Option<f64>| -> (f64, Vec<(usize, usize, Option<usize>, f64)>) {
            let r = fold.right(p);
            let lambda = fold.eigen.values[p];
            let unit: Vec<C64> = (0..n).map(|q| if q == p { C64::ONE } else { C64::ZERO }).collect();
            let mut parts = vec![];
            let (mut num, mut den) = (0.0, 0.0);
            for (j, rep) in fold.replicas.iter().enumerate().filter(|(_, rep)| rep.group == fold.groups[p]) {
                let wr = rep.w.apply(&r);
                let sigma = lambda + C64::real(rep.frequency);
                let x0 = rep.k.resolvent(sigma, &wr);
                let d: Vec<C64> = match shift {
                    None => x0.clone(),
                    Some(delta) => {
                        let shifted = rep.k.resolvent(sigma + C64::real(delta), &wr);
                        let second = rep.k.resolvent(sigma, &x0);
                        (0..x0.len()).map(|i| shifted[i] - x0[i] + second[i] * delta).collect()
                    }
                };
                let mut on = vec![0.0; blocks.len()];
                for (i, &e) in rep.q.iter().enumerate() {
                    if let Some(b) = block_of(e) {
                        on[b] += d[i].norm_sqr();
                    }
                    num += d[i].norm_sqr();
                    den += x0[i].norm_sqr();
                }
                let mut responses = None;
                for (b, &x) in on.iter().enumerate().filter(|(_, x)| **x > 0.0) {
                    let responses = responses.get_or_insert_with(|| fold.feed_responses(j, &unit));
                    for (class, share) in class_shares(responses, &rep.q, &blocks[b]) {
                        parts.push((b, j, class, x * share));
                    }
                }
            }
            let scale = if shift.is_some() { den } else { 1.0 };
            if scale > 0.0 {
                parts.iter_mut().for_each(|x| x.3 /= scale);
            }
            (if shift.is_some() { if den > 0.0 { (num / den).sqrt() } else { 0.0 } } else { num.sqrt() }, parts)
        };
        // Moves: (block, replica, class, the eigenstate whose part it is), largest part first per state, until the rest
        // is within `limit`; one replica a block and class.
        let mut moves: Vec<(usize, usize, usize, usize)> = vec![];
        let mut remainder: f64 = 0.0;
        let pick = |p: usize, mut order: Vec<(usize, usize, Option<usize>, f64)>, total: f64, limit: f64, moves: &mut Vec<(usize, usize, usize, usize)>| {
            order.sort_by(|x, y| y.3.total_cmp(&x.3));
            let mut rest = total;
            for (b, j, class, x) in order {
                if rest <= limit {
                    break;
                }
                rest -= x;
                if let Some(c) = class {
                    if !moves.iter().any(|m| m.0 == b && m.2 == c) {
                        moves.push((b, j, c, p));
                    }
                }
            }
        };
        if !fold.exists {
            // One move a round: the largest part of any ground-like eigenstate's excited share, of a class.
            let mut largest: Option<(usize, usize, usize, usize, f64)> = None;
            for &p in &ground_like {
                let (_, parts) = parts_of(p, None);
                for (b, j, class, x) in parts {
                    if let Some(c) = class.filter(|_| x > largest.map_or(0.0, |m| m.4)) {
                        largest = Some((b, j, c, p, x));
                    }
                }
            }
            moves.extend(largest.map(|(b, j, c, p, _)| (b, j, c, p)));
        } else {
            let delta = residual + 2.0 * fastest * speed;
            for &p in &ground_like {
                let mut worst = (0.0, vec![]);
                for shift in [delta, -delta] {
                    let (rho, parts) = parts_of(p, Some(shift));
                    if rho >= worst.0 {
                        worst = (rho, parts);
                    }
                }
                remainder = remainder.max(worst.0);
                if worst.0 > share {
                    pick(p, worst.1, worst.0 * worst.0, share * share, &mut moves);
                }
            }
        }
        let mut grew = false;
        for (b, j, c, p) in moves {
            // The class's drives at the replica that holds the part, from the ground states of the eigenstate's group.
            let rep = &fold.replicas[j];
            let held = slow.groups[fold.dominant(p)];
            let links: Vec<Link> = blocks[b]
                .iter()
                .flat_map(|&e| (0..ground).map(move |g| (e, g)))
                .filter(|&(e, g)| slow.groups[g] == held && replicas.driven[c][g] < replicas.frequencies.len() && same(replicas.frequencies[replicas.driven[c][g]], rep.frequency) && classes[c].beams.iter().any(|beam| beam.coupling[(e, g)] != C64::ZERO))
                .map(|(e, g)| Link { a: ground + e, b: g, classes: (None, Some(c)) })
                .filter(|l| !kept.links.contains(l))
                .collect();
            if !links.is_empty() {
                if !kept.blocks.contains(&b) {
                    kept.blocks.push(b);
                }
                kept.links.extend(links);
                grew = true;
            }
        }
        if grew {
            continue;
        }
        // Criterion two.
        let label = |p: usize| {
            let s = fold.dominant(p);
            if s < ground { s } else { ground + fold.explicit[s - ground] }
        };
        let mut mixing: f64 = 0.0;
        for x in fold.components(velocity) {
            let m = x.mixing();
            let link = Link { a: label(x.row), b: label(x.column), classes: x.classes };
            if 2.0 * m * m > share {
                if link.a != link.b && !kept.links.contains(&link) {
                    kept.links.push(link);
                    grew = true;
                }
            } else {
                mixing = mixing.max(m);
            }
        }
        for x in fold.photons(levels) {
            if x.size > share * x.beat.abs() {
                let pair = x.classes;
                if !kept.photons.contains(&pair) {
                    kept.photons.push(pair);
                }
            }
        }
        if grew {
            continue;
        }
        // |K|: the largest wave vector an entry of the generator holds -- a class's beams against each other (its
        // pattern), and the classes of a kept link against each other.
        let pairs: Vec<(usize, usize)> = (0..classes.len()).map(|c| (c, c)).chain(kept.links.iter().filter_map(|l| Some((l.classes.0?, l.classes.1?)))).collect();
        let pattern = pairs
            .iter()
            .flat_map(|&(c, d)| classes[c].beams.iter().flat_map(move |b| classes[d].beams.iter().map(move |b2| [b.wave[0] - b2.wave[0], b.wave[1] - b2.wave[1], b.wave[2] - b2.wave[2]])))
            .map(|k| (k[0] * k[0] + k[1] * k[1] + k[2] * k[2]).sqrt())
            .fold(0.0, f64::max);
        return Closed { requirement: residual + pattern * speed, kept, frames: slow.frames.clone(), residual, fold, remainder, mixing };
    }
}

/// The classes' shares of a replica's part on an excited block (ELIMINATION.md §6: a block is kept for one class of the
/// light, `Unit::Block`): per class its feeds' responses on the block's states `on` (`Fold::feed_responses`, Q's places
/// `q`), ‖R W_feed x|_on‖², over their sum -- their cross terms the classes' interference, no class's alone. None where
/// nothing of the replica reaches the block.
fn class_shares(responses: &[(Option<usize>, Vec<C64>)], q: &[usize], on: &[usize]) -> Vec<(Option<usize>, f64)> {
    let mut out: Vec<(Option<usize>, f64)> = vec![];
    for (class, v) in responses {
        let size: f64 = q.iter().zip(v).filter(|(e, _)| on.contains(e)).map(|(_, z)| z.norm_sqr()).sum();
        match out.iter_mut().find(|o| o.0 == *class) {
            Some(o) => o.1 += size,
            None => out.push((*class, size)),
        }
    }
    let total: f64 = out.iter().map(|o| o.1).sum();
    if !(total > 0.0) {
        return vec![];
    }
    out.iter_mut().for_each(|o| o.1 /= total);
    out
}

/// The components of n places joined by the pairs `joined`: per place the least place of its component.
fn components(n: usize, joined: impl Iterator<Item = (usize, usize)>) -> Vec<usize> {
    let mut parent: Vec<usize> = (0..n).collect();
    fn root(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    for (a, b) in joined {
        let (a, b) = (root(&mut parent, a), root(&mut parent, b));
        parent[a.max(b)] = a.min(b);
    }
    (0..n).map(|x| root(&mut parent, x)).collect()
}

/// What a model keeps or folds, the keep rule's objects (ELIMINATION.md §6): an excited block, a P-layer link, a pair of
/// photon frequencies' cross term.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Object {
    /// An excited block for one class of the light (None: the explicit states' own couplings into it).
    Block(usize, Option<usize>),
    Link(Link),
    Photons(Option<usize>, Option<usize>),
}

/// A model's error at a state (ELIMINATION.md §6, v2, v3): the populations' (absolute) and T's (relative), its four
/// terms, and per object the part its keep would take away.
#[derive(Clone, Debug)]
pub(crate) struct Errors {
    pub population: f64,
    pub temperature: f64,
    pub remainder: f64,
    pub transient: f64,
    pub averaged: f64,
    pub photons: f64,
    pub parts: Vec<(Object, f64)>,
}

/// The model's error at a state ψ (P amplitudes in the frames) of an atom at the velocity v (µm/µs), the frequency range
/// its amplitude spans Δ (rad/µs; `closure`): one function for the prior (a start, each ground-like eigenstate full in
/// turn) and for the readings (a trajectory's state).
/// - The remainder (criterion one): a rate's relative error is 2ρ_p(Δ) of the eigenstate the state holds, weighted by
///   its share w_p of the state; each excited block's part of ρ² is its part.
/// - The transient (`Fold::transient`): after the next jump the atom lands in a bare ground state -- at the state's
///   own ground populations p_g --, whose waiting time the model takes from its slow part alone with the threshold law
///   P(T > t) = min(1, ‖y(t)‖²) (step.rs `jump`). The exact one is longer by τ and the law's clamp leaves out
///   (N − 1 − ln N)/γ where the slow part's norm N = ‖(x₀, T x₀)‖² is above one (γ its decay there): the rate's relative
///   error is |τ + clamp| over the model's mean wait, N/γ or (1 + ln N)/γ. Its part goes to the blocks in proportion to
///   their share of the landing's excited amplitude T x₀: making a block explicit takes its fast part into the dynamics.
/// - The averaged components (criterion two): 2(a/δ)² of each one not kept (v5), on its two eigenstates' shares.
/// - The photons' cross terms not kept: γ/|Ω| of each, on its source eigenstate's share.
/// A relative rate error x moves the populations by at most x (absolute) and T, which the light sets through its rates
/// and shifts, by x relative: the four terms add on both.
#[allow(clippy::too_many_arguments)]
/// `linked`: whether a P-layer link is the model's (kept, or static in its frames: in its generator as it is);
/// `photons`: whether a pair of photon frequencies' cross term is kept -- None where the caller books the photons'
/// cross terms itself (the run's photon classes, which land in every cluster: model.rs `Model::readings`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn errors(fold: &Fold, levels: &Levels, blocks: &[Vec<usize>], linked: &dyn Fn(&Link) -> bool, photons: Option<&dyn Fn((Option<usize>, Option<usize>)) -> bool>, psi: &[C64], velocity: [f64; 3], range: f64) -> Errors {
    let n = fold.size();
    let ground = fold.ground;
    let excited = levels.excited.rows;
    let coefficients: Vec<C64> = (0..n).map(|p| (0..n).map(|i| fold.eigen.inverse[(p, i)] * psi[i]).sum()).collect();
    let total: f64 = coefficients.iter().map(|z| z.norm_sqr()).sum();
    let weight: Vec<f64> = coefficients.iter().map(|z| if total > 0.0 { z.norm_sqr() / total } else { 0.0 }).collect();
    let block_of = |e: usize| blocks.iter().position(|b| b.contains(&e));
    let mut parts: Vec<(Object, f64)> = vec![];
    let add = |object: Object, x: f64, parts: &mut Vec<(Object, f64)>| {
        if x > 0.0 {
            match parts.iter_mut().find(|p| p.0 == object) {
                Some(p) => p.1 += x,
                None => parts.push((object, x)),
            }
        }
    };
    // The remainder.
    let mut remainder = 0.0;
    for p in (0..n).filter(|&p| weight[p] > 0.0) {
        let (mut rho, mut each) = (0.0, vec![]);
        for shift in [range, -range] {
            let (r, e) = fold.remainder(p, shift, excited);
            if r >= rho {
                (rho, each) = (r, e);
            }
        }
        remainder += weight[p] * 2.0 * rho;
        if rho > 0.0 {
            let unit: Vec<C64> = (0..n).map(|q| if q == p { C64::ONE } else { C64::ZERO }).collect();
            for (j, row) in each.iter().enumerate() {
                let mut responses = None;
                for (b, block) in blocks.iter().enumerate() {
                    let x: f64 = block.iter().map(|&e| row[e]).sum();
                    if x > 0.0 {
                        let responses = responses.get_or_insert_with(|| fold.feed_responses(j, &unit));
                        for (class, share) in class_shares(responses, &fold.replicas[j].q, block) {
                            add(Object::Block(b, class), weight[p] * 2.0 * rho * x * share / (rho * rho), &mut parts);
                        }
                    }
                }
            }
        }
    }
    // The transient after the next jump, from the state's own ground populations.
    let mut transient = 0.0;
    if let Some(left) = fold.left_operators() {
        let norm: f64 = psi.iter().map(|z| z.norm_sqr()).sum();
        let metric = fold.metric();
        let landings: Vec<usize> = (0..ground).filter(|&g| psi[g].norm_sqr() > 0.0).collect();
        let bares: Vec<Vec<C64>> = landings.iter().map(|&g| (0..n).map(|i| if i == g { C64::ONE } else { C64::ZERO }).collect()).collect();
        for (&g, (x0, tau)) in landings.iter().zip(fold.transients(&left, &bares).unwrap_or_default()) {
            let population = psi[g].norm_sqr() / norm;
            let held = crate::linalg::inner(&x0, &metric.apply(&x0)).re;
            let big = crate::linalg::norm_sqr(&x0) + held;
            let mx: Vec<C64> = {
                let hx = fold.generator.apply(&x0);
                let mut v = hx.clone();
                let m = metric.apply(&hx);
                v.iter_mut().zip(&m).for_each(|(a, b)| *a += *b);
                v
            };
            let gamma = -2.0 * crate::linalg::inner(&x0, &mx).im / big;
            if !(gamma > 0.0) {
                continue;
            }
            let (wait, clamp) = if big <= 1.0 { (big / gamma, 0.0) } else { ((1.0 + big.ln()) / gamma, (big - 1.0 - big.ln()) / gamma) };
            let x = population * (tau + clamp).abs() / wait;
            transient += x;
            // Its parts by the blocks' shares of the landing's excited amplitude, replica by replica, each split among its
            // classes (`class_shares`).
            let mut shares = vec![vec![0.0; blocks.len()]; fold.replicas.len()];
            for (j, rep) in fold.replicas.iter().enumerate() {
                let y = rep.t.apply(&x0);
                for (i, &e) in rep.q.iter().enumerate() {
                    if let Some(b) = block_of(e) {
                        shares[j][b] += y[i].norm_sqr();
                    }
                }
            }
            let sum: f64 = shares.iter().flatten().sum();
            if sum > 0.0 {
                let coefficients: Vec<C64> = (0..n).map(|p| (0..n).map(|i| fold.eigen.inverse[(p, i)] * x0[i]).sum()).collect();
                for (j, row) in shares.iter().enumerate() {
                    let mut responses = None;
                    for (b, &share) in row.iter().enumerate().filter(|(_, s)| **s > 0.0) {
                        let responses = responses.get_or_insert_with(|| fold.feed_responses(j, &coefficients));
                        for (class, part) in class_shares(responses, &fold.replicas[j].q, &blocks[b]) {
                            add(Object::Block(b, class), x * share * part / sum, &mut parts);
                        }
                    }
                }
            }
        }
    }
    // The averaged components and the photons' cross terms not kept.
    let label = |p: usize| {
        let s = fold.dominant(p);
        if s < ground { s } else { ground + fold.explicit[s - ground] }
    };
    let mut averaged = 0.0;
    for x in fold.components(velocity) {
        let link = Link { a: label(x.row), b: label(x.column), classes: x.classes };
        if linked(&link) || (weight[x.row] == 0.0 && weight[x.column] == 0.0) {
            continue;
        }
        let e = 2.0 * x.mixing() * x.mixing() * (weight[x.row] + weight[x.column]);
        averaged += e;
        add(Object::Link(link), e, &mut parts);
    }
    let mut cross = 0.0;
    if let Some(kept) = photons {
        for x in fold.photons(levels) {
            if kept(x.classes) || weight[x.from] == 0.0 {
                continue;
            }
            let e = x.size / x.beat.abs() * weight[x.from];
            cross += e;
            add(Object::Photons(x.classes.0, x.classes.1), e, &mut parts);
        }
    }
    let sum = remainder + transient + averaged + cross;
    Errors { population: sum, temperature: sum, remainder, transient, averaged, photons: cross, parts }
}

#[cfg(test)]
mod tests;
