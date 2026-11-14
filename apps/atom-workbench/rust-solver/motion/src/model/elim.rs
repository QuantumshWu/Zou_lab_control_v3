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
//! (−∂T/∂f). The first two terms are the model's; the third is the remainder, by the resolvent's identity
//! R(z + Δ) = R − Δ R² + Δ² R² R(z + Δ), z = λ_p + f, exact at any Δ -- no series, no radius. The resolvents of one K
//! commute, so ‖Δ² R² R(z + Δ) W r_p‖ = Δ² ‖R(z + Δ) R² W r_p‖ ≤ Δ² ‖R(z + Δ)‖ ‖R² W r_p‖, and K = H_E − (i/2)D on Q is
//! normal where D is each line's excited states' common rate times one (D = Σ C†C: the line's total decay, the same from
//! every excited state of it) and H_E holds no coupling across lines -- every scene's --, so that ‖R(z + Δ)‖ is exactly
//! 1/min_k |z + Δ − κ_k|, κ_k K's eigenvalues, its Schur form's diagonal (`Schur::distance`; where K is not normal the
//! numerical range's bound 1/dist(z + Δ, W(K)), W(K) within the Gershgorin rectangle of the Hermitian parts). Over the
//! frequencies a P amplitude spans, each component c at its offset δ_c with its share w_c of the amplitude's population,
//! the remainders add as the shares weigh them: the first criterion's relative size is
//!   ρ_p = ‖R² W r_p‖/‖R W r_p‖ · Σ_c w_c δ_c² / min_k |λ_p + f + δ_c − κ_k|,
//! the replicas added in squares, each with its own sum (`Fold::offsets`, `Fold::block_parts`): one product of the
//! resolvent with the response and, per component, its share and its distance from K's spectrum. At small δ it is the
//! second moment of the amplitude's spectrum over the distance; near a resonance of the folded block it grows without
//! bound, and the block is made explicit by its part as any part makes it; at a share of zero it is nothing -- continuous
//! in every coupling, whatever its offset (a far replica's beat that a tilt's ∝ sin θ coupling brings is no resonance
//! of the block).
//!
//! The offsets a P amplitude spans (`Fold::offsets`; ELIMINATION v8): the pattern a moving atom sees, ±2 k_max v (the
//! beams' Doppler shifts, at most k_max v, and their pattern's turn, |k_l − k_l'| v ≤ 2 k_max v), with the amplitude
//! whole; and every component into its eigenstate (# The criteria's numbers: a turning direct entry, or two replicas'
//! coupling) at its combined frequency δ = Ω + λ_p − λ_q -- the same in every gauge, of the actual spectrum -- as a
//! sideband of the first-order amplitude a/|δ|, its share of the population min(1, a/|δ|)². The share is that
//! frequency's spectral weight in the state: an entry a e^{−iΩt} from q′ into q drives, to the first order,
//! x_q ⊃ (a/δ) x_q′ e^{−i(λ_q′ + Ω)t}, a component of the amplitude at λ_q + δ whose part of the norm is (a/|δ|)² of
//! the source's -- the part of the response the fold reads at the shifted frequency, and so the part of its remainder
//! there that is the state's (a rate's relative error ρ on a population share w is w ρ); at |δ| ≤ a the mixing is whole
//! and the share one. A point's components are its fold's eigenstate pairs over the replicas' feed pairs and the
//! turning direct entries (`Fold::components`): on tilt2's prior atoms 2 500 to 10 000 a point, about a thousand into
//! each ground-like eigenstate -- each a term of ρ_p's sum: its share and its distance from K's spectrum, n_Q numbers. Criterion one reads ρ_p of
//! every ground-like eigenstate (`closure`), the error at a state the same at the state's populations: the main line at
//! its eigenstate's, each sideband at its source's times its share (`errors`). A loop's weak closing edge whose frame
//! beat is B is a sideband of a/B on its eigenstates: its term is (a/B)² B² = a², what the coupling's own first order
//! moves the amplitude by. (The frames' residual -- Karp's least largest, a bound of every residual whatever its size
//! -- was read before, the whole amplitude at it: it depended on the frames, and the frames now hold the strongest
//! couplings static, cluster.rs `frames`, the weakest turning at a loop's whole miss.)
//!
//! # The criteria's numbers (`Fold::components`; the photons' on the run's photon classes, `Model::readings`)
//! - The remainder ρ_p of each P eigenstate at the offsets its amplitude spans (# First order in the frequency offsets),
//!   the replicas added in squares (different frequencies), with each excited state's part of it: the excited block that
//!   leaves the most is the one a partition makes explicit (criterion one).
//! - The P-layer components: two replicas f ≠ f' make the entries W'_{f'} T_f of P, turning at f − f' less the
//!   two beams' Doppler beat (k_l − k_l')·v; in H's eigenbasis an entry of size a across the distance
//!   δ = Ω + λ_p − λ_p' mixes a/|δ| (criterion two, v5: its error averaged is 2(a/δ)² -- the population's time mean after the light
//!   is switched on, 4(a/δ)² sin²(δt/2) --, explicit where that is above the share, else averaged into its static shift
//!   |a|²/δ).
//! - The photons' cross terms: the jumps C_a T_f of two replicas land in the same ground states at two photon
//!   frequencies; their interference turns at the difference, and of a rate γ it moves the populations by γ/Ω (the
//!   secular approximation, first order: kept above the share). Read where the photons are the run's own -- its photon
//!   classes, which land in every cluster (`Model::readings`) --, not in the prior's closure: no keep the prior makes
//!   reads them.
use crate::linalg::{schur, Eigen, Mat, C64};
use crate::units::rounding;

/// The atom's levels at a point (rad/µs): the ground block in the lab frame, the excited block in the excited states' frame,
/// and the decay D = Σ_a C_a†C_a on the excited states (C_a the emission, ground ← excited).
#[derive(Clone, Debug)]
pub(crate) struct Levels {
    pub ground: Mat,
    pub excited: Mat,
    pub decay: Mat,
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
/// frame, itself a sum of at most a component per state along the frames' paths (cluster.rs `frames`): γ_STATES of their sizes.
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
    /// The Schur form of K_f = (H_E − (i/2)D) on Q_f.
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
    /// 1/‖(z − A)⁻¹‖ (module documentation, # First order in the frequency offsets): where A is normal -- its Schur form
    /// diagonal to the rounding of its entries -- exactly the distance of z from A's nearest eigenvalue; otherwise a bound
    /// of it from below, z's distance from the Gershgorin rectangle of the Hermitian parts (S + S†)/2 and (S − S†)/2i,
    /// which holds A's numerical range W(A) (‖(z − A)⁻¹‖ ≤ 1/dist(z, W(A))); zero where z lies in it.
    fn distance(&self, z: C64) -> f64 {
        let n = self.n;
        let scale = self.s.iter().map(|x| x.abs()).fold(0.0, f64::max);
        let normal = (0..n).all(|i| (i + 1..n).all(|j| self.s[i * n + j].abs() <= rounding(n) * scale));
        if normal {
            return (0..n).map(|i| (z - self.s[i * n + i]).abs()).fold(f64::INFINITY, f64::min);
        }
        let mut range = [f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY];
        for i in 0..n {
            let (mut re, mut im) = (0.0, 0.0);
            for j in (0..n).filter(|&j| j != i) {
                let (a, b) = (self.s[i * n + j], self.s[j * n + i].conj());
                (re, im) = (re + 0.5 * (a + b).abs(), im + 0.5 * (a - b).abs());
            }
            let d = self.s[i * n + i];
            range = [range[0].min(d.re - re), range[1].max(d.re + re), range[2].min(d.im - im), range[3].max(d.im + im)];
        }
        let (x, y) = ((range[0] - z.re).max(z.re - range[1]).max(0.0), (range[2] - z.im).max(z.im - range[3]).max(0.0));
        x.hypot(y)
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
/// step's operations (`terms`) times the scale, or no longer falls.
fn settle(start: &Mat, mut step: impl FnMut(&Mat) -> (Mat, f64), terms: usize) -> (Mat, bool) {
    let (mut x, mut last, mut iterations, mut first) = (start.clone(), f64::INFINITY, 0, None);
    loop {
        iterations += 1;
        let (next, scale) = step(&x);
        let change = difference(&next, &x);
        x = next;
        if change <= rounding(terms) * scale {
            return (x, true);
        }
        match first {
            None => first = Some(change),
            Some(f) if iterations == 2 && change > 0.25 * f => return (x, false),
            _ => {}
        }
        if !(change < last) {
            return (x, true);
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
    let mut forms: Vec<(Vec<usize>, Schur)> = vec![];
    let mut folded: Vec<Folded> = replicas
        .frequencies
        .iter()
        .zip(&replicas.groups)
        .map(|(&f, &group)| {
            let q: Vec<usize> = (0..excited).filter(|&e| !slow.explicit.iter().enumerate().any(|(i, &x)| x == e && slow.groups[ground + i] == group && same(frame(ground + i), f))).collect();
            if !forms.iter().any(|x| x.0 == q) {
                let k = Schur::of(&k_full.block(&q, &q)).expect("the Schur form of the excited block");
                forms.push((q.clone(), k));
            }
            let (_, k) = forms.iter().find(|x| x.0 == q).expect("its form");
            Folded { frequency: f, group: group_of(group), q, feeds: vec![], w: Mat::zeros(0, 0), back: Mat::zeros(0, 0), t: Mat::zeros(0, 0), k: k.clone() }
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
    let (h, exists) = match reference {
        None => settle(&a, step, (n + excited) * (n + excited)),
        Some(z0) => (step(&Mat::from_fn(n, n, |i, j| if i == j { C64::real(z0) } else { C64::ZERO })).0, true),
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
    Fold { ground, explicit: slow.explicit.clone(), frames: slow.frames.clone(), direct: a, generator: h, replicas: folded, eigen, exists, reference, turning, driven, members, groups, schurs: sh }
}

/// One P-layer component's reading (criterion two): its size a in H's eigenbasis, its distance δ = Ω + λ_p − λ_p',
/// its residual beat Ω, the eigenstates (p' ← p) and what turns it (two classes, or a turning direct entry).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Component {
    pub row: usize,
    pub column: usize,
    pub size: f64,
    pub distance: C64,
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

impl Fold {
    /// P's size.
    pub(crate) fn size(&self) -> usize {
        self.generator.rows
    }
    /// The right eigenvector r_p (unit) of H.
    pub(crate) fn right(&self, p: usize) -> Vec<C64> {
        (0..self.size()).map(|i| self.eigen.vectors[(i, p)]).collect()
    }
    /// Its ground weight ‖P_G r_p‖².
    pub(crate) fn ground_weight(&self, p: usize) -> f64 {
        (0..self.ground).map(|i| self.eigen.vectors[(i, p)].norm_sqr()).sum()
    }
    /// The P state an eigenstate holds most of.
    pub(crate) fn dominant(&self, p: usize) -> usize {
        (0..self.size()).max_by(|&a, &b| self.eigen.vectors[(a, p)].norm_sqr().total_cmp(&self.eigen.vectors[(b, p)].norm_sqr())).unwrap_or(0)
    }

    /// The offsets eigenstate p's amplitude spans, per replica (module documentation, # First order in the frequency
    /// offsets): Σ_c w_c δ_c² / ‖R(λ_p + f + δ_c)‖⁻¹ at each replica of p's group (`Schur::distance`; none at the others)
    /// -- `pattern` (2 k_max v) at the share `main`, the larger of its two signs, and each component into p of
    /// `components` (`components`, at the atom's velocity) at its combined frequency δ's real part, at its source's share
    /// `source(p′)` times its own, min(1, a/|δ|)².
    pub(crate) fn offsets(&self, p: usize, pattern: f64, components: &[Component], main: f64, source: &dyn Fn(usize) -> f64) -> Vec<f64> {
        let lambda = self.eigen.values[p];
        let into: Vec<(f64, f64)> = components.iter().filter(|x| x.row == p && x.size > 0.0).map(|x| (source(x.column) * x.mixing().min(1.0).powi(2), x.distance.re)).filter(|x| x.0 > 0.0).collect();
        self.replicas
            .iter()
            .map(|rep| {
                if rep.group != self.groups[p] {
                    return 0.0;
                }
                let z = lambda + C64::real(rep.frequency);
                let term = |delta: f64| delta * delta / rep.k.distance(z + C64::real(delta));
                let main = if main > 0.0 { main * term(pattern).max(term(-pattern)) } else { 0.0 };
                main + into.iter().map(|&(share, delta)| share * term(delta)).sum::<f64>()
            })
            .collect()
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
        let lambda = &self.eigen.values;
        let doppler = |k: [f64; 3]| k[0] * velocity[0] + k[1] * velocity[1] + k[2] * velocity[2];
        // Per group its eigenstates (# Groups: each is one group's), and the left and right eigenvectors on its places:
        // an entry block of the places of group `to` ← group `from` is, in H's eigenbasis, l_to · m · r_from.
        let groups = self.members.len();
        let states: Vec<Vec<usize>> = (0..groups).map(|g| (0..self.size()).filter(|&p| self.groups[p] == g).collect()).collect();
        let lefts: Vec<Mat> = (0..groups).map(|g| Mat::from_fn(states[g].len(), self.members[g].len(), |a, x| self.eigen.inverse[(states[g][a], self.members[g][x])])).collect();
        let rights: Vec<Mat> = (0..groups).map(|g| Mat::from_fn(self.members[g].len(), states[g].len(), |x, b| self.eigen.vectors[(self.members[g][x], states[g][b])])).collect();
        let mut out = vec![];
        for (i, rep) in self.replicas.iter().enumerate() {
            let columns = &self.members[rep.group];
            for feed in &rep.feeds {
                let response = sylvester_in(&rep.k, &self.schurs[rep.group], rep.frequency, &feed.w, columns);
                for (j, other) in self.replicas.iter().enumerate() {
                    if i == j {
                        continue;
                    }
                    let rows = &self.members[other.group];
                    for back in &other.feeds {
                        // The other replica's light emitted from this one's excited amplitude: its coupling back from Q_f,
                        // on the other replica's group's places ← this one's.
                        let m = Mat::from_fn(rows.len(), columns.len(), |r, c| rep.q.iter().enumerate().map(|(y, &e)| back.emitted[(rows[r], e)] * response[(y, columns[c])]).sum());
                        if m.max_abs() == 0.0 {
                            continue;
                        }
                        let beat = rep.frequency - other.frequency - (doppler(feed.wave) - doppler(back.wave));
                        let e = lefts[other.group].product(&m.product(&rights[rep.group]));
                        let floor = rounding(self.size() + rep.q.len()) * back.emitted.frobenius() * response.frobenius();
                        for (b, &p) in states[rep.group].iter().enumerate() {
                            for (a, &q) in states[other.group].iter().enumerate() {
                                let size = e[(a, b)].abs();
                                if size <= floor {
                                    continue;
                                }
                                out.push(Component { row: q, column: p, size, distance: C64::real(beat) + lambda[p] - lambda[q], classes: (back.class, feed.class) });
                            }
                        }
                    }
                }
            }
        }
        // A turning direct entry row ← column: of the eigenstates of their two groups, l_q[row] value r_p[column].
        let group_of = |x: usize| self.members.iter().position(|m| m.contains(&x)).expect("a place's group");
        for &(row, column, beat, value) in &self.turning {
            for &p in &states[group_of(column)] {
                for &q in &states[group_of(row)] {
                    let size = (self.eigen.inverse[(q, row)] * value * self.eigen.vectors[(column, p)]).abs();
                    if size > 0.0 {
                        out.push(Component { row: q, column: p, size, distance: C64::real(beat) + lambda[p] - lambda[q], classes: (None, None) });
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

    /// The replicas' responses about z₀ to order K (`series`): per replica R₀^{k+1}W, k ≤ K, R₀ = (z₀ + f − K)⁻¹ on its
    /// group's places.
    pub(crate) fn responses_about(&self, z0: C64, orders: usize) -> Vec<Vec<Mat>> {
        self.replicas
            .iter()
            .map(|rep| {
                let sigma = z0 + C64::real(rep.frequency);
                let mut x = rep.w.clone();
                (0..=orders)
                    .map(|_| {
                        for &j in &self.members[rep.group] {
                            let column: Vec<C64> = (0..x.rows).map(|i| x[(i, j)]).collect();
                            let solved = rep.k.resolvent(sigma, &column);
                            for i in 0..x.rows {
                                x[(i, j)] = solved[i];
                            }
                        }
                        x.clone()
                    })
                    .collect()
            })
            .collect()
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
    /// The excited state that sets the series' distance (`series_distance`): of the excited block's Schur forms over the
    /// replicas -- each block of the states K's entries join, where the drive reaches one of them --, the diagonal entry
    /// nearest z₀ + f, the state its Schur vector holds most of (E's index): the block criterion one makes explicit where
    /// the series does not converge (`Series::moved`), and on which a run's state books what the evaluation's orders
    /// leave of its error against the definition (`Model::readings`).
    pub(crate) fn nearest(&self, z0: C64) -> Option<usize> {
        let mut best: Option<(f64, usize)> = None;
        for rep in &self.replicas {
            let driven = |state: usize| (0..rep.w.cols).any(|s| rep.w[(state, s)] != C64::ZERO);
            for b in rep.k.blocks.iter().filter(|b| b.states.iter().any(|&state| driven(state))) {
                let m = b.states.len();
                for k in 0..m {
                    let distance = (z0 + C64::real(rep.frequency) - b.s[k * m + k]).abs();
                    if best.is_none_or(|x| distance < x.0) {
                        let most = (0..m).max_by(|&x, &y| b.q[k * m + x].norm_sqr().total_cmp(&b.q[k * m + y].norm_sqr())).expect("a block's state");
                        best = Some((distance, rep.q[b.states[most]]));
                    }
                }
            }
        }
        best.map(|b| b.1)
    }
    /// The excited state that holds the most of the excited share X = Σ_f T_f†T_f: of the responses' rows, the largest
    /// Σ_s |T[e, s]|² over the replicas (E's index).
    pub(crate) fn most_excited(&self) -> Option<usize> {
        let mut weight: Vec<(usize, f64)> = vec![];
        for rep in &self.replicas {
            for (i, &e) in rep.q.iter().enumerate() {
                let w: f64 = (0..rep.t.cols).map(|s| rep.t[(i, s)].norm_sqr()).sum();
                match weight.iter_mut().find(|x| x.0 == e) {
                    Some(x) => x.1 += w,
                    None => weight.push((e, w)),
                }
            }
        }
        weight.into_iter().filter(|x| x.1 > 0.0).max_by(|a, b| a.1.total_cmp(&b.1)).map(|x| x.0)
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

    /// The parts of an eigenstate p's response (or, with the `offsets` it spans per replica, `offsets`, of its remainder
    /// there, ‖R² W r_p‖ by each replica's sum: module documentation, # First order in the frequency offsets) per excited
    /// block and
    /// replica, each split among the replica's classes (`class_shares`): (block, replica, class, part) -- of the replicas
    /// of p's own group; and the response's (or remainder's) size. What the closure moves a block by (`closure`), and
    /// what a state's error on a block is split by (`Fold::state_block_parts`).
    pub(crate) fn block_parts(&self, blocks: &[Vec<usize>], p: usize, offsets: Option<&[f64]>) -> (f64, Vec<(usize, usize, Option<usize>, f64)>) {
        let block_of = |e: usize| blocks.iter().position(|b| b.contains(&e));
            let (n, r) = (self.size(), self.right(p));
            let lambda = self.eigen.values[p];
            let unit: Vec<C64> = (0..n).map(|q| if q == p { C64::ONE } else { C64::ZERO }).collect();
            let mut parts = vec![];
            let (mut num, mut den) = (0.0, 0.0);
            for (j, rep) in self.replicas.iter().enumerate().filter(|(_, rep)| rep.group == self.groups[p]) {
                let wr = rep.w.apply(&r);
                let sigma = lambda + C64::real(rep.frequency);
                let x0 = rep.k.resolvent(sigma, &wr);
                let d: Vec<C64> = match offsets {
                    None => x0.clone(),
                    Some(offsets) => rep.k.resolvent(sigma, &x0).into_iter().map(|z| z * offsets[j]).collect(),
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
                    let responses = responses.get_or_insert_with(|| self.feed_responses(j, &unit));
                    for (class, share) in class_shares(responses, &rep.q, &blocks[b]) {
                        parts.push((b, j, class, x * share));
                    }
                }
            }
            let scale = if offsets.is_some() { den } else { 1.0 };
            if scale > 0.0 {
                parts.iter_mut().for_each(|x| x.3 /= scale);
            }
            (if offsets.is_some() { if den > 0.0 { (num / den).sqrt() } else { 0.0 } } else { num.sqrt() }, parts)
            }

    /// A state ψ's error `x` on the excited block b, split as an eigenstate's part is (`block_parts`): over the state's
    /// eigenstates by their weights |⟨l_p|ψ⟩|², each by its replicas' and classes' parts on the block; the classes' shares
    /// of x (none where nothing of the state reaches the block).
    pub(crate) fn state_block_parts(&self, blocks: &[Vec<usize>], b: usize, psi: &[C64], x: f64) -> Vec<(Option<usize>, f64)> {
        let n = self.size();
        let coefficients: Vec<C64> = (0..n).map(|p| (0..n).map(|i| self.eigen.inverse[(p, i)] * psi[i]).sum()).collect();
        let total: f64 = coefficients.iter().map(|z| z.norm_sqr()).sum();
        let mut shares: Vec<(Option<usize>, f64)> = vec![];
        for p in (0..n).filter(|&p| total > 0.0 && coefficients[p].norm_sqr() > 0.0) {
            let weight = coefficients[p].norm_sqr() / total;
            for (_, _, class, part) in self.block_parts(blocks, p, None).1.into_iter().filter(|part| part.0 == b) {
                match shares.iter_mut().find(|s| s.0 == class) {
                    Some(s) => s.1 += weight * part,
                    None => shares.push((class, weight * part)),
                }
            }
        }
        let sum: f64 = shares.iter().map(|s| s.1).sum();
        if !(sum > 0.0) {
            return vec![];
        }
        shares.into_iter().map(|(class, s)| (class, x * s / sum)).collect()
    }

    /// The block move where a series does not converge on the excited block b (criterion one's existence, as the closure
    /// moves where a fold does not exist, `closure`): of the ground-like eigenstates' parts on b, the largest of a class
    /// -- its replica's, its eigenstate's group's ground places --; none where no class reaches b.
    pub(crate) fn block_move(&self, blocks: &[Vec<usize>], b: usize) -> Option<(usize, Vec<usize>)> {
        let mut largest: Option<(usize, usize, f64)> = None;
        for p in (0..self.size()).filter(|&p| self.ground_weight(p) >= 0.5) {
            for (_, _, class, x) in self.block_parts(blocks, p, None).1.into_iter().filter(|part| part.0 == b) {
                if let Some(c) = class.filter(|_| x > largest.map_or(0.0, |m| m.2)) {
                    largest = Some((c, p, x));
                }
            }
        }
        let (class, p, _) = largest?;
        let group = self.groups[self.dominant(p)];
        Some((class, self.members[group].iter().copied().filter(|&place| place < self.ground).collect()))
    }

    /// The fold's largest excited share x, X's largest eigenvalue (`metric`; `Series`): the normalization's series
    /// S⁻¹ = Σ_j b_j X^j converges where x < 1.
    pub(crate) fn excited_share(&self) -> f64 {
        crate::linalg::hermitian_eigen(&self.metric()).values.iter().fold(0.0_f64, |m, &v| m.max(v))
    }

    /// The evaluation's normalized generator at the orders `orders` (assemble.rs `normal`; ELIMINATION v7) of this fold's
    /// series about z₀ -- the reference's forms, what the tables hold --: H₀ = Herm(A + N₀) − (i/2)L, L = Σ (R₀W)†D(R₀W),
    /// at K = 1 H = H₀ − N₁(H₀ − z₀); its Hermitian part Herm H and its decay S_J⁻¹ (Σ T†D T) S_J⁻¹ of the jumps' wave
    /// operator T = R₀W (− R₀²W E at K = 1, E = H − z₀) and S_J⁻¹ = Σ_{j≤J} b_j X^j, X = Σ (R₀W)†(R₀W): Herm H −
    /// (i/2) decay.
    pub(crate) fn evaluated(&self, levels: &Levels, z0: C64, orders: Orders) -> Mat {
        let n = self.size();
        let responses = self.responses_about(z0, orders.series);
        let forms = |k: usize| {
            let mut m = Mat::zeros(n, n);
            for (rep, r) in self.replicas.iter().zip(&responses) {
                m.add_scaled(C64::ONE, &rep.back.product(&r[k]));
            }
            m
        };
        let (mut metric, mut loss) = (Mat::zeros(n, n), Mat::zeros(n, n));
        for (rep, r) in self.replicas.iter().zip(&responses) {
            metric.add_scaled(C64::ONE, &r[0].adjoint_product(&r[0]));
            loss.add_scaled(C64::ONE, &r[0].adjoint_product(&levels.decay.block(&rep.q, &rep.q).product(&r[0])));
        }
        let mut whole = self.direct.clone();
        whole.add_scaled(C64::ONE, &forms(0));
        let mut h = whole.hermitian_part();
        h.add_scaled(C64::new(0.0, -0.5), &loss);
        let offset = |h: &Mat| {
            let mut e = h.clone();
            for i in 0..n {
                e[(i, i)] -= z0;
            }
            e
        };
        if orders.series >= 1 {
            let next = forms(1).product(&offset(&h));
            h.add_scaled(-C64::ONE, &next);
        }
        let e = offset(&h);
        let mut inner = Mat::zeros(n, n);
        for (rep, r) in self.replicas.iter().zip(&responses) {
            let mut wave = r[0].clone();
            if orders.series >= 1 {
                wave.add_scaled(-C64::ONE, &r[1].product(&e));
            }
            inner.add_scaled(C64::ONE, &wave.adjoint_product(&levels.decay.block(&rep.q, &rep.q).product(&wave)));
        }
        let mut bent = Mat::identity(n);
        let mut power = Mat::identity(n);
        for j in 1..=orders.normalization {
            power = power.product(&metric);
            bent.add_scaled(C64::real(binomial(-0.5, j)), &power);
        }
        let mut out = h.hermitian_part();
        out.add_scaled(C64::new(0.0, -0.5), &bent.product(&inner).product(&bent));
        out
    }

    /// The definition's normalized generator (ELIMINATION v6, v7): of this fold -- the self-consistent one, its excited
    /// block whole (`Model::definition_fold`) -- the des Cloizeaux generator's Hermitian part Herm(S H S⁻¹) and its decay
    /// S⁻¹(Σ_f T_f†D T_f)S⁻¹, S = (1 + X)^{1/2} of its own metric (`normalization`): Herm(S H S⁻¹) − (i/2) decay.
    pub(crate) fn defined(&self, levels: &Levels) -> Mat {
        let n = self.size();
        let (s, s_inverse) = self.normalization();
        let mut loss = Mat::zeros(n, n);
        for rep in &self.replicas {
            loss.add_scaled(C64::ONE, &rep.t.adjoint_product(&levels.decay.block(&rep.q, &rep.q).product(&rep.t)));
        }
        let mut out = s.product(&self.generator).product(&s_inverse).hermitian_part();
        out.add_scaled(C64::new(0.0, -0.5), &s_inverse.product(&loss).product(&s_inverse));
        out
    }

    /// The evaluation's error on a state ψ against the definition, measured (ELIMINATION v7; a unit's error where the atoms
    /// went): r = ‖(G_e − G_d)ψ‖ / ‖(G_d − z₀)ψ‖ of the two normalized generators (`evaluated`, `defined`), a rate's
    /// relative error 2r + r². Every truncation of the evaluation -- the series' order, the normalization's, the
    /// excited block the reference leaves out (δK), the Hermitian part's normalization -- in the one comparison.
    pub(crate) fn evaluation_error(evaluated: &Mat, defined: &Mat, z0: C64, psi: &[C64]) -> f64 {
        let (e, d) = (evaluated.apply(psi), defined.apply(psi));
        rate_error(norm_of(e.iter().zip(&d).map(|(a, b)| *a - *b)), norm_of(d.iter().zip(psi).map(|(d, p)| *d - z0 * *p)))
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
        let (_, settled) = settle(
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

    /// The transient of bare states f (ELIMINATION v3 3 and 4, one term): what the waiting time for the next photon from
    /// f holds beyond its slow part's, E_exact − ∫‖slow‖² dt. The bare state is its slow part (x₀, T x₀) and its fast
    /// part (−S y₀, y₀), y₀ = −T x₀, x₀ = (1 + Σ_f S_f T_f)⁻¹ f its landing; the fast part is the homogeneous solution of
    /// the excited block, e^{−i(K − f)t} -- it decays at the excited states' Γ/2 within 1/Γ, and it is the same term
    /// whether the bare state is a jump's landing or the light being switched on. Its survival adds ∫‖fast‖² +
    /// 2 Re∫⟨fast|slow⟩; to the leading order in the fold, per replica,
    ///   τ = (T x₀)† 𝓛 (T x₀) − 2 Re (T x₀)† 𝒳 x₀,
    /// 𝓛 = ∫ e^{iK†t} e^{−iKt} dt (𝓛 K − K† 𝓛 = −i; 𝓛 = 1/Γ on a line of one width) and 𝒳 = ∫ e^{i(K−f)†t} (T − S†)
    /// e^{−iHt} dt (𝒳 H − (K† − f) 𝒳 = −i(T − S†)): the norm the fast part holds over its life, and its interference
    /// with the slow part's excited and ground amplitudes. Without decay S† = T and only the first term is left. On one
    /// line it is the whole difference between the mean scattering rate after a jump and the optical Bloch steady state
    /// beside the threshold's clamp (tests). Per f (x₀, τ), each replica's 𝓛 and 𝒳 and the landing's overlap made once.
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
    let levels = Levels { ground: h.block(&ground, &ground), excited: h.block(&excited, &excited), decay };
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

/// A model's series constants (module documentation, # The definition and the route it is evaluated on; ELIMINATION v5,
/// v7): per cluster the centre c of its near light's scalar shift over the atoms read (the reference z₀ is the cluster's
/// far levels' mean and c), the series' largest ratio q over them and the largest excited share x (X's largest
/// eigenvalue) -- each series' existence, q < 1 and x < 1 (the Neumann series of the resolvent and of (1 + X)^{−1/2}
/// converge: criterion one's condition, `Series::exists`), nothing booked of them --, and where one does not, the block
/// move on the excited state that stops it -- the one that sets the distance where the ratio is largest
/// (`Fold::nearest`), or that holds the most of the excited share where x is largest (`Fold::most_excited`) --; and the
/// orders every evaluation takes, the kept units' (`Series::kept_orders`):
/// their errors are measured where the atoms went (`Model::readings`).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Series {
    pub centre: Vec<f64>,
    pub ratio: f64,
    pub order: usize,
    pub excited: f64,
    pub normalization: usize,
    /// Where a series does not converge: the block move (`Fold::block_move`) -- the block, the class (a carrier), the
    /// cluster's ground states of the eigenstate's group.
    pub moved: Option<(usize, usize, Vec<usize>)>,
}

impl Series {
    /// The series of no light: every order 0 (a model's own before its atoms are read, `Model::referenced`).
    pub(crate) fn none(clusters: usize) -> Series {
        Series { centre: vec![0.0; clusters], ratio: 0.0, order: 0, excited: 0.0, normalization: 0, moved: None }
    }

    /// Whether both series converge at the atoms read: q < 1 and x < 1 (`Series`; where one does not, criterion one makes
    /// a block explicit, `Model::series_block`).
    pub(crate) fn exists(&self) -> bool {
        self.ratio < 1.0 && self.excited < 1.0
    }

    /// The orders the kept units ask (`Unit::Series`, `Unit::Normalization`): the series' first where it is kept, else
    /// order 0; the normalization's highest kept order, else 0.
    pub(crate) fn kept_orders(kept: &[super::Unit]) -> (usize, usize) {
        let normalization = kept.iter().filter_map(|u| if let super::Unit::Normalization { order } = u { Some(*order) } else { None }).max().unwrap_or(0);
        (usize::from(kept.contains(&super::Unit::Series)), normalization)
    }

    /// The orders an evaluation takes.
    pub(crate) fn orders(&self) -> Orders {
        Orders { series: self.order, normalization: self.normalization }
    }
}

/// The orders of the normalized generator's evaluation (ELIMINATION v7; `Series`): the series' K (0, or 1 where
/// `Unit::Series` is kept: the near forms hold N₀ and N₁), the normalization's J. The Hermitian part is the series' own:
/// what the normalization would do to it is in the evaluation's error against the definition (`Fold::evaluation_error`),
/// booked on the block nearest the reference (`Model::readings`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Orders {
    pub series: usize,
    pub normalization: usize,
}

/// A vector's 2-norm.
fn norm_of(v: impl Iterator<Item = C64>) -> f64 {
    v.map(|z| z.norm_sqr()).sum::<f64>().sqrt()
}

/// A rate's relative error 2r + r² of an amplitude's r = `apart`/`scale` (of T†T): nothing where nothing is apart, all
/// of it where the scale is nothing and something is.
fn rate_error(apart: f64, scale: f64) -> f64 {
    if apart == 0.0 {
        return 0.0;
    }
    let r = apart / scale;
    if r.is_finite() { (2.0 * r + r * r).min(1.0) } else { 1.0 }
}

/// The binomial coefficient binom(α, j) = α(α − 1)…(α − j + 1)/j!: the power series' j-th coefficient of (1 + x)^α.
pub(crate) fn binomial(alpha: f64, j: usize) -> f64 {
    (0..j).fold(1.0, |c, i| c * (alpha - i as f64) / (i + 1) as f64)
}

impl super::Model {
    /// The fold of a model's cluster at a point, as the run's series stands for it (`Series`): the levels with the far
    /// light alone (eliminate.rs `Model::far_hamiltonian`), every carrier a class of its coherent field's couplings
    /// folded at the atom and explicit, the cluster's explicit states and frames; with the far levels' mean in the
    /// frames and the norm of what the far levels leave out of the excited block -- the carriers' own static forms there,
    /// bilinear in the near field (eliminate.rs `level_pair`) -- in the Frobenius norm, a bound of the spectral one.
    pub(crate) fn cluster_fold(&self, cluster: usize, amplitudes: &[crate::field::Amplitude], about: bool) -> Option<(Fold, Levels, f64, f64)> {
        let (fold, levels, far, left) = self.cluster_fold_with(cluster, amplitudes, about, false)?;
        Some((fold, levels, far, left.frobenius()))
    }

    /// The definition at a point (ELIMINATION v6, v7): the cluster's self-consistent fold with its excited block whole --
    /// the far levels and the carriers' own static forms on it (δK, which the series' reference leaves out) --, and its
    /// levels: what every truncation of the evaluation is measured against (`Fold::evaluation_error`).
    pub(crate) fn definition_fold(&self, cluster: usize, amplitudes: &[crate::field::Amplitude]) -> Option<(Fold, Levels)> {
        self.cluster_fold_with(cluster, amplitudes, false, true).map(|(fold, levels, _, _)| (fold, levels))
    }

    /// `cluster_fold`, about the reference or self-consistent, its excited block the far levels' or, `whole`, with the
    /// carriers' own static forms on it; the fold, its levels, the far levels' mean and those forms.
    fn cluster_fold_with(&self, cluster: usize, amplitudes: &[crate::field::Amplitude], about: bool, whole: bool) -> Option<(Fold, Levels, f64, Mat)> {
        use crate::request::{GROUND, STATES};
        let running = self.running();
        let h = &self.far_hamiltonian(amplitudes, &running, 0)[0];
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
        let mut block = h.block(&excited, &excited);
        if whole {
            block.add_scaled(C64::ONE, &left);
        }
        let levels = Levels { ground: h.block(&grounds, &grounds), excited: block, decay: self.decay.clone() };
        let frames: Vec<f64> = c.frames.iter().map(|f| self.omega(f)).collect();
        let far = (0..g).map(|y| h[(grounds[y], grounds[y])].re - frames[y]).sum::<f64>() / g as f64;
        let slow = Slow::single(c.states[g..].iter().map(|&s| s - GROUND).collect(), frames);
        let replicas = Replicas::of(&classes, &slow, g);
        let fold = fold_about(&levels, &classes, &slow, &replicas, about.then(|| far + self.series.centre[cluster]));
        Some((fold, levels, far, left))
    }

    /// The model's series constants at the atoms `points` (`Series`): per cluster the centre of the near light's scalar
    /// shift -- the mean of the fold's ground-like eigenvalues less the far levels' mean, over the atoms its range's
    /// middle --, then at each atom the ratio of the series about z₀ = far mean + c, q = (‖H − z₀‖₂ + ‖δK‖)/d (the
    /// Neumann series of the resolvent in both its arguments: `Fold::series_norm` over `Fold::series_distance`, and the excited block the reference
    /// leaves out), the largest; and the order it asks for. An atom where a fold does not exist (criterion one's: its
    /// closure makes a block explicit) is read as the series' end: q = ∞. And at each atom the fold's excited share x
    /// (`Fold::excited_share`), the largest. Where either series does not converge, the closure's block move on the state
    /// that stops it (`Series::moved`).
    pub(crate) fn series_at(&self, points: &[([f64; 3], [f64; 3])]) -> Series {
        use rayon::prelude::*;
        let clusters = self.clusters.len();
        let read: Vec<Vec<Option<(f64, Fold, f64, f64, f64)>>> = points
            .par_iter()
            .map(|&(r, _)| {
                let amplitudes = self.light.amplitudes(r);
                (0..clusters)
                    .map(|cluster| {
                        let (fold, _, far, left) = self.cluster_fold(cluster, &amplitudes, false)?;
                        let like: Vec<usize> = (0..fold.size()).filter(|&p| fold.ground_weight(p) >= 0.5).collect();
                        let shift = like.iter().map(|&p| fold.eigen.values[p].re).sum::<f64>() / like.len().max(1) as f64 - far;
                        let share = fold.excited_share();
                        Some((shift, fold, far, left, share))
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
        let (ratio, nearest) = read
            .iter()
            .flat_map(|at| at.iter().enumerate())
            .filter_map(|(cluster, x)| x.as_ref().map(|x| (cluster, x)))
            .map(|(cluster, (_, fold, far, left, _))| {
                let z0 = C64::real(far + centre[cluster]);
                let ratio = if fold.exists { (fold.series_norm(z0) + left) / fold.series_distance(z0) } else { f64::INFINITY };
                (ratio, (cluster, fold, z0))
            })
            .fold((0.0, None), |(most, at), (ratio, of)| if ratio > most { (ratio, Some(of)) } else { (most, at) });
        let densest = read
            .iter()
            .flat_map(|at| at.iter().enumerate())
            .filter_map(|(cluster, x)| x.as_ref().map(|x| (cluster, x)))
            .fold((f64::NEG_INFINITY, None), |(most, at), (cluster, x)| if x.4 > most { (x.4, Some((cluster, &x.1))) } else { (most, at) })
            .1;
        let excited = read.iter().flatten().flatten().fold(0.0_f64, |m, x| m.max(x.4));
        // Where a series does not converge, the closure's move on the block that stops it (criterion one).
        let blocks = self.blocks();
        let move_on = |cluster: usize, fold: &Fold, e: Option<usize>| -> Option<(usize, usize, Vec<usize>)> {
            let b = e.and_then(|e| blocks.iter().position(|b| b.contains(&e)))?;
            let (class, places) = fold.block_move(&blocks, b)?;
            Some((b, class, places.iter().map(|&place| self.clusters[cluster].states[place]).collect()))
        };
        let moved = if !(ratio < 1.0) {
            nearest.and_then(|(cluster, fold, z0)| move_on(cluster, fold, fold.nearest(z0)))
        } else if !(excited < 1.0) {
            densest.and_then(|(cluster, fold)| move_on(cluster, fold, fold.most_excited()))
        } else {
            None
        };
        let (order, normalization) = Series::kept_orders(&self.kept);
        Series { centre, ratio, order, excited, normalization, moved }
    }

    /// The excited hyperfine blocks F′ of the model's states (`blocks`), from its labels.
    pub(crate) fn blocks(&self) -> Vec<Vec<usize>> {
        excited_blocks(&self.labels)
    }

    /// Where a series does not converge at the model's atoms (`Series::exists`: criterion one's condition, of the same
    /// kind as a fold that does not exist): the block the closure moves there (`Series::moved`, `Fold::block_move`) --
    /// the block of the state that sets the series' distance where its ratio is largest (q ≥ 1), or of the state that
    /// holds the most of the excited share (x ≥ 1); its class and its eigenstate's group's ground states --; none where
    /// both converge.
    pub(crate) fn series_block(&self) -> Option<super::Unit> {
        let (block, class, grounds) = self.series.moved.as_ref()?;
        Some(super::Unit::Block { block: *block, component: super::Component { frequency: self.carriers[*class].frequency, counter: false }, grounds: ground_set(grounds) })
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
        // The evaluation's next orders where a fold has the normalized generator (`Unit::Series`, `Unit::Normalization`).
        let running = self.running();
        if (0..self.clusters.len()).any(|c| self.plan(c, &running).fourth.is_some()) {
            if self.series.order == 0 {
                out.push(super::Unit::Series);
            }
            out.push(super::Unit::Normalization { order: self.series.normalization + 1 });
        }
        out.retain(|u| !self.kept.contains(u));
        out
    }

    /// The candidates' errors at a trajectory state (`candidates`; ELIMINATION.md §6: the prior's function at a run's
    /// state): the fold of the sample's cluster at its point (`cluster_fold`, the definition the run's series stands for)
    /// read by `errors` at the state, the atom's velocity and the offsets its amplitude spans -- the pattern its moving
    /// atom sees, 2 k_max v, and the components' combined frequencies over their distances from K's spectrum
    /// (`Fold::offsets`) --, each part on its
    /// object's unit: a block's on `Unit::Block`, a link's on the `Unit::Term` of its two states at its lab beat. The
    /// photons' cross terms on the run's photon classes, which land in every cluster: of each pair landing in one, the
    /// rate of their interference at the state γ = |Σ_a ⟨u_a|u′_a⟩| over the state's norm, against their beat Ω -- γ/|Ω|
    /// (v3 1: the secular approximation's first order), on `Unit::Coherence`. A relative rate error x moves the
    /// populations by at most x (absolute) and T by x (relative). And beside the units, per transition to a level
    /// outside the model (`Model::outside`), its error at the state (`errors`; `outside_errors` where no carrier folds:
    /// the P states' populations alone), booked so too.
    pub(crate) fn readings(&self, units: &[super::Unit], sample: &super::Sample, tables: Option<&super::tabulate::Tables>) -> (Vec<super::UnitError>, Vec<super::UnitError>) {
        use crate::request::GROUND;
        use super::{Unit, UnitError};
        let mut out = vec![UnitError::NONE; units.len()];
        let book = |unit: &Unit, x: f64, out: &mut Vec<UnitError>| {
            if let Some(i) = units.iter().position(|u| u == unit) {
                let e = &mut out[i];
                (e.population, e.temperature) = (e.population + x, e.temperature + x);
            }
        };
        let c = &self.clusters[sample.cluster];
        let g = c.ground;
        let amplitudes = self.light.amplitudes(sample.at.r);
        let velocity = sample.at.kappa.map(|k| k * self.hbar_over_mass);
        let speed = crate::linalg::norm3(velocity);
        let outside: Vec<Outside> = self
            .outside_at(&amplitudes)
            .iter()
            .map(|(m, o)| Outside { slow: c.states.iter().map(|&s| m[s]).collect(), excited: (GROUND..crate::request::STATES).map(|s| m[s] * o[s]).collect() })
            .collect();
        let mut beyond = None;
        if let Some((fold, levels, far, _)) = self.cluster_fold(sample.cluster, &amplitudes, true) {
            let fastest = self.light.beams.iter().map(|b| b.wave_number).fold(0.0, f64::max);
            let state = |label: usize| if label < g { c.states[label] } else { GROUND + label - g };
            let class = |k: Option<usize>| k.map_or(self.axes.zero(), |k| self.carriers[k].beat.clone());
            let term = |l: &Link| super::eliminate::term_unit(state(l.a), state(l.b), &class(l.classes.0).plus(-1, &class(l.classes.1)));
            let linked = |l: &Link| {
                let Unit::Term { a, b, beat } = term(l) else { unreachable!("a link's unit is a term") };
                self.in_generator(a, b, &beat)
            };
            let blocks = self.blocks();
            let errors = errors(&fold, &blocks, &linked, &sample.psi, velocity, 2.0 * fastest * speed, &outside);
            beyond = Some(errors.outside.clone());
            for (object, x) in &errors.parts {
                match object {
                    Object::Block(b, Some(k)) => book(&Unit::Block { block: *b, component: super::Component { frequency: self.carriers[*k].frequency, counter: false }, grounds: ground_set(&c.states[..g]) }, *x, &mut out),
                    Object::Block(_, None) => {}
                    Object::Link(l) => book(&term(l), *x, &mut out),
                }
            }
            // A part on a block, split among its classes as `errors` splits a block's (`Fold::state_block_parts`).
            let on_block = |e: Option<usize>, x: f64, out: &mut Vec<UnitError>| {
                let Some(b) = e.and_then(|e| blocks.iter().position(|b| b.contains(&e))) else { return };
                for (class, part) in fold.state_block_parts(&blocks, b, &sample.psi, x) {
                    if let Some(k) = class {
                        book(&Unit::Block { block: b, component: super::Component { frequency: self.carriers[k].frequency, counter: false }, grounds: ground_set(&c.states[..g]) }, part, out);
                    }
                }
            };
            // The evaluation against the definition (ELIMINATION v7; `Fold::evaluation_error`): one comparison at the
            // run's orders, and at each next order the part it takes away -- the series' first (`Unit::Series`) where the
            // run takes order 0, the normalization's next (`Unit::Normalization`) --; the rest (δK, the orders beyond, the
            // Hermitian part's normalization) on the block nearest the reference (`Fold::nearest`), which criterion one
            // makes explicit.
            let z0 = C64::real(far + self.series.centre[sample.cluster]);
            if let Some((definition, whole)) = self.definition_fold(sample.cluster, &amplitudes).filter(|d| d.0.exists) {
                let defined = definition.defined(&whole);
                let psi = &sample.psi;
                let error = |orders: Orders| Fold::evaluation_error(&fold.evaluated(&levels, z0, orders), &defined, z0, psi);
                let own = self.series.orders();
                let at = error(own);
                let series = if own.series == 0 { (at - error(Orders { series: 1, ..own })).max(0.0) } else { 0.0 };
                let normalization = (at - error(Orders { normalization: own.normalization + 1, ..own })).max(0.0);
                if own.series == 0 {
                    book(&Unit::Series, series, &mut out);
                }
                book(&Unit::Normalization { order: own.normalization + 1 }, normalization, &mut out);
                on_block(fold.nearest(z0), (at - series - normalization).max(0.0), &mut out);
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
        let beyond = beyond.unwrap_or_else(|| outside_errors(None, &sample.psi, &outside));
        (out, beyond.iter().map(|&x| UnitError { population: x, temperature: x }).collect())
    }

    /// The transitions to the levels outside the model (`Model::outside`).
    pub(crate) fn outside_count(&self) -> usize {
        self.outside.len()
    }

    /// Per transition to a level outside the model (`Model::outside`) its mixing and its shift at a point of the beams'
    /// amplitudes `amplitudes`, per model state: m_s = Σ_i x_i(E) mixing_i[s] and o_s = Σ_i x_i(E) shift_i[s], E its
    /// frequency's coherent field there (request.rs `coordinates`).
    pub(crate) fn outside_at(&self, amplitudes: &[crate::field::Amplitude]) -> Vec<(Vec<f64>, Vec<f64>)> {
        let x: Vec<[f64; 9]> = self.light.frequencies.iter().map(|f| crate::request::coordinates(&f.field(amplitudes).value)).collect();
        let at = |w: usize, form: &[Vec<f64>; 9]| (0..crate::request::STATES).map(|s| (0..9).map(|i| x[w][i] * form[i][s]).sum()).collect::<Vec<f64>>();
        self.outside.iter().map(|(w, t)| (at(*w, &t.mixing), at(*w, &t.shift))).collect()
    }

    /// The model's prior (ELIMINATION.md §3–§4, the brief's "先验"): at the reach's own atoms `points` (position,
    /// velocity) the point's closure (`closure`) from what is kept already, the atoms together until no atom adds
    /// anything; and per class its fold's route, at the point or at its bare levels (`bare_remainders`, within the share
    /// no output resolves at every atom: `estimate::unseen`). What it keeps, as the model's units: every drive of a
    /// class folded at the point (`Unit::Bare`: its fold taken at the levels there), the explicit drives into the
    /// blocks (`Unit::Drive`), the P-layer links of two ground states (`Unit::Term`, their lab beat exactly).
    pub(crate) fn prior(&self, points: &[([f64; 3], [f64; 3])]) -> Vec<super::Unit> {
        use rayon::prelude::*;
        let blocks = self.blocks();
        let bare = &self.energies;
        let unseen = crate::estimate::unseen(crate::units::MOST_TRAJECTORIES as f64);
        let mut kept = Kept::default();
        let mut routes: Vec<f64> = vec![];
        let mut sources: Vec<(usize, bool)> = vec![];
        loop {
            let read: Vec<(Kept, Vec<f64>, Vec<(usize, bool)>)> = points
                .par_iter()
                .map(|&(r, v)| {
                    let (levels, classes, sources) = system(&self.request, &self.light, r);
                    let closed = closure(&levels, &classes, &blocks, v, &kept);
                    let routes = bare_remainders(&closed.fold, classes.len(), bare);
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

    /// Its strength at the point (rad/µs; the frames' order, cluster.rs `frames`): an explicit drive's coupling, the
    /// class's beams summed, |C_c(e, g)|; a coupling of two P states through two classes its second-order entry through
    /// the excited states the partition folds (all but `explicit`), |Σ_e C₀(e, a)* C₁(e, b) / (ν₁ + E_b − E_e + (i/2)Γ_e)|
    /// on the levels' diagonals -- the entry the fold makes, to its leading order --; none of a P state with itself.
    pub(crate) fn strength(&self, classes: &[Class], levels: &Levels, explicit: &[usize]) -> f64 {
        let ground = levels.ground.rows;
        let coupling = |c: usize, e: usize, g: usize| classes[c].beams.iter().map(|b| b.coupling[(e, g)]).sum::<C64>();
        match self.classes {
            (None, Some(c)) if self.a >= ground && self.b < ground => coupling(c, self.a - ground, self.b).abs(),
            (Some(c), None) if self.b >= ground && self.a < ground => coupling(c, self.b - ground, self.a).abs(),
            (Some(emitted), Some(absorbed)) if self.a < ground && self.b < ground => (0..levels.excited.rows)
                .filter(|e| !explicit.contains(e))
                .map(|e| {
                    let distance = C64::new(classes[absorbed].nu + levels.ground[(self.b, self.b)].re - levels.excited[(e, e)].re, 0.5 * levels.decay[(e, e)].re);
                    coupling(emitted, e, self.a).conj() * coupling(absorbed, e, self.b) / distance
                })
                .sum::<C64>()
                .abs(),
            _ => 0.0,
        }
    }
}

/// What a model keeps explicit (ELIMINATION.md §6: the keep rule's two kinds of objects): excited blocks, and P-layer
/// links -- the explicit components, the explicit drives into the blocks with them.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Kept {
    pub blocks: Vec<usize>,
    pub links: Vec<Link>,
}

/// A point's closure (criteria one and two at one atom, ELIMINATION.md §3–§4 with v2 and v3): the kept set from
/// `start` on, made larger until the point needs nothing more, with its fold.
#[derive(Clone, Debug)]
pub(crate) struct Closed {
    pub kept: Kept,
    pub fold: Fold,
}

/// The P states (labels as in `Link`) of a kept set, the frames from its links at their strengths (cluster.rs `frames`:
/// the strongest static; `Link::strength`) and its groups (`components` of the links, each explicit block's states one
/// group with them: a block is moved whole, its sublevels coupled by the Zeeman and tensor shifts): the ground states,
/// then the kept blocks' states.
fn slow_of(blocks: &[Vec<usize>], kept: &Kept, classes: &[Class], levels: &Levels) -> Slow {
    let ground = levels.ground.rows;
    let explicit: Vec<usize> = kept.blocks.iter().flat_map(|&b| blocks[b].iter().copied()).collect();
    let place = |s: usize| if s < ground { Some(s) } else { explicit.iter().position(|&e| e == s - ground).map(|i| ground + i) };
    let edges: Vec<(usize, usize, f64, f64)> = kept.links.iter().filter_map(|l| Some((place(l.a)?, place(l.b)?, l.beat(classes), l.strength(classes, levels, &explicit)))).collect();
    let n = ground + explicit.len();
    let frames = super::cluster::frames(n, &edges, &0.0);
    let within = kept.blocks.iter().flat_map(|&b| blocks[b].windows(2).filter_map(|w| Some((place(ground + w[0])?, place(ground + w[1])?))));
    let groups = components(n, edges.iter().map(|e| (e.0, e.1)).chain(within));
    Slow { explicit, frames, groups }
}

/// The closure at one point (`Closed`): from `start`, at the atom's velocity v (µm/µs) --
/// 1. the fold in the frames of the kept links; where it does not exist (`settle`), the excited block with the largest
///    part of the ground-like eigenstates' excited share -- over all of them -- is made explicit;
/// 2. criterion one: every ground-like eigenstate's remainder at the offsets its amplitude spans (`Fold::offsets`: the
///    pattern a moving atom sees, 2 k_max v, whole, and each component into it at its combined frequency at its share,
///    each over its distance from the folded block's spectrum), within the share; beyond it the blocks with the largest
///    parts are made explicit until the rest is within it -- near a resonance of a folded block its part grows without
///    bound, and that block is made explicit;
/// - a block made explicit for an eigenstate is explicit at the replica that holds its part and in the P component the
///   eigenstate holds (`components`): the drives into it at that replica from that component's ground states are kept
///   (its frame). A drive from another component -- the other ground hyperfine level, 6.8 GHz away -- turns at that
///   component's own frequency, where the block is a Q state folded like any other (ELIMINATION §2–§3, v2 5); a block
///   explicit in one component is joined by another's drives only where that component's own part asks for it;
/// 3. criterion two: every component of two replicas whose averaged error 2(a/δ)² is beyond the share (v5: the
///    population's time mean after a sudden switch-on) is kept (a link of its two dominant states and classes);
/// until nothing more is kept. Each round only adds. (The photons' cross terms are the run's to read, `Model::readings`.)
pub(crate) fn closure(levels: &Levels, classes: &[Class], blocks: &[Vec<usize>], velocity: [f64; 3], start: &Kept) -> Closed {
    let ground = levels.ground.rows;
    let share = share();
    let speed = (velocity[0] * velocity[0] + velocity[1] * velocity[1] + velocity[2] * velocity[2]).sqrt();
    let fastest = classes.iter().flat_map(|c| c.beams.iter()).map(|b| (b.wave[0] * b.wave[0] + b.wave[1] * b.wave[1] + b.wave[2] * b.wave[2]).sqrt()).fold(0.0, f64::max);
    let pattern = 2.0 * fastest * speed;
    let mut kept = start.clone();
    loop {
        let slow = slow_of(blocks, &kept, classes, levels);
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
        let components = fold.components(velocity);
        // The parts of a ground-like eigenstate's response (or remainder) per block and replica, each split among the
        // replica's classes (`Fold::block_parts`): (block, replica, class, part).
        let parts_of = |p: usize, offsets: Option<&[f64]>| fold.block_parts(blocks, p, offsets);
        // Moves: (block, replica, class, the eigenstate whose part it is), largest part first per state, until the rest
        // is within `limit`; one replica a block and class.
        let mut moves: Vec<(usize, usize, usize, usize)> = vec![];
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
            // Each eigenstate's remainder at the offsets it spans (`Fold::offsets`), each component at its share, the
            // source's whole, its blocks' parts with it.
            for &p in &ground_like {
                let offsets = fold.offsets(p, pattern, &components, 1.0, &|_| 1.0);
                let (rho, parts) = parts_of(p, Some(&offsets));
                if rho > share {
                    pick(p, parts, rho * rho, share * share, &mut moves);
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
        for x in components {
            let m = x.mixing();
            let link = Link { a: label(x.row), b: label(x.column), classes: x.classes };
            if 2.0 * m * m > share {
                if link.a != link.b && !kept.links.contains(&link) {
                    kept.links.push(link);
                    grew = true;
                }
            }
        }
        if grew {
            continue;
        }
        return Closed { kept, fold };
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

/// What a model keeps or folds at a point, the keep rule's objects (ELIMINATION.md §6): an excited block, a P-layer link.
/// (The photons' cross terms are read on the run's photon classes, `Model::readings`.)
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Object {
    /// An excited block for one class of the light (None: the explicit states' own couplings into it).
    Block(usize, Option<usize>),
    Link(Link),
}

/// A model's error at a state (ELIMINATION.md §6, v2, v3): per object the part its keep would take away (the fold's three
/// terms, each split among the objects that take it away).
#[derive(Clone, Debug)]
pub(crate) struct Errors {
    pub parts: Vec<(Object, f64)>,
    /// Per transition to a level outside the model its error (`outside_errors`): no object's part, nothing keeps it.
    pub outside: Vec<f64>,
}

/// One transition to a level outside the model at a state's point (request.rs `Transition`, `Model::outside_at`): its
/// mixing m_s -- the population a state s holds in the level, (coupling/detuning)² summed over its sublevels -- per P
/// state; and per excited state the error of the level that transition's second-order shift o_e puts it at, m_e o_e
/// (rad/µs: the term wrong by its own small parameter, relative).
pub(crate) struct Outside {
    pub slow: Vec<f64>,
    pub excited: Vec<f64>,
}

/// The errors of the transitions to the levels outside the model at a state ψ (P amplitudes; `errors`), per transition,
/// each as criterion one reads what the model holds:
/// - a P state s leaks its population into the level: |ψ_s|²/‖ψ‖² m_s;
/// - the excited states the fold eliminates hold their levels wrong by δK = diag(m_e o_e), which acts on the slow
///   dynamics through the fold's response, as any error of the excited block does: of each eigenstate p of the fold
///   (`fold`, with its share w_p of ψ) ρ_p = ‖R(λ_p) δK T r_p‖/‖T r_p‖ over the replicas in squares -- `Fold::remainder`'s
///   ratio, this error in its stead --, a rate's relative error 2ρ_p, Σ_p w_p 2ρ_p. Not by the excited states' populations:
///   a transition that moves a level away from the light lowers its population and raises the error it makes.
/// The response T r_p of each eigenstate and replica is made once, then a resolvent per transition that is not zero there.
pub(crate) fn outside_errors(fold: Option<(&Fold, &[f64])>, psi: &[C64], outside: &[Outside]) -> Vec<f64> {
    let norm = crate::linalg::norm_sqr(psi);
    let mut out: Vec<f64> = outside.iter().map(|o| if norm > 0.0 { psi.iter().zip(&o.slow).map(|(z, m)| z.norm_sqr() * m).sum::<f64>() / norm } else { 0.0 }).collect();
    let Some((fold, weight)) = fold else { return out };
    for p in (0..fold.size()).filter(|&p| weight[p] > 0.0) {
        let (r, lambda) = (fold.right(p), fold.eigen.values[p]);
        let responses: Vec<(C64, Vec<C64>)> = fold.replicas.iter().map(|rep| {
            let sigma = lambda + C64::real(rep.frequency);
            (sigma, rep.k.resolvent(sigma, &rep.w.apply(&r)))
        }).collect();
        let held: f64 = responses.iter().map(|(_, x)| crate::linalg::norm_sqr(x)).sum();
        if !(held > 0.0) {
            continue;
        }
        for (o, error) in outside.iter().zip(out.iter_mut()) {
            let mut moved = 0.0;
            for (rep, (sigma, x)) in fold.replicas.iter().zip(&responses) {
                let y: Vec<C64> = rep.q.iter().zip(x).map(|(&e, z)| *z * o.excited[e]).collect();
                if y.iter().any(|z| *z != C64::ZERO) {
                    moved += crate::linalg::norm_sqr(&rep.k.resolvent(*sigma, &y));
                }
            }
            *error += weight[p] * 2.0 * (moved / held).sqrt();
        }
    }
    out
}

/// The model's error at a state ψ (P amplitudes in the frames) of an atom at the velocity v (µm/µs), the frequency range
/// its amplitude spans Δ (rad/µs; `closure`): one function for the prior (a start, each ground-like eigenstate full in
/// turn) and for the readings (a trajectory's state).
/// - The remainder (criterion one): a rate's relative error is 2ρ_p(Δ) of the eigenstate the state holds, weighted by
///   its share w_p of the state; each excited block's part of ρ² is its part.
/// - The transient (`Fold::transients`): after the next jump the atom lands in a bare ground state -- at the state's
///   own ground populations p_g --, whose waiting time the model takes from its slow part alone with the threshold law
///   P(T > t) = min(1, ‖y(t)‖²) (step.rs `jump`). The exact one is longer by τ and the law's clamp leaves out
///   (N − 1 − ln N)/γ where the slow part's norm N = ‖(x₀, T x₀)‖² is above one (γ its decay there): the rate's relative
///   error is |τ + clamp| over the model's mean wait, N/γ or (1 + ln N)/γ. Its part goes to the blocks in proportion to
///   their share of the landing's excited amplitude T x₀: making a block explicit takes its fast part into the dynamics.
/// - The averaged components (criterion two): 2(a/δ)² of each one not kept (v5), on its two eigenstates' shares.
/// - The levels outside the model (request.rs `Transition`, `outside`): each transition's second-order term errs by its
///   own small parameter m = (coupling/detuning)²: on a P state the population it leaks into the level, on an excited
///   state the fold eliminates the level error m o it makes, through the fold's response as criterion one reads one
///   (`outside_errors`). Irreducible: those levels are not in the model, and no object's part.
/// The photons' cross terms are the caller's (the run's photon classes, which land in every cluster: model.rs
/// `Model::readings`). A relative rate error x moves the populations by at most x (absolute) and T, which the light sets
/// through its rates and shifts, by x relative: the terms add on both.
/// `linked`: whether a P-layer link is the model's (kept, or static in its frames: in its generator as it is).
#[allow(clippy::too_many_arguments)]
pub(crate) fn errors(fold: &Fold, blocks: &[Vec<usize>], linked: &dyn Fn(&Link) -> bool, psi: &[C64], velocity: [f64; 3], pattern: f64, outside: &[Outside]) -> Errors {
    let n = fold.size();
    let ground = fold.ground;
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
    // The remainder at the offsets each eigenstate spans, at the populations there (`Fold::offsets`): the pattern's at the
    // eigenstate's own, each component's sideband at its source's times its share; booked as 2ρ on the blocks by their
    // parts of it.
    let components = fold.components(velocity);
    for p in 0..n {
        let offsets = fold.offsets(p, pattern, &components, weight[p], &|q| weight[q]);
        if offsets.iter().all(|x| *x == 0.0) {
            continue;
        }
        let (rho, each) = fold.block_parts(blocks, p, Some(&offsets));
        let total: f64 = each.iter().map(|x| x.3).sum();
        for (b, _, class, x) in each.into_iter().filter(|_| total > 0.0) {
            add(Object::Block(b, class), 2.0 * rho * x / total, &mut parts);
        }
    }
    // The transient after the next jump, from the state's own ground populations.
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
    // The averaged components not kept.
    let label = |p: usize| {
        let s = fold.dominant(p);
        if s < ground { s } else { ground + fold.explicit[s - ground] }
    };
    for x in components {
        let link = Link { a: label(x.row), b: label(x.column), classes: x.classes };
        if linked(&link) || (weight[x.row] == 0.0 && weight[x.column] == 0.0) {
            continue;
        }
        let e = 2.0 * x.mixing() * x.mixing() * (weight[x.row] + weight[x.column]);
        add(Object::Link(link), e, &mut parts);
    }
    let outside = outside_errors(Some((fold, &weight)), psi, outside);
    Errors { parts, outside }
}

#[cfg(test)]
mod tests;
