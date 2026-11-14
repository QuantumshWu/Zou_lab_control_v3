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
//!   δ = Ω + λ_p − λ_p' mixes a/|δ| (criterion two: explicit where it is above √share, else averaged into its static
//!   shift |a|²/δ, its error (a/δ)²).
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

/// P beside the ground states: the explicit excited states (indices into E), and the frame θ_s of every P state (rad/µs;
/// the ground states first, then the explicit ones in their order).
#[derive(Clone, Debug)]
pub(crate) struct Slow {
    pub explicit: Vec<usize>,
    pub frames: Vec<f64>,
}

/// The replicas (module documentation, # P, Q and the frames): their frequencies f, the replica each class drives each
/// ground state's amplitude into, and the one each explicit state's static couplings drive. Made of exact frequencies
/// by the caller; `Replicas::of` groups equal values.
#[derive(Clone, Debug)]
pub(crate) struct Replicas {
    pub frequencies: Vec<f64>,
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
    /// The replicas of frequencies that are one (`same`).
    pub(crate) fn of(classes: &[Class], slow: &Slow, ground: usize) -> Replicas {
        let mut frequencies: Vec<f64> = vec![];
        let mut place = |f: f64| match frequencies.iter().position(|&x| same(x, f)) {
            Some(at) => at,
            None => {
                frequencies.push(f);
                frequencies.len() - 1
            }
        };
        let driven = classes.iter().map(|c| (0..ground).map(|g| place(c.nu + slow.frames[g])).collect()).collect();
        let held = (0..slow.explicit.len()).map(|i| place(slow.frames[ground + i])).collect();
        Replicas { frequencies, driven, held }
    }
}

/// What drives one replica: a class's beam from the ground states (`class` Some), or the excited block's static
/// couplings from the explicit states of its frame (`class` None, its wave vector zero); its coupling W (Q_f ← P), the
/// coupling back W' (P ← Q_f), and its response T (the Sylvester solution of W alone: the responses are linear in W).
#[derive(Clone, Debug)]
pub(crate) struct Feed {
    pub class: Option<usize>,
    pub beam: usize,
    pub wave: [f64; 3],
    pub w: Mat,
    pub back: Mat,
    /// The coupling back from every excited state (P ← E): what the feed's light does to an excited amplitude at another
    /// replica (a P-layer component's emitted leg).
    pub emitted: Mat,
    pub t: Mat,
}

/// One replica folded: its frequency, Q_f, the sum of its feeds' W, W', T, and T2 (module documentation).
#[derive(Clone, Debug)]
pub(crate) struct Folded {
    pub frequency: f64,
    pub q: Vec<usize>,
    pub feeds: Vec<Feed>,
    pub w: Mat,
    pub back: Mat,
    pub t: Mat,
    pub t2: Mat,
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
    /// The direct entries of P that turn in the frames: (row, column, residual beat Ω, value), the entry ∝ e^{−iΩt}.
    pub turning: Vec<(usize, usize, f64, C64)>,
}

/// A Schur form A = Q S Q† (S upper triangular row by row, Q column by column: linalg.rs `schur`).
#[derive(Clone, Debug)]
struct Schur {
    n: usize,
    s: Vec<C64>,
    q: Vec<C64>,
}

impl Schur {
    fn of(a: &Mat) -> Option<Schur> {
        let n = a.rows;
        let (mut s, mut q) = (a.data.clone(), vec![C64::ZERO; n * n]);
        let (mut v, mut sums) = (vec![C64::ZERO; n], vec![C64::ZERO; n]);
        if n > 0 {
            schur(&mut s, &mut q, &mut v, &mut sums, n)?;
        }
        Some(Schur { n, s, q })
    }
    fn q(&self, i: usize, k: usize) -> C64 {
        self.q[k * self.n + i]
    }
    /// Q† x (x with n rows).
    fn left_in(&self, x: &Mat) -> Mat {
        Mat::from_fn(self.n, x.cols, |k, j| (0..self.n).map(|i| self.q(i, k).conj() * x[(i, j)]).sum())
    }
    /// Q y.
    fn left_out(&self, y: &Mat) -> Mat {
        Mat::from_fn(self.n, y.cols, |i, j| (0..self.n).map(|k| self.q(i, k) * y[(k, j)]).sum())
    }
    /// x Q (x with n columns).
    fn right_in(&self, x: &Mat) -> Mat {
        Mat::from_fn(x.rows, self.n, |i, k| (0..self.n).map(|j| x[(i, j)] * self.q(j, k)).sum())
    }
    /// y Q†.
    fn right_out(&self, y: &Mat) -> Mat {
        Mat::from_fn(y.rows, self.n, |i, j| (0..self.n).map(|k| y[(i, k)] * self.q(j, k).conj()).sum())
    }
    /// (σ − S)⁻¹ y in place (back substitution).
    fn shifted_solve(&self, sigma: C64, y: &mut [C64]) {
        for m in (0..self.n).rev() {
            let mut acc = y[m];
            for k in m + 1..self.n {
                acc += self.s[m * self.n + k] * y[k];
            }
            y[m] = acc / (sigma - self.s[m * self.n + m]);
        }
    }
    /// (σ − A)⁻¹ v.
    fn resolvent(&self, sigma: C64, v: &[C64]) -> Vec<C64> {
        let mut y: Vec<C64> = (0..self.n).map(|k| (0..self.n).map(|i| self.q(i, k).conj() * v[i]).sum()).collect();
        self.shifted_solve(sigma, &mut y);
        (0..self.n).map(|i| (0..self.n).map(|k| self.q(i, k) * y[k]).sum()).collect()
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
            column[m] = fq[(m, j)] - (0..j).map(|i| right.s[i * n + j] * y[(m, i)]).sum::<C64>();
        }
        left.shifted_solve(right.s[j * n + j] + r - l, &mut column);
        for m in 0..left.n {
            y[(m, j)] = column[m];
        }
    }
    left.left_out(&right.right_out(&y))
}

/// The fold's Sylvester equation T H − (K − f) T = W (module documentation, # The fold).
fn sylvester(k: &Schur, h: &Schur, f: f64, w: &Mat) -> Mat {
    solve_sylvester(k, C64::real(-f), h, C64::ZERO, w)
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
/// fold's existence bound).
pub(crate) fn fold(levels: &Levels, classes: &[Class], slow: &Slow, replicas: &Replicas) -> Fold {
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
    // A: the direct static entries; the turning ones apart.
    let mut a = Mat::zeros(n, n);
    let mut turning = vec![];
    let mut put = |a: &mut Mat, row: usize, column: usize, value: C64| {
        if value == C64::ZERO {
            return;
        }
        if same(frame(column), frame(row)) {
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
    // The explicit drives at residual zero are A's; the others are folded at their replica (an explicit state of
    // another frame is Q there).
    for class in classes {
        for beam in &class.beams {
            for (i, &x) in slow.explicit.iter().enumerate() {
                for g in 0..ground {
                    let v = beam.coupling[(x, g)];
                    if v != C64::ZERO && same(class.nu + frame(g), frame(ground + i)) {
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
        .map(|&f| {
            let q: Vec<usize> = (0..excited).filter(|&e| !slow.explicit.iter().enumerate().any(|(i, &x)| x == e && same(frame(ground + i), f))).collect();
            if !forms.iter().any(|x| x.0 == q) {
                let kq = k_full.block(&q, &q);
                let k = Schur::of(&kq).expect("the Schur form of the excited block");
                forms.push((q.clone(), kq, k));
            }
            let (_, kq, k) = forms.iter().find(|x| x.0 == q).expect("its form");
            Folded { frequency: f, q, feeds: vec![], w: Mat::zeros(0, 0), back: Mat::zeros(0, 0), t: Mat::zeros(0, 0), t2: Mat::zeros(0, 0), kq: kq.clone(), k: k.clone() }
        })
        .collect();
    for (c, class) in classes.iter().enumerate() {
        for (l, beam) in class.beams.iter().enumerate() {
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
                rep.feeds.push(Feed { class: Some(c), beam: l, wave: beam.wave, w, back, emitted, t: Mat::zeros(0, 0) });
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
        rep.feeds.push(Feed { class: None, beam: 0, wave: [0.0; 3], w, back, emitted, t: Mat::zeros(0, 0) });
    }
    folded.retain(|rep| !rep.feeds.is_empty());
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
    let (h, iterations, exists) = settle(
        &a,
        |h| {
            let sh = Schur::of(h).expect("the Schur form of P's generator");
            let (mut next, mut scale) = (a.clone(), a.frobenius());
            for rep in folded.iter_mut() {
                rep.t = sylvester(&rep.k, &sh, rep.frequency, &rep.w);
                next.add_scaled(C64::ONE, &rep.back.product(&rep.t));
                scale += rep.back.frobenius() * rep.t.frobenius();
            }
            (next, scale)
        },
        (n + excited) * (n + excited),
    );
    // The responses at the generator they made it of, and their frequency partners and the feeds' own, at its final form.
    let sh = Schur::of(&h).expect("the Schur form of P's generator");
    for rep in folded.iter_mut() {
        rep.t = sylvester(&rep.k, &sh, rep.frequency, &rep.w);
        rep.t2 = sylvester(&rep.k, &sh, rep.frequency, &rep.t);
        for feed in rep.feeds.iter_mut() {
            feed.t = sylvester(&rep.k, &sh, rep.frequency, &feed.w);
        }
    }
    let mut eigen = Eigen::default();
    eigen.decompose(&h);
    Fold { ground, explicit: slow.explicit.clone(), frames: slow.frames.clone(), direct: a, generator: h, replicas: folded, eigen, exists, iterations, turning }
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
    /// (module documentation, # First order): ρ_p(Δ), and per excited state (E's index) its part of the remainder's square
    /// over the response's square -- the parts sum to ρ_p².
    pub(crate) fn remainder(&self, p: usize, shift: f64, excited: usize) -> (f64, Vec<f64>) {
        let r = self.right(p);
        let lambda = self.eigen.values[p];
        let mut parts = vec![0.0; excited];
        let (mut num, mut den) = (0.0, 0.0);
        for rep in &self.replicas {
            let wr = rep.w.apply(&r);
            let sigma = lambda + C64::real(rep.frequency);
            let x0 = rep.k.resolvent(sigma, &wr);
            let shifted = rep.k.resolvent(sigma + C64::real(shift), &wr);
            let second = rep.k.resolvent(sigma, &x0);
            for (i, &e) in rep.q.iter().enumerate() {
                let d = shifted[i] - x0[i] + second[i] * shift;
                parts[e] += d.norm_sqr();
                num += d.norm_sqr();
                den += x0[i].norm_sqr();
            }
        }
        if den == 0.0 {
            return (0.0, vec![0.0; excited]);
        }
        parts.iter_mut().for_each(|x| *x /= den);
        ((num / den).sqrt(), parts)
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
            for (j, other) in self.replicas.iter().enumerate() {
                if i == j {
                    continue;
                }
                for feed in &rep.feeds {
                    for back in &other.feeds {
                        // The other replica's light emitted from this one's excited amplitude: its coupling back from Q_f.
                        let m = Mat::from_fn(n, n, |r, c| rep.q.iter().enumerate().map(|(y, &e)| back.emitted[(r, e)] * feed.t[(y, c)]).sum());
                        if m.max_abs() == 0.0 {
                            continue;
                        }
                        let beat = rep.frequency - other.frequency - (doppler(feed.wave) - doppler(back.wave));
                        let e = eigenbasis(&m);
                        let floor = rounding(n + rep.q.len()) * back.emitted.frobenius() * feed.t.frobenius();
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
        let adjoints: Vec<Schur> = self.replicas.iter().map(|rep| Schur::of(&rep.kq.adjoint()).expect("the Schur form of the excited block")).collect();
        let mut s: Vec<Mat> = self.replicas.iter().map(|rep| Mat::zeros(n, rep.q.len())).collect();
        let excited = self.replicas.iter().map(|rep| rep.q.len()).max().unwrap_or(1);
        let (_, _, settled) = settle(
            &self.direct,
            |m| {
                let sm = Schur::of(&m.adjoint()).expect("the Schur form of P's left generator");
                let (mut next, mut scale) = (self.direct.clone(), self.direct.frobenius());
                for ((rep, k), si) in self.replicas.iter().zip(&adjoints).zip(s.iter_mut()) {
                    *si = sylvester(k, &sm, rep.frequency, &rep.back.adjoint()).adjoint();
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
        let x0 = self.landed_with(s, f)?;
        let h = Schur::of(&self.generator)?;
        let mut tau = 0.0;
        for (rep, si) in self.replicas.iter().zip(s) {
            let nq = rep.q.len();
            if nq == 0 {
                continue;
            }
            let k = &rep.k;
            let kh = Schur::of(&rep.kq.adjoint())?;
            let identity = Mat::from_fn(nq, nq, |i, j| if i == j { C64::new(0.0, -1.0) } else { C64::ZERO });
            let lyapunov = solve_sylvester(&kh, C64::ZERO, k, C64::ZERO, &identity);
            let mut b = rep.t.clone();
            b.add_scaled(-C64::ONE, &si.adjoint());
            let cross = solve_sylvester(&kh, C64::real(-rep.frequency), &h, C64::ZERO, &b.scaled(C64::new(0.0, -1.0)));
            let y = rep.t.apply(&x0);
            let ly = lyapunov.apply(&y);
            let cx = cross.apply(&x0);
            tau += crate::linalg::inner(&y, &ly).re - 2.0 * crate::linalg::inner(&y, &cx).re;
        }
        Some((x0, tau))
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
pub(crate) fn system(request: &crate::request::Request, light: &crate::field::Light, r: [f64; 3]) -> (Levels, Vec<Class>) {
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
    let mut classes = vec![];
    for f in &light.frequencies {
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
            }
        }
    }
    (levels, classes)
}

/// A request's excited hyperfine blocks F' (indices into E): the states of one manifold and F, the unit a partition
/// moves -- the Zeeman shifts and the light's tensor shifts couple a block's sublevels, so splitting one breaks the
/// block structure (ELIMINATION.md §3).
pub(crate) fn blocks(request: &crate::request::Request) -> Vec<Vec<usize>> {
    use crate::request::{Level, GROUND};
    let mut out: Vec<(String, Vec<usize>)> = vec![];
    for (e, label) in request.states[GROUND..].iter().enumerate() {
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

/// The P states (labels as in `Link`) of a kept set and the frames from its links (`frames`): the ground states, then
/// the kept blocks' states.
fn slow_of(blocks: &[Vec<usize>], kept: &Kept, classes: &[Class], ground: usize) -> (Slow, f64) {
    let explicit: Vec<usize> = kept.blocks.iter().flat_map(|&b| blocks[b].iter().copied()).collect();
    let place = |s: usize| if s < ground { Some(s) } else { explicit.iter().position(|&e| e == s - ground).map(|i| ground + i) };
    let edges: Vec<(usize, usize, f64)> = kept.links.iter().filter_map(|l| Some((place(l.a)?, place(l.b)?, l.beat(classes)))).collect();
    let (frames, residual) = frames(ground + explicit.len(), &edges);
    (Slow { explicit, frames }, residual)
}

/// The closure at one point (`Closed`): from `start`, at the atom's velocity v (µm/µs) --
/// 1. the fold in the frames of the kept links; where it does not exist (`settle`), the excited block with the largest
///    part of the ground-like eigenstates' excited share is made explicit;
/// 2. criterion one: every ground-like eigenstate's remainder ρ(±Δ), Δ = t* + 2 k_max v the frequency range its
///    amplitude spans (the frames' residual, and the Doppler shifts and the pattern a moving atom sees: at most
///    k_max v and |k_l − k_l'| v ≤ 2 k_max v), within the share; beyond it the blocks with the largest parts are made
///    explicit until the rest is within it -- a block already explicit at another replica keeps that drive instead;
///    a new block takes the drive whose replica holds its largest part (its frame);
/// 3. criterion two: every component of two replicas whose mixing a/|δ| is above √share, its error (a/δ)² beyond the
///    share, is kept (a link of its two dominant states and classes), and every photon cross term whose γ/|Ω| is
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
        let replicas = Replicas::of(classes, &slow, ground);
        let fold = fold(levels, classes, &slow, &replicas);
        let n = fold.size();
        let ground_like: Vec<usize> = (0..n).filter(|&p| fold.ground_weight(p) >= 0.5).collect();
        let excited = levels.excited.rows;
        // The parts of a ground-like eigenstate's response (or remainder) per block and replica.
        let parts_of = |p: usize, shift: Option<f64>| -> (f64, Vec<Vec<f64>>) {
            let r = fold.right(p);
            let lambda = fold.eigen.values[p];
            let mut parts = vec![vec![0.0; fold.replicas.len()]; blocks.len()];
            let (mut num, mut den) = (0.0, 0.0);
            for (j, rep) in fold.replicas.iter().enumerate() {
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
                for (i, &e) in rep.q.iter().enumerate() {
                    if let Some(b) = block_of(e) {
                        parts[b][j] += d[i].norm_sqr();
                    }
                    num += d[i].norm_sqr();
                    den += x0[i].norm_sqr();
                }
            }
            let scale = if shift.is_some() { den } else { 1.0 };
            if scale > 0.0 {
                parts.iter_mut().flatten().for_each(|x| *x /= scale);
            }
            (if shift.is_some() { if den > 0.0 { (num / den).sqrt() } else { 0.0 } } else { num.sqrt() }, parts)
        };
        // Moves: (block, replica) with its part, largest first per state, until the rest is within `limit`.
        let mut moves: Vec<(usize, usize)> = vec![];
        let mut remainder: f64 = 0.0;
        let mut pick = |parts: &[Vec<f64>], total: f64, limit: f64, moves: &mut Vec<(usize, usize)>| {
            let mut order: Vec<(usize, usize, f64)> = parts.iter().enumerate().flat_map(|(b, row)| row.iter().enumerate().map(move |(j, &x)| (b, j, x))).filter(|x| x.2 > 0.0).collect();
            order.sort_by(|x, y| y.2.total_cmp(&x.2));
            let mut rest = total;
            for (b, j, x) in order {
                if rest <= limit {
                    break;
                }
                rest -= x;
                if !moves.iter().any(|m| m.0 == b) {
                    moves.push((b, j));
                }
            }
        };
        if !fold.exists {
            for &p in &ground_like {
                let (_, parts) = parts_of(p, None);
                let total: f64 = parts.iter().flatten().sum();
                pick(&parts, total, 0.0, &mut moves);
                moves.truncate(1);
                break;
            }
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
                    pick(&worst.1, worst.0 * worst.0, share * share, &mut moves);
                }
            }
        }
        let mut grew = false;
        for (b, j) in moves {
            let rep = &fold.replicas[j];
            // The drive whose replica holds the part: its class and the ground states that feed it.
            let feeds: Vec<(usize, usize)> = rep.feeds.iter().filter_map(|f| f.class.map(|c| (c, f.beam))).collect();
            let Some(&(c, _)) = feeds.first() else { continue };
            if !kept.blocks.contains(&b) {
                kept.blocks.push(b);
            }
            for &e in &blocks[b] {
                for g in 0..ground {
                    if replicas.driven[c][g] < replicas.frequencies.len() && same(replicas.frequencies[replicas.driven[c][g]], rep.frequency) && classes[c].beams.iter().any(|beam| beam.coupling[(e, g)] != C64::ZERO) {
                        let link = Link { a: ground + e, b: g, classes: (None, Some(c)) };
                        if !kept.links.contains(&link) {
                            kept.links.push(link);
                            grew = true;
                        }
                    }
                }
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
            if m > share.sqrt() {
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

/// What a model keeps or folds, the keep rule's objects (ELIMINATION.md §6): an excited block, a P-layer link, a pair of
/// photon frequencies' cross term.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Object {
    Block(usize),
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
/// - The averaged components (criterion two): (a/δ)² of each one not kept, on its two eigenstates' shares.
/// - The photons' cross terms not kept: γ/|Ω| of each, on its source eigenstate's share.
/// A relative rate error x moves the populations by at most x (absolute) and T, which the light sets through its rates
/// and shifts, by x relative: the four terms add on both.
#[allow(clippy::too_many_arguments)]
pub(crate) fn errors(fold: &Fold, levels: &Levels, blocks: &[Vec<usize>], kept: &Kept, psi: &[C64], velocity: [f64; 3], range: f64) -> Errors {
    let n = fold.size();
    let ground = fold.ground;
    let excited = levels.excited.rows;
    let coefficients: Vec<C64> = (0..n).map(|p| (0..n).map(|i| fold.eigen.inverse[(p, i)] * psi[i]).sum()).collect();
    let total: f64 = coefficients.iter().map(|z| z.norm_sqr()).sum();
    let weight: Vec<f64> = coefficients.iter().map(|z| if total > 0.0 { z.norm_sqr() / total } else { 0.0 }).collect();
    let block_of = |e: usize| blocks.iter().position(|b| b.contains(&e));
    let mut parts: Vec<(Object, f64)> = vec![];
    let mut add = |object: Object, x: f64, parts: &mut Vec<(Object, f64)>| {
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
        let (mut rho, mut each) = (0.0, vec![0.0; excited]);
        for shift in [range, -range] {
            let (r, e) = fold.remainder(p, shift, excited);
            if r >= rho {
                (rho, each) = (r, e);
            }
        }
        remainder += weight[p] * 2.0 * rho;
        if rho > 0.0 {
            for (e, x) in each.iter().enumerate() {
                if let Some(b) = block_of(e) {
                    add(Object::Block(b), weight[p] * 2.0 * rho * x / (rho * rho), &mut parts);
                }
            }
        }
    }
    // The transient after the next jump, from the state's own ground populations.
    let mut transient = 0.0;
    if let Some(left) = fold.left_operators() {
        let norm: f64 = psi.iter().map(|z| z.norm_sqr()).sum();
        let metric = fold.metric();
        for g in (0..ground).filter(|&g| psi[g].norm_sqr() > 0.0) {
            let population = psi[g].norm_sqr() / norm;
            let bare: Vec<C64> = (0..n).map(|i| if i == g { C64::ONE } else { C64::ZERO }).collect();
            let Some((x0, tau)) = fold.transient(&left, &bare) else { continue };
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
            // Its parts by the blocks' shares of the landing's excited amplitude.
            let mut shares = vec![0.0; blocks.len()];
            for rep in &fold.replicas {
                let y = rep.t.apply(&x0);
                for (i, &e) in rep.q.iter().enumerate() {
                    if let Some(b) = block_of(e) {
                        shares[b] += y[i].norm_sqr();
                    }
                }
            }
            let sum: f64 = shares.iter().sum();
            if sum > 0.0 {
                for (b, share) in shares.iter().enumerate() {
                    add(Object::Block(b), x * share / sum, &mut parts);
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
        if kept.links.contains(&link) || (weight[x.row] == 0.0 && weight[x.column] == 0.0) {
            continue;
        }
        let e = x.mixing() * x.mixing() * (weight[x.row] + weight[x.column]);
        averaged += e;
        add(Object::Link(link), e, &mut parts);
    }
    let mut photons = 0.0;
    for x in fold.photons(levels) {
        if kept.photons.contains(&x.classes) || weight[x.from] == 0.0 {
            continue;
        }
        let e = x.size / x.beat.abs() * weight[x.from];
        photons += e;
        add(Object::Photons(x.classes.0, x.classes.1), e, &mut parts);
    }
    let sum = remainder + transient + averaged + photons;
    Errors { population: sum, temperature: sum, remainder, transient, averaged, photons, parts }
}

#[cfg(test)]
mod tests;
