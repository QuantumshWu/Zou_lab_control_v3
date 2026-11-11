//! A generator along a step's path in the interaction picture of its value at the step's middle (SPEC §2.1; step.rs,
//! # The kick): what the step's one propagation is made of.
//!
//! # The generator
//! Along the path Ĥ(u), u the time from the step's middle, is Ĥ₀ + D(u): Ĥ₀ = Ĥ(0) holds every frequency of the atom's
//! own -- detunings, light shifts, decay rates, of any size against the step --, and D(u) = Ĥ(u) − Ĥ₀ what the motion
//! along the path and the lights' beats change, slow against the step by the step's own choice. D is taken through the
//! step's three nodes, the middle and the two ends ±h: D(u) = uA + u²B, A = (Ĥ(h) − Ĥ(−h))/2h and
//! B = (Ĥ(h) + Ĥ(−h) − 2Ĥ₀)/2h². Its error is the cubic's, (u³ − uh²)D‴/6: third order in the slow rate at a time of
//! the step, whatever Ĥ₀ is, and odd about the middle, so that the step's integrals take it at the fourth; a half step
//! takes its own three nodes on half the spacing.
//!
//! # The state at any time of the step
//! With Ĥ₀ = c + RΛR⁻¹ (its scalar part c apart, `linalg::eigen`) and x = R d, i ḋ = (Λ + Dᵉ(u)) d, Dᵉ = R⁻¹DR,
//! ω_ab = λ_a − λ_b. What D does to a pair of states depends on how far the pair turns between the step's nodes.
//!
//! **A fast pair, |ω_ab| |h| ≥ 1** -- it turns a radian and more between two nodes: an explicit state's detuning, its
//! decay. The state follows the change: d = (1 + X(u)) g, with X on the fast pairs the polynomial that solves
//! ω_ab X_ab − i Ẋ_ab + Dᵉ_ab = 0 exactly,
//!   X = X₀ + uX₁ + u²X₂,  X₂ = −B/ω,  X₁ = (2iX₂ − A)/ω,  X₀ = iX₁/ω  (element by element)
//! -- the generator's own eigenvectors along the path to the first order of the change, with the corrections of its
//! rate. Then g obeys i ġ = (Λ + D₂(u)) g with (1 + X) D₂ = D_s + D X exactly (D_s: D on the slow pairs). What that
//! leaves on the fast pairs in the second order, F = [D X − X D_s]_f, is followed in its turn: F is made at the step's
//! two ends (it is zero at the middle, as D is), taken through the three nodes like D, and its polynomial is added to
//! X. With it (1 + X) D₂ = D_s + D X − F at the two ends, and D₂'s fast pairs are of the third order in the change.
//! They are left out. No time of the step is resolved for a fast pair and nothing of it is integrated: 1 + X(u) is a
//! matrix at any time, at any |ω| |h| however large.
//!
//! **A slow pair**, and every state with itself, stays in D₂: its slow pairs at the two ends, from
//! (1 + X) D₂ = D_s + D X − F by the factors of 1 + X there -- exactly, so that the change's own light shift
//! −Σ_e D_ge D_eg′/ω_eg′ is in it (a hop through unresolved pairs left it out) and a generator without a decay keeps
//! the state's norm (measured: within 10⁻⁷ over a step where the following is 0.04, 10⁻¹⁰ where it is 0.01; taken to
//! the second order alone it drifted by a sixth of the following's square there) --, and through the three nodes.
//!
//! g = e^{−iΛ(u + h)} m: Ĥ₀'s phases and decays are factors, exact over any width, and i ṁ = D̃₂(u) m with
//! D̃₂_ab = e^{iω_ab(u + h)} (D₂)_ab on the slow pairs. m is hopped from the step's start to its middle and to its end
//! -- two hops a step, whatever Ĥ₀ --, each the first Magnus term, m ← e^{−iJ} m with J = ∫D̃₂ over the hop, the
//! integral over the pair's turn exact: for a hop from u over w, with D₂ = uA + u²B,
//!   J_ab = e^{iω_ab(u + h)} w [A_ab (u I₀ + w I₁) + B_ab (u² I₀ + 2uw I₁ + w² I₂)],  I_p = ∫₀¹ s^p e^{zs} ds,  z = iω_ab w,
//! the moments by their series Σ_j z^j/(j!(j + p + 1)) -- |z| < 1 for a slow pair over a half step: no cancellation,
//! and summed until its terms are within the round-off of the state the exponent acts on or of the exponent's own
//! first term (`series`, `Picture::exponent`). A hop asked wider than a turn of the slow pairs (beyond the step's
//! end: `Picture::extend`) is taken in the fewest equal parts within it.
//!
//! Between the three nodes m is their quintic: its values there and its rates ṁ = −iD̃₂m (zero at the middle), the
//! Hermite interpolant. m changes by the slow pairs alone, so its sixth derivative is of the size |ω|⁵ |D₂| with
//! |ω| |h| < 1, and the quintic errs by no more than (4/27)/720 of h⁶ times it. So the state at any time u of the step
//! is x(u) = R (1 + X(u)) e^{−iΛ(u + h)} m(u) -- n exponentials and two products of n² (`Picture::at`), no hop --, and
//! at the step's three nodes it is the hops' own.
//!
//! Where nothing moves (D = 0: an atom at rest, a generator still along every path) X = 0 and m is constant: the state
//! is e^{−i(c + Λ)(u + h)} on its start, exact at any detuning times step.
//!
//! What is left out, all of the change alone:
//! - of the slow pairs, the hop's second Magnus term, −½∫∫[D̃₂(t₁), D̃₂(t₂)]: over a half step it is
//!   (h⁵/60)[A, [Λ, A]] and terms of [A, B] -- the fifth order in the step, so the propagation is of the fourth order
//!   there like the three nodes of D (Simpson's rule's: their cubic is odd about the middle; what stays is τ⁵D⁗/2880 a
//!   step);
//! - of the fast pairs, the third order of the following and what the three nodes miss of F between them: within
//!   the square of the following at the step's end, |X|² ≤ (|D|/|ω|)² -- with |D| ≈ k·v |h| times the coupling V that is
//!   (k·v h)² (V/ω)², falling with the step and with the detuning --, and measured an order and more below it.
//!   (A hop through such a pair left the light shift of the change, (k·v τ)²/24 of the light shift's turn over the
//!   step at any detuning.)
//! (step/tests.rs, `a_step_is_of_the_fourth_order_in_the_path_s_change_where_its_hops_resolve_the_generator`.)
//!
//! A radian between nodes is where the two treatments meet: a pair slower than that is resolved by the nodes (its
//! moments' series converge without cancellation, its quintic holds), a pair faster is followed more closely than a
//! hop over a half step would take it (|X| ≤ |D| |h| there, the size of the hop's own exponent). A half step takes
//! the pairs by its own nodes.
//!
//! # What follows the generator's own frequencies, and what does not
//! The propagation's own work -- the decomposition, the following, two hops -- is the same whatever Ĥ₀'s frequencies
//! times the step are. The step's forms are read on the state at times of a quadrature (step.rs `propagate`), and
//! their number still follows the forms' bandwidth: the kick is a quotient by the state's norm, which with a decay is a
//! sum of the pairs' exponentials of its own -- ringing wherever the state holds a fast pair's two states --, and a
//! quotient of two such sums has no closed form. (The forms on the unnormalized state have one, pair by pair: the
//! moments above. They would serve where the norm is one, which is no cluster that scatters.) A state there costs n
//! exponentials and two products of n², against a hop's series and its exponential before.
//!
//! # Towards a defective generator
//! A state taken through the eigenvectors rounds by κ(R) u (1 + |τ| × the decay rates' spread) relative to its norm
//! (`linalg::Eigen`), and κ(R) grows without bound towards an exceptional point. There the state is taken without the
//! decomposition (`Picture::basis` None): from stop to stop of the caller's (`Picture::advance`) by CF4 (Blanes & Moan
//! 2006) of the same Ĥ₀ + uA + u²B, two exponentials on the vector to round-off (`linalg::exp_apply`), whose rounding
//! does not know κ. It is exact where nothing moves; with D it is of fourth order in the stride's width, which is the
//! caller's nodes' spacing -- a few radians of Ĥ₀'s bound at most --, so its error is a share of D's own effect, not an
//! error of Ĥ₀'s frequencies. The rule is the rounding's, not a tuned number: the decomposition where a state through
//! it rounds no more than through the exponential of Ĥ₀ over the step (`linalg::Chebyshev::rounding`), the exponential
//! otherwise.

use crate::linalg::{self, C64, Mat};
use crate::units::UNIT_ROUNDOFF;

/// A generator along a step's path (module documentation): filled by the caller with Ĥ at the middle (`base`) and at
/// the two ends (`ends`: at −h, at +h), then prepared (`prepare`) and launched from the step's start state (`launch`);
/// kept by a trajectory from step to step, its memory made once.
pub struct Picture {
    n: usize,
    /// Ĥ₀ (n × n; rad/µs), the caller's: kept as it is given.
    pub base: Mat,
    /// Ĥ(−h) and Ĥ(+h), the caller's; after `prepare` the slopes the hops take: A and B of D₂ on the slow pairs in
    /// the eigenbasis, or of D as it is given where there is no decomposition.
    pub ends: [Mat; 2],
    /// A hop's exponent.
    matrix: Mat,
    /// Ĥ₀'s decomposition of its traceless part, where the state goes through it.
    basis: Option<linalg::Eigen>,
    /// Ĥ₀'s eigenvalues, the scalar part with them.
    lambda: Vec<C64>,
    /// The slow pairs' moments I_p = ∫₀¹ s^p e^{iω_ab w s} ds, p = 0, 1, 2 (3n²), of the hops of the width `width`.
    moments: Vec<C64>,
    width: f64,
    /// The exponential's work vectors (2n), a state's coordinates (2n) and the pairs' turns (n).
    vectors: Vec<C64>,
    coordinates: Vec<C64>,
    turns: Vec<C64>,
    /// Work matrices (7n²): D, X and D_s at an end, the second order and D₂ at the two ends, and 1 + X's factors
    /// at the step's end.
    product: Vec<C64>,
    /// Whether D is anything at all, and the step's half h.
    moving: bool,
    half: f64,
    /// The fast pairs (n²), their following X₀, X₁, X₂ (3n²), whether any pair is followed at all, and 1 + X(−h)
    /// factored (LU with partial pivoting, n² and its pivots): the start state's way into g.
    fast: Vec<bool>,
    follow: Vec<C64>,
    followed: bool,
    factors: Vec<C64>,
    pivots: Vec<usize>,
    /// The path: m and ṁ at −h, 0 and h and at the far end of an extension beyond h (8n), that end, and the rounding
    /// of the hops' exponentials relative to the state's norm.
    path: Vec<C64>,
    far: f64,
    bound: f64,
    /// Without the decomposition: the frame's state at the last two stops (2n) and their times.
    stops: Vec<C64>,
    times: [f64; 2],
    /// max|λ_a − λ_b| over the slow pairs (rad/µs): what a hop's series is summed against.
    spread: f64,
    /// The bounds of Ĥ₀ less its scalar part (`linalg::bounds`), and the slopes' Frobenius norms as the hops take them:
    /// what bounds a hop's exponent, and what a hop is priced by.
    bounds: [f64; 2],
    slopes: [f64; 2],
    /// max|λ_a* − λ_b| (rad/µs): the bandwidth of the forms ⟨x|·|x⟩ on a state propagated by Ĥ₀ -- their frequencies and
    /// decay rates; without the decomposition its bound 2(ρ + γ).
    pub bandwidth: f64,
    /// The rounding of a state taken through the decomposition, relative to its norm (none without it: the
    /// exponentials' are the strides').
    pub rounding: f64,
    /// The complex multiply-adds counted so far (`decide::Work` counts them four real ones each).
    pub work: f64,
    /// The hops taken since `prepare` (through the decomposition: of m; without it: the strides between stops).
    pub hops: usize,
}

impl Default for Picture {
    fn default() -> Picture {
        let none = || Mat::zeros(0, 0);
        Picture {
            n: 0,
            base: none(),
            ends: [none(), none()],
            matrix: none(),
            basis: None,
            lambda: Vec::new(),
            moments: Vec::new(),
            width: f64::NAN,
            vectors: Vec::new(),
            coordinates: Vec::new(),
            turns: Vec::new(),
            product: Vec::new(),
            moving: false,
            half: 0.0,
            fast: Vec::new(),
            follow: Vec::new(),
            followed: false,
            factors: Vec::new(),
            pivots: Vec::new(),
            path: Vec::new(),
            far: 0.0,
            bound: 0.0,
            stops: Vec::new(),
            times: [0.0; 2],
            spread: 0.0,
            bounds: [0.0; 2],
            slopes: [0.0; 2],
            bandwidth: 0.0,
            rounding: 0.0,
            work: 0.0,
            hops: 0,
        }
    }
}

/// The terms of ∫₀¹ t^p e^{zt} dt's series Σ_j z^j/(j!(j + p + 1)) at |z| ≤ x for an integral of the size `size` at
/// most: the least T with x^T/T! times the size within the unit round-off, the size no more than one -- the terms from T
/// on are within the round-off of the state the integral acts on, or of its own first term where that is more.
fn series(x: f64, size: f64) -> usize {
    let (mut terms, mut next) = (1, x * size.min(1.0));
    while next > UNIT_ROUNDOFF {
        terms += 1;
        next *= x / terms as f64;
    }
    terms
}

/// The quintic through three nodes at s = −1, 0, 1 (the Hermite interpolant of values and rates): the weights of the
/// values at −1, 0, 1 and of the rates d/ds at −1 and 1 (the middle's rate is none here).
fn quintic(s: f64) -> ([f64; 3], [f64; 2]) {
    let (minus, middle, plus) = (0.5 * s * (s - 1.0), 1.0 - s * s, 0.5 * s * (s + 1.0));
    let (m2, p2) = (minus * minus, plus * plus);
    ([(1.0 + 3.0 * (s + 1.0)) * m2, middle * middle, (1.0 - 3.0 * (s - 1.0)) * p2], [(s + 1.0) * m2, (s - 1.0) * p2])
}

/// P A = L U of the n × n matrix in `f`, in its place (L unit lower, U upper; partial pivoting, the pivots' rows).
fn factor(f: &mut [C64], pivots: &mut [usize], n: usize) {
    for k in 0..n {
        let p = (k..n).max_by(|&i, &j| f[i * n + k].abs1().total_cmp(&f[j * n + k].abs1())).unwrap();
        pivots[k] = p;
        if p != k {
            for j in 0..n {
                f.swap(k * n + j, p * n + j);
            }
        }
        let (above, below) = f.split_at_mut((k + 1) * n);
        let pivot_row = &above[k * n..];
        for row in below.chunks_mut(n) {
            let l = row[k] / pivot_row[k];
            row[k] = l;
            for (z, upper) in row[k + 1..].iter_mut().zip(&pivot_row[k + 1..]) {
                *z -= l * *upper;
            }
        }
    }
}

/// B ← A⁻¹B by A's factors (`factor`), B of n rows of `columns` numbers, row by row.
fn solve(f: &[C64], pivots: &[usize], n: usize, b: &mut [C64], columns: usize) {
    for (k, &p) in pivots.iter().enumerate() {
        if p != k {
            for j in 0..columns {
                b.swap(k * columns + j, p * columns + j);
            }
        }
    }
    for i in 0..n {
        let (above, row) = b.split_at_mut(i * columns);
        for j in 0..i {
            let l = f[i * n + j];
            for (z, other) in row[..columns].iter_mut().zip(&above[j * columns..(j + 1) * columns]) {
                *z -= l * *other;
            }
        }
    }
    for i in (0..n).rev() {
        let (rest, below) = b.split_at_mut((i + 1) * columns);
        let row = &mut rest[i * columns..];
        for j in i + 1..n {
            let upper = f[i * n + j];
            for (z, other) in row.iter_mut().zip(&below[(j - i - 1) * columns..(j - i) * columns]) {
                *z -= upper * *other;
            }
        }
        let diagonal = f[i * n + i];
        for z in row.iter_mut() {
            *z = *z / diagonal;
        }
    }
}

impl Picture {
    /// The picture for n states.
    pub fn fit(&mut self, n: usize) {
        if self.n != n {
            self.n = n;
            for m in [&mut self.base, &mut self.matrix].into_iter().chain(&mut self.ends) {
                (m.rows, m.cols) = (n, n);
                m.data.resize(n * n, C64::ZERO);
            }
            let size = n * n;
            for (v, length) in [
                (&mut self.lambda, n),
                (&mut self.vectors, 2 * n),
                (&mut self.coordinates, 2 * n),
                (&mut self.turns, n),
                (&mut self.product, 7 * size),
                (&mut self.follow, 3 * size),
                (&mut self.moments, 3 * size),
                (&mut self.factors, size),
                (&mut self.path, 8 * n),
                (&mut self.stops, 2 * n),
            ] {
                v.resize(length, C64::ZERO);
            }
            self.fast.resize(size, false);
            self.pivots.resize(n, 0);
        }
    }

    /// The states.
    pub fn states(&self) -> usize {
        self.n
    }

    /// Whether the state goes through Ĥ₀'s decomposition.
    pub fn decomposed(&self) -> bool {
        self.basis.is_some()
    }

    /// Whether the generator changes along the path at all.
    pub fn moving(&self) -> bool {
        self.moving
    }

    /// Prepares the picture of a step of 2h from `base` and, unless the generator is `still` along the path (then they
    /// are not read), `ends`: the slopes, Ĥ₀'s decomposition and the route of the state (module documentation, #
    /// Towards a defective generator); through the decomposition the fast pairs' following and the slow pairs' slopes
    /// (`follow`). `work` begins anew with what this counted.
    pub fn prepare(&mut self, half: f64, still: bool) {
        let n = self.n;
        (self.work, self.width, self.moving, self.hops) = (0.0, f64::NAN, false, 0);
        (self.half, self.followed, self.spread) = (half, false, 0.0);
        if !still {
            let [minus, plus] = &mut self.ends;
            let (first, second) = (0.5 / half, 0.5 / (half * half));
            for ((m, p), c) in minus.data.iter_mut().zip(plus.data.iter_mut()).zip(&self.base.data) {
                let (a, b) = ((*p - *m) * first, (*p + *m - *c * 2.0) * second);
                (*m, *p) = (a, b);
                self.moving |= a != C64::ZERO || b != C64::ZERO;
            }
            self.work += (2 * n * n) as f64;
        }
        self.matrix.data.copy_from_slice(&self.base.data);
        let scalar = self.matrix.trace() / n as f64;
        for i in 0..n {
            self.matrix[(i, i)] -= scalar;
        }
        let linalg::Bounds { rho, gamma, .. } = linalg::bounds(&self.matrix, &mut self.vectors);
        self.bounds = [rho, gamma];
        let tau = 2.0 * half.abs();
        let exponential = linalg::Chebyshev::plan(rho * tau, gamma * tau).rounding(n);
        let mut rounding = 0.0;
        self.basis = linalg::eigen(&self.matrix).filter(|e| {
            let decay = |pick: fn(f64, f64) -> f64, from: f64| e.values.iter().map(|l| l.im).fold(from, pick);
            rounding = e.condition * UNIT_ROUNDOFF * (1.0 + tau * (decay(f64::max, f64::NEG_INFINITY) - decay(f64::min, f64::INFINITY)));
            rounding <= exponential
        });
        match &self.basis {
            Some(e) => {
                for (l, value) in self.lambda.iter_mut().zip(&e.values) {
                    *l = *value + scalar;
                }
                self.bandwidth = self.lambda.iter().flat_map(|a| self.lambda.iter().map(move |b| (a.conj() - *b).abs())).fold(0.0, f64::max);
                self.rounding = rounding;
                self.work += e.work;
                if self.moving {
                    // A ← R⁻¹AR and B ← R⁻¹BR: two products of n³ each.
                    for slope in &mut self.ends {
                        for (row, out) in self.product[..n * n].chunks_mut(n).enumerate() {
                            for (b, z) in out.iter_mut().enumerate() {
                                *z = (0..n).map(|k| slope.data[row * n + k] * e.vectors.data[k * n + b]).sum();
                            }
                        }
                        for (row, out) in slope.data.chunks_mut(n).enumerate() {
                            for (b, z) in out.iter_mut().enumerate() {
                                *z = (0..n).map(|k| e.inverse.data[row * n + k] * self.product[k * n + b]).sum();
                            }
                        }
                    }
                    self.work += (4 * n * n * n) as f64;
                }
            }
            None => (self.bandwidth, self.rounding) = (2.0 * (rho + gamma), 0.0),
        }
        if self.basis.is_some() && self.moving {
            self.follow();
        }
        self.slopes = if self.moving { [0, 1].map(|i| self.ends[i].data.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt()) } else { [0.0; 2] };
    }

    /// The fast pairs and their following X (module documentation, # The state at any time of the step), and in the
    /// place of the slopes of D those of D₂ on the slow pairs.
    /// - X's first order from the slopes of D, pair by pair.
    /// - At the two ends the second order D X − X D_s. Its fast part F is followed in its turn (X's second order: the
    ///   same polynomial of F through the three nodes -- F is zero at the middle, as D is).
    /// - At the two ends D₂ whole, (1 + X) D₂ = D_s + D X − F by the factors of 1 + X there, and of it the slow pairs,
    ///   through the three nodes (zero at the middle). The factors at −h are kept: the start state's way into g.
    fn follow(&mut self) {
        let (n, h) = (self.n, self.half);
        let size = n * n;
        let [a, b] = &mut self.ends;
        // A pair's following of the change uA + u²B: the polynomial X₀ + uX₁ + u²X₂ with ωX − iẊ + uA + u²B = 0.
        let following = |omega: C64, a: C64, b: C64| {
            let two = -(b / omega);
            let one = (C64::I * two * 2.0 - a) / omega;
            [C64::I * one / omega, one, two]
        };
        for row in 0..n {
            for column in 0..n {
                let (k, omega) = (row * n + column, self.lambda[row] - self.lambda[column]);
                let fast = row != column && omega.abs() * h.abs() >= 1.0;
                self.fast[k] = fast;
                let mut x = [C64::ZERO; 3];
                if fast {
                    x = following(omega, a.data[k], b.data[k]);
                    self.followed |= a.data[k] != C64::ZERO || b.data[k] != C64::ZERO;
                } else if row != column {
                    self.spread = self.spread.max(omega.abs());
                }
                for (j, x) in x.into_iter().enumerate() {
                    self.follow[j * size + k] = x;
                }
            }
        }
        self.work += (3 * size) as f64;
        if !self.followed {
            return;
        }
        let (work, rest) = self.product.split_at_mut(3 * size);
        let (d, work) = work.split_at_mut(size);
        let (x, slow) = work.split_at_mut(size);
        let (ends, rest) = rest.split_at_mut(2 * size);
        let (whole, factors) = rest.split_at_mut(size);
        // D, X and D_s at the time u.
        let at = |u: f64, follow: &[C64], d: &mut [C64], x: &mut [C64], slow: &mut [C64]| {
            for k in 0..size {
                d[k] = a.data[k] * u + b.data[k] * (u * u);
                x[k] = follow[k] + follow[size + k] * u + follow[2 * size + k] * (u * u);
                slow[k] = if self.fast[k] { C64::ZERO } else { d[k] };
            }
        };
        // The second order D X − X D_s at the two ends (X is on the fast pairs alone, D_s on the slow ones), row by row.
        for (end, u) in [-h, h].into_iter().enumerate() {
            at(u, &self.follow, d, x, slow);
            let out = &mut ends[end * size..(end + 1) * size];
            out.fill(C64::ZERO);
            for (row, out) in out.chunks_mut(n).enumerate() {
                for k in 0..n {
                    let (left, right) = (d[row * n + k], x[row * n + k]);
                    for ((z, x), slow) in out.iter_mut().zip(&x[k * n..(k + 1) * n]).zip(&slow[k * n..(k + 1) * n]) {
                        *z += left * *x - right * *slow;
                    }
                }
            }
        }
        // Its fast part through the three nodes, followed: X's second order.
        let (first, second) = (0.5 / h, 0.5 / (h * h));
        for row in 0..n {
            for column in 0..n {
                let k = row * n + column;
                if self.fast[k] {
                    let (minus, plus) = (ends[k], ends[size + k]);
                    for (j, x) in following(self.lambda[row] - self.lambda[column], (plus - minus) * first, (plus + minus) * second).into_iter().enumerate() {
                        self.follow[j * size + k] += x;
                    }
                }
            }
        }
        // D₂ at the two ends: (1 + X) D₂ = D_s + D X − F, and of it the slow pairs.
        let mut spare = [0; crate::model::STATES];
        for (end, u) in [-h, h].into_iter().enumerate() {
            at(u, &self.follow, d, x, slow);
            let out = &mut ends[end * size..(end + 1) * size];
            for k in 0..size {
                whole[k] = if self.fast[k] { -out[k] } else { slow[k] };
            }
            for (row, whole) in whole.chunks_mut(n).enumerate() {
                for k in 0..n {
                    let left = d[row * n + k];
                    for (z, x) in whole.iter_mut().zip(&x[k * n..(k + 1) * n]) {
                        *z += left * *x;
                    }
                }
            }
            let (f, pivots) = if end == 0 { (&mut self.factors[..], &mut self.pivots[..]) } else { (&mut factors[..], &mut spare[..n]) };
            f.copy_from_slice(x);
            for i in 0..n {
                f[i * n + i] += C64::ONE;
            }
            factor(f, pivots, n);
            solve(f, pivots, n, whole, n);
            out.copy_from_slice(whole);
        }
        for k in 0..size {
            (a.data[k], b.data[k]) = if self.fast[k] { (C64::ZERO, C64::ZERO) } else { ((ends[size + k] - ends[k]) * first, (ends[size + k] + ends[k]) * second) };
        }
        self.work += (6 * n * size + 2 * (n * size / 3 + n * size) + 16 * size) as f64;
    }

    /// The bound of a hop's exponent from the time u over the width w: ∫‖D‖ at the slopes' norms, D(t) = tA + t²B at the
    /// farther end of the hop from the middle.
    fn moved(&self, u: f64, w: f64) -> f64 {
        let far = u.abs().max((u + w).abs());
        (far * self.slopes[0] + far * far * self.slopes[1]) * w.abs()
    }

    /// A hop's exponent ∫₀ʷ e^{iω_ab t}(D₂)_ab(u + t) dt into `matrix` (module documentation), from the time u over the
    /// width w: w [A_ab (u I₀ + w I₁) + B_ab (u² I₀ + 2uw I₁ + w² I₂)] with the pair's moments
    /// I_p = ∫₀¹ s^p e^{zs} ds = Σ_j z^j/(j!(j + p + 1)), z = iω_ab w, each pair's series as far as its own |z| and
    /// the exponent's bound ask (`series`). The moments are a width's: made once for the hops of one width.
    fn exponent(&mut self, u: f64, w: f64) {
        let n = self.n;
        let size = n * n;
        if w != self.width {
            // A pair and its reverse have z and −z: the series' even and odd parts, summed once for the two.
            let bound = self.moved(u, w);
            let mut summed = 0;
            for a in 0..n {
                for b in a..n {
                    let k = a * n + b;
                    if self.fast[k] {
                        continue;
                    }
                    let z = C64::I * (self.lambda[a] - self.lambda[b]) * w;
                    let terms = series(z.abs(), bound);
                    let (mut power, mut parts) = (C64::ONE, [[C64::ZERO; 3]; 2]);
                    for j in 0..terms {
                        for (p, moment) in parts[j % 2].iter_mut().enumerate() {
                            *moment += power * (1.0 / (j + p + 1) as f64);
                        }
                        power = power * z * (1.0 / (j + 1) as f64);
                    }
                    for p in 0..3 {
                        (self.moments[p * size + k], self.moments[p * size + b * n + a]) = (parts[0][p] + parts[1][p], parts[0][p] - parts[1][p]);
                    }
                    summed += terms;
                }
            }
            self.width = w;
            self.work += (4 * summed) as f64;
        }
        let (zero, rest) = self.moments.split_at(size);
        let (one, two) = rest.split_at(size);
        for k in 0..size {
            self.matrix.data[k] = if self.fast[k] {
                C64::ZERO
            } else {
                (self.ends[0].data[k] * (zero[k] * u + one[k] * w) + self.ends[1].data[k] * (zero[k] * (u * u) + one[k] * (2.0 * u * w) + two[k] * (w * w))) * w
            };
        }
        self.work += (4 * size) as f64;
    }

    /// `matrix`'s slow pairs turned by e^{iω_ab s}: from a pair's own time to the path's, s = u + h from the step's
    /// start. A slow pair's rates differ by less than 1/|h|, so the factor is bounded whatever the states' own decays
    /// over s are: the turn by the two states' phases, the modulus by the rates' difference.
    fn turn(&mut self, s: f64) {
        if s == 0.0 {
            return;
        }
        let n = self.n;
        for (turn, l) in self.turns.iter_mut().zip(&self.lambda) {
            *turn = C64::cis(l.re * s);
        }
        for a in 0..n {
            for b in 0..n {
                if a != b && !self.fast[a * n + b] {
                    self.matrix.data[a * n + b] *= self.turns[a] * self.turns[b].conj() * ((self.lambda[b].im - self.lambda[a].im) * s).exp();
                }
            }
        }
        self.work += (2 * n * n) as f64;
    }

    /// One hop of m from the time u over the width w, either sign (module documentation): m ← e^{−iJ} m, in the fewest
    /// equal parts within a turn of the slow pairs. Returns the rounding of its exponentials relative to the state's
    /// norm (`linalg::Chebyshev::rounding`).
    fn hop(&mut self, m: &mut [C64], u: f64, w: f64) -> f64 {
        let n = self.n;
        let parts = ((self.spread * w.abs() / std::f64::consts::TAU).ceil() as usize).max(1);
        let (part, mut bound) = (w / parts as f64, 0.0);
        for k in 0..parts {
            let from = u + k as f64 * part;
            self.exponent(from, part);
            self.turn(from + self.half);
            let plan = linalg::exp_apply(&mut self.matrix, C64::new(0.0, -1.0), m, &mut self.vectors);
            bound += plan.rounding(n);
            self.work += (plan.parts * plan.terms * n * n) as f64;
        }
        self.hops += parts;
        bound
    }

    /// ṁ = −i D̃₂(u) m at the time u, into `rate`.
    fn rate(&mut self, u: f64, m: &[C64], rate: &mut [C64]) {
        let n = self.n;
        for ((z, a), b) in self.matrix.data.iter_mut().zip(&self.ends[0].data).zip(&self.ends[1].data) {
            *z = *a * u + *b * (u * u);
        }
        self.turn(u + self.half);
        for (row, out) in self.matrix.data.chunks(n).zip(rate.iter_mut()) {
            let z: C64 = row.iter().zip(m).map(|(d, m)| *d * *m).sum();
            *out = C64::new(z.im, -z.re);
        }
        self.work += (2 * n * n) as f64;
    }

    /// One stride of the frame's state without the decomposition, from the time u over the width w, either sign: CF4
    /// of Ĥ₀ + uA + u²B (module documentation, # Towards a defective generator). Returns its exponentials' rounding.
    fn stride(&mut self, state: &mut [C64], u: f64, w: f64) -> f64 {
        let n = self.n;
        let mut bound = 0.0;
        // Ĥ at the stride's two Gauss nodes c = ½ ∓ √3/6, the weights ¼ ± √3/6.
        let r = 3.0_f64.sqrt() / 6.0;
        let (u1, u2) = (u + (0.5 - r) * w, u + (0.5 + r) * w);
        for [w1, w2] in [[0.25 + r, 0.25 - r], [0.25 - r, 0.25 + r]] {
            let (first, second) = if self.moving { (w1 * u1 + w2 * u2, w1 * u1 * u1 + w2 * u2 * u2) } else { (0.0, 0.0) };
            for (((z, c), a), b) in self.matrix.data.iter_mut().zip(&self.base.data).zip(&self.ends[0].data).zip(&self.ends[1].data) {
                *z = *c * (w1 + w2);
                if self.moving {
                    *z += *a * first + *b * second;
                }
            }
            let plan = linalg::exp_apply(&mut self.matrix, C64::new(0.0, -w), state, &mut self.vectors);
            bound += plan.rounding(n);
            self.work += ((3 + plan.parts * plan.terms) * n * n) as f64;
        }
        self.hops += 1;
        bound
    }

    /// Launches the picture from the frame's state x at the step's start, the time −h. Through the decomposition:
    /// m there, (1 + X(−h))⁻¹R⁻¹x, and where the generator moves m's two hops to the middle and to the end with its
    /// rates at the nodes -- the state is then known at every time of the step (`at`). Without it: the first stop.
    pub fn launch(&mut self, x: &[C64]) {
        let (n, h) = (self.n, self.half);
        (self.far, self.bound) = (h, 0.0);
        let Some(e) = &self.basis else {
            self.stops[..n].copy_from_slice(x);
            self.stops[n..].copy_from_slice(x);
            self.times = [-h; 2];
            return;
        };
        let mut path = std::mem::take(&mut self.path);
        for (a, d) in path[..n].iter_mut().enumerate() {
            *d = e.inverse.data[a * n..(a + 1) * n].iter().zip(x).map(|(r, x)| *r * *x).sum();
        }
        self.work += (n * n) as f64;
        if self.moving {
            if self.followed {
                // g = (1 + X(−h))⁻¹ d by the factors.
                solve(&self.factors, &self.pivots, n, &mut path[..n], 1);
                self.work += (n * n) as f64;
            }
            let (start, rest) = path.split_at_mut(n);
            let (rate, rest) = rest.split_at_mut(n);
            self.rate(-h, start, rate);
            let (middle, rest) = rest.split_at_mut(2 * n);
            middle[..n].copy_from_slice(start);
            self.bound += self.hop(&mut middle[..n], -h, h);
            let (end, rest) = rest.split_at_mut(n);
            end.copy_from_slice(&middle[..n]);
            self.bound += self.hop(end, 0.0, h);
            self.rate(h, end, &mut rest[..n]);
        }
        self.path = path;
    }

    /// Follows the path beyond the step's end as far as the time `far` (from the middle), on the same generator
    /// uA + u²B continued: one more hop of m and its rate there. Nothing where the state takes strides or nothing moves.
    pub fn extend(&mut self, far: f64) {
        let (n, h) = (self.n, self.half);
        if self.basis.is_some() && self.moving && far != h {
            let mut path = std::mem::take(&mut self.path);
            let (before, beyond) = path.split_at_mut(6 * n);
            let (m, rate) = beyond.split_at_mut(n);
            m.copy_from_slice(&before[4 * n..5 * n]);
            self.bound += self.hop(m, h, far - h);
            self.rate(far, m, rate);
            self.path = path;
            self.far = far;
        }
    }

    /// The frame's state at the time u (from the middle) through the decomposition (module documentation):
    /// x = R (1 + X(u)) e^{−iΛ(u + h)} m(u), m the nodes' quintic within the step and the cubic of the end's and the
    /// far end's values and rates beyond it.
    fn at(&mut self, u: f64, x: &mut [C64]) {
        let (n, h) = (self.n, self.half);
        let size = n * n;
        let e = self.basis.as_ref().expect("a state through the decomposition");
        let (d, v) = self.coordinates.split_at_mut(n);
        let path = &self.path;
        let node = |k: usize| &path[k * n..(k + 1) * n];
        if !self.moving {
            d.copy_from_slice(node(0));
        } else if self.far == h || (u - h) / h <= 0.0 {
            let (values, rates) = quintic(u / h);
            let (start, rate, middle, end, last) = (node(0), node(1), node(2), node(4), node(5));
            for a in 0..n {
                d[a] = start[a] * values[0] + middle[a] * values[1] + end[a] * values[2] + (rate[a] * rates[0] + last[a] * rates[1]) * h;
            }
        } else {
            // The cubic on [h, far]: values and rates at its two ends.
            let w = self.far - h;
            let t = (u - h) / w;
            let (values, rates) = ([(1.0 + 2.0 * t) * (1.0 - t) * (1.0 - t), t * t * (3.0 - 2.0 * t)], [t * (1.0 - t) * (1.0 - t) * w, t * t * (t - 1.0) * w]);
            let (end, last, far, there) = (node(4), node(5), node(6), node(7));
            for a in 0..n {
                d[a] = end[a] * values[0] + far[a] * values[1] + last[a] * rates[0] + there[a] * rates[1];
            }
        }
        for (z, l) in d.iter_mut().zip(&self.lambda) {
            *z *= (C64::new(0.0, -(u + h)) * *l).exp();
        }
        let state: &[C64] = if self.followed {
            let (x0, rest) = self.follow.split_at(size);
            let (x1, x2) = rest.split_at(size);
            let square = u * u;
            for (a, z) in v.iter_mut().enumerate() {
                let row = a * n..(a + 1) * n;
                let followed: C64 = x0[row.clone()].iter().zip(&x1[row.clone()]).zip(&x2[row]).zip(d.iter()).map(|(((x0, x1), x2), d)| (*x0 + *x1 * u + *x2 * square) * *d).sum();
                *z = d[a] + followed;
            }
            v
        } else {
            d
        };
        for (a, x) in x.iter_mut().enumerate() {
            *x = e.vectors.data[a * n..(a + 1) * n].iter().zip(state).map(|(r, d)| *r * *d).sum();
        }
        self.work += ((if self.followed { 2 } else { 1 }) * size + 4 * n) as f64;
    }

    /// The frame's state at the next stop, the time u from the middle, ahead of the last one on the path. Through
    /// the decomposition it is the state at that time (`at`); without it the stride from the last stop, which stays
    /// kept for `between`. Returns the rounding of the exponentials behind it relative to the state's norm.
    pub fn advance(&mut self, u: f64, x: &mut [C64]) -> f64 {
        let n = self.n;
        if self.basis.is_some() {
            self.at(u, x);
            return self.bound;
        }
        let mut stops = std::mem::take(&mut self.stops);
        let (before, now) = stops.split_at_mut(n);
        before.copy_from_slice(now);
        let from = self.times[1];
        let bound = self.stride(now, from, u - from);
        x.copy_from_slice(now);
        self.stops = stops;
        self.times = [from, u];
        bound
    }

    /// The frame's state at a time u between the last two stops: the state there through the decomposition, the
    /// stride from the earlier stop without it.
    pub fn between(&mut self, u: f64, x: &mut [C64]) {
        let n = self.n;
        if self.basis.is_some() {
            return self.at(u, x);
        }
        x.copy_from_slice(&self.stops[..n]);
        let from = self.times[0];
        self.stride(x, from, u - from);
    }

    /// The counted complex multiply-adds of a launch and of the states at `stops` stops over the step, priced without
    /// taking them: through the decomposition the start's coordinates, the two hops of m at the slopes' bound of D₂
    /// over a half step (their series, their turn, their exponentials' plan, the slopes' powers) with the rates at the
    /// two ends, and a state per stop; without it a stride per stop at Ĥ₀'s bounds over the stops' mean spacing.
    pub fn price(&self, stops: usize) -> f64 {
        let (n, h) = (self.n, self.half.abs());
        let size = (n * n) as f64;
        if self.basis.is_none() {
            let w = 2.0 * h / stops as f64;
            let moved = self.moved(h - w, w);
            let plan = linalg::Chebyshev::plan(0.5 * self.bounds[0] * w + moved, 0.5 * self.bounds[1] * w + moved);
            return (stops * 2 * (3 + plan.parts * plan.terms)) as f64 * size;
        }
        let state = stops as f64 * ((if self.followed { 2.0 } else { 1.0 }) * size + 4.0 * n as f64);
        if !self.moving {
            return size + state;
        }
        let moved = self.moved(0.0, h);
        let (plan, terms) = (linalg::Chebyshev::plan(moved, moved), series(self.spread * h, moved));
        (2 + 4 * terms + 2 * (6 + plan.parts * plan.terms) + 4) as f64 * size + state
    }

    /// The largest row sum of |X| at the step's end: the size of the following there.
    #[cfg(test)]
    fn following(&self) -> f64 {
        let (n, h) = (self.n, self.half);
        let size = n * n;
        (0..n).map(|a| (0..n).map(|b| (self.follow[a * n + b] + self.follow[size + a * n + b] * h + self.follow[2 * size + a * n + b] * (h * h)).abs()).sum::<f64>()).fold(0.0, f64::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dense RK4 of iẋ = (Ĥ₀ + uA + u²B)x from −h to h in `steps` steps: the tests' reference of a picture's state.
    fn flow(base: &Mat, a: &Mat, b: &Mat, half: f64, start: &[C64], steps: usize) -> Vec<C64> {
        let n = start.len();
        let at = |u: f64| Mat::from_fn(n, n, |i, j| base[(i, j)] + a[(i, j)] * u + b[(i, j)] * (u * u));
        let slope = |u: f64, y: &[C64]| -> Vec<C64> { at(u).apply(y).into_iter().map(|z| C64::new(z.im, -z.re)).collect() };
        let add = |y: &[C64], k: &[C64], s: f64| -> Vec<C64> { y.iter().zip(k).map(|(y, k)| *y + *k * s).collect() };
        let dt = 2.0 * half / steps as f64;
        let mut x = start.to_vec();
        for i in 0..steps {
            let u = -half + i as f64 * dt;
            let k1 = slope(u, &x);
            let k2 = slope(u + 0.5 * dt, &add(&x, &k1, 0.5 * dt));
            let k3 = slope(u + 0.5 * dt, &add(&x, &k2, 0.5 * dt));
            let k4 = slope(u + dt, &add(&x, &k3, dt));
            x = (0..n).map(|s| x[s] + (k1[s] + k2[s] * 2.0 + k3[s] * 2.0 + k4[s]) * (dt / 6.0)).collect();
        }
        x
    }

    /// A picture of Ĥ₀ + uA + u²B over a step of 2h, prepared.
    fn pictured(base: &Mat, a: &Mat, b: &Mat, half: f64, moving: bool) -> Picture {
        let n = base.rows;
        let mut picture = Picture::default();
        picture.fit(n);
        picture.base = base.clone();
        picture.ends = [-half, half].map(|u| Mat::from_fn(n, n, |i, j| base[(i, j)] + a[(i, j)] * u + b[(i, j)] * (u * u)));
        picture.prepare(half, !moving);
        picture
    }

    /// The largest error of a launched picture's state at the times `at` (from the middle) against the flow of the
    /// same Ĥ₀ + uA + u²B from the step's start to each: that part of the step is a step of its own middle.
    fn errors(picture: &mut Picture, base: &Mat, a: &Mat, b: &Mat, half: f64, start: &[C64], at: &[f64], steps: usize) -> f64 {
        let n = start.len();
        let mut x = vec![C64::ZERO; n];
        let mut worst = 0.0_f64;
        for &u in at {
            picture.advance(u, &mut x);
            let (middle, part) = (0.5 * (u - half), 0.5 * (u + half));
            let shifted = Mat::from_fn(n, n, |i, j| base[(i, j)] + a[(i, j)] * middle + b[(i, j)] * (middle * middle));
            let slope = Mat::from_fn(n, n, |i, j| a[(i, j)] + b[(i, j)] * (2.0 * middle));
            let exact = flow(&shifted, &slope, b, part, start, steps);
            worst = worst.max(x.iter().zip(&exact).map(|(x, e)| (*x - *e).abs()).fold(0.0, f64::max));
        }
        worst
    }

    /// Towards a defective generator the state leaves the decomposition (module documentation): a resonant two-level
    /// atom whose coupling is a quarter of its decay rate, [[0, c], [c, −iΓ/2]] at c = Γ/4, has one eigenvector. At that
    /// point and a millionth of the coupling off it the picture takes the exponential's strides; a hundredth off it,
    /// where the eigenvectors' condition is some hundreds, it takes the decomposition's. Each way the state at the
    /// step's end, through sixteen stops, is the generator's flow: to the reference's own accuracy where nothing
    /// moves, and with slopes within the second order of what they move between two stops.
    #[test]
    fn a_defective_generator_s_strides_are_the_exponential_s() {
        let (gamma, half, stops) = (38.117, 0.03, 16);
        let start = [C64::ONE, C64::new(0.2, -0.1)];
        for (off, decomposed) in [(0.0, false), (1e-6, false), (1e-2, true)] {
            let c = 0.25 * gamma * (1.0 + off);
            let base = Mat::from_fn(2, 2, |i, j| if i != j { C64::real(c) } else if i == 1 { C64::new(0.0, -0.5 * gamma) } else { C64::ZERO });
            for moving in [false, true] {
                let slope = |size: f64| Mat::from_fn(2, 2, |i, j| if i != j && moving { C64::new(size, 0.3 * size * (i as f64 - j as f64)) } else { C64::ZERO });
                let (a, b) = (slope(2.0), slope(30.0));
                let exact = flow(&base, &a, &b, half, &start, 100_000);
                let mut picture = pictured(&base, &a, &b, half, moving);
                assert_eq!(picture.decomposed(), decomposed, "{off} off the exceptional point");
                let mut x = [C64::ZERO; 2];
                picture.launch(&start);
                let w = 2.0 * half / stops as f64;
                for k in 1..=stops {
                    picture.advance(if k == stops { half } else { -half + k as f64 * w }, &mut x);
                }
                let error = x.iter().zip(&exact).map(|(x, e)| (*x - *e).abs()).fold(0.0, f64::max);
                // What the slopes move between two stops at the step's end: (h‖A‖ + h²‖B‖) w, its square the second order.
                let moved = (half * 2.0 + half * half * 30.0) * 2.0_f64.sqrt() * w;
                eprintln!("{off} off the exceptional point, moving {moving}: the state's error {error:e} (the slopes move {moved:e} between stops)");
                assert!(error < if moving { moved * moved } else { 1e-12 }, "{off} off the exceptional point, moving {moving}: {error:e}");
            }
        }
    }

    /// The picture's state is the generator's flow where every pair is fast: Ĥ₀ of detunings up to 600 rad/µs with a
    /// decay and couplings of 40 rad/µs on four states, against RK4 at 4·10⁵ steps of the same Ĥ₀ + uA + u²B. Without
    /// slopes the state is exact, at the end and between the nodes. With slopes that turn the couplings by a tenth of
    /// a radian over the step it is the following alone (no pair is slow: two hops of nothing): it errs by the second
    /// order of the following, within the square of X's size at the step's end, and by a quarter of that at half the
    /// slopes.
    #[test]
    fn a_fast_pair_s_state_follows_the_change() {
        let (n, half) = (4, 0.05);
        let entry = |seed: f64, i: usize, j: usize| C64::new((seed * (1.0 + i as f64) + 0.7 * j as f64).sin(), (seed * (2.0 + j as f64) - 0.3 * i as f64).cos());
        let hermitian = |seed: f64, size: f64| Mat::from_fn(n, n, |i, j| (entry(seed, i, j) + entry(seed, j, i).conj()) * (0.5 * size));
        let mut base = hermitian(1.3, 40.0);
        for (i, shift) in [0.0, 5.0, -600.0, -620.0].into_iter().enumerate() {
            base[(i, i)] += C64::new(shift, if i >= 2 { -19.0 } else { 0.0 });
        }
        let start: Vec<C64> = (0..n).map(|i| C64::cis(0.9 * i as f64) * 0.5).collect();
        let mut found = Vec::new();
        for scale in [0.0, 1.0, 0.5] {
            let (a, b) = (hermitian(2.1, 2.0 * scale), hermitian(0.4, 20.0 * scale));
            let moving = scale > 0.0;
            let mut picture = pictured(&base, &a, &b, half, moving);
            assert!(picture.decomposed() && picture.moving() == moving);
            assert!(!moving || picture.fast.iter().enumerate().all(|(k, fast)| *fast || k / n == k % n), "a slow pair");
            picture.launch(&start);
            let error = errors(&mut picture, &base, &a, &b, half, &start, &[half, 0.37 * half, -0.6 * half], 400_000);
            eprintln!("slopes × {scale}: the state's error {error:e}, the following's size {:e}, {} hops", picture.following(), picture.hops);
            found.push((error, picture.following()));
        }
        assert!(found[0].0 < 1e-11, "{found:?}");
        assert!(found[1].0 < found[1].1 * found[1].1 && found[2].0 < 0.4 * found[1].0, "{found:?}");
    }

    /// Slow pairs are hopped (module documentation): Ĥ₀ of light shifts that turn less than a radian over a half step,
    /// with a small decay, and slopes that change it by a tenth over the step, on three states -- no pair fast. The
    /// state at the end and between the nodes is the flow within the fifth order of the step -- the hop's second Magnus
    /// term, (h⁵/60)‖[A, [Λ, A]]‖ ≈ (|ω|h)(h²‖A‖)²/60 = 0.4 × 0.064²/60 = 3·10⁻⁵ at the half 0.08 --: at half the step
    /// it errs twenty-four times less and more.
    #[test]
    fn a_slow_pair_s_hops_are_of_the_fourth_order() {
        let n = 3;
        let entry = |seed: f64, i: usize, j: usize| C64::new((seed * (1.0 + i as f64) + 0.7 * j as f64).sin(), (seed * (2.0 + j as f64) - 0.3 * i as f64).cos());
        let hermitian = |seed: f64, size: f64| Mat::from_fn(n, n, |i, j| (entry(seed, i, j) + entry(seed, j, i).conj()) * (0.5 * size));
        let mut base = hermitian(1.3, 2.0);
        for i in 0..n {
            base[(i, i)] += C64::new(0.0, -0.05 * i as f64);
        }
        let (a, b) = (hermitian(2.1, 4.0), hermitian(0.4, 20.0));
        let start: Vec<C64> = (0..n).map(|i| C64::cis(0.9 * i as f64) * 0.5).collect();
        let mut found = Vec::new();
        for half in [0.08, 0.04] {
            let mut picture = pictured(&base, &a, &b, half, true);
            assert!(picture.decomposed() && !picture.followed, "a fast pair at the half {half}");
            picture.launch(&start);
            let error = errors(&mut picture, &base, &a, &b, half, &start, &[half, 0.0, -0.41 * half, 0.73 * half], 100_000);
            eprintln!("half {half}: the state's error {error:e}, {} hops", picture.hops);
            found.push(error);
        }
        assert!(found[0] < 1e-4 && found[1] < found[0] / 24.0, "{found:?}");
    }
}
