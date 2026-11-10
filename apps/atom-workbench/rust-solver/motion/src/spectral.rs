//! A generator along a step's path in the interaction picture of its value at the step's middle (SPEC §2.1; step.rs,
//! # The kick): what the step's one propagation is made of.
//!
//! # The generator
//! Along the path Ĥ(u), u the time from the step's middle, is Ĥ₀ + D(u): Ĥ₀ = Ĥ(0) holds every frequency of the atom's
//! own -- detunings, light shifts, decay rates, of any size against the step --, and D(u) = Ĥ(u) − Ĥ₀ what the motion
//! along the path and the lights' beats change, slow against the step by the step's own choice. D is taken through the
//! step's three nodes, the middle and the two ends ±h: D(u) = uA + u²B, A = (Ĥ(h) − Ĥ(−h))/2h and
//! B = (Ĥ(h) + Ĥ(−h) − 2Ĥ₀)/2h². Its error is the cubic's, (u³ − uh²)D‴/6: third order in the slow rate at a time of
//! the step, whatever Ĥ₀ is, and odd about the middle, so that the step's integrals take it at the fourth (# A hop,
//! below); a half step takes its own three nodes on half the spacing.
//!
//! # A hop
//! With Ĥ₀ = c + RΛR⁻¹ (its scalar part c apart, `linalg::eigen`) and x = R d, i ḋ = (c + Λ + Dᵉ(u)) d, Dᵉ = R⁻¹DR. Over
//! a hop from u to u + w, d(u + t) = e^{−i(c + Λ)t} y(t) and i ẏ = D̃(t) y with D̃_ab(t) = e^{iω_ab t} Dᵉ_ab(u + t),
//! ω_ab = λ_a − λ_b: the interaction picture, in which Ĥ₀ is gone -- its phases and decays are factors, exact over any
//! width. The hop takes the first Magnus term, y(w) = e^{−iJ} y(0) with J = ∫₀ʷ D̃ dt, whose integral over the
//! oscillation is exact: element by element
//!   J_ab = Σ_j (iω_ab)^j/j! · w^{j+1} [A_ab (u/(j+1) + w/(j+2)) + B_ab (u²/(j+1) + 2uw/(j+2) + w²/(j+3))],
//! the series of ∫₀ʷ t^p e^{iωt} dt, summed until its terms are within the round-off of the state the exponent acts on
//! or of the exponent's own first term, whichever is the larger (`Picture::exponent`).
//! The hops are the caller's, between the nodes of a quadrature that resolves Ĥ₀'s frequencies (step.rs `propagate`).
//! Nodes that resolve a frequency lie within a turn of it of each other (a Gauss–Legendre rule of m nodes is exact to
//! the degree 2m − 1, which e^{iνs} over a panel needs beyond νw/2: νw < 4m, and its widest gap is π/(2m + 1) of the
//! panel's width), so |ω|w < 2π there -- 3.5 rad at the 64 nodes the rules end with: the series is some thirty terms
//! at its longest and loses no more than e^{|ω|w} of the round-off to its own cancellation. A hop asked wider than
//! that turn is taken in the fewest equal parts within it. The series' powers (iω_ab)^j/j! on A and on B are made once
//! a step, as far as its hops ask.
//!
//! Where nothing moves (D = 0: an atom at rest, a generator still along every path) a hop is e^{−i(c + Λ)w} alone:
//! exact, at any detuning times step.
//!
//! What a hop leaves out is the second Magnus term, −½∫∫[D̃(t₁), D̃(t₂)] = −(w²/2) Σ_l [m_{l+1}, m_l] with m_l the
//! Legendre moments of D̃ over the hop, (1/w)∫₀ʷ P_l(2t/w − 1) D̃(t) dt: second order in D. Its order in the step
//! depends on what the hop resolves:
//! - a hop that resolves Ĥ₀'s frequencies (‖Ĥ₀‖w ≪ 1: a folded cluster's light shifts) leaves w³u²[A, B]/12, u the
//!   hop's middle -- over the hop D is its value there, u A + u²B, and its slope A + 2uB, which commute but for that --
//!   and terms of w⁵: the fifth order in the step, so the propagation is of the fourth order there, like the three
//!   nodes of D (Simpson's rule's: their cubic is odd about the middle; what stays is τ⁵D⁗/2880 a step, and a first
//!   moment -- the lag a friction lives in -- short by (ω τ)²/60 of itself for a beat of the rate ω);
//! - a hop that does not (an explicit far-detuned state: the hops follow the detuning, not the step) leaves the light
//!   shift of the change itself, bounded by ½(∫‖D‖)² whatever ‖Ĥ₀‖τ is -- the second order in the step, and no
//!   growth with the detuning.
//! (step/tests.rs, `a_step_is_of_the_fourth_order_in_the_path_s_change_where_its_hops_resolve_the_generator`.)
//!
//! # Towards a defective generator
//! A state taken through the eigenvectors rounds by κ(R) u (1 + |τ| × the decay rates' spread) relative to its norm
//! (`linalg::Eigen`), and κ(R) grows without bound towards an exceptional point. There the hops are taken without the
//! decomposition (`Picture::basis` None): CF4 (Blanes & Moan 2006) of the same Ĥ₀ + uA + u²B over each hop, two
//! exponentials on the vector to round-off (`linalg::exp_apply`), whose rounding does not know κ. It is exact where
//! nothing moves, like the eigenbasis's hop; with D it is of fourth order in the hop's width, which is the caller's
//! nodes' spacing -- a few radians of Ĥ₀'s bound at most --, so its error is a share of D's own effect, not an error of
//! Ĥ₀'s frequencies. The rule is the rounding's, not a tuned number: the decomposition where a state through it rounds no
//! more than through the exponential of Ĥ₀ over the step (`linalg::Chebyshev::rounding`), the exponential otherwise.

use crate::linalg::{self, C64, Mat};
use crate::units::UNIT_ROUNDOFF;

/// A generator along a step's path (module documentation): filled by the caller with Ĥ at the middle (`base`) and at
/// the two ends (`ends`: at −h, at +h), then prepared (`prepare`); kept by a trajectory from step to step, its memory
/// made once.
pub struct Picture {
    n: usize,
    /// Ĥ₀ (n × n; rad/µs), the caller's: kept as it is given.
    pub base: Mat,
    /// Ĥ(−h) and Ĥ(+h), the caller's; after `prepare` the slopes A and B, in the eigenbasis where there is one.
    pub ends: [Mat; 2],
    /// A hop's exponent.
    matrix: Mat,
    /// Ĥ₀'s decomposition of its traceless part, where the hops go through it.
    basis: Option<linalg::Eigen>,
    /// Ĥ₀'s eigenvalues, the scalar part with them.
    lambda: Vec<C64>,
    /// The slopes' powers (iω_ab)^j/j! A_ab and (iω_ab)^j/j! B_ab, j = 0…`terms` − 1 (2n² numbers each).
    powers: Vec<C64>,
    terms: usize,
    /// The exponential's work vectors (2n) and the similarity's (n²).
    vectors: Vec<C64>,
    product: Vec<C64>,
    /// Whether D is anything at all.
    moving: bool,
    /// max|λ_a − λ_b| (rad/µs): what a hop's series is summed against.
    spread: f64,
    /// The bounds of Ĥ₀ less its scalar part (`linalg::bounds`), and the slopes' Frobenius norms as the hops take them
    /// (in the eigenbasis where there is one): what bounds a hop's exponent, and what a hop is priced by.
    bounds: [f64; 2],
    slopes: [f64; 2],
    /// max|λ_a* − λ_b| (rad/µs): the bandwidth of the forms ⟨x|·|x⟩ on a state propagated by Ĥ₀ -- their frequencies and
    /// decay rates; without the decomposition its bound 2(ρ + γ).
    pub bandwidth: f64,
    /// The rounding of a state taken through the decomposition, relative to its norm (none without it: the
    /// exponentials' are the hops').
    pub rounding: f64,
    /// The complex multiply-adds counted so far (`decide::Work` counts them four real ones each).
    pub work: f64,
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
            powers: Vec::new(),
            terms: 0,
            vectors: Vec::new(),
            product: Vec::new(),
            moving: false,
            spread: 0.0,
            bounds: [0.0; 2],
            slopes: [0.0; 2],
            bandwidth: 0.0,
            rounding: 0.0,
            work: 0.0,
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

impl Picture {
    /// The picture for n states.
    pub fn fit(&mut self, n: usize) {
        if self.n != n {
            self.n = n;
            for m in [&mut self.base, &mut self.matrix].into_iter().chain(&mut self.ends) {
                (m.rows, m.cols) = (n, n);
                m.data.resize(n * n, C64::ZERO);
            }
            self.lambda.resize(n, C64::ZERO);
            self.vectors.resize(2 * n, C64::ZERO);
            self.product.resize(n * n, C64::ZERO);
        }
    }

    /// The states.
    pub fn states(&self) -> usize {
        self.n
    }

    /// Whether the hops go through Ĥ₀'s decomposition.
    pub fn decomposed(&self) -> bool {
        self.basis.is_some()
    }

    /// Whether the generator changes along the path at all.
    pub fn moving(&self) -> bool {
        self.moving
    }

    /// Prepares the picture of a step of 2h from `base` and, unless the generator is `still` along the path (then they
    /// are not read), `ends`: the slopes, Ĥ₀'s decomposition and the route of the hops (module documentation, # Towards
    /// a defective generator), the slopes in the eigenbasis. `work` begins anew with what this counted.
    pub fn prepare(&mut self, half: f64, still: bool) {
        let n = self.n;
        (self.work, self.terms, self.moving) = (0.0, 0, false);
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
                let pairs = |of: fn(C64, C64) -> f64| self.lambda.iter().flat_map(|a| self.lambda.iter().map(move |b| of(*a, *b))).fold(0.0, f64::max);
                (self.bandwidth, self.spread, self.rounding) = (pairs(|a, b| (a.conj() - b).abs()), pairs(|a, b| (a - b).abs()), rounding);
                self.work += e.work;
                if self.moving {
                    // A ← R⁻¹AR and B ← R⁻¹BR: two products of n³ each.
                    for slope in &mut self.ends {
                        for (row, out) in self.product.chunks_mut(n).enumerate() {
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
            None => (self.bandwidth, self.spread, self.rounding) = (2.0 * (rho + gamma), 0.0, 0.0),
        }
        self.slopes = if self.moving { [0, 1].map(|i| self.ends[i].data.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt()) } else { [0.0; 2] };
    }

    /// The bound of a hop's exponent from the time u over the width w: ∫‖D‖ at the slopes' norms, D(t) = tA + t²B at the
    /// farther end of the hop from the middle.
    fn moved(&self, u: f64, w: f64) -> f64 {
        let far = u.abs().max((u + w).abs());
        (far * self.slopes[0] + far * far * self.slopes[1]) * w.abs()
    }

    /// A frame's state x in the picture's coordinates: R⁻¹x through the decomposition, x itself without it.
    pub fn enter(&mut self, x: &[C64], state: &mut [C64]) {
        let n = self.n;
        match &self.basis {
            Some(e) => {
                for (a, d) in state.iter_mut().enumerate() {
                    *d = e.inverse.data[a * n..(a + 1) * n].iter().zip(x).map(|(r, x)| *r * *x).sum();
                }
                self.work += (n * n) as f64;
            }
            None => state.copy_from_slice(x),
        }
    }

    /// The frame's state of the picture's coordinates: x = R d.
    pub fn leave(&mut self, state: &[C64], x: &mut [C64]) {
        let n = self.n;
        match &self.basis {
            Some(e) => {
                for (a, x) in x.iter_mut().enumerate() {
                    *x = e.vectors.data[a * n..(a + 1) * n].iter().zip(state).map(|(r, d)| *r * *d).sum();
                }
                self.work += (n * n) as f64;
            }
            None => x.copy_from_slice(state),
        }
    }

    /// A hop's exponent J into `matrix` (module documentation, # A hop): from the time u over the width w, the series
    /// against the largest |ω|w and the exponent's bound, the slopes' powers made as far as it goes.
    fn exponent(&mut self, u: f64, w: f64) {
        let n = self.n;
        let (terms, size) = (series(self.spread * w.abs(), self.moved(u, w)), 2 * n * n);
        while self.terms < terms {
            let j = self.terms;
            self.powers.resize((j + 1) * size, C64::ZERO);
            let (before, power) = self.powers.split_at_mut(j * size);
            if j == 0 {
                power[..n * n].copy_from_slice(&self.ends[0].data);
                power[n * n..].copy_from_slice(&self.ends[1].data);
            } else {
                let last = &before[(j - 1) * size..];
                for a in 0..n {
                    for b in 0..n {
                        let factor = C64::I * (self.lambda[a] - self.lambda[b]) * (1.0 / j as f64);
                        power[a * n + b] = last[a * n + b] * factor;
                        power[n * n + a * n + b] = last[n * n + a * n + b] * factor;
                    }
                }
            }
            self.terms += 1;
            self.work += size as f64;
        }
        self.matrix.data.fill(C64::ZERO);
        let mut power = w;
        for (j, slopes) in self.powers.chunks(size).take(terms).enumerate() {
            let (one, two, three) = (1.0 / (j + 1) as f64, 1.0 / (j + 2) as f64, 1.0 / (j + 3) as f64);
            let (alpha, beta) = (power * (u * one + w * two), power * (u * u * one + 2.0 * u * w * two + w * w * three));
            let (a, b) = slopes.split_at(n * n);
            for ((z, a), b) in self.matrix.data.iter_mut().zip(a).zip(b) {
                *z += *a * alpha + *b * beta;
            }
            power *= w;
        }
        self.work += (terms * n * n) as f64;
    }

    /// One hop of the picture's coordinates from the time u (from the middle) over the width w, either sign (module
    /// documentation, # A hop; without the decomposition CF4 of the same generator, # Towards a defective generator).
    /// Returns the rounding of its exponentials relative to the state's norm (`linalg::Chebyshev::rounding`).
    pub fn hop(&mut self, state: &mut [C64], u: f64, w: f64) -> f64 {
        let n = self.n;
        let mut bound = 0.0;
        if self.basis.is_some() {
            // Where nothing moves the hop is its phases at any width; else in the fewest equal parts within the series' reach.
            let parts = if self.moving { ((self.spread * w.abs() / std::f64::consts::TAU).ceil() as usize).max(1) } else { 1 };
            let part = w / parts as f64;
            for k in 0..parts {
                if self.moving {
                    self.exponent(u + k as f64 * part, part);
                    let plan = linalg::exp_apply(&mut self.matrix, C64::new(0.0, -1.0), state, &mut self.vectors);
                    bound += plan.rounding(n);
                    self.work += (plan.parts * plan.terms * n * n) as f64;
                }
                for (z, l) in state.iter_mut().zip(&self.lambda) {
                    *z *= (C64::new(0.0, -part) * *l).exp();
                }
                self.work += n as f64;
            }
        } else {
            // CF4 over the hop: Ĥ at its two Gauss nodes c = ½ ∓ √3/6, the weights ¼ ± √3/6.
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
        }
        bound
    }

    /// The counted complex multiply-adds of one hop of the width w near the step's end at h, priced without taking it:
    /// its exponent's series and its exponential's plan at the slopes' bound of D there (through the decomposition), or
    /// CF4's two exponentials at Ĥ₀'s bounds (without it).
    pub fn price(&self, half: f64, w: f64) -> f64 {
        let (n, w, h) = (self.n, w.abs(), half.abs());
        let moved = self.moved(h - w, w);
        if self.basis.is_some() {
            let exponent = if self.moving {
                let plan = linalg::Chebyshev::plan(moved, moved);
                ((series(self.spread * w, moved) + plan.parts * plan.terms) * n * n) as f64
            } else {
                0.0
            };
            exponent + n as f64
        } else {
            let plan = linalg::Chebyshev::plan(0.5 * self.bounds[0] * w + moved, 0.5 * self.bounds[1] * w + moved);
            (2 * (3 + plan.parts * plan.terms) * n * n) as f64
        }
    }

    /// The counted complex multiply-adds of the slopes' powers a hop of the width w asks for: made once a step.
    pub fn powers(&self, w: f64) -> f64 {
        if self.basis.is_some() && self.moving { (series(self.spread * w.abs(), 1.0) * 2 * self.n * self.n) as f64 } else { 0.0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dense RK4 of iẋ = (Ĥ₀ + uA + u²B)x from −h to h in `steps` steps: the tests' reference of a picture's hops.
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

    /// Towards a defective generator the hops leave the decomposition (module documentation): a resonant two-level atom
    /// whose coupling is a quarter of its decay rate, [[0, c], [c, −iΓ/2]] at c = Γ/4, has one eigenvector. At that
    /// point and a millionth of the coupling off it the picture takes the exponential's hops; a hundredth off it, where
    /// the eigenvectors' condition is some hundreds, it takes the decomposition's. Each way a hop is the generator's flow:
    /// to the reference's own accuracy where nothing moves, and with slopes within the second order of what they move
    /// over a hop.
    #[test]
    fn a_defective_generator_s_hops_are_the_exponential_s() {
        let (gamma, half, hops) = (38.117, 0.03, 16);
        let start = [C64::ONE, C64::new(0.2, -0.1)];
        for (off, decomposed) in [(0.0, false), (1e-6, false), (1e-2, true)] {
            let c = 0.25 * gamma * (1.0 + off);
            let base = Mat::from_fn(2, 2, |i, j| if i != j { C64::real(c) } else if i == 1 { C64::new(0.0, -0.5 * gamma) } else { C64::ZERO });
            for moving in [false, true] {
                let slope = |size: f64| Mat::from_fn(2, 2, |i, j| if i != j && moving { C64::new(size, 0.3 * size * (i as f64 - j as f64)) } else { C64::ZERO });
                let (a, b) = (slope(2.0), slope(30.0));
                let exact = flow(&base, &a, &b, half, &start, 100_000);
                let mut picture = Picture::default();
                picture.fit(2);
                picture.base = base.clone();
                picture.ends = [-half, half].map(|u| Mat::from_fn(2, 2, |i, j| base[(i, j)] + a[(i, j)] * u + b[(i, j)] * (u * u)));
                picture.prepare(half, !moving);
                assert_eq!(picture.decomposed(), decomposed, "{off} off the exceptional point");
                let (mut state, mut x) = ([C64::ZERO; 2], [C64::ZERO; 2]);
                picture.enter(&start, &mut state);
                let w = 2.0 * half / hops as f64;
                for k in 0..hops {
                    picture.hop(&mut state, -half + k as f64 * w, w);
                }
                picture.leave(&state, &mut x);
                let error = x.iter().zip(&exact).map(|(x, e)| (*x - *e).abs()).fold(0.0, f64::max);
                // What the slopes move over a hop at the step's end: (h‖A‖ + h²‖B‖) w, its square the second order.
                let moved = (half * 2.0 + half * half * 30.0) * 2.0_f64.sqrt() * w;
                eprintln!("{off} off the exceptional point, moving {moving}: the state's error {error:e} (the slopes move {moved:e} a hop)");
                assert!(error < if moving { moved * moved } else { 1e-12 }, "{off} off the exceptional point, moving {moving}: {error:e}");
            }
        }
    }

    /// A hop is the generator's flow: Ĥ₀ of detunings up to 600 rad/µs with a decay, slopes that turn the couplings by a
    /// tenth of a radian over the step, on four states -- one hop over the step (60 rad of the detuning: taken in its
    /// parts of a turn), and 320 hops, against RK4 at 4·10⁵ steps of the same Ĥ₀ + uA + u²B. Without slopes a hop
    /// is exact; with them its error is the second Magnus term's, and it falls with the hops.
    #[test]
    fn a_hop_is_the_generator_s_flow() {
        let (n, half) = (4, 0.05);
        let entry = |seed: f64, i: usize, j: usize| C64::new((seed * (1.0 + i as f64) + 0.7 * j as f64).sin(), (seed * (2.0 + j as f64) - 0.3 * i as f64).cos());
        let hermitian = |seed: f64, size: f64| Mat::from_fn(n, n, |i, j| (entry(seed, i, j) + entry(seed, j, i).conj()) * (0.5 * size));
        let mut base = hermitian(1.3, 40.0);
        for (i, shift) in [0.0, 5.0, -600.0, -620.0].into_iter().enumerate() {
            base[(i, i)] += C64::new(shift, if i >= 2 { -19.0 } else { 0.0 });
        }
        let (a, b) = (hermitian(2.1, 2.0), hermitian(0.4, 20.0));
        for moving in [false, true] {
            let zero = Mat::zeros(n, n);
            let (a, b) = if moving { (&a, &b) } else { (&zero, &zero) };
            let at = |u: f64| Mat::from_fn(n, n, |i, j| base[(i, j)] + a[(i, j)] * u + b[(i, j)] * (u * u));
            let start: Vec<C64> = (0..n).map(|i| C64::cis(0.9 * i as f64) * 0.5).collect();
            let exact = flow(&base, a, b, half, &start, 400_000);
            let mut errors = Vec::new();
            for hops in [1, 320] {
                let mut picture = Picture::default();
                picture.fit(n);
                picture.base = base.clone();
                picture.ends = [at(-half), at(half)];
                picture.prepare(half, !moving);
                assert!(picture.decomposed() && picture.moving() == moving);
                let (mut state, mut x) = (vec![C64::ZERO; n], vec![C64::ZERO; n]);
                picture.enter(&start, &mut state);
                let w = 2.0 * half / hops as f64;
                for k in 0..hops {
                    picture.hop(&mut state, -half + k as f64 * w, w);
                }
                picture.leave(&state, &mut x);
                errors.push(x.iter().zip(&exact).map(|(x, e)| (*x - *e).abs()).fold(0.0, f64::max));
            }
            eprintln!("moving {moving}: the state's error over one hop {:e}, over 320 {:e}", errors[0], errors[1]);
            if moving {
                assert!(errors[0] < 1e-4 && errors[1] < 0.05 * errors[0], "{errors:?}");
            } else {
                assert!(errors[0] < 1e-11 && errors[1] < 1e-11, "{errors:?}");
            }
        }
    }
}
