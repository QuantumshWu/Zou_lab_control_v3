//! The special functions more than one module takes, each written once: the incomplete gamma function's two expansions
//! (Abramowitz & Stegun 6.5.29 and 6.5.31; Numerical Recipes §6.2, gser and gcf), each summed to the format's round-off,
//! for the normal's tail (estimate.rs) and the lowered isothermal's momenta (ensemble.rs); and Gauss–Legendre's nodes and
//! weights (`gauss_legendre`, O(n) to the round-off), for the ensemble's integrals over the trap (ensemble.rs), the
//! level's share (ensemble.rs `Ensemble::beyond`) and the step's quadrature of an entire integrand (step.rs).
//!
//! With P(s, x) = γ(s, x)/Γ(s) the regularized lower function and Q = 1 − P the upper one, for x > 0:
//! - the lower one's series, e^x x^{−s} Γ(s + 1) P(s, x) = 1 + Σ_{n ≥ 1} x^n/((s + 1)…(s + n)) -- `series` is the sum from
//!   n = 1, so that what a caller takes from P's leading power x^s e^{−x}/Γ(s + 1) it takes exactly (ensemble.rs `kept`);
//! - the upper one's continued fraction, e^x x^{−s} Γ(s) Q(s, x) = 1/(x + 1 − s − 1(1 − s)/(x + 3 − s − 2(2 − s)/(x + 5 − s − …)))
//!   (`fraction`), which ends after s steps at a whole s.
//! Both hold at every x > 0. The series' terms fall from the first on only below x = s + 1, and the fraction converges the
//! faster the further x lies above it: the series is the one to take below s + 1, the fraction from there on (`by_series`).
use crate::units::UNIT_ROUNDOFF;
use std::f64::consts::PI;

/// Γ(s) at a positive multiple of ½, each one exact in the format: Γ(1) = 1 or Γ(½) = √π, then Γ(t + 1) = t Γ(t) up to s.
pub fn gamma(s: f64) -> f64 {
    let (mut g, mut t) = if (2. * s) as i64 % 2 == 0 { (1., 1.) } else { (PI.sqrt(), 0.5) };
    while t < s {
        g *= t;
        t += 1.;
    }
    g
}

/// Whether the lower function's series is the expansion to take at x (module documentation): below s + 1.
pub fn by_series(s: f64, x: f64) -> bool {
    x < s + 1.
}

/// Σ_{n ≥ 1} x^n/((s + 1)…(s + n)) for x ≥ 0 (module documentation): positive terms, summed until one no longer changes
/// the sum.
pub fn series(s: f64, x: f64) -> f64 {
    let (mut term, mut sum, mut a) = (1., 0., s);
    loop {
        a += 1.;
        term *= x / a;
        sum += term;
        if term <= sum * UNIT_ROUNDOFF {
            break sum;
        }
    }
}

/// The upper function's continued fraction at x > 0 (module documentation) by the modified Lentz method: the convergents'
/// ratios multiplied up until one is 1 to the format's spacing, a vanishing denominator replaced by the least number whose
/// reciprocal the format holds.
fn fraction(s: f64, x: f64) -> f64 {
    let tiny = f64::MIN_POSITIVE / f64::EPSILON;
    let (mut b, mut c) = (x + 1. - s, 1. / tiny);
    let mut d = 1. / b;
    let mut h = d;
    let mut i = 1.;
    loop {
        let an = -i * (i - s);
        b += 2.;
        d = an * d + b;
        if d.abs() < tiny {
            d = tiny;
        }
        c = b + an / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1. / d;
        let step = d * c;
        h *= step;
        if (step - 1.).abs() <= f64::EPSILON {
            break h;
        }
        i += 1.;
    }
}

/// Q(s, x), the regularized upper incomplete gamma function, for x ≥ 0 and s a positive multiple of ½ (`gamma`): one less
/// the lower one by its series below s + 1, by its continued fraction from there on.
pub fn upper(s: f64, x: f64) -> f64 {
    if !(x > 0.) {
        return 1.;
    }
    let power = (s * x.ln() - x).exp();
    if by_series(s, x) {
        1. - power / gamma(s + 1.) * (1. + series(s, x))
    } else {
        power / gamma(s) * fraction(s, x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::units::rounding;

    /// The most terms the series sums at x: its terms' ratio x/(s + n) is at most ½ from n = 2x − s on and no term exceeds
    /// the sum, so 53 terms further one is below the sum's round-off. A term costs three operations (the ratio, the
    /// product, the sum): the sum's rounding is within γ of three times its terms (`units::rounding`).
    fn terms(s: f64, x: f64) -> usize {
        crate::linalg::larger((2. * x - s).ceil(), 0.) as usize + 54
    }
    /// The round-offs that x^s e^{−x} = e^{s ln x − x} carries from its exponent: the logarithm's and the product's on
    /// s ln x, the difference's on the exponent, each a round-off of the result per unit of its size, and the exponential's
    /// own.
    fn power(s: f64, x: f64) -> usize {
        (2. * (s * x.ln()).abs() + (s * x.ln() - x).abs()).ceil() as usize + 1
    }
    /// The rounding of one less the lower function by its series, absolute since P ≤ 1: the power's, the series', Γ(s + 1)'s
    /// (a root and three products at most at the orders here), and the four operations that join them and take the result
    /// from one.
    fn lower(s: f64, x: f64) -> f64 {
        rounding(power(s, x) + 3 * terms(s, x) + 4 + 4)
    }

    /// Γ at the half-integers by its recurrence from Γ(1) = 1 and Γ(½) = √π, product for product: Γ(5) = 24,
    /// Γ(7/2) = (5/2)(3/2)(1/2)√π.
    #[test]
    fn gamma_follows_its_recurrence() {
        assert_eq!((gamma(1.), gamma(2.), gamma(5.)), (1., 1., 24.));
        assert_eq!((gamma(0.5), gamma(3.5)), (PI.sqrt(), PI.sqrt() * 0.5 * 1.5 * 2.5));
    }

    /// The series against its closed form at s = 1, Σ_{n ≥ 1} x^n/(n + 1)! = (e^x − 1 − x)/x, below s + 1 and above (it
    /// holds at every x): within its own rounding (`terms`) and the closed form's, three operations whose difference
    /// e^x − 1 − x loses e^x/(e^x − 1 − x) of the format's digits.
    #[test]
    fn the_series_is_its_closed_form() {
        for x in [1e-3_f64, 0.3, 1., 1.9, 2.1, 5., 30.] {
            let exact = (x.exp_m1() - x) / x;
            let within = rounding(3 * terms(1., x)) + rounding(3) * x.exp() / (x.exp_m1() - x);
            assert!((series(1., x) / exact - 1.).abs() <= within, "x {x}: {} against {exact} within {within:e}", series(1., x));
        }
    }

    /// Q at the whole orders against its closed forms, Q(1, x) = e^{−x} and Q(2, x) = e^{−x}(1 + x) (three operations), on
    /// both sides of s + 1 and next to it, and Q(s, 0) = 1. Below s + 1 within the lower function's rounding (`lower`).
    /// Above, the fraction ends after s steps and nothing in it cancels, so the rounding is relative to the result: the
    /// power's, the fraction's -- its first denominator x + 1 − s (two operations, the second's result no less than half its
    /// operand) and its reciprocal, then ten operations a step --, and the two that join them.
    #[test]
    fn the_upper_function_is_its_closed_forms_at_whole_orders() {
        for (s, closed) in [(1_f64, (|x: f64| (-x).exp()) as fn(f64) -> f64), (2., |x: f64| (-x).exp() * (1. + x))] {
            for x in [1e-6_f64, 0.25, 1., s + 0.99, s + 1.01, 7., 40., 700.] {
                let (value, exact) = (upper(s, x), closed(x));
                let within = rounding(3) * exact + if by_series(s, x) { lower(s, x) } else { rounding(power(s, x) + 4 + 10 * s as usize + 2) * exact };
                assert!((value - exact).abs() <= within, "Q({s}, {x}) = {value:e} against {exact:e} within {within:e}");
            }
            assert_eq!(upper(s, 0.), 1.);
        }
    }

    /// Q at the half orders. Q(½, z²/2) is the normal's two-sided tail beyond z: 0.31731050786291415 at z = 1, where the
    /// series is taken, within the lower function's rounding. From s + 1 on, the fraction against the series, which holds
    /// there too: one less the lower function by its series, within that one's rounding (the fraction's own, relative to a
    /// Q below one, is asserted to lie within it).
    #[test]
    fn the_upper_function_s_two_expansions_agree_at_half_orders() {
        assert!((upper(0.5, 0.5) - 0.31731050786291415).abs() <= lower(0.5, 0.5), "{}", upper(0.5, 0.5));
        for s in [0.5_f64, 1.5, 2.5, 3.5] {
            for x in [s + 1., s + 1.5, s + 4., 20.] {
                let by_its_series = 1. - (s * x.ln() - x).exp() / gamma(s + 1.) * (1. + series(s, x));
                assert!(!by_series(s, x) && (upper(s, x) - by_its_series).abs() <= lower(s, x), "Q({s}, {x}) = {:e} against {by_its_series:e}", upper(s, x));
            }
        }
    }

    /// The rule before 2026-11-20, in O(n²): Newton's iteration in x on P_n by the three-term recurrence from Tricomi's
    /// start cos(π(i − ¼)/(n + ½)), i = 1…n, until the step stops shrinking; the weight 2/((1 − x²)P_n′(x)²). The
    /// reference `gauss_legendre` is read against.
    fn by_recurrence(n: usize) -> Vec<(f64, f64)> {
        (1..=n)
            .map(|i| {
                let mut x = (PI * (i as f64 - 0.25) / (n as f64 + 0.5)).cos();
                let mut last = f64::INFINITY;
                let slope = loop {
                    let (p, slope) = legendre(n, x);
                    let dx = p / slope;
                    if !(dx.abs() < last) {
                        break slope;
                    }
                    last = dx.abs();
                    x -= dx;
                };
                (x, 2. / ((1. - x * x) * slope * slope))
            })
            .collect()
    }

    /// P_n(x) and P_n′(x) by the three-term recurrence.
    fn legendre(n: usize, x: f64) -> (f64, f64) {
        let (mut p0, mut p1) = (1., x);
        for k in 2..=n {
            let p2 = ((2 * k - 1) as f64 * x * p1 - (k - 1) as f64 * p0) / k as f64;
            p0 = p1;
            p1 = p2;
        }
        (p1, n as f64 * (x * p1 - p0) / (x * x - 1.))
    }


    /// A number to twice the format's digits, hi + lo with |lo| ≤ ½ulp(hi) (Dekker 1971; Bailey's double-double): the
    /// error-free sum and product (the product's error by the fused multiply-add) and the operations built on them, each to
    /// some 2⁻¹⁰⁴ relative -- the reference rule's arithmetic (`by_twice`).
    #[derive(Clone, Copy, Debug)]
    struct Twice(f64, f64);

    impl Twice {
        fn sum(a: f64, b: f64) -> Twice {
            let s = a + b;
            let v = s - a;
            Twice(s, (a - (s - v)) + (b - v))
        }
        fn quick(a: f64, b: f64) -> Twice {
            let s = a + b;
            Twice(s, b - (s - a))
        }
        fn add(self, b: Twice) -> Twice {
            let Twice(s, e) = Twice::sum(self.0, b.0);
            Twice::quick(s, e + self.1 + b.1)
        }
        fn neg(self) -> Twice {
            Twice(-self.0, -self.1)
        }
        fn mul(self, b: Twice) -> Twice {
            let p = self.0 * b.0;
            let e = self.0.mul_add(b.0, -p) + (self.0 * b.1 + self.1 * b.0);
            Twice::quick(p, e)
        }
        fn div(self, b: Twice) -> Twice {
            let q1 = self.0 / b.0;
            let r = self.add(b.mul(Twice(q1, 0.0)).neg());
            let q2 = r.0 / b.0;
            let r = r.add(b.mul(Twice(q2, 0.0)).neg());
            Twice::quick(q1, q2).add(Twice(r.0 / b.0, 0.0))
        }
        fn of(x: f64) -> Twice {
            Twice(x, 0.0)
        }
    }

    /// The reference rule: each node of `gauss_legendre`'s taken on by Newton's iteration in twice the digits on P_n by the
    /// three-term recurrence (as `by_recurrence`, in `Twice`), three steps from a start within some round-offs of the root
    /// (quadratic: the third is past 2⁻¹⁰⁴), and its weight 2/((1 − x²)P_n′(x)²) there, in twice the digits -- the
    /// recurrence's and 1 − x²'s roundings 2⁻⁵³ smaller than the format's.
    fn by_twice(n: usize, start: &[(f64, f64)]) -> Vec<(Twice, Twice)> {
        let one = Twice::of(1.0);
        let value = |x: Twice| -> (Twice, Twice) {
            let (mut p0, mut p1) = (one, x);
            for k in 2..=n {
                let p2 = Twice::of((2 * k - 1) as f64).mul(x).mul(p1).add(Twice::of((k - 1) as f64).mul(p0).neg()).div(Twice::of(k as f64));
                (p0, p1) = (p1, p2);
            }
            let p = if n == 0 { one } else { p1 };
            (p, Twice::of(n as f64).mul(x.mul(p1).add(p0.neg())).div(x.mul(x).add(one.neg())))
        };
        start
            .iter()
            .map(|&(x0, _)| {
                let mut x = Twice::of(x0);
                for _ in 0..3 {
                    let (p, slope) = value(x);
                    x = x.add(p.div(slope).neg());
                }
                let (_, slope) = value(x);
                (x, Twice::of(2.0).div(one.add(x.mul(x).neg()).mul(slope).mul(slope)))
            })
            .collect()
    }

    /// `gauss_legendre` against the rule in twice the digits (`by_twice`), node by node, and against the rule it replaced
    /// (`by_recurrence`), at every n up to 64 and at 99, 100, 101, 128, 200, 256, 300, 400 and 500 -- the recurrence's
    /// nodes alone below n ≈ 8, the Stieltjes expansion's from the first few on at each end above. Within its rounding:
    /// - a node x = sin(ψ − δ) within 8 round-offs of the true one: ψ = π(n + 1 − 2k)/(2n + 1)'s three roundings and the
    ///   difference's, at ψ ≤ π/2 (some 6.3), the sine's, and Newton's floor, the step it stops at, below θ's round-off;
    /// - an expansion node's weight within 24 of itself: its factor's five (π, (n + ½)², the quotient, e^{−2s} and the
    ///   series' cut), sin θ's one and θ's two through it, twice the slope's seven (its leading term's four, the phase's
    ///   two, the sum's) and the product's and the quotient's;
    /// - a recurrence node's weight within 6n + 12: the recurrence's six roundings a step over n steps, a bound (they are
    ///   some √n round-offs measured: R1-PROP), and the weight's.
    /// The nodes the rule replaced lie within the same 8 and the reference's 2 of them; its weights carry 1 − x²'s
    /// cancellation near the ends and the recurrence's rounding in P_n′ (printed: hundreds to tens of thousands of
    /// round-offs from n ≈ 50 on, where the O(n) rule's are a few) -- the reason the rule was replaced as well as its n².
    #[test]
    fn gauss_legendre_is_its_rule_in_twice_the_digits() {
        let u = UNIT_ROUNDOFF;
        for n in (1..=64).chain([99, 100, 101, 128, 200, 256, 300, 400, 500]) {
            let (new, old) = (gauss_legendre(n), by_recurrence(n));
            let reference = by_twice(n, &new);
            let constant = stieltjes_constant(n);
            let (mut worst, mut before) = ([0.0_f64; 3], [0.0_f64; 2]);
            for (i, ((&(x, w), &(xo, wo)), &(rx, rw))) in new.iter().zip(&old).zip(&reference).enumerate() {
                let k = (i + 1).min(n - i);
                let start = PI * (4 * k - 1) as f64 / (4 * n + 2) as f64;
                let theta = rx.0.abs().acos();
                let expanded = constant.and_then(|c| stieltjes(n, k, theta, theta - start, c)).is_some();
                let (node, weight) = (Twice::of(x).add(rx.neg()).0.abs(), Twice::of(w).add(rw.neg()).0.abs() / rw.0);
                let within = if expanded { rounding(24) } else { rounding(6 * n + 12) };
                assert!(node <= rounding(8) && weight <= within, "n {n}, node {}: x {x} off by {:.1} u, w {w:e} by {:.1} u relative ({})", i + 1, node / u, weight / u, if expanded { "expansion" } else { "recurrence" });
                assert!((x - xo).abs() <= rounding(8) + rounding(2), "n {n}, node {}: {x} against the recurrence's {xo}", i + 1);
                worst = [crate::linalg::larger(worst[0], node / u), if expanded { crate::linalg::larger(worst[1], weight / u) } else { worst[1] }, if expanded { worst[2] } else { crate::linalg::larger(worst[2], weight / u) }];
                before = [crate::linalg::larger(before[0], Twice::of(xo).add(rx.neg()).0.abs() / u), crate::linalg::larger(before[1], Twice::of(wo).add(rw.neg()).0.abs() / (rw.0 * u))];
            }
            eprintln!("n {n}: node {:.2} u, weight {:.1} u (expansion), {:.1} u (recurrence); the rule before: node {:.2} u, weight {:.1} u", worst[0], worst[1], worst[2], before[0], before[1]);
        }
    }
}

/// Gauss–Legendre's nodes and weights on [−1, 1], from the largest node down, to the round-off in O(n) operations (Hale &
/// Townsend, SIAM J. Sci. Comput. 35 (2013) A652, the interior's construction): node k = 1…⌈n/2⌉ from x = 1 (the others
/// by the symmetry x → −x, exactly) at x = cos θ by Newton's iteration in θ from Tricomi's start θ⁰ + δ⁰ -- θ⁰ =
/// π(k − ¼)/(n + ½), and x = (1 − (1 − 1/n)/(8n²)) cos θ⁰ to O(n⁻⁴) (Tricomi 1950), δ⁰ = (1 − 1/n)/(8n²) cot θ⁰ --
/// until the step stops shrinking (the round-off's floor), on P_n(cos θ) and dP_n/dθ by Stieltjes' expansion
/// (`stieltjes`), whose terms do not grow in number with n, wherever it reaches the round-off, else by the three-term
/// recurrence (`recurrence`, n terms). The recurrence takes the nodes where n sin θ is below some twenty -- a few at
/// each end, as many at every n -- and every node of a rule of fewer than some twenty-five: the rule costs O(n). The
/// weight is 2/(dP_n/dθ)² = 2/((1 − x²)P_n′(x)²); the node is written x = sin(π(n + 1 − 2k)/(2n + 1) − δ) = cos(θ⁰ + δ),
/// to x's own round-off near the middle; the middle node of an odd rule is 0 (P_n is odd: δ = 0).
pub fn gauss_legendre(n: usize) -> Vec<(f64, f64)> {
    let constant = stieltjes_constant(n);
    let mut rule = vec![(0.0, 0.0); n];
    for k in 1..=n.div_ceil(2) {
        let (x, weight) = node(n, k, constant);
        rule[n - k] = (-x, weight);
        rule[k - 1] = (x, weight);
    }
    rule
}

/// Node k of the rule of n nodes and its weight (`gauss_legendre`); `constant` the Stieltjes expansion's weight factor
/// where its series reaches the round-off (`stieltjes_constant`).
fn node(n: usize, k: usize, constant: Option<f64>) -> (f64, f64) {
    let (nf, middle) = (n as f64, 2 * k - 1 == n);
    let start = PI * (4 * k - 1) as f64 / (4 * n + 2) as f64;
    let mut delta = if middle { 0.0 } else { (1.0 - 1.0 / nf) / (8.0 * nf * nf) / start.tan() };
    let mut last = f64::INFINITY;
    let weight = loop {
        let (step, weight) = constant.and_then(|c| stieltjes(n, k, start + delta, delta, c)).unwrap_or_else(|| recurrence(n, start + delta));
        if middle || !(step.abs() < last) {
            break weight;
        }
        last = step.abs();
        delta -= step;
    };
    ((PI * (n + 1 - 2 * k) as f64 / (2 * n + 1) as f64 - delta).sin(), weight)
}

/// Newton's step P_n/(dP_n/dθ) and the weight 2/(dP_n/dθ)² at θ = θ⁰ + δ (node k's start θ⁰, `node`) by Stieltjes'
/// expansion (Szegő, Orthogonal Polynomials, §8.21; DLMF §18.15(iii)),
/// P_n(cos θ) = C_n Σ_{m ≥ 0} h_m cos α_m/(2 sin θ)^{m + ½}, α_m = (n + m + ½)θ − (m + ½)π/2,
/// h_0 = 1, h_{m+1} = h_m (m + ½)²/((m + 1)(n + m + 3/2)), C_n = (4/π) Π_{j ≤ n} j/(j + ½),
/// which cut after M terms errs by less than twice the first term left out with its cosine taken as one. The terms fall
/// while m is below about 2n sin θ; the expansion is summed until twice the next term is within the round-off of the
/// first -- the value's envelope, and of the derivative (the expansion differentiated term by term: −(n + m + ½) sin α_m
/// − (m + ½) cot θ cos α_m over the same power, its own first term the slope's size at a node, where sin α_0 = ±1) --,
/// and is not taken (None) where its terms stop falling before. The phases are e^{iα_m} = e^{iφ}(e^{i(θ − π/2)})^m with
/// φ = (n + ½)θ − π/4 = π(k − ½) + (n + ½)δ exactly: e^{iφ} = (−1)^k (sin β − i cos β), β = (n + ½)δ -- no phase of
/// size nθ is rounded. C_n(2 sin θ)^{−½} is common to both sums and cancels in the step; the weight is
/// 2/(C_n²(2 sin θ)^{−1}S′²) = `constant` sin θ/S′², S′ the derivative's sum.
fn stieltjes(n: usize, k: usize, theta: f64, delta: f64, constant: f64) -> Option<(f64, f64)> {
    let nf = n as f64;
    let (sin, cos) = theta.sin_cos();
    let (r, cot) = (0.5 / sin, cos / sin);
    let (sb, cb) = ((nf + 0.5) * delta).sin_cos();
    let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
    // e^{iα_m} from m = 0, turned by e^{i(θ − π/2)} = sin θ − i cos θ a term; the amplitude h_m (2 sin θ)^{−m} and the
    // derivative's, (n + m + ½ + (m + ½)|cot θ|) times it.
    let (mut re, mut im) = (sign * sb, -sign * cb);
    let first = nf + 0.5 + 0.5 * cot.abs();
    let (mut amplitude, mut size, mut value, mut slope) = (1.0_f64, first, 0.0, 0.0);
    for m in 0usize.. {
        let half = m as f64 + 0.5;
        value += amplitude * re;
        slope -= amplitude * ((nf + half) * im + half * cot * re);
        let next = amplitude * half * half / ((m + 1) as f64 * (nf + half + 1.0)) * r;
        let next_size = next * (nf + half + 1.0 + (half + 1.0) * cot.abs());
        if 2.0 * next <= UNIT_ROUNDOFF && 2.0 * next_size <= UNIT_ROUNDOFF * first {
            break;
        }
        if next >= amplitude || next_size >= size {
            return None;
        }
        (amplitude, size) = (next, next_size);
        (re, im) = (re * sin + im * cos, im * sin - re * cos);
    }
    Some((value / slope, constant * sin / (slope * slope)))
}

/// The Stieltjes expansion's weight factor 4/C_n² = π(n + ½)² e^{−2s}/n (`stieltjes`) where s's series reaches the
/// round-off, else None. C_n = (4/π)Γ(3/2)Γ(n + 1)/Γ(n + 3/2) = (2/√π)√n e^s/(n + ½), s = ln(Γ(n + 1)/Γ(n + ½)) −
/// ½ ln n: the difference of Stirling's series of the two, ln Γ(z + h) ~ (z + h − ½) ln z − z + ½ ln 2π +
/// Σ_k (−1)^{k+1} B_{k+1}(h)/(k(k + 1)z^k) at z = n, h = 1 and ½ (B_j(1) = B_j, B_j(½) = (2^{1−j} − 1)B_j), is
/// s ~ Σ_{j ≥ 1} (2 − 2^{1−2j}) B_{2j}/((2j − 1)2j n^{2j−1}) = 1/(8n) − 1/(192n³) + 1/(640n⁵) − …, summed until twice
/// the next term -- the factor's relative error, from e^{−2s} -- is within the round-off (from n ≈ 10 on, where the
/// Bernoulli numbers up to B₂₀ are enough). The product of n factors would round n times.
fn stieltjes_constant(n: usize) -> Option<f64> {
    const BERNOULLI: [f64; 10] = [1.0 / 6.0, -1.0 / 30.0, 1.0 / 42.0, -1.0 / 30.0, 5.0 / 66.0, -691.0 / 2730.0, 7.0 / 6.0, -3617.0 / 510.0, 43867.0 / 798.0, -174611.0 / 330.0];
    let nf = n as f64;
    let (mut s, mut power, mut last) = (0.0_f64, 1.0 / nf, f64::INFINITY);
    for (i, bernoulli) in BERNOULLI.iter().enumerate() {
        let j = i as i32 + 1;
        let term = (2.0 - 2f64.powi(1 - 2 * j)) * bernoulli / ((2 * j - 1) * 2 * j) as f64 * power;
        if 2.0 * term.abs() <= UNIT_ROUNDOFF {
            return Some(PI * (nf + 0.5) * (nf + 0.5) / nf * (-2.0 * s).exp());
        }
        if !(term.abs() < last) {
            return None;
        }
        (s, power, last) = (s + term, power / (nf * nf), term.abs());
    }
    None
}

/// Newton's step P_n/(dP_n/dθ) and the weight 2/(dP_n/dθ)² at θ by the three-term recurrence carried in the differences
/// D_k = P_k − P_{k−1} and y = 1 − cos θ = 2 sin²(θ/2): D_{k+1} = (k D_k − (2k + 1) y P_k)/(k + 1), P_{k+1} = P_k + D_{k+1},
/// from P_0 = 1 -- near x = 1, where P_k and P_{k−1} agree to many digits, the difference is carried rather than left to
/// cancel (Reinsch's modification) --; dP_n/dθ = n(cos θ P_n − P_{n−1})/sin θ = n(D_n − y P_n)/sin θ.
fn recurrence(n: usize, theta: f64) -> (f64, f64) {
    let (sin, half) = (theta.sin(), (0.5 * theta).sin());
    let y = 2.0 * half * half;
    let (mut p, mut d) = (1.0_f64, 0.0_f64);
    for k in 0..n {
        d = (k as f64 * d - (2 * k + 1) as f64 * y * p) / (k + 1) as f64;
        p += d;
    }
    let slope = n as f64 * (d - y * p) / sin;
    (p / slope, 2.0 / (slope * slope))
}
