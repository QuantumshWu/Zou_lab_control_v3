//! The incomplete gamma function's two expansions (Abramowitz & Stegun 6.5.29 and 6.5.31; Numerical Recipes §6.2, gser and
//! gcf), each summed to the format's round-off, written once for the two that take them: the normal's tail (estimate.rs)
//! and the lowered isothermal's momenta (ensemble.rs).
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
        (2. * x - s).ceil().max(0.) as usize + 54
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
}
