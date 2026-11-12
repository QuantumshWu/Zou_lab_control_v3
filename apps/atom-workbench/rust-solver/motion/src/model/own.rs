//! A real quantity over a cell in its own form (reach.rs, # Own forms): its value at the cell's centre, its first order
//! along each side -- h_a ∂_a f(c), the gradient at the centre times the cell's half-width -- and the bound of its
//! remainder over the cell, |f(c + d) − f(c) − ∇f(c)·d| ≤ R for every displacement d within the half-widths h -- so f lies
//! within f(c) ± (Σ_a h_a |∂_a f(c)| + R), its own first-order variation and a remainder of the second order --, and beside
//! it a range [lo, hi] that holds f over the cell, kept the narrower of the two from the form's making on. The arithmetic
//! keeps both: each operation's value and first order by the chain rule at the centre and its remainder from its operands'
//! remainders and widths (each rule derived at its function), and its range by the operation's exact range over the
//! operands' (interval arithmetic). A quantity assembled from parts that share a term -- a difference of two levels that
//! move together, a ratio of two couplings that grow with one intensity -- so reads that term's cancellation at the first
//! order instead of adding the parts' ranges: its bound tends to its value as fast as the quantity itself varies; and where
//! a cell is too coarse for the first order (a turn of the light's phase across it) the range bounds it, a monotone
//! function's to its ends. Where an operation is not smooth over the cell (an absolute value, a maximum or a quotient whose
//! argument's range holds the kink or the pole) the form is the operation's range, its first order lost: that is the
//! operation's own variation there. On a point (h = 0) every width is nothing and the form is the value.

use crate::linalg::C64;

/// A real quantity over a cell (module documentation): `first` its first order along each side, h_a ∂_a f(c); [lo, hi]
/// within f(c) ± (Σ_a |first_a| + rest).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Own {
    pub(crate) value: f64,
    pub(crate) first: [f64; 3],
    pub(crate) rest: f64,
    pub(crate) lo: f64,
    pub(crate) hi: f64,
}

/// x y with 0 ∞ read as nothing: a part that is exactly nothing over the cell (its value and its width) makes its products
/// nothing whatever the other's range -- and a gradient's component that is nothing, nothing along an unbounded side.
fn times(x: f64, y: f64) -> f64 {
    if x == 0.0 || y == 0.0 {
        0.0
    } else {
        x * y
    }
}

/// The range of a product of two ranges.
fn product(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    let ends = [times(a[0], b[0]), times(a[0], b[1]), times(a[1], b[0]), times(a[1], b[1])];
    [ends.iter().copied().fold(f64::INFINITY, f64::min), ends.iter().copied().fold(f64::NEG_INFINITY, f64::max)]
}

impl Own {
    /// The form of its parts, its range the given one within its own form's (`f64::max`/`min` pass a NaN end by: an
    /// infinite value's form is its range).
    fn made(value: f64, first: [f64; 3], rest: f64, lo: f64, hi: f64) -> Own {
        let width = first[0].abs() + first[1].abs() + first[2].abs() + rest;
        Own { value, first, rest, lo: lo.max(value - width), hi: hi.min(value + width) }
    }

    /// A form of its own over a cell of half-widths `half`: value, gradient at the centre and remainder, no range beside it.
    pub(crate) fn new(value: f64, gradient: [f64; 3], rest: f64, half: [f64; 3]) -> Own {
        Own::made(value, std::array::from_fn(|a| times(gradient[a], half[a])), rest, f64::NEG_INFINITY, f64::INFINITY)
    }

    /// A constant.
    pub(crate) fn constant(value: f64) -> Own {
        Own { value, first: [0.0; 3], rest: 0.0, lo: value, hi: value }
    }

    /// A quantity known only to lie in [lo, hi] over the cell, its value at the centre `at`: no first order.
    pub(crate) fn range(at: f64, lo: f64, hi: f64) -> Own {
        Own { value: at, first: [0.0; 3], rest: (hi - at).max(at - lo).max(0.0), lo, hi }
    }

    /// Σ_a h_a |∂_a f(c)|, its first-order width.
    pub(crate) fn lean(&self) -> f64 {
        self.first.iter().map(|x| x.abs()).sum()
    }

    /// How far its own form lies from its centre value over the cell, at most.
    pub(crate) fn width(&self) -> f64 {
        self.lean() + self.rest
    }

    pub(crate) fn upper(&self) -> f64 {
        self.hi
    }

    pub(crate) fn lower(&self) -> f64 {
        self.lo
    }

    /// The same quantity known also to lie in [lo, hi].
    pub(crate) fn within(self, lo: f64, hi: f64) -> Own {
        Own { lo: self.lo.max(lo), hi: self.hi.min(hi), ..self }
    }

    pub(crate) fn add(self, o: Own) -> Own {
        Own::made(self.value + o.value, std::array::from_fn(|a| self.first[a] + o.first[a]), self.rest + o.rest, self.lo + o.lo, self.hi + o.hi)
    }

    pub(crate) fn sub(self, o: Own) -> Own {
        self.add(o.scale(-1.0))
    }

    pub(crate) fn plus(self, c: f64) -> Own {
        Own { value: self.value + c, lo: self.lo + c, hi: self.hi + c, ..self }
    }

    pub(crate) fn scale(self, c: f64) -> Own {
        let ends = [times(self.lo, c), times(self.hi, c)];
        Own { value: self.value * c, first: self.first.map(|x| x * c), rest: times(self.rest, c.abs()), lo: ends[0].min(ends[1]), hi: ends[0].max(ends[1]) }
    }

    /// a b: (a + δa)(b + δb) − ab − (a∇b + b∇a)·d = a R_b + b R_a + δa δb, |δ| at most the width; its range the product's.
    pub(crate) fn mul(self, o: Own) -> Own {
        let [lo, hi] = product([self.lo, self.hi], [o.lo, o.hi]);
        let first = std::array::from_fn(|a| self.value * o.first[a] + o.value * self.first[a]);
        Own::made(self.value * o.value, first, times(self.value.abs(), o.rest) + times(o.value.abs(), self.rest) + times(self.width(), o.width()), lo, hi)
    }

    pub(crate) fn square(self) -> Own {
        let (l, h) = (self.lo, self.hi);
        let (lo, hi) = if l >= 0.0 { (l * l, h * h) } else if h <= 0.0 { (h * h, l * l) } else { (0.0, (l * l).max(h * h)) };
        let w = self.width();
        Own::made(self.value * self.value, self.first.map(|x| 2.0 * self.value * x), 2.0 * times(self.value.abs(), self.rest) + w * w, lo, hi)
    }

    /// 1/a, where a's range holds no zero: 1/(a + δ) = 1/a − δ/a² + δ²/(a²(a + δ)), so the remainder is at most R_a/a² +
    /// W_a²/(a² m), m = min |a| over the cell; its range [1/max, 1/min]. Where its range holds zero, everything beyond the
    /// nearer end: at least 1/max for a quantity that is not negative.
    pub(crate) fn recip(self) -> Own {
        let (lo, hi, a) = (self.lo, self.hi, self.value);
        if lo > 0.0 || hi < 0.0 {
            let m = lo.abs().min(hi.abs());
            let w = self.width();
            let rest = self.rest / (a * a) + times(w * w, 1.0 / (a * a * m));
            Own::made(1.0 / a, self.first.map(|x| -x / (a * a)), rest, 1.0 / hi, 1.0 / lo)
        } else {
            let value = if a != 0.0 { 1.0 / a } else { f64::INFINITY };
            Own::range(value, if lo >= 0.0 { 1.0 / hi } else { f64::NEG_INFINITY }, f64::INFINITY)
        }
    }

    pub(crate) fn div(self, o: Own) -> Own {
        self.mul(o.recip())
    }

    /// √a of a ≥ 0: with s = √(a + δ), t = √a, s − t − δ/(2t) = −(s − t)²/(2t) = −δ²/(2t(s + t)²), so the remainder is at
    /// most R_a/(2t) + W_a²/(2t(√m + t)²), m the least of a over the cell; its range [√min, √max]. Where a's range reaches
    /// nothing, the range alone.
    pub(crate) fn sqrt(self) -> Own {
        let (lo, hi, a) = (self.lo, self.hi, self.value);
        if lo > 0.0 && a > 0.0 {
            let (t, w) = (a.sqrt(), self.width());
            let rest = self.rest / (2.0 * t) + w * w / (2.0 * t * (lo.sqrt() + t).powi(2));
            Own::made(t, self.first.map(|x| x / (2.0 * t)), rest, lo.sqrt(), hi.sqrt())
        } else {
            Own::range(a.max(0.0).sqrt(), lo.max(0.0).sqrt(), hi.max(0.0).sqrt())
        }
    }

    /// |a|: itself or its negative where its range keeps one sign, else the range [0, max |end|].
    pub(crate) fn abs(self) -> Own {
        if self.lo >= 0.0 {
            self
        } else if self.hi <= 0.0 {
            self.scale(-1.0)
        } else {
            Own::range(self.value.abs(), 0.0, self.lo.abs().max(self.hi.abs()))
        }
    }

    /// max(a, b): the one whose range lies above the other's, else the range of the maximum.
    pub(crate) fn max(self, o: Own) -> Own {
        if self.lo >= o.hi {
            self
        } else if o.lo >= self.hi {
            o
        } else {
            Own::range(self.value.max(o.value), self.lo.max(o.lo), self.hi.max(o.hi))
        }
    }

    /// min(a, b): the one whose range lies below the other's, else the range of the minimum.
    pub(crate) fn min(self, o: Own) -> Own {
        if self.hi <= o.lo {
            self
        } else if o.hi <= self.lo {
            o
        } else {
            Own::range(self.value.min(o.value), self.lo.min(o.lo), self.hi.min(o.hi))
        }
    }

    /// (a² + c²)^{1/2}, c a constant: the modulus of a + ic.
    pub(crate) fn hypot(self, c: f64) -> Own {
        self.square().plus(c * c).sqrt()
    }

    /// The norm ‖u‖ over a cell of half-widths `half` of a complex vector u(c + d) = u(c) + Σ_a d_a ∂_a u(c) + R, ‖R‖ ≤
    /// `remainder`: with δ = u(c + d) − u(c), ‖δ‖ ≤ w = Σ_a h_a ‖∂_a u(c)‖ + R, and x = Re⟨u, δ⟩/‖u‖, ‖u + δ‖² = (‖u‖ + x)² +
    /// ‖δ‖² − x², so 0 ≤ ‖u + δ‖ − ‖u‖ − x ≤ ‖δ‖²/(2(‖u‖ − ‖δ‖)): its gradient Re⟨u, ∂_a u⟩/‖u‖ and its remainder R +
    /// w²/(2(‖u‖ − w)), where ‖u‖ > w; else the range ‖u‖ ± w, at least nothing. Its range ‖u(c)‖ ± `first` within
    /// [0, `blind`]. A turning phase at constant modulus (a counter-propagating pair's a₁* a₂ = |a₁||a₂| e^{2ikx}) moves the
    /// modulus by the second order, not by the 2k h each part moves. On the whole space, [0, blind].
    pub(crate) fn norm(value: &[C64], slopes: [&[C64]; 3], remainder: f64, first: f64, blind: f64, half: [f64; 3]) -> Own {
        let n = value.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        if half.iter().any(|h| !h.is_finite()) {
            return Own::range(0.0, 0.0, blind);
        }
        let steep: f64 = (0..3).map(|a| times(half[a], slopes[a].iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt())).sum();
        let w = steep + remainder;
        let own = if n > w {
            let gradient = std::array::from_fn(|a| value.iter().zip(slopes[a]).map(|(u, g)| (u.conj() * *g).re).sum::<f64>() / n);
            Own::new(n, gradient, remainder + w * w / (2.0 * (n - w)), half)
        } else {
            Own::range(n, (n - w).max(0.0), n + w)
        };
        own.within((n - first).max(0.0), (n + first).min(blind))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each operation's form bounds its function over a cell: on random cells about random centres, the quantities
    /// a = 1.3 + 0.4 sin(x + 2y) z, b = 0.7 + 0.2 cos(3z − y) x² and their sums, products, quotients, square roots,
    /// absolute values of their difference, maxima, minima, b's modulus with a constant and a phase-turning modulus
    /// |2 + e^{3ix}|, built from the operands' values, gradients and Lagrange remainders, each hold at 2000 points of the
    /// cell.
    #[test]
    fn an_own_form_bounds_its_function_over_the_cell() {
        let mut stream = crate::random::Stream::new(3, crate::random::Purpose::Initial, 0, 0);
        // A function's own form at c over half-widths h: the value, the gradient and the remainder by a bound of its
        // second derivatives (40, the test's own, large) times ½ (Σ h)².
        let form = |f: &dyn Fn([f64; 3]) -> f64, c: [f64; 3], h: [f64; 3]| -> Own {
            let e = 1e-6;
            let gradient = std::array::from_fn(|a| { let mut p = c; let mut m = c; p[a] += e; m[a] -= e; (f(p) - f(m)) / (2.0 * e) });
            let span: f64 = h.iter().sum();
            Own::new(f(c), gradient, 0.5 * 40.0 * span * span + 1e-9, h)
        };
        let a = |r: [f64; 3]| 1.3 + 0.4 * (r[0] + 2.0 * r[1]).sin() * r[2];
        let b = |r: [f64; 3]| 0.7 + 0.2 * (3.0 * r[2] - r[1]).cos() * r[0] * r[0];
        for _ in 0..200 {
            let c: [f64; 3] = std::array::from_fn(|_| 2.0 * stream.uniform() - 1.0);
            let size = 0.3 * stream.uniform();
            let h: [f64; 3] = std::array::from_fn(|_| size * (0.2 + 0.8 * stream.uniform()));
            let (fa, fb) = (form(&a, c, h), form(&b, c, h));
            let cases: Vec<(Own, Box<dyn Fn([f64; 3]) -> f64>)> = vec![
                (fa.add(fb), Box::new(move |r| a(r) + b(r))),
                (fa.mul(fb), Box::new(move |r| a(r) * b(r))),
                (fa.div(fb), Box::new(move |r| a(r) / b(r))),
                (fa.sqrt(), Box::new(move |r| a(r).sqrt())),
                (fa.sub(fb).abs(), Box::new(move |r| (a(r) - b(r)).abs())),
                (fa.max(fb), Box::new(move |r| a(r).max(b(r)))),
                (fa.min(fb), Box::new(move |r| a(r).min(b(r)))),
                (fb.hypot(0.5), Box::new(move |r| (b(r) * b(r) + 0.25).sqrt())),
                (fa.sub(fb).square(), Box::new(move |r| (a(r) - b(r)).powi(2))),
            ];
            for (own, f) in &cases {
                for _ in 0..2000 {
                    let r: [f64; 3] = std::array::from_fn(|x| c[x] + h[x] * (2.0 * stream.uniform() - 1.0));
                    let v = f(r);
                    assert!(v <= own.upper() + 1e-9 && v >= own.lower() - 1e-9, "{v} outside [{}, {}] at {r:?} about {c:?}", own.lower(), own.upper());
                }
            }
            // A phase-turning modulus: u = 2 + e^{3ix}, its own norm form.
            let u = |r: [f64; 3]| C64::real(2.0) + C64::cis(3.0 * r[0]);
            let value = [u(c)];
            let slopes = [vec![C64::cis(3.0 * c[0]) * C64::new(0.0, 3.0)], vec![C64::ZERO], vec![C64::ZERO]];
            let own = Own::norm(&value, [&slopes[0], &slopes[1], &slopes[2]], 0.5 * 9.0 * h[0] * h[0], 3.0 * h[0], 3.0, h);
            for _ in 0..2000 {
                let r: [f64; 3] = std::array::from_fn(|x| c[x] + h[x] * (2.0 * stream.uniform() - 1.0));
                let v = u(r).abs();
                assert!(v <= own.upper() + 1e-12 && v >= own.lower() - 1e-12, "|u| {v} outside [{}, {}]", own.lower(), own.upper());
            }
        }
    }
}
