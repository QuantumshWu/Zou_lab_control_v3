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

/// The larger of two bounds, and the smaller (`least`): a bound that is no number makes the result none -- `f64::max` and
/// `min` return the other operand, and a NaN Gershgorin radius read through them as a distance's band became nothing: a
/// drive's distance fell to Γ/2 and a counter-rotating drive's share across 4.8e9 rad/µs read 1.33 (a random scene of R4_LIN_2's
/// atom, the model's forms none on one excited state -- a degenerate Raman pair, now joined: units.rs
/// `every_model_s_forms_are_numbers_whatever_it_keeps`). One principle for every bound and form: a number that is none
/// stays none, and a statement read of it fails (units.rs `Gate::exists`: a gate that is none is not below ½).
pub(crate) fn most(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() { f64::NAN } else { a.max(b) }
}
pub(crate) fn least(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() { f64::NAN } else { a.min(b) }
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
fn ends(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    let ends = [times(a[0], b[0]), times(a[0], b[1]), times(a[1], b[0]), times(a[1], b[1])];
    [ends.iter().copied().fold(f64::INFINITY, least), ends.iter().copied().fold(f64::NEG_INFINITY, most)]
}

impl Own {
    /// A quantity that is no number: every own form made of it is none (`most`).
    pub(crate) const NONE: Own = Own { value: f64::NAN, first: [f64::NAN; 3], rest: f64::NAN, lo: f64::NAN, hi: f64::NAN };

    /// Whether any part of it is no number.
    pub(crate) fn is_none(&self) -> bool {
        self.value.is_nan() || self.rest.is_nan() || self.lo.is_nan() || self.hi.is_nan() || self.first.iter().any(|x| x.is_nan())
    }

    /// The form of its parts, its range the given one within its own form's -- an infinite value's form is its range (∞ −
    /// ∞ is none of its ends) --; none where a part is none (`most`).
    fn made(value: f64, first: [f64; 3], rest: f64, lo: f64, hi: f64) -> Own {
        if value.is_nan() || rest.is_nan() || lo.is_nan() || hi.is_nan() || first.iter().any(|x| x.is_nan()) {
            return Own::NONE;
        }
        let width = first[0].abs() + first[1].abs() + first[2].abs() + rest;
        let (below, above) = if value.is_infinite() && width.is_infinite() { (lo, hi) } else { (most(lo, value - width), least(hi, value + width)) };
        Own { value, first, rest, lo: below, hi: above }
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
        if at.is_nan() || lo.is_nan() || hi.is_nan() {
            return Own::NONE;
        }
        // An infinite value's remainder is infinite (∞ − ∞ is no remainder).
        let rest = if at.is_infinite() { f64::INFINITY } else { most(most(hi - at, at - lo), 0.0) };
        Own { value: at, first: [0.0; 3], rest, lo, hi }
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
        if self.is_none() || lo.is_nan() || hi.is_nan() {
            return Own::NONE;
        }
        Own { lo: most(self.lo, lo), hi: least(self.hi, hi), ..self }
    }

    pub(crate) fn add(self, o: Own) -> Own {
        if self.is_none() || o.is_none() {
            return Own::NONE;
        }
        Own::made(self.value + o.value, std::array::from_fn(|a| self.first[a] + o.first[a]), self.rest + o.rest, self.lo + o.lo, self.hi + o.hi)
    }

    pub(crate) fn plus(self, c: f64) -> Own {
        if self.is_none() || c.is_nan() {
            return Own::NONE;
        }
        Own { value: self.value + c, lo: self.lo + c, hi: self.hi + c, ..self }
    }

    pub(crate) fn scale(self, c: f64) -> Own {
        if self.is_none() || c.is_nan() {
            return Own::NONE;
        }
        let ends = [times(self.lo, c), times(self.hi, c)];
        Own { value: times(self.value, c), first: self.first.map(|x| times(x, c)), rest: times(self.rest, c.abs()), lo: least(ends[0], ends[1]), hi: most(ends[0], ends[1]) }
    }

    /// a b: (a + δa)(b + δb) − ab − (a∇b + b∇a)·d = a R_b + b R_a + δa δb, |δ| at most the width; its range the product's.
    pub(crate) fn mul(self, o: Own) -> Own {
        if self.is_none() || o.is_none() {
            return Own::NONE;
        }
        let [lo, hi] = ends([self.lo, self.hi], [o.lo, o.hi]);
        let first = std::array::from_fn(|a| times(self.value, o.first[a]) + times(o.value, self.first[a]));
        Own::made(self.value * o.value, first, times(self.value.abs(), o.rest) + times(o.value.abs(), self.rest) + times(self.width(), o.width()), lo, hi)
    }

    pub(crate) fn square(self) -> Own {
        if self.is_none() {
            return Own::NONE;
        }
        let (l, h) = (self.lo, self.hi);
        let (lo, hi) = if l >= 0.0 { (l * l, h * h) } else if h <= 0.0 { (h * h, l * l) } else { (0.0, (l * l).max(h * h)) };
        let w = self.width();
        Own::made(self.value * self.value, self.first.map(|x| 2.0 * times(self.value, x)), 2.0 * times(self.value.abs(), self.rest) + w * w, lo, hi)
    }

    /// 1/a, where a's range holds no zero: 1/(a + δ) = 1/a − δ/a² + δ²/(a²(a + δ)), so the remainder is at most R_a/a² +
    /// W_a²/(a² m), m = min |a| over the cell; its range [1/max, 1/min]. Where its range holds zero, everything beyond the
    /// nearer end: at least 1/max for a quantity that is not negative.
    pub(crate) fn recip(self) -> Own {
        if self.is_none() {
            return Own::NONE;
        }
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

}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each operation's form bounds its function over a cell: on random cells about random centres, the quantities
    /// a = 1.3 + 0.4 sin(x + 2y) z, b = 0.7 + 0.2 cos(3z − y) x² and their sums, products, quotients, square roots,
    /// absolute values of their difference, maxima, minima, b's modulus with a constant, a dispersive shape of their
    /// difference through its resonance and its slope, a phase-turning modulus |2 + e^{3ix}|, and of the forms of other
    /// spaces (`Form`, `product`) the complex u = a + ib's |u|² and |u|, the vector (a, b, ab)'s norm and its product with a −
    /// b, a band about a with b its radius, a resolvent's modulus 1/|3(a − b) − 0.3i| and the norm of (u, z/((a − b) − 0.2i)),
    /// built from the operands' values, gradients and Lagrange remainders, each hold at 2000 points of the cell.
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
                (fa.sub(fb).scale(3.0).dispersion(0.3), Box::new(move |r| { let x = 3.0 * (a(r) - b(r)); x / (x * x + 0.09) })),
                (fa.sub(fb).square(), Box::new(move |r| (a(r) - b(r)).powi(2))),
                (fa.sub(fb).scale(3.0).dispersion_slope(0.3), Box::new(move |r| { let x = 3.0 * (a(r) - b(r)); (0.09 - x * x) / (x * x + 0.09).powi(2) })),
            ];
            // The forms of other spaces, of the two quantities' real forms: u = a + ib, its |u|² and |u|; v = (a, b, ab), |v|
            // and |(a − b) v|; and a band a ± b, its least modulus and its largest.
            let (ra, rb) = (fa.form(), fb.form());
            let unit = |z: C64| Form { value: z, first: [C64::ZERO; 3], rest: 0.0, moved: 0.0, most: z.abs() };
            let u = product(&ra, &unit(C64::ONE), |x: f64, z: C64| z * x).add(product(&rb, &unit(C64::I), |x: f64, z: C64| z * x));
            let axis = |a: usize| Form { value: std::array::from_fn::<f64, 3, _>(|b| if a == b { 1.0 } else { 0.0 }), first: [[0.0; 3]; 3], rest: 0.0, moved: 0.0, most: 1.0 };
            let v = product(&ra, &axis(0), |x: f64, e: [f64; 3]| e.map(|y| y * x)).add(product(&rb, &axis(1), |x: f64, e: [f64; 3]| e.map(|y| y * x))).add(product(&fa.mul(fb).form(), &axis(2), |x: f64, e: [f64; 3]| e.map(|y| y * x)));
            let band = Band { mid: fa, rad: fb };
            let vector = move |r: [f64; 3]| [a(r), b(r), a(r) * b(r)];
            let length = |v: [f64; 3]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
            let others: Vec<(Own, Box<dyn Fn([f64; 3]) -> f64>)> = vec![
                (product(&u, &u, |x: C64, y: C64| (x.conj() * y).re).own(), Box::new(move |r| a(r) * a(r) + b(r) * b(r))),
                (u.norm(), Box::new(move |r| a(r).hypot(b(r)))),
                (v.norm(), Box::new(move |r| length(vector(r)))),
                (v.by(&fa.sub(fb)).norm(), Box::new(move |r| (a(r) - b(r)).abs() * length(vector(r)))),
                (band.apart(), Box::new(move |r| (a(r).abs() - b(r)).max(0.0))),
                (band.most(), Box::new(move |r| a(r).abs() + b(r))),
            ];
            for (own, f) in &others {
                for _ in 0..2000 {
                    let r: [f64; 3] = std::array::from_fn(|x| c[x] + h[x] * (2.0 * stream.uniform() - 1.0));
                    let v = f(r);
                    assert!(v <= own.upper() + 1e-9 && v >= own.lower() - 1e-9, "a form of another space: {v} outside [{}, {}] at {r:?} about {c:?}", own.lower(), own.upper());
                }
            }
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

    /// A number that is none stays none (`most`): fed into any combinator of the own forms -- as an operand's value, an
    /// end of its range, its remainder or a first order, or as a constant -- the result is none, and its bounds are no
    /// numbers: a statement read of it is not below any threshold. And ∞ is not none: an infinite value's form is its
    /// range.
    #[test]
    fn a_number_that_is_none_surfaces_through_every_combinator() {
        let fine = Own::new(2.0, [0.3, -0.1, 0.2], 0.05, [1.0, 1.0, 1.0]);
        let nan = f64::NAN;
        let poisoned = [
            Own { value: nan, ..fine },
            Own { lo: nan, ..fine },
            Own { hi: nan, ..fine },
            Own { rest: nan, ..fine },
            Own { first: [0.3, nan, 0.2], ..fine },
            Own::range(nan, 0.0, 1.0),
            Own::range(1.0, nan, 2.0),
            Own::constant(nan),
        ];
        let none = |o: Own, what: &str| assert!(o.is_none() && o.upper().is_nan() && o.lower().is_nan() && !(o.upper() < 0.5), "{what}: {o:?}");
        for (k, p) in poisoned.iter().enumerate() {
            assert!(p.is_none(), "{k}: {p:?}");
            for (name, o) in [
                ("add", p.add(fine)), ("add'", fine.add(*p)), ("sub", p.sub(fine)), ("sub'", fine.sub(*p)), ("mul", p.mul(fine)), ("mul'", fine.mul(*p)),
                ("div", p.div(fine)), ("div'", fine.div(*p)), ("max", p.max(fine)), ("max'", fine.max(*p)), ("min", p.min(fine)), ("min'", fine.min(*p)),
                ("square", p.square()), ("recip", p.recip()), ("sqrt", p.sqrt()), ("abs", p.abs()), ("scale", p.scale(2.0)), ("plus", p.plus(1.0)),
                ("within", p.within(0.0, 1.0)), ("hypot", p.hypot(1.0)), ("dispersion", p.dispersion(1.0)), ("dispersion slope", p.dispersion_slope(1.0)),
                ("dispersion ends", p.dispersion_ends(fine, 1.0)[1]), ("dispersion ends'", fine.dispersion_ends(*p, 1.0)[0]),
            ] {
                none(o, &format!("{k} {name}"));
            }
            assert!(p.form().moved.is_nan() || p.form().most.is_nan(), "{k} form: {:?}", p.form());
        }
        for (name, o) in [("scale", fine.scale(nan)), ("plus", fine.plus(nan)), ("within", fine.within(nan, 1.0)), ("within'", fine.within(0.0, nan))] {
            none(o, name);
        }
        // The band of the case found: a radius none, its distance read through max(|Δ| − moved, 0) and Γ/2 -- none.
        let distance = Own::constant(4.8e9).abs().sub(Own::range(nan, 2.27, nan)).max(Own::constant(0.0)).hypot(19.06);
        none(distance, "the distance of a band that is none");
        assert!(most(nan, 1.0).is_nan() && most(1.0, nan).is_nan() && least(nan, 1.0).is_nan() && least(1.0, nan).is_nan() && most(1.0, 2.0) == 2.0 && least(1.0, 2.0) == 1.0);
        // ∞ is a number: an infinite value's form is its range.
        let infinite = Own::range(f64::INFINITY, 0.0, f64::INFINITY);
        assert!(!infinite.is_none() && infinite.upper() == f64::INFINITY && infinite.lower() == 0.0, "{infinite:?}");
        assert!(!infinite.add(fine).is_none() && !fine.div(Own::range(0.0, 0.0, 1.0)).is_none(), "infinite forms stay numbers");
    }
}
