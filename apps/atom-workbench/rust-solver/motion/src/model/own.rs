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
//!
//! # Other spaces
//! A quantity whose values are complex numbers or vectors -- a beam sum's amplitude, a level's gradient -- has the same form
//! (`Form`), with the norm in the place of the modulus and, beside it, how far it moves from its centre value and how large
//! it is over the cell in the place of the range. A product of two forms by a bilinear map (`product`) has its value and first
//! order by the product rule and its remainder from the factors' remainders and widths, as `Own::mul`'s; the norm of a form
//! is an `Own` (`Form::norm`). So a rate that reads the slope of a difference -- of two levels, of a level and a shift that
//! moves with it -- reads the norm of the difference's own gradient, never the sum of the parts' slopes: two levels the same
//! light shifts alike move apart by the difference of their gradients.
//!
//! # Bands
//! What the bounds read at an atom depends beside the place on parameters that are not functions of it: the Doppler shift of
//! the reach's speeds, a coupling's or a self-energy's effect bounded and not computed. Such a quantity is a band (`Band`),
//! mid ± rad: `mid` the smooth function of the place with those parameters at their centre, `rad` ≥ 0 how far they move it at
//! most -- each its own form. Put into a form's remainder instead, a parameter's interval does not shrink with the cell, and
//! every product with a first order of the place keeps a first-order cross term (the band's width times the other factor's
//! first order), which the quantity at any one point does not have.

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
fn ends(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
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
        let [lo, hi] = ends([self.lo, self.hi], [o.lo, o.hi]);
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

    /// f(a) of a smooth f: `value` and `slope` f(a), f'(a) at the operand's centre value and `curved` a bound of |f''|
    /// between the operand's values over the cell -- f(a + δ) = f(a) + f'(a) δ + ½ f''(ξ) δ², so its first order f'(a) times
    /// the operand's and its remainder |f'(a)| R_a + ½ `curved` W_a² --; its range [lo, hi], f's over the operand's.
    fn through(self, value: f64, slope: f64, curved: f64, lo: f64, hi: f64) -> Own {
        let w = self.width();
        Own::made(value, self.first.map(|x| slope * x), times(slope.abs(), self.rest) + 0.5 * times(curved, w * w), lo, hi)
    }

    /// The least modulus of its values over the cell (its centre value's with its range).
    fn least_size(&self) -> f64 {
        let (lo, hi) = (self.lo.min(self.value), self.hi.max(self.value));
        if lo <= 0.0 && hi >= 0.0 { 0.0 } else { lo.abs().min(hi.abs()) }
    }

    /// φ(a) = a/(a² + γ²), γ > 0 a constant: a two-level atom's dispersive shape, its level's light shift per |v|² at the
    /// detuning a and the half-width γ, φ(x) = Re 1/(x + iγ). Smooth everywhere, its derivatives φ⁽ⁿ⁾(x) = Re (−1)ⁿ n!/(x +
    /// iγ)ⁿ⁺¹ at most n!/|x + iγ|ⁿ⁺¹: between the operand's values, whose least modulus is m, |φ''| ≤ 2/(m² + γ²)^{3/2} --
    /// and everywhere at most K/γ³, K = 2 max |t(t² − 3)|/(1 + t²)³ at t = √2 − 1, where t⁴ − 6t² + 1 = 0: 1.457 (`through`).
    /// A drive far from its line, |a| ≫ γ, curves as 2/|a|³ there, not as the line's own 1.457/γ³. Its range is the shape's
    /// over the operand's: monotone on each side of ±γ, its extremes ±1/(2γ) there.
    pub(crate) fn dispersion(self, gamma: f64) -> Own {
        let f = |x: f64| if x.is_infinite() { 0.0 } else { x / (x * x + gamma * gamma) };
        let a = self.value;
        let slope = (gamma * gamma - a * a) / (a * a + gamma * gamma).powi(2);
        let t = std::f64::consts::SQRT_2 - 1.0;
        let curved = (2.0 * (t * (3.0 - t * t)) / (1.0 + t * t).powi(3) / gamma.powi(3)).min(2.0 / self.least_size().hypot(gamma).powi(3));
        let ends = [f(self.lo), f(self.hi)];
        let mut lo = ends[0].min(ends[1]);
        let mut hi = ends[0].max(ends[1]);
        if self.lo <= gamma && gamma <= self.hi {
            hi = 0.5 / gamma;
        }
        if self.lo <= -gamma && -gamma <= self.hi {
            lo = -0.5 / gamma;
        }
        self.through(f(a), slope, curved, lo, hi)
    }

    /// φ'(a) = (γ² − a²)/(a² + γ²)², the dispersive shape's slope (`dispersion`): its own slope φ''(a) = 2a(a² − 3γ²)/(a² +
    /// γ²)³ and |φ'''| ≤ 6/(m² + γ²)² between the operand's values (`through`). Its range: φ' is even, falls from 1/γ² at 0 to
    /// −1/(8γ²) at |x| = √3 γ and rises from there towards 0 -- over the operand's moduli [m, M] the larger of its ends'
    /// values, the least −1/(8γ²) where √3 γ is among them, else the smaller end's.
    pub(crate) fn dispersion_slope(self, gamma: f64) -> Own {
        let g2 = gamma * gamma;
        let f = |x: f64| if x.is_infinite() { 0.0 } else { (g2 - x * x) / (x * x + g2).powi(2) };
        let a = self.value;
        let slope = 2.0 * a * (a * a - 3.0 * g2) / (a * a + g2).powi(3);
        let m = self.least_size();
        let most = self.lo.abs().max(self.hi.abs()).max(a.abs());
        let (at_least, at_most) = (f(m), f(most));
        let lo = if m <= 3.0_f64.sqrt() * gamma && 3.0_f64.sqrt() * gamma <= most { -0.125 / g2 } else { at_least.min(at_most) };
        self.through(f(a), slope, 6.0 / (m * m + g2).powi(2), lo, at_least.max(at_most))
    }

    /// The same quantity as a real form (`Form`): how far it moves, the farther of its range's ends from its value; how large,
    /// the larger end's modulus.
    pub(crate) fn form(&self) -> Form<f64> {
        Form { value: self.value, first: self.first, rest: self.rest, moved: (self.hi - self.value).max(self.value - self.lo).max(0.0), most: self.lo.abs().max(self.hi.abs()) }
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
        let dots = std::array::from_fn(|a| times(half[a], value.iter().zip(slopes[a]).map(|(u, g)| (u.conj() * *g).re).sum::<f64>()));
        measured(n, dots, remainder, (steep + remainder).min(first), first, blind)
    }
}

/// The norm of a quantity over a cell (`Own::norm`, `Form::norm`) from its centre value's norm n, Re⟨u(c), first_a⟩ per side
/// (its first order times n), its remainder R, a bound w of |u − u(c)| over the cell, how far it moves and how large it is:
/// where n > w, the first order Re⟨u, first_a⟩/n and the remainder R + w²/(2(n − w)); else the range n ± w, at least
/// nothing; within n ± `moved`, at most `most`.
fn measured(n: f64, dots: [f64; 3], rest: f64, w: f64, moved: f64, most: f64) -> Own {
    let own = if n > w { Own::made(n, dots.map(|x| x / n), rest + w * w / (2.0 * (n - w)), f64::NEG_INFINITY, f64::INFINITY) } else { Own::range(n, (n - w).max(0.0), n + w) };
    own.within((n - moved).max(0.0), (n + moved).min(most))
}

/// A vector space a quantity over a cell takes its values in (`Form`): the reals, the complex numbers, real and complex
/// 3-vectors (a gradient), each with the real part of its inner product and the Euclidean norm it makes.
pub(crate) trait Space: Copy {
    const ZERO: Self;
    fn plus(self, o: Self) -> Self;
    fn times(self, c: f64) -> Self;
    /// Re⟨self, o⟩.
    fn dot(self, o: Self) -> f64;
    fn size(self) -> f64 {
        self.dot(self).sqrt()
    }
}

impl Space for f64 {
    const ZERO: f64 = 0.0;
    fn plus(self, o: f64) -> f64 {
        self + o
    }
    fn times(self, c: f64) -> f64 {
        self * c
    }
    fn dot(self, o: f64) -> f64 {
        self * o
    }
    fn size(self) -> f64 {
        self.abs()
    }
}

impl Space for C64 {
    const ZERO: C64 = C64::ZERO;
    fn plus(self, o: C64) -> C64 {
        self + o
    }
    fn times(self, c: f64) -> C64 {
        self * c
    }
    fn dot(self, o: C64) -> f64 {
        (self.conj() * o).re
    }
    fn size(self) -> f64 {
        self.abs()
    }
}

impl<T: Space> Space for [T; 3] {
    const ZERO: [T; 3] = [T::ZERO; 3];
    fn plus(self, o: [T; 3]) -> [T; 3] {
        std::array::from_fn(|a| self[a].plus(o[a]))
    }
    fn times(self, c: f64) -> [T; 3] {
        self.map(|x| x.times(c))
    }
    fn dot(self, o: [T; 3]) -> f64 {
        (0..3).map(|a| self[a].dot(o[a])).sum()
    }
}

/// A quantity over a cell valued in a space (module documentation, # Other spaces), in its own form as `Own` is a real one's:
/// its value at the centre, its first order along each side h_a ∂_a f(c) and the bound of its remainder, and beside them --
/// each bounded on its own, made with the quantity -- how far it moves from its centre value over the cell (`moved`) and how
/// large it is there (`most`). On a point every width is nothing; on the whole space its value is nothing and `most` its bound.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Form<T> {
    pub(crate) value: T,
    pub(crate) first: [T; 3],
    pub(crate) rest: f64,
    pub(crate) moved: f64,
    pub(crate) most: f64,
}

impl<T: Space> Form<T> {
    /// Nothing.
    pub(crate) fn zero() -> Form<T> {
        Form { value: T::ZERO, first: [T::ZERO; 3], rest: 0.0, moved: 0.0, most: 0.0 }
    }

    /// A constant.
    pub(crate) fn constant(value: T) -> Form<T> {
        Form { value, first: [T::ZERO; 3], rest: 0.0, moved: 0.0, most: value.size() }
    }

    /// A quantity over the whole space, known only to be at most `most`.
    pub(crate) fn whole(most: f64) -> Form<T> {
        Form { value: T::ZERO, first: [T::ZERO; 3], rest: f64::INFINITY, moved: f64::INFINITY, most }
    }

    /// How far it lies from its centre value over the cell, at most: its own form's width, within how far it moves.
    pub(crate) fn width(&self) -> f64 {
        (self.first.iter().map(|x| x.size()).sum::<f64>() + self.rest).min(self.moved)
    }

    pub(crate) fn add(self, o: Form<T>) -> Form<T> {
        Form { value: self.value.plus(o.value), first: std::array::from_fn(|a| self.first[a].plus(o.first[a])), rest: self.rest + o.rest, moved: self.moved + o.moved, most: self.most + o.most }
    }

    pub(crate) fn scale(self, c: f64) -> Form<T> {
        let size = c.abs();
        Form { value: self.value.times(c), first: self.first.map(|x| x.times(c)), rest: times(self.rest, size), moved: times(self.moved, size), most: times(self.most, size) }
    }

    pub(crate) fn sub(self, o: Form<T>) -> Form<T> {
        self.add(o.scale(-1.0))
    }

    /// Its norm over the cell in its own form (`measured`): the value's norm, its first order Re⟨f, first_a⟩/|f| and its
    /// remainder R + w²/(2(|f| − w)) -- with δ = f(c + d) − f(c), |δ| ≤ w, and x = Re⟨f, δ⟩/|f|, |f + δ|² = (|f| + x)² + |δ|²
    /// − x², so 0 ≤ |f + δ| − |f| − x ≤ |δ|²/(2(|f| − |δ|)) --, within |f| ± `moved` and at most `most`. A vector that turns
    /// at constant length (a counter-propagating pair's coefficient, a gradient along a fringe's crest) moves its norm by the
    /// second order, not by its first order's size.
    pub(crate) fn norm(&self) -> Own {
        let n = self.value.size();
        measured(n, self.first.map(|x| self.value.dot(x)), self.rest, self.width(), self.moved, self.most)
    }
}

impl Form<f64> {
    /// The real quantity as an `Own`: within its value ± how far it moves and ± how large it is.
    pub(crate) fn own(&self) -> Own {
        Own::made(self.value, self.first, self.rest, (self.value - self.moved).max(-self.most), (self.value + self.moved).min(self.most))
    }
}

impl Form<C64> {
    /// z u of a complex constant z.
    pub(crate) fn turned(self, z: C64) -> Form<C64> {
        let size = z.abs();
        Form { value: self.value * z, first: self.first.map(|x| x * z), rest: times(self.rest, size), moved: times(self.moved, size), most: times(self.most, size) }
    }

    /// |u|² and its gradient 2 Re(u* ∇u) over the cell, of u's form and its gradient's (`product`): |u|² not negative.
    pub(crate) fn square(&self, gradient: &Form<[C64; 3]>) -> (Own, Form<[f64; 3]>) {
        let square = product(self, self, |a: C64, b: C64| (a.conj() * b).re).own().within(0.0, f64::INFINITY);
        (square, product(self, gradient, |a: C64, g: [C64; 3]| g.map(|z| (a.conj() * z).re)).scale(2.0))
    }
}

impl Form<[f64; 3]> {
    /// s v of a real quantity s over the cell (`product`).
    pub(crate) fn by(&self, s: &Own) -> Form<[f64; 3]> {
        product(&s.form(), self, |a: f64, v: [f64; 3]| v.map(|x| x * a))
    }
}

impl Form<[C64; 3]> {
    /// Its real part: each part's at most the whole's.
    pub(crate) fn re(&self) -> Form<[f64; 3]> {
        let re = |v: [C64; 3]| v.map(|z| z.re);
        Form { value: re(self.value), first: self.first.map(re), rest: self.rest, moved: self.moved, most: self.most }
    }
}

/// p(x, y) over a cell of two forms and a bilinear map p with |p(a, b)| ≤ |a| |b| (module documentation, # Other spaces): with
/// x + δx, y + δy their values at c + d, p(x + δx, y + δy) − p(x, y) − p(x, ∂y·d) − p(∂x·d, y) = p(x, δy − ∂y·d) + p(δx −
/// ∂x·d, y) + p(δx, δy), so its remainder is at most |x| R_y + |y| R_x + W_x W_y, W a bound of |δ| (`Form::width`); it moves
/// by |p(x, δy) + p(δx, y + δy)| ≤ |x| moved_y + moved_x most_y, and is at most most_x most_y.
pub(crate) fn product<A: Space, B: Space, R: Space>(x: &Form<A>, y: &Form<B>, p: impl Fn(A, B) -> R) -> Form<R> {
    let (sx, sy) = (x.value.size(), y.value.size());
    Form {
        value: p(x.value, y.value),
        first: std::array::from_fn(|a| p(x.value, y.first[a]).plus(p(x.first[a], y.value))),
        rest: times(sx, y.rest) + times(sy, x.rest) + times(x.width(), y.width()),
        moved: times(sx, y.moved) + times(x.moved, y.most),
        most: times(x.most, y.most),
    }
}

/// The norm of a vector of forms over the cell (`Form::norm`): its parts' remainders, widths, movements and sizes taken
/// together as the Euclidean norm of theirs.
pub(crate) fn norm_of<T: Space>(parts: &[Form<T>]) -> Own {
    let euclid = |f: &dyn Fn(&Form<T>) -> f64| parts.iter().map(|x| f(x).powi(2)).sum::<f64>().sqrt();
    let dots = std::array::from_fn(|a| parts.iter().map(|x| x.value.dot(x.first[a])).sum());
    measured(euclid(&|x| x.value.size()), dots, euclid(&|x| x.rest), euclid(&|x| x.width()), euclid(&|x| x.moved), euclid(&|x| x.most))
}

impl Own {
    /// 1/(a − iγ), γ > 0 a constant, a complex quantity (`Form`): a drive's response per unit coupling at the detuning a.
    /// With D = a − iγ and δ the operand's change, 1/(D + δ) = 1/D − δ/D² + δ²/(D²(D + δ)): its first order −first/D² and
    /// its remainder R/|D|² + W²/(|D|² m), m = (min|a|² + γ²)^{1/2} the least |D| over the cell; it moves by W/(|D| m) and is
    /// at most 1/m.
    pub(crate) fn resolvent(self, gamma: f64) -> Form<C64> {
        let d = C64::new(self.value, -gamma);
        let (size, least, w) = (d.abs(), self.least_size().hypot(gamma), self.width());
        let slope = -(d * d).recip();
        Form { value: d.recip(), first: self.first.map(|x| slope * x), rest: self.rest / (size * size) + times(w * w, 1.0 / (size * size * least)), moved: times(w, 1.0 / (size * least)), most: 1.0 / least }
    }
}

/// A real quantity known within a band about a smooth function of the place (module documentation, # Bands): mid ± rad, each
/// its own form, `rad` not negative.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Band {
    pub(crate) mid: Own,
    pub(crate) rad: Own,
}

impl Band {
    /// A quantity of no width: its own form.
    pub(crate) fn exact(mid: Own) -> Band {
        Band { mid, rad: Own::constant(0.0) }
    }

    pub(crate) fn add(self, o: Band) -> Band {
        Band { mid: self.mid.add(o.mid), rad: self.rad.add(o.rad) }
    }

    pub(crate) fn sub(self, o: Band) -> Band {
        Band { mid: self.mid.sub(o.mid), rad: self.rad.add(o.rad) }
    }

    /// The least of its modulus over the cell's places and the parameters: |mid| − rad (a distance less what it does not
    /// know), in its own form.
    pub(crate) fn apart(&self) -> Own {
        self.mid.abs().sub(self.rad)
    }

    /// The largest of its modulus: |mid| + rad.
    pub(crate) fn most(&self) -> Own {
        self.mid.abs().add(self.rad)
    }

    /// Its bounds over the cell and the parameters.
    #[cfg(test)]
    pub(crate) fn upper(&self) -> f64 {
        self.mid.add(self.rad).upper()
    }

    #[cfg(test)]
    pub(crate) fn lower(&self) -> f64 {
        self.mid.sub(self.rad).lower()
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
            let unit = |z: C64| Form::constant(z);
            let u = product(&ra, &unit(C64::ONE), |x: f64, z: C64| z * x).add(product(&rb, &unit(C64::I), |x: f64, z: C64| z * x));
            let axis = |a: usize| Form::constant(std::array::from_fn::<f64, 3, _>(|b| if a == b { 1.0 } else { 0.0 }));
            let v = product(&ra, &axis(0), |x: f64, e: [f64; 3]| e.map(|y| y * x)).add(product(&rb, &axis(1), |x: f64, e: [f64; 3]| e.map(|y| y * x))).add(product(&fa.mul(fb).form(), &axis(2), |x: f64, e: [f64; 3]| e.map(|y| y * x)));
            let band = Band { mid: fa, rad: fb };
            let vector = move |r: [f64; 3]| [a(r), b(r), a(r) * b(r)];
            let length = |v: [f64; 3]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
            let others: Vec<(Own, Box<dyn Fn([f64; 3]) -> f64>)> = vec![
                (u.square(&Form::zero()).0, Box::new(move |r| a(r) * a(r) + b(r) * b(r))),
                (u.norm(), Box::new(move |r| a(r).hypot(b(r)))),
                (v.norm(), Box::new(move |r| length(vector(r)))),
                (v.by(&fa.sub(fb)).norm(), Box::new(move |r| (a(r) - b(r)).abs() * length(vector(r)))),
                (band.apart(), Box::new(move |r| a(r).abs() - b(r))),
                (band.most(), Box::new(move |r| a(r).abs() + b(r))),
                (fa.sub(fb).scale(3.0).resolvent(0.3).norm(), Box::new(move |r| 1.0 / (3.0 * (a(r) - b(r))).hypot(0.3))),
                (norm_of(&[u, fa.sub(fb).resolvent(0.2).turned(C64::new(0.3, -0.4))]), Box::new(move |r| (a(r).powi(2) + b(r).powi(2) + 0.25 / ((a(r) - b(r)).powi(2) + 0.04)).sqrt())),
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
}
