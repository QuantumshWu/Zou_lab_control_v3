//! Per-axis cubic tables (SPEC §2.3): a function of a few bounded coordinates, given by the model (the near light's
//! eliminated nonlinear part, model/tabulate.rs), tabulated on a tensor grid and interpolated per axis by the cubic Hermite
//! interpolant with Catmull–Rom slopes; each axis refined by halving its spacing, every node kept; the work bound checked
//! before the first grid is built -- how far an axis is refined is its caller's (model/tabulate.rs, # Refinement). Generic
//! over what is tabulated: `width` reals per node. A table is asked on its axes' ranges
//! alone: nothing extrapolates, and a bounded coordinate outside its range (or no number) is its caller's error, refused
//! aloud (`Axis::place`).
//!
//! # The interpolant
//! On a uniform axis of spacing h the interval [x_i, x_{i+1}] takes the cubic Hermite polynomial through f_i, f_{i+1}
//! with slopes m_i = (f_{i+1} − f_{i−1})/2h (Catmull–Rom), and at a bounded axis's ends the second-order one-sided slopes
//! (−3f₀ + 4f₁ − f₂)/2h and (3f_{n−1} − 4f_{n−2} + f_{n−3})/2h; a periodic axis (a phase) takes the central slopes
//! around its period. The interpolant is C¹ (the force it gives is continuous), its stencil four nodes per axis, 4^D
//! per point on D axes.
//!
//! Its error (field.md §1.6): the central slopes err by (h²/6) f‴, so inside an interval the value errs by
//! (h³ f‴/6) u(1 − u)(1 − 2u) + O(h⁴) (u ∈ [0, 1] the place in the interval): third order (`VALUE_ORDER`); its derivative
//! by (h² f‴/6)(1 − 6u + 6u²) + O(h³): second order (`GRADIENT_ORDER`). The one-sided end slopes err by −(h²/3) f‴, so a
//! bounded axis's end interval errs by −(h³ f‴/6) u(u − 1)(u − 2): the same orders, four times the mean error of an
//! inner interval. On D axes the errors of the axes add to leading order.
//!
//! # The error read of an axis (SPEC §2.3)
//! An axis whose node count is a nested refinement (bounded 2m − 1, periodic 2m, m ≥ 4) holds the coarser grid of
//! every other node. Comparing at a point the table's interpolant I with the one of that coarser grid along axis f,
//! I_f: with an error of order p, e(2h) ≈ 2^p e(h), so the table's own error along f is about (I_f − I)/(2^p − 1)
//! -- p = 3 for the value, 2 for the gradient (`Table::axis_error`). Pointwise this is a read, not a bound: the error's
//! shape differs between the two grids at a point (inside, u(1 − u)(1 − 2u) vanishes at the coarse midpoints).
//! Averaged over points spread through the intervals it is never below the actual error: inside an axis |I_f − I| is
//! 8 times the mean |e| of the value (the read conservative by 8/7) and 3.9 times that of the gradient (by 1.3); a
//! bounded axis's end intervals make it more conservative on coarse grids (1.46 and 1.48 on 13 nodes, for a constant
//! f‴). The reads at the atoms' coordinates, each as the error it makes of the atom's generator, are the model's
//! (model/tabulate.rs `Model::table_reads`).

use rayon::prelude::*;

/// The interpolant's order in its values: third (module documentation, # The interpolant).
pub const VALUE_ORDER: i32 = 3;
/// The interpolant's order in its gradient: second.
pub const GRADIENT_ORDER: i32 = 2;
/// A cubic's stencil: four nodes per axis.
const STENCIL: usize = 4;
/// The most axes of a table: a lookup holds its axes' stencils in place, and at 4^D rows a point a table of more axes
/// is beyond any work bound long before. A capacity, not derived: the axes' own rules (model/tabulate.rs, the cost of
/// an interpolation against the elimination's) stop a table before it.
pub const MOST_AXES: usize = 6;
/// The reals a lookup takes at a time through its rows, its value and its slopes together: 2 kB a row -- the four rows
/// and the sums stay in the first-level cache. A machine's number, not derived; it changes no value (the sums are the
/// same sums, stretch by stretch), the pace alone.
const BLOCK: usize = 256;

/// One axis of a table: its range [lo, hi] (a periodic axis's period is hi − lo, its node n the node 0 again), whether
/// it is periodic, and its node count.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Axis {
    pub lo: f64,
    pub hi: f64,
    pub periodic: bool,
    pub nodes: usize,
}

/// One axis's stencil at a coordinate: its four nodes (indices on the axis), their weights in the value and in its
/// derivative along the coordinate, and which of them the interpolant takes there (`live`: from the first of them up
/// to the one past the last) -- all four inside an axis; on a bounded axis's end intervals the three its one-sided
/// slope is made of, the fourth having no part in the interpolant (its weights are the zeros that fill the array).
#[derive(Clone, Copy, Debug)]
struct Stencil {
    nodes: [usize; STENCIL],
    value: [f64; STENCIL],
    slope: [f64; STENCIL],
    live: (usize, usize),
}

/// A point's place on a table's grid (`Table::place`): per axis its stencil there, and the axes' strides among the
/// nodes. Every read of the table at the point takes it (`Table::read`): a point is placed once, whatever is read
/// there.
#[derive(Clone, Copy, Debug)]
pub struct Place {
    stencils: [Stencil; MOST_AXES],
    strides: [usize; MOST_AXES],
}

impl Axis {
    /// The first grid of a coordinate's range: the fewest nodes whose grid of every other node still holds a cubic's
    /// stencil of four, so that the first table already reads every axis's error: 7 on a bounded axis (its coarser
    /// grid 4), 8 on a periodic one (4).
    pub fn first(lo: f64, hi: f64, periodic: bool) -> Axis {
        Axis { lo, hi, periodic, nodes: if periodic { 2 * STENCIL } else { 2 * STENCIL - 1 } }
    }
    /// The node spacing h: (hi − lo)/(n − 1) bounded, (hi − lo)/n periodic.
    pub fn spacing(&self) -> f64 {
        (self.hi - self.lo) / if self.periodic { self.nodes } else { self.nodes - 1 } as f64
    }
    /// The coordinate of node i: lo + i h, a bounded axis' last node hi itself (the sum's rounding could leave the range).
    pub fn node(&self, i: usize) -> f64 {
        if !self.periodic && i + 1 == self.nodes {
            return self.hi;
        }
        self.lo + i as f64 * self.spacing()
    }
    /// The axis with its spacing halved `times` times, every node kept: n → 2n − 1 bounded, 2n periodic (a count
    /// beyond the machine's saturates, for the work bound to refuse).
    pub fn halved(&self, times: u32) -> Axis {
        let doubled = |n: usize| n.checked_shl(times).filter(|m| m >> times == n).unwrap_or(usize::MAX);
        let nodes = if self.periodic { doubled(self.nodes) } else { doubled(self.nodes - 1).saturating_add(1) };
        Axis { nodes, ..*self }
    }
    /// Whether the grid of every other node is itself a grid of at least a stencil's nodes: the axis's error can be read.
    pub fn readable(&self) -> bool {
        if self.periodic {
            self.nodes % 2 == 0 && self.nodes / 2 >= STENCIL
        } else {
            self.nodes % 2 == 1 && (self.nodes + 1) / 2 >= STENCIL
        }
    }
    /// The grid of every other node.
    fn coarser(&self) -> Axis {
        Axis { nodes: if self.periodic { self.nodes / 2 } else { (self.nodes + 1) / 2 }, ..*self }
    }
    /// How many halvings make this axis `finer`, if `finer` is a nested refinement of it.
    pub fn halvings_to(&self, finer: &Axis) -> Option<u32> {
        if (finer.lo, finer.hi, finer.periodic) != (self.lo, self.hi, self.periodic) {
            return None;
        }
        (0..usize::BITS).find(|&k| self.halved(k).nodes >= finer.nodes).filter(|&k| self.halved(k).nodes == finer.nodes)
    }
    /// The interval holding s and the place u ∈ [0, 1] in it: a periodic coordinate modulo the period; a bounded one on
    /// its range (its last node in the last interval) -- outside it, or no number, the table has no value and the call
    /// is refused: the caller's coordinates are bounded by their construction (module documentation).
    fn place(&self, s: f64) -> (usize, f64) {
        assert!(if self.periodic { s.is_finite() } else { self.lo <= s && s <= self.hi }, "a table's axis on [{}, {}] asked at {s}", self.lo, self.hi);
        let (n, h) = (self.nodes, self.spacing());
        let (x, last) = if self.periodic { (periodic((s - self.lo) / h, n as f64), n - 1) } else { ((s - self.lo) / h, n - 2) };
        let i = (x.floor() as usize).min(last);
        (i, x - i as f64)
    }
    /// The stencil at s (module documentation, # The interpolant).
    fn stencil(&self, s: f64) -> Stencil {
        let (i, u) = self.place(s);
        self.weights(i, u)
    }
    /// The stencil at place u of interval i: the Hermite basis on the interval and its derivative, combined with the
    /// slopes' weights -- central inside, the one-sided ones on a bounded axis's end intervals, around the period on a
    /// periodic one.
    fn weights(&self, i: usize, u: f64) -> Stencil {
        let (n, h) = (self.nodes, self.spacing());
        // The Hermite basis (f_i, h m_i, f_{i+1}, h m_{i+1}) and its derivative in u.
        let basis = [2.0 * u * u * u - 3.0 * u * u + 1.0, u * u * u - 2.0 * u * u + u, -2.0 * u * u * u + 3.0 * u * u, u * u * u - u * u];
        let derivative = [6.0 * u * u - 6.0 * u, 3.0 * u * u - 4.0 * u + 1.0, -6.0 * u * u + 6.0 * u, 3.0 * u * u - 2.0 * u];
        // The values' weights on the stencil from the slopes h m: central (f_{j+1} − f_{j−1})/2, at a bounded axis's
        // first node (−3f₀ + 4f₁ − f₂)/2, at its last (3f_{n−1} − 4f_{n−2} + f_{n−3})/2.
        let first_interval = !self.periodic && i == 0;
        let last_interval = !self.periodic && i == n - 2;
        let combine = |[a, b, c, d]: [f64; 4]| {
            if first_interval {
                [a - 1.5 * b - 0.5 * d, 2.0 * b + c, -0.5 * b + 0.5 * d, 0.0]
            } else if last_interval {
                [0.0, -0.5 * b + 0.5 * d, a - 2.0 * d, 0.5 * b + c + 1.5 * d]
            } else {
                [-0.5 * b, a - 0.5 * d, c + 0.5 * b, 0.5 * d]
            }
        };
        let nodes = if self.periodic {
            std::array::from_fn(|j| (i + n + j - 1) % n)
        } else {
            let first = if first_interval { 0 } else if last_interval { n - STENCIL } else { i - 1 };
            std::array::from_fn(|j| first + j)
        };
        let live = if first_interval { (0, STENCIL - 1) } else if last_interval { (1, STENCIL) } else { (0, STENCIL) };
        Stencil { nodes, value: combine(basis), slope: combine(derivative).map(|x| x / h), live }
    }
}

/// x.rem_euclid(period) of a period > 0, the remainder taken without the library's division where x lies within a
/// period of the origin: there x % period is x itself, exactly (the remainder of a quotient that truncates to zero), and
/// the rest is `rem_euclid`'s own sum -- the same number whichever way.
fn periodic(x: f64, period: f64) -> f64 {
    if x >= 0.0 && x < period {
        x
    } else if x < 0.0 && x > -period {
        x + period
    } else {
        x.rem_euclid(period)
    }
}

/// A table of `width` reals per node on the tensor grid of its axes (row-major: the last axis's index varies fastest).
#[derive(Clone, Debug)]
pub struct Table {
    axes: Vec<Axis>,
    width: usize,
    values: Vec<f64>,
}

/// The tabulated function: f(s, out) writes the `width` values at the chart coordinates s.
pub type Function<'a> = &'a (dyn Fn(&[f64], &mut [f64]) + Sync);

impl Table {
    pub fn axes(&self) -> &[Axis] {
        &self.axes
    }
    pub fn width(&self) -> usize {
        self.width
    }
    /// The nodes of a grid of these axes: the count the work bound is checked on before any grid is built (saturating).
    pub fn nodes(axes: &[Axis]) -> usize {
        axes.iter().fold(1usize, |n, a| n.saturating_mul(a.nodes))
    }
    /// Node `node`'s index on each axis and its reals (the nodes in the grid's row-major order, `Table::nodes` of them).
    pub fn node(&self, node: usize) -> (Vec<usize>, &[f64]) {
        (Table::indices(&self.axes, node), &self.values[node * self.width..(node + 1) * self.width])
    }
    /// A node's index on each axis.
    fn indices(axes: &[Axis], mut node: usize) -> Vec<usize> {
        let mut out = vec![0; axes.len()];
        for (f, a) in axes.iter().enumerate().rev() {
            out[f] = node % a.nodes;
            node /= a.nodes;
        }
        out
    }
    /// The table of f on the grid of `axes` (each of at least a stencil's nodes, a range of positive length), its nodes
    /// evaluated in parallel. Err(the grid's nodes) when they are more than `most`: nothing is built.
    pub fn build(axes: Vec<Axis>, width: usize, most: usize, f: Function) -> Result<Table, usize> {
        let nodes = Table::nodes(&axes);
        if nodes > most {
            return Err(nodes);
        }
        Ok(Table::empty(axes, width).refine_from(None, f))
    }
    /// The nodes `refine` evaluates for `axes`: those of the finer grid not in this one.
    pub fn new_nodes(&self, axes: &[Axis]) -> usize {
        Table::nodes(axes).saturating_sub(Table::nodes(&self.axes))
    }
    /// This table on finer axes, each a nested refinement of this one's (`Axis::halved`): its nodes copied, only the
    /// new ones evaluated. How far to refine is the caller's: its rule bounds the axes it asks for.
    pub fn refine(&self, axes: Vec<Axis>, f: Function) -> Table {
        assert_eq!(axes.len(), self.axes.len(), "a refinement keeps the table's axes");
        Table::empty(axes, self.width).refine_from(Some(self), f)
    }
    /// A table of these axes with no values yet.
    fn empty(axes: Vec<Axis>, width: usize) -> Table {
        assert!(width > 0, "a table holds at least one value per node");
        assert!(axes.len() <= MOST_AXES, "a table of {} axes (at most {MOST_AXES})", axes.len());
        for a in &axes {
            assert!(a.nodes >= STENCIL && a.hi > a.lo, "an axis of {} nodes on [{}, {}]", a.nodes, a.lo, a.hi);
        }
        Table { axes, width, values: vec![] }
    }
    /// Fills this table's values: from `old`'s nodes where it has them (its axes nested in this one's), from f
    /// elsewhere, in parallel.
    fn refine_from(mut self, old: Option<&Table>, f: Function) -> Table {
        let steps: Vec<u32> = match old {
            Some(t) => t.axes.iter().zip(&self.axes).map(|(a, b)| a.halvings_to(b).expect("a refinement's axes are nested in the table's")).collect(),
            None => vec![],
        };
        let (axes, width) = (&self.axes, self.width);
        let mut values = vec![0.0; Table::nodes(axes) * width];
        values.par_chunks_mut(width).enumerate().for_each(|(node, out)| {
            let at = Table::indices(axes, node);
            if let Some(t) = old {
                if at.iter().zip(&steps).all(|(&i, &k)| i % (1 << k) == 0) {
                    let from = at.iter().zip(&steps).zip(&t.axes).fold(0, |n, ((&i, &k), a)| n * a.nodes + (i >> k));
                    out.copy_from_slice(&t.values[from * width..(from + 1) * width]);
                    return;
                }
            }
            let s: Vec<f64> = at.iter().zip(axes).map(|(&i, a)| a.node(i)).collect();
            f(&s, out);
        });
        self.values = values;
        self
    }
    /// The interpolant at s: `value` (width) and, when asked, `gradient` (axes × width, axis-major: ∂I/∂s_f); s on the
    /// axes' ranges (`Axis::place`).
    pub fn eval(&self, s: &[f64], value: &mut [f64], gradient: Option<&mut [f64]>) {
        self.read(&self.place(s), 0, None, value, gradient);
    }
    /// The place of s on the grid (`Place`): each axis's stencil there; s on the axes' ranges (`Axis::place`).
    pub fn place(&self, s: &[f64]) -> Place {
        self.placed(s, None)
    }
    /// The read of axis f's error at s (module documentation, # The error read of an axis): the value
    /// (I_f − I)/(2³ − 1) into `value` and the gradient (∇I_f − ∇I)/(2² − 1) into `gradient` (axes × width), I_f the
    /// interpolant of the grid of every other node along f. The axis must be `readable`.
    pub fn axis_error(&self, f: usize, s: &[f64], value: &mut [f64], gradient: &mut [f64]) {
        assert!(self.axes[f].readable(), "axis {f} of {} nodes has no coarser grid to read against", self.axes[f].nodes);
        let (w, d) = (self.width, self.axes.len());
        let (mut fine, mut slope) = (vec![0.0; w], vec![0.0; d * w]);
        self.read(&self.placed(s, None), 0, None, &mut fine, Some(&mut slope));
        self.read(&self.placed(s, Some(f)), 0, None, value, Some(gradient));
        let (p, q) = ((1 << VALUE_ORDER) - 1, (1 << GRADIENT_ORDER) - 1);
        for (x, y) in value.iter_mut().zip(&fine) {
            *x = (*x - y) / p as f64;
        }
        for (x, y) in gradient.iter_mut().zip(&slope) {
            *x = (*x - y) / q as f64;
        }
    }
    /// The place of s on the grid, axis `coarse` (if any) on its grid of every other node.
    fn placed(&self, s: &[f64], coarse: Option<usize>) -> Place {
        let d = self.axes.len();
        assert!(s.len() == d, "a table of {d} axes asked at {} coordinates", s.len());
        let mut stencils = [Stencil { nodes: [0; STENCIL], value: [0.0; STENCIL], slope: [0.0; STENCIL], live: (0, STENCIL) }; MOST_AXES];
        let mut strides = [1; MOST_AXES];
        for f in 0..d {
            stencils[f] = if coarse == Some(f) {
                let mut st = self.axes[f].coarser().stencil(s[f]);
                st.nodes = st.nodes.map(|i| 2 * i);
                st
            } else {
                self.axes[f].stencil(s[f])
            };
        }
        for f in (0..d.saturating_sub(1)).rev() {
            strides[f] = strides[f + 1] * self.axes[f + 1].nodes;
        }
        Place { stencils, strides }
    }
    /// The tensor-product interpolant at a place (`Table::place`) of the node's reals from `first` on, as many as
    /// `value` holds: `value` = `start` (where given: what a caller adds its constants to is made in the
    /// interpolation's own pass over the rows) + Σ over the 4^D stencil nodes of the product of the axes' weights
    /// times the node's values; and, when asked, `gradient` (axes × that many, axis-major: ∂I/∂s_f), whose axis f takes
    /// its slope weights in place of its value weights. Nothing is allocated and nothing is written before it is
    /// known: the last axis' nodes are one sum (`blend`) -- those its stencil takes (`Stencil::live`): the rows of a
    /// node without a part in the interpolant are not read --, and each stretch of `BLOCK` reals takes its value and
    /// its slopes together, so the rows pass through the cache once.
    pub fn read(&self, at: &Place, first: usize, start: Option<&[f64]>, value: &mut [f64], mut gradient: Option<&mut [f64]>) {
        let (w, d) = (value.len(), self.axes.len());
        assert!(first + w <= self.width && gradient.as_ref().is_none_or(|g| g.len() == d * w) && start.is_none_or(|b| b.len() == w));
        if d == 0 {
            // No axis: the one node's reals.
            let row = &self.values[first..first + w];
            match start {
                Some(base) => value.iter_mut().zip(base).zip(row).for_each(|((x, b), y)| *x = b + y),
                None => value.copy_from_slice(row),
            }
            return;
        }
        let Place { stencils, strides } = at;
        let last = &stencils[d - 1];
        let live = last.live.0..last.live.1;
        let mut at = 0;
        while at < w {
            let stretch = at..(at + BLOCK).min(w);
            // The combinations of the axes before the last; the last's nodes are one sum each.
            for combination in 0..STENCIL.pow(d as u32 - 1) {
                let digit = |f: usize| (combination / STENCIL.pow((d - 2 - f) as u32)) % STENCIL;
                let node: usize = (0..d - 1).map(|f| stencils[f].nodes[digit(f)] * strides[f]).sum();
                let rows: [&[f64]; STENCIL] = std::array::from_fn(|j| {
                    let row = (node + last.nodes[j]) * self.width + first;
                    &self.values[row + stretch.start..row + stretch.end]
                });
                // The weights of the axes before the last, axis `slope` (if any) by its slope weights.
                let before = |slope: Option<usize>| -> f64 { (0..d - 1).map(|f| if slope == Some(f) { stencils[f].slope[digit(f)] } else { stencils[f].value[digit(f)] }).product() };
                let onto = match (combination, start) {
                    (0, Some(base)) => Onto::Start(&base[stretch.clone()]),
                    (0, None) => Onto::Nothing,
                    _ => Onto::Itself,
                };
                blend(&mut value[stretch.clone()], onto, &rows[live.clone()], &last.value.map(|x| x * before(None))[live.clone()]);
                if let Some(g) = gradient.as_deref_mut() {
                    let onto = if combination == 0 { Onto::Nothing } else { Onto::Itself };
                    for f in 0..d {
                        let weights = if f == d - 1 { last.slope.map(|x| x * before(None)) } else { last.value.map(|x| x * before(Some(f))) };
                        blend(&mut g[f * w + stretch.start..f * w + stretch.end], onto, &rows[live.clone()], &weights[live.clone()]);
                    }
                }
            }
            at = stretch.end;
        }
    }
}

/// What a sum of rows is added to (`blend`): nothing (the sum alone), a start, or what is there.
#[derive(Clone, Copy)]
enum Onto<'a> {
    Nothing,
    Start(&'a [f64]),
    Itself,
}

/// out ← (what `onto` says) + Σ_j w_j rows_j over a stencil's live nodes, four or three (`summed`).
fn blend(out: &mut [f64], onto: Onto, rows: &[&[f64]], w: &[f64]) {
    match (<[&[f64]; STENCIL]>::try_from(rows), <[f64; STENCIL]>::try_from(w)) {
        (Ok(rows), Ok(w)) => summed(out, onto, rows, w),
        _ => summed::<{ STENCIL - 1 }>(out, onto, rows.try_into().expect("a stencil's three or four nodes"), w.try_into().expect("a weight a node")),
    }
}

/// out ← (what `onto` says) + Σ_j w_j rows_j, element by element, the sum from the last row in: one pass over the
/// rows, no element depending on another -- the compiler's vector loop, a fused multiply-add a row and real
/// (linalg.rs: the build's instructions).
fn summed<const N: usize>(out: &mut [f64], onto: Onto, rows: [&[f64]; N], w: [f64; N]) {
    let n = out.len();
    let rows = rows.map(|row| &row[..n]);
    let sum = |i: usize, onto: f64| (0..N).rev().fold(onto, |sum, j| w[j].mul_add(rows[j][i], sum));
    match onto {
        Onto::Nothing => (0..n).for_each(|i| out[i] = sum(i, 0.0)),
        Onto::Start(base) => {
            let base = &base[..n];
            (0..n).for_each(|i| out[i] = sum(i, base[i]))
        }
        Onto::Itself => (0..n).for_each(|i| out[i] = sum(i, out[i])),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::{Purpose, Stream};
    use crate::units::rounding;
    use std::f64::consts::PI;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// The table of a function of one axis.
    fn one(axis: Axis, f: &(dyn Fn(f64) -> f64 + Sync)) -> Table {
        Table::build(vec![axis], 1, usize::MAX, &|s: &[f64], out: &mut [f64]| out[0] = f(s[0])).unwrap()
    }
    /// The interpolant's value and gradient at s.
    fn at(t: &Table, s: &[f64]) -> (f64, Vec<f64>) {
        let (mut v, mut g) = ([0.0], vec![0.0; t.axes().len()]);
        t.eval(s, &mut v, Some(&mut g));
        (v[0], g)
    }
    /// A periodic coordinate's remainder is `rem_euclid`'s bit for bit, within a period of the origin and beyond it.
    #[test]
    fn the_periodic_remainder_is_rem_euclid_s() {
        let mut s = Stream::new(31, Purpose::Initial, 0, 0);
        for k in 0..200_000 {
            let period = [16.0, 24.0, 64.0, 2.0 * PI][k % 4];
            let x = match k % 5 {
                0 => 0.0,
                1 => -0.0,
                2 => (s.uniform() - 0.5) * 2.0 * period,
                3 => (s.uniform() - 0.5) * 40.0 * period,
                _ => [period, -period, period * (1.0 - f64::EPSILON), -period * (1.0 - f64::EPSILON)][k % 4],
            };
            assert_eq!(periodic(x, period).to_bits(), x.rem_euclid(period).to_bits(), "{x} modulo {period}");
        }
    }

    /// The bounded axis [0, 1.3] and the periodic [−π, π) with smooth functions on them and their derivatives.
    fn cases() -> [(Axis, fn(f64) -> f64, fn(f64) -> f64); 2] {
        [
            (Axis { lo: 0.0, hi: 1.3, periodic: false, nodes: 7 }, |x| (3.0 * x).sin() + x * x * x, |x| 3.0 * (3.0 * x).cos() + 3.0 * x * x),
            (Axis { lo: -PI, hi: PI, periodic: true, nodes: 8 }, |x| x.cos() + 0.3 * (2.0 * x).sin() + 0.1 * (3.0 * x + 1.0).cos(), |x| {
                -x.sin() + 0.6 * (2.0 * x).cos() - 0.3 * (3.0 * x + 1.0).sin()
            }),
        ]
    }

    /// Quadratics are reproduced exactly, value and slope: the central and the one-sided slopes are exact for them;
    /// to the rounding of the stencil's sums, γ₁₆ of the values (of the values over h for the slope).
    #[test]
    fn the_interpolant_reproduces_quadratics() {
        let f = |x: f64| 0.3 - 1.7 * x + 2.2 * x * x;
        let axis = Axis { lo: -0.5, hi: 1.3, periodic: false, nodes: 9 };
        let t = one(axis, &f);
        let scale = 5.0;
        for k in 0..=200 {
            let x = -0.5 + 1.8 * k as f64 / 200.0;
            let (v, g) = at(&t, &[x]);
            assert!((v - f(x)).abs() <= rounding(16) * scale, "{x}: {v} {}", f(x));
            assert!((g[0] - (-1.7 + 4.4 * x)).abs() <= rounding(16) * scale / axis.spacing(), "{x}: {} {}", g[0], -1.7 + 4.4 * x);
        }
    }

    /// Third order in the value and second in the gradient (module documentation, # The interpolant), on a bounded and
    /// a periodic axis: over the last of three halvings the largest error over 2000 points falls by 2^p, its measured
    /// order log₂(ratio) within ½ of p (the next order's share at these spacings).
    #[test]
    fn the_interpolant_is_third_order_in_value_and_second_in_gradient() {
        for (axis, f, df) in cases() {
            let mut errors = vec![];
            for k in 0..4 {
                let a = axis.halved(k);
                let t = one(a, &f);
                let (mut ev, mut eg) = (0.0_f64, 0.0_f64);
                for j in 0..2000 {
                    let x = a.lo + (a.hi - a.lo) * (j as f64 + 0.37) / 2000.0;
                    let (v, g) = at(&t, &[x]);
                    ev = ev.max((v - f(x)).abs());
                    eg = eg.max((g[0] - df(x)).abs());
                }
                errors.push((ev, eg));
            }
            let (coarse, fine) = (errors[2], errors[3]);
            let (pv, pg) = ((coarse.0 / fine.0).log2(), (coarse.1 / fine.1).log2());
            assert!((pv - VALUE_ORDER as f64).abs() <= 0.5, "{axis:?}: value order {pv} ({errors:?})");
            assert!((pg - GRADIENT_ORDER as f64).abs() <= 0.5, "{axis:?}: gradient order {pg} ({errors:?})");
        }
    }

    /// C⁰ and C¹: at every node the stencil of the interval before it at u = 1 and the one after it at u = 0 are the
    /// same functional of the node values, in the value and in the slope (to the rounding of the weights), on bounded
    /// axes of 4 to 13 nodes (their end intervals included) and periodic ones, whose seam (node 0 ≡ node n) is a node
    /// like the others; and a periodic interpolant repeats with its period.
    #[test]
    fn the_interpolant_is_c1_across_its_nodes_and_its_seam() {
        let functional = |st: &Stencil, slope: bool| {
            let mut out = std::collections::BTreeMap::new();
            for (j, &node) in st.nodes.iter().enumerate() {
                *out.entry(node).or_insert(0.0) += if slope { st.slope[j] } else { st.value[j] };
            }
            out
        };
        let same = |a: &std::collections::BTreeMap<usize, f64>, b: &std::collections::BTreeMap<usize, f64>, scale: f64| {
            a.keys().chain(b.keys()).all(|k| (a.get(k).unwrap_or(&0.0) - b.get(k).unwrap_or(&0.0)).abs() <= rounding(16) * scale)
        };
        for axis in [Axis { lo: 0.0, hi: 1.3, periodic: false, nodes: 4 }, Axis { lo: 0.0, hi: 1.3, periodic: false, nodes: 13 }, Axis { lo: -PI, hi: PI, periodic: true, nodes: 8 }] {
            let intervals = if axis.periodic { axis.nodes } else { axis.nodes - 1 };
            for k in 1..=intervals {
                if !axis.periodic && k == intervals {
                    break;
                }
                let (before, after) = (axis.weights(k - 1, 1.0), axis.weights(k % axis.nodes, 0.0));
                for slope in [false, true] {
                    let scale = if slope { 1.0 / axis.spacing() } else { 1.0 };
                    assert!(same(&functional(&before, slope), &functional(&after, slope), scale), "{axis:?} node {k}, slope {slope}");
                }
            }
        }
        let [_, (periodic, p, _)] = cases();
        let t = one(periodic.halved(1), &p);
        for x in [-2.0, 0.3, 3.0] {
            let ((v, g), (w, h)) = (at(&t, &[x]), at(&t, &[x + 2.0 * PI]));
            assert!((v - w).abs() <= rounding(16) * 3.0 + g[0].abs() * 8.0 * f64::EPSILON && (g[0] - h[0]).abs() <= rounding(64) * 3.0 / t.axes()[0].spacing(), "{x}");
        }
    }

    /// The work bound before the first grid: a build over its bound returns the nodes it would take and calls the
    /// function on none; a refinement evaluates exactly its new nodes and equals the table built at once on the finer
    /// axes (its nodes' coordinates are the same numbers); the first axes are readable and nested.
    #[test]
    fn the_work_bound_comes_first_and_refinement_reuses_nodes() {
        let calls = AtomicUsize::new(0);
        let f = |s: &[f64], out: &mut [f64]| {
            calls.fetch_add(1, Ordering::Relaxed);
            out[0] = (2.0 * s[0]).sin() * s[1].cos();
            out[1] = s[0] * s[0] + (s[1] + 0.5).sin();
        };
        let axes = vec![Axis::first(0.0, 1.0, false), Axis::first(-PI, PI, true)];
        assert_eq!((axes[0].nodes, axes[1].nodes, Table::nodes(&axes)), (7, 8, 56));
        assert!(axes.iter().all(Axis::readable) && !Axis { nodes: 6, ..axes[0] }.readable());
        assert_eq!(Table::build(axes.clone(), 2, 55, &f).unwrap_err(), 56);
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        let coarse = Table::build(axes.clone(), 2, 56, &f).unwrap();
        let finer = vec![axes[0].halved(2), axes[1].halved(1)];
        assert_eq!((finer[0].nodes, finer[1].nodes), (25, 16));
        let fresh = coarse.new_nodes(&finer);
        assert_eq!(fresh, 25 * 16 - 56);
        calls.store(0, Ordering::Relaxed);
        let refined = coarse.refine(finer.clone(), &f);
        assert_eq!(calls.load(Ordering::Relaxed), fresh);
        let direct = Table::build(finer, 2, usize::MAX, &f).unwrap();
        assert_eq!(refined.values, direct.values);
    }

    /// A table has no value outside a bounded axis' range: the call is refused, not answered with the edge's value.
    #[test]
    #[should_panic(expected = "a table's axis on [0, 1.3] asked at 1.3000001")]
    fn a_bounded_axis_is_asked_on_its_range_alone() {
        let [(bounded, f, _), _] = cases();
        at(&one(bounded, &f), &[1.3000001]);
    }

    /// The error read of an axis (module documentation, # The error read of an axis), averaged over 20000 points spread
    /// through the intervals, against the table's actual error: never below it; inside a fine axis (a periodic one, no
    /// end intervals, 256 nodes) conservative by the derived 8/7 in the value and 3.9/3 in the gradient, within a tenth
    /// (the next order's share at h = 2π/256); on the coarsest readable bounded axis (13 nodes) within a factor two above
    /// it (its end intervals' one-sided slopes: 1.46 and 1.48 for a constant third derivative). On two axes the read of
    /// a product g(x)h(φ) along x is the one-axis read of g times h's interpolant (the tensor product's own algebra).
    #[test]
    fn an_axis_s_error_read_matches_its_actual_error() {
        let ratios = |axis: Axis, f: fn(f64) -> f64, df: fn(f64) -> f64| {
            let t = one(axis, &f);
            let mut stream = Stream::new(5, Purpose::Initial, axis.nodes as u64, 0);
            let (mut read, mut actual, mut read_slope, mut actual_slope) = (0.0, 0.0, 0.0, 0.0);
            let (mut v, mut g) = ([0.0], [0.0]);
            for _ in 0..20000 {
                let x = axis.lo + (axis.hi - axis.lo) * stream.uniform();
                let (value, slope) = at(&t, &[x]);
                actual += (value - f(x)).abs();
                actual_slope += (slope[0] - df(x)).abs();
                t.axis_error(0, &[x], &mut v, &mut g);
                read += v[0].abs();
                read_slope += g[0].abs();
            }
            (read / actual, read_slope / actual_slope)
        };
        let [(bounded, f, df), (periodic, p, dp)] = cases();
        let (value, slope) = ratios(periodic.halved(5), p, dp);
        assert!((value / (8.0 / 7.0) - 1.0).abs() <= 0.1 && (slope / (3.9 / 3.0) - 1.0).abs() <= 0.1, "inside: {value} {slope}");
        let (value, slope) = ratios(bounded.halved(1), f, df);
        assert!((1.0..=2.0).contains(&value) && (1.0..=2.0).contains(&slope), "coarsest: {value} {slope}");
        let axes = vec![bounded.halved(1), periodic.halved(1)];
        let t = Table::build(axes.clone(), 1, usize::MAX, &|s: &[f64], out: &mut [f64]| out[0] = f(s[0]) * p(s[1])).unwrap();
        let (gx, hp) = (one(axes[0], &f), one(axes[1], &p));
        let (mut v, mut g, mut v1, mut g1) = ([0.0], [0.0; 2], [0.0], [0.0]);
        for s in [[0.05, -3.0], [0.71, 0.2], [1.29, 2.9]] {
            t.axis_error(0, &s, &mut v, &mut g);
            gx.axis_error(0, &s[..1], &mut v1, &mut g1);
            let (h, _) = at(&hp, &s[1..]);
            // The two reads differ by the rounding of the interpolants they difference (|f p| ≤ 3), over h for the slope.
            let within = rounding(64) * 3.0;
            assert!((v[0] - v1[0] * h).abs() <= within && (g[0] - g1[0] * h).abs() <= within / axes[0].spacing(), "{s:?}: {} {}", v[0], v1[0] * h);
        }
    }
}
