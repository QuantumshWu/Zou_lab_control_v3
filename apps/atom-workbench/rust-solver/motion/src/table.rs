//! Per-axis cubic tables (SPEC §2.3): a function of a few chart coordinates, given by the model (the near light's
//! eliminated nonlinear part), tabulated on a tensor grid and interpolated per axis by the cubic Hermite interpolant
//! with Catmull–Rom slopes; each axis refined by its own measured error; the work bound checked before any grid is built.
//! Generic over what is tabulated: `width` reals per node.
//!
//! # The interpolant
//! On a uniform axis of spacing h the interval [x_i, x_{i+1}] takes the cubic Hermite polynomial through f_i, f_{i+1}
//! with slopes m_i = (f_{i+1} − f_{i−1})/2h (Catmull–Rom), and at a bounded axis's ends the second-order one-sided slopes
//! (−3f₀ + 4f₁ − f₂)/2h and (3f_{n−1} − 4f_{n−2} + f_{n−3})/2h; a periodic axis (a phase) takes the central slopes
//! around its period. The interpolant is C¹ (the force it gives is continuous), its stencil four nodes per axis, 4^D
//! per point on D axes.
//!
//! Its error (field.md §1.6): the slopes err by (h²/6) f‴, so inside an interval the value errs by
//! (h³ f‴/6) u(1 − u)(1 − 2u) + O(h⁴) (u ∈ [0, 1] the place in the interval): third order (`VALUE_ORDER`); its derivative
//! by (h² f‴/6)(1 − 6u + 6u²) + O(h³): second order (`GRADIENT_ORDER`). On D axes the errors of the axes add to leading
//! order.
//!
//! # The error read of an axis (SPEC §2.3)
//! An axis whose node count is a nested refinement (bounded 2m − 1, periodic 2m, m ≥ 4) holds the coarser grid of
//! every other node. Comparing at a point the table's interpolant I with the one of that coarser grid along axis f,
//! I_f: with an error of order p, e(2h) ≈ 2^p e(h), so the table's own error along f is about (I_f − I)/(2^p − 1)
//! -- p = 3 for the value, 2 for the gradient (`Table::axis_error`). Pointwise this is a read, not a bound: the error's
//! shape u(1 − u)(1 − 2u) differs between the two grids at a point (it vanishes at the coarse midpoints); averaged
//! over points spread through the intervals, |I_f − I| is 8 times the mean |e| of the value (2^p − 1 = 7: the read is
//! conservative by 8/7) and 3.9 times that of the gradient (3: by 1.3). The reads at the atom's chart coordinates
//! (the pilot's states), weighted by each term's weight in the generator, are the model's and the decision's (SPEC
//! §5.1); an axis beyond its share is refined by `Axis::halvings` at the order of what exceeds.

/// The interpolant's order in its values: third (module documentation, # The interpolant).
pub const VALUE_ORDER: i32 = 3;
/// The interpolant's order in its gradient: second.
pub const GRADIENT_ORDER: i32 = 2;
/// A cubic's stencil: four nodes per axis.
const STENCIL: usize = 4;

/// One axis of a table: its range [lo, hi] (a periodic axis's period is hi − lo, its node n the node 0 again), whether
/// it is periodic, and its node count.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Axis {
    pub lo: f64,
    pub hi: f64,
    pub periodic: bool,
    pub nodes: usize,
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
    /// The coordinate of node i.
    pub fn node(&self, i: usize) -> f64 {
        self.lo + i as f64 * self.spacing()
    }
    /// The axis with its spacing halved `times` times, every node kept: n → 2n − 1 bounded, 2n periodic.
    pub fn halved(&self, times: u32) -> Axis {
        let nodes = if self.periodic { self.nodes << times } else { ((self.nodes - 1) << times) + 1 };
        Axis { nodes, ..*self }
    }
    /// The halvings that bring an error e read at an order p down to a share: the least k ≥ 0 with e 2^{−kp} ≤ share,
    /// k = ⌈log₂(e/share)/p⌉ (SPEC §5.1: refined by the measured ratio at the layer's order).
    pub fn halvings(error: f64, share: f64, order: i32) -> u32 {
        let _ = (error, share, order);
        todo!("phase 2")
    }
    /// Whether the grid of every other node is itself a grid of at least a stencil's nodes: the axis's error can be read.
    pub fn readable(&self) -> bool {
        if self.periodic {
            self.nodes % 2 == 0 && self.nodes / 2 >= STENCIL
        } else {
            self.nodes % 2 == 1 && (self.nodes + 1) / 2 >= STENCIL
        }
    }
}

/// A table of `width` reals per node on the tensor grid of its axes.
#[derive(Clone, Debug)]
pub struct Table {
    axes: Vec<Axis>,
    width: usize,
    values: Vec<f64>,
}

impl Table {
    pub fn axes(&self) -> &[Axis] {
        &self.axes
    }
    pub fn width(&self) -> usize {
        self.width
    }
    /// The nodes of a grid of these axes: the count the work bound is checked on before any grid is built.
    pub fn nodes(axes: &[Axis]) -> usize {
        axes.iter().map(|a| a.nodes).product()
    }
    /// The table of f on the grid of `axes`: f(s, out) writes the `width` values at the chart coordinates s; the nodes
    /// are evaluated in parallel. Err(the grid's nodes) when they are more than `most`: nothing is built.
    pub fn build(axes: Vec<Axis>, width: usize, most: usize, f: &(dyn Fn(&[f64], &mut [f64]) + Sync)) -> Result<Table, usize> {
        let _ = (axes, width, most, f);
        todo!("phase 2")
    }
    /// The nodes `refine` would evaluate for `axes`: those of the finer grid not in this one.
    pub fn new_nodes(&self, axes: &[Axis]) -> usize {
        Table::nodes(axes) - Table::nodes(&self.axes)
    }
    /// This table on finer axes, each a nested refinement of this one's (`Axis::halved`): its nodes copied, only the new
    /// ones evaluated. Err(the new nodes) when they are more than `most`: nothing is built.
    pub fn refine(&self, axes: Vec<Axis>, most: usize, f: &(dyn Fn(&[f64], &mut [f64]) + Sync)) -> Result<Table, usize> {
        let _ = (axes, most, f);
        todo!("phase 2")
    }
    /// The interpolant at s: `value` (width) and, when asked, `gradient` (axes × width, axis-major: ∂I/∂s_f). A bounded
    /// coordinate is taken within its range, a periodic one modulo its period.
    pub fn eval(&self, s: &[f64], value: &mut [f64], gradient: Option<&mut [f64]>) {
        let _ = (s, value, gradient);
        todo!("phase 2")
    }
    /// The read of axis f's error at s (module documentation, # The error read of an axis): the value
    /// (I_f − I)/(2³ − 1) into `value` and the gradient (∇I_f − ∇I)/(2² − 1) into `gradient` (axes × width), I_f the
    /// interpolant of the grid of every other node along f. The axis must be `readable`.
    pub fn axis_error(&self, f: usize, s: &[f64], value: &mut [f64], gradient: &mut [f64]) {
        let _ = (f, s, value, gradient);
        todo!("phase 2")
    }
}
