//! Matplotlib's affine transform and bbox arithmetic, bit for bit.
//!
//! * `np.dot` of two 3x3 matrices (what every composite's `get_matrix`
//!   is) runs through OpenBLAS's dgemm, whose kernel accumulates each entry
//!   as an FMA chain `fma(a2, b2, fma(a1, b1, a0 * b0))` -- measured on the
//!   oracle machine: 20000/20000 random products, against 11833/20000 for the
//!   unfused sum. [`M3::dot`] is that chain.
//! * `affine_transform` (C++ `_path.h`), used by every `Transform.transform`
//!   of points, is unfused: `(sx * x + shx * y) + tx` (20000/20000).
//! * Boxes are `[x0, y0, x1, y1]`; `bounds` is `(x0, y0, x1 - x0, y1 - y0)`.
use crate::ir::{BboxNode, Node, NodeId, TransformNode};
use crate::{invalid, unsupported, Result};

/// A full 3x3 matrix, row major, as numpy holds it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct M3(pub [[f64; 3]; 3]);

impl M3 {
    pub const IDENTITY: M3 = M3([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);

    /// `[[a, b, c], [d, e, f], [0, 0, 1]]`.
    pub fn from6(m: [f64; 6]) -> M3 { M3([[m[0], m[1], m[2]], [m[3], m[4], m[5]], [0.0, 0.0, 1.0]]) }

    pub fn to6(&self) -> [f64; 6] {
        let m = &self.0;
        [m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2]]
    }

    /// `np.dot(self, other)`.
    pub fn dot(&self, other: &M3) -> M3 {
        let (a, b) = (&self.0, &other.0);
        let mut c = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                c[i][j] = a[i][2].mul_add(b[2][j], a[i][1].mul_add(b[1][j], a[i][0] * b[0][j]));
            }
        }
        M3(c)
    }

    /// `affine_transform` of one point.
    #[inline]
    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        let m = &self.0;
        (m[0][0] * x + m[0][1] * y + m[0][2], m[1][0] * x + m[1][1] * y + m[1][2])
    }

    /// `Affine2D.scale(sx, sy)` in place (element-wise, as matplotlib does it).
    pub fn scale(mut self, sx: f64, sy: f64) -> M3 {
        for v in &mut self.0[0] { *v *= sx; }
        for v in &mut self.0[1] { *v *= sy; }
        self
    }

    /// `Affine2D.translate(tx, ty)` in place.
    pub fn translate(mut self, tx: f64, ty: f64) -> M3 {
        self.0[0][2] += tx;
        self.0[1][2] += ty;
        self
    }

    /// `Affine2D.rotate(theta)` in place (radians), with matplotlib's
    /// scalar formulas: new = [[a, -b], [b, a]] @ old.
    pub fn rotate(mut self, theta: f64) -> M3 {
        let a = theta.cos();
        let b = theta.sin();
        let m = &mut self.0;
        let (xx, xy, x0) = (m[0][0], m[0][1], m[0][2]);
        let (yx, yy, y0) = (m[1][0], m[1][1], m[1][2]);
        m[0][0] = a * xx - b * yx;
        m[0][1] = a * xy - b * yy;
        m[0][2] = a * x0 - b * y0;
        m[1][0] = b * xx + a * yx;
        m[1][1] = b * xy + a * yy;
        m[1][2] = b * x0 + a * y0;
        self
    }

    /// `Affine2D.rotate_deg(degrees)`: `rotate(math.radians(degrees))`.
    pub fn rotate_deg(self, degrees: f64) -> M3 { self.rotate(degrees.to_radians()) }
}

/// A box by its points `[x0, y0, x1, y1]`.
pub type Points = [f64; 4];

pub fn bounds(p: &Points) -> (f64, f64, f64, f64) { (p[0], p[1], p[2] - p[0], p[3] - p[1]) }
pub fn width(p: &Points) -> f64 { p[2] - p[0] }
pub fn height(p: &Points) -> f64 { p[3] - p[1] }

/// Python's `min` over a sequence: the FIRST smallest element.
fn py_min(values: &[f64]) -> f64 {
    let mut best = values[0];
    for &v in &values[1..] { if v < best { best = v; } }
    best
}

fn py_max(values: &[f64]) -> f64 {
    let mut best = values[0];
    for &v in &values[1..] { if v > best { best = v; } }
    best
}

/// `TransformedBbox.get_points` for an affine transform: the four corners
/// transformed, then min/max with the input's orientation kept.
pub fn transformed_points(m: &M3, p: &Points) -> Points {
    let corners = [(p[0], p[1]), (p[2], p[1]), (p[0], p[3]), (p[2], p[3])];
    let mut xs = [0.0; 4];
    let mut ys = [0.0; 4];
    for (k, &(x, y)) in corners.iter().enumerate() {
        let (tx, ty) = m.apply(x, y);
        xs[k] = tx;
        ys[k] = ty;
    }
    let (mut x0, mut x1) = (py_min(&xs), py_max(&xs));
    if p[0] > p[2] { std::mem::swap(&mut x0, &mut x1); }
    let (mut y0, mut y1) = (py_min(&ys), py_max(&ys));
    if p[1] > p[3] { std::mem::swap(&mut y0, &mut y1); }
    [x0, y0, x1, y1]
}

/// `Bbox.transformed(transform)`: ll, ul, lr transformed; `[ll, [lr.x, ul.y]]`.
pub fn bbox_transformed(m: &M3, p: &Points) -> Points {
    let ll = m.apply(p[0], p[1]);
    let ul = m.apply(p[0], p[3]);
    let lr = m.apply(p[2], p[1]);
    [ll.0, ll.1, lr.0, ul.1]
}

/// `Bbox.union(bboxes)`.
pub fn union(boxes: &[Points]) -> Points {
    let mut x0 = f64::INFINITY;
    let mut y0 = f64::INFINITY;
    let mut x1 = f64::NEG_INFINITY;
    let mut y1 = f64::NEG_INFINITY;
    // np.min(np.minimum.reduce)... matplotlib: x0 = np.min([bbox.xmin for bbox in bboxes]) etc.
    let xmins: Vec<f64> = boxes.iter().map(|b| b[0].min(b[2])).collect();
    let xmaxs: Vec<f64> = boxes.iter().map(|b| b[0].max(b[2])).collect();
    let ymins: Vec<f64> = boxes.iter().map(|b| b[1].min(b[3])).collect();
    let ymaxs: Vec<f64> = boxes.iter().map(|b| b[1].max(b[3])).collect();
    for v in xmins { x0 = np_min2(x0, v); }
    for v in xmaxs { x1 = np_max2(x1, v); }
    for v in ymins { y0 = np_min2(y0, v); }
    for v in ymaxs { y1 = np_max2(y1, v); }
    [x0, y0, x1, y1]
}

fn np_min2(a: f64, b: f64) -> f64 { if a.is_nan() || b.is_nan() { f64::NAN } else if b < a { b } else { a } }
fn np_max2(a: f64, b: f64) -> f64 { if a.is_nan() || b.is_nan() { f64::NAN } else if b > a { b } else { a } }

#[derive(Clone, Copy, Debug)]
enum Value { Matrix(M3), Points(Points) }

/// Evaluates the IR's transform/bbox DAG against the live boxes, caching
/// every node until a live box changes.
///
/// Affine nodes are matrices. The non-affine ones (a log scale, zlc's
/// piecewise coordinate scale) follow matplotlib's split: `transform(p)` is
/// `get_affine()` applied to `transform_non_affine(p)`, with the composite
/// and blended rules of `CompositeGenericTransform`/`BlendedGenericTransform`.
pub struct Evaluator<'a> {
    nodes: &'a [Node],
    pub dpi: f64,
    pub size_inches: [f64; 2],
    /// Active position per axes.
    pub positions: Vec<Points>,
    pub view_lims: Vec<Points>,
    cache: Vec<Option<Value>>,
}

impl<'a> Evaluator<'a> {
    pub fn new(figure: &'a crate::ir::Figure) -> Self {
        Evaluator {
            nodes: &figure.nodes,
            dpi: figure.dpi,
            size_inches: figure.size_inches,
            positions: figure.axes.iter().map(|a| a.position_active).collect(),
            view_lims: figure.axes.iter().map(|a| a.view_lim).collect(),
            cache: vec![None; figure.nodes.len()],
        }
    }

    /// Set an axes' active position (`_set_position(pos, 'active')`).
    pub fn set_position(&mut self, axes: usize, p: Points) {
        if self.positions[axes].map(f64::to_bits) != p.map(f64::to_bits) {
            self.positions[axes] = p;
            self.cache.iter_mut().for_each(|v| *v = None);
        }
    }

    fn node(&self, id: NodeId) -> Result<&'a Node> {
        match self.nodes.get(id as usize) {
            Some(node) => Ok(node),
            None => invalid(format!("node {id} out of range")),
        }
    }

    /// `transform.is_affine`.
    pub fn is_affine(&self, id: NodeId) -> Result<bool> {
        Ok(match self.node(id)? {
            Node::Transform(t) => match t {
                TransformNode::Log { .. } | TransformNode::Piecewise { .. } => false,
                TransformNode::Composite { a, b, .. } => self.is_affine(*a)? && self.is_affine(*b)?,
                TransformNode::Blended { x, y, .. } => self.is_affine(*x)? && self.is_affine(*y)?,
                _ => true,
            },
            Node::Bbox(_) => return invalid(format!("node {id} is a bbox, not a transform")),
        })
    }

    /// `transform.input_dims`.
    fn input_dims(&self, id: NodeId) -> Result<u8> {
        Ok(match self.node(id)? {
            Node::Transform(TransformNode::Log { .. } | TransformNode::Piecewise { .. }) => 1,
            Node::Transform(TransformNode::Composite { a, .. }) => self.input_dims(*a)?,
            _ => 2,
        })
    }

    /// The matrix of an AFFINE transform node; a non-affine one is refused.
    pub fn matrix(&mut self, id: NodeId) -> Result<M3> {
        if !self.is_affine(id)? {
            return unsupported(format!("non-affine transform (node {id}) where an affine one is needed"));
        }
        self.affine(id)
    }

    /// `get_affine().get_matrix()` of a transform node.
    pub fn affine(&mut self, id: NodeId) -> Result<M3> {
        if let Some(Value::Matrix(m)) = self.cache[id as usize] { return Ok(m); }
        let m = match self.node(id)? {
            Node::Transform(t) => match t {
                TransformNode::Identity {} | TransformNode::Log { .. } | TransformNode::Piecewise { .. } => M3::IDENTITY,
                TransformNode::DpiScale {} => M3::IDENTITY.scale(self.dpi, self.dpi),
                TransformNode::Affine { m } => M3::from6(*m),
                TransformNode::Composite { a, b, .. } => {
                    let (a, b) = (*a, *b);
                    if !self.is_affine(b)? {
                        self.affine(b)?
                    } else {
                        let ma = self.affine(a)?;
                        let mb = self.affine(b)?;
                        mb.dot(&ma)
                    }
                }
                TransformNode::Blended { x, y, x_eq_y, .. } => {
                    let (x, y, x_eq_y) = (*x, *y, *x_eq_y);
                    if x_eq_y {
                        self.affine(x)?
                    } else {
                        let mx = self.affine(x)?;
                        let my = self.affine(y)?;
                        M3([mx.0[0], my.0[1], [0.0, 0.0, 1.0]])
                    }
                }
                TransformNode::BboxTo { bbox } => {
                    let (l, b, w, h) = bounds(&self.points(*bbox)?);
                    M3([[w, 0.0, l], [0.0, h, b], [0.0, 0.0, 1.0]])
                }
                TransformNode::BboxFrom { bbox } => {
                    let (l, b, w, h) = bounds(&self.points(*bbox)?);
                    let xs = 1.0 / w;
                    let ys = 1.0 / h;
                    M3([[xs, 0.0, -l * xs], [0.0, ys, -b * ys], [0.0, 0.0, 1.0]])
                }
                TransformNode::ScaledTranslation { xt, yt, scale } => {
                    let (xt, yt, scale) = (*xt, *yt, *scale);
                    let s = self.matrix(scale)?;
                    let (tx, ty) = s.apply(xt, yt);
                    let mut m = M3::IDENTITY;
                    m.0[0][2] = tx;
                    m.0[1][2] = ty;
                    m
                }
            },
            Node::Bbox(_) => return invalid(format!("node {id} is a bbox, not a transform")),
        };
        self.cache[id as usize] = Some(Value::Matrix(m));
        Ok(m)
    }

    /// A one-dimensional scale transform of a scalar (`transform_non_affine`
    /// of a 1-D transform; identity for the affine ones).
    pub fn non_affine_1d(&mut self, id: NodeId, v: f64) -> Result<f64> {
        match self.node(id)? {
            Node::Transform(TransformNode::Log { base, clip }) => {
                let (base, clip) = (*base, *clip);
                let mut out = if base == 10.0 {
                    v.log10()
                } else if base == 2.0 {
                    v.log2()
                } else if base == std::f64::consts::E {
                    v.ln()
                } else {
                    v.ln() / base.ln()
                };
                if clip && v <= 0.0 { out = -1000.0; }
                Ok(out)
            }
            Node::Transform(TransformNode::Piecewise { coordinates }) => Ok(piecewise(v, coordinates)),
            _ if self.is_affine(id)? => Ok(v),
            _ => unsupported(format!("1-D transform of node {id}")),
        }
    }

    /// `transform.transform_non_affine(point)`.
    pub fn non_affine(&mut self, id: NodeId, x: f64, y: f64) -> Result<(f64, f64)> {
        if self.is_affine(id)? { return Ok((x, y)); }
        match self.node(id)? {
            Node::Transform(TransformNode::Composite { a, b, .. }) => {
                let (a, b) = (*a, *b);
                if self.is_affine(b)? {
                    self.non_affine(a, x, y)
                } else {
                    let (x, y) = self.transform(a, x, y)?;
                    self.non_affine(b, x, y)
                }
            }
            Node::Transform(TransformNode::Blended { x: tx, y: ty, x_eq_y, .. }) => {
                let (tx, ty, x_eq_y) = (*tx, *ty, *x_eq_y);
                if x_eq_y && self.input_dims(tx)? == 2 { return self.non_affine(tx, x, y); }
                let xo = if self.input_dims(tx)? == 2 { self.non_affine(tx, x, y)?.0 } else { self.non_affine_1d(tx, x)? };
                let yo = if self.input_dims(ty)? == 2 { self.non_affine(ty, x, y)?.1 } else { self.non_affine_1d(ty, y)? };
                Ok((xo, yo))
            }
            _ => unsupported(format!("2-D non-affine transform of node {id}")),
        }
    }

    /// `transform.transform(point)`: the non-affine part, then the affine.
    pub fn transform(&mut self, id: NodeId, x: f64, y: f64) -> Result<(f64, f64)> {
        if self.is_affine(id)? { return Ok(self.affine(id)?.apply(x, y)); }
        let (x, y) = self.non_affine(id, x, y)?;
        Ok(self.affine(id)?.apply(x, y))
    }

    /// `transform.transform((x, y))` (any transform node).
    pub fn apply(&mut self, id: NodeId, x: f64, y: f64) -> Result<(f64, f64)> { self.transform(id, x, y) }

    /// The points of a bbox node.
    pub fn points(&mut self, id: NodeId) -> Result<Points> {
        if let Some(Value::Points(p)) = self.cache[id as usize] { return Ok(p); }
        let p = match self.node(id)? {
            Node::Bbox(b) => match *b {
                BboxNode::AxesPosition { axes } => match self.positions.get(axes) {
                    Some(p) => *p,
                    None => return invalid(format!("axes {axes} out of range")),
                },
                BboxNode::ViewLim { axes } => match self.view_lims.get(axes) {
                    Some(p) => *p,
                    None => return invalid(format!("axes {axes} out of range")),
                },
                BboxNode::FigureInches {} => [0.0, 0.0, self.size_inches[0], self.size_inches[1]],
                BboxNode::Literal { p } => p,
                BboxNode::Transformed { bbox, transform } => {
                    let inner = self.points(bbox)?;
                    if self.is_affine(transform)? {
                        let m = self.affine(transform)?;
                        transformed_points(&m, &inner)
                    } else {
                        self.transformed_points_general(transform, &inner)?
                    }
                }
            },
            Node::Transform(_) => return unsupported(format!("node {id} is a transform, not a bbox")),
        };
        self.cache[id as usize] = Some(Value::Points(p));
        Ok(p)
    }

    /// `TransformedBbox.get_points` through a non-affine transform.
    fn transformed_points_general(&mut self, id: NodeId, p: &Points) -> Result<Points> {
        let corners = [(p[0], p[1]), (p[2], p[1]), (p[0], p[3]), (p[2], p[3])];
        let mut xs = [0.0; 4];
        let mut ys = [0.0; 4];
        for (k, &(x, y)) in corners.iter().enumerate() {
            let (tx, ty) = self.transform(id, x, y)?;
            xs[k] = tx;
            ys[k] = ty;
        }
        let (mut x0, mut x1) = (py_min(&xs), py_max(&xs));
        if p[0] > p[2] { std::mem::swap(&mut x0, &mut x1); }
        let (mut y0, mut y1) = (py_min(&ys), py_max(&ys));
        if p[1] > p[3] { std::mem::swap(&mut y0, &mut y1); }
        Ok([x0, y0, x1, y1])
    }

    /// Whether `transform.transform_path_non_affine(path)` hands back a new
    /// `Path` (its simplification rcParams', as `Path.__init__` sets it)
    /// rather than the input's internals: an affine transform returns the
    /// path itself, a scale or blended transform keeps its internals
    /// (`_fast_from_codes_and_verts`), and a composite whose second part is
    /// non-affine runs the first part's `transform_path`, which ends in
    /// `AffineBase.transform_path_affine`: `Path(...)`.
    pub fn path_non_affine_is_new(&self, id: NodeId) -> Result<bool> {
        if self.is_affine(id)? { return Ok(false); }
        match self.node(id)? {
            Node::Transform(TransformNode::Composite { a, b, .. }) => {
                if self.is_affine(*b)? { self.path_non_affine_is_new(*a) } else { Ok(true) }
            }
            Node::Transform(TransformNode::Blended { .. } | TransformNode::Log { .. } | TransformNode::Piecewise { .. }) => Ok(false),
            _ => unsupported(format!("transform_path_non_affine of node {id}")),
        }
    }

    /// `transform.transform_path_non_affine(path).vertices` and
    /// `transform.get_affine()`: what a draw hands the renderer.
    pub fn path_parts(&mut self, id: NodeId, vertices: &[[f64; 2]]) -> Result<(Option<Vec<[f64; 2]>>, M3)> {
        if self.is_affine(id)? { return Ok((None, self.affine(id)?)); }
        let mut out = Vec::with_capacity(vertices.len());
        for v in vertices {
            let (x, y) = self.non_affine(id, v[0], v[1])?;
            out.push([x, y]);
        }
        Ok((Some(out), self.affine(id)?))
    }
}

/// zlc's `_axis_scale._piecewise(value, coordinates, arange(len))`: the axis
/// position of a value on a nonuniform coordinate table, extended linearly.
pub fn piecewise(value: f64, coordinates: &[f64]) -> f64 {
    let n = coordinates.len();
    let mut positions: Vec<f64> = (0..n).map(|i| i as f64).collect();
    if n > 0 && coordinates[n - 1] < coordinates[0] { positions.reverse(); }
    if n == 1 { return value - coordinates[0] + positions[0]; }
    let (c, p): (Vec<f64>, Vec<f64>) = if coordinates[n - 1] < coordinates[0] {
        (coordinates.iter().rev().copied().collect(), positions.iter().rev().copied().collect())
    } else {
        (coordinates.to_vec(), positions)
    };
    // searchsorted(c, value, side="right") - 1, clipped to [0, n - 2].
    let (mut lo, mut hi) = (0usize, n);
    while lo < hi {
        let mid = (lo + hi) / 2;
        if c[mid] <= value { lo = mid + 1; } else { hi = mid; }
    }
    let index = (lo as i64 - 1).clamp(0, n as i64 - 2) as usize;
    let fraction = (value - c[index]) / (c[index + 1] - c[index]);
    p[index] + fraction * (p[index + 1] - p[index])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotate_zero_is_exact_identity_on_translation() {
        let m = M3::IDENTITY.translate(-2.5, -1.25).rotate_deg(0.0).translate(2.5, 1.25);
        assert_eq!(m.0[0][2], 0.0);
        assert_eq!(m.0[1][2], 0.0);
    }
}
