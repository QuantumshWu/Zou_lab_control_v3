//! The transform/bbox DAG of a figure, built the way matplotlib builds its
//! transform tree (`Figure.__init__`, `Axes._set_lim_and_transforms`, the
//! axis and title transforms), so that every artist refers to the same
//! structure matplotlib's artists do.
use zlc_figure::ir::{BboxNode, Node, NodeId, TransformNode};

/// The nodes every figure has.
#[derive(Clone, Copy, Debug)]
pub struct FigureNodes {
    pub identity: NodeId,
    pub dpi_scale: NodeId,
    /// `figure.bbox` = TransformedBbox(bbox_inches, dpi_scale_trans).
    pub bbox: NodeId,
    /// `figure.transFigure` = BboxTransformTo(figure.bbox).
    pub trans_figure: NodeId,
}

/// An axis' scale as its transform: linear (IdentityTransform), matplotlib's
/// `log` (LogTransform(10, clip)) or zlc's FuncScale over a nonuniform lattice.
#[derive(Clone, Copy, Debug)]
pub enum ScaleNode<'a> { Linear, Log, Lattice(&'a [f64]) }

/// The nodes one axes has (`_set_lim_and_transforms`, then `_update_transScale`).
#[derive(Clone, Copy, Debug)]
pub struct AxesNodes {
    /// `axes.bbox` = TransformedBbox(position, transFigure).
    pub bbox: NodeId,
    pub trans_axes: NodeId,
    /// The axes' x and y scale transforms (`axis.get_transform()`).
    pub x_scale: NodeId,
    pub y_scale: NodeId,
    /// `transScale` after `_update_transScale`.
    pub trans_scale: NodeId,
    /// `transLimits + transAxes`: the data transform past the scales.
    pub limits_axes: NodeId,
    /// transData equals transAxes by value (`transAxes == transData`).
    pub data_is_axes: bool,
    pub trans_data: NodeId,
    /// `get_xaxis_transform()` = blended(transData, transAxes).
    pub xaxis: NodeId,
    /// `get_yaxis_transform()` = blended(transAxes, transData).
    pub yaxis: NodeId,
    /// The clip box `set_clip_path(axes.patch)` gives a child.
    pub patch_clip: NodeId,
    /// `axes.patch` transform: BboxTransformTo(Bbox.unit()) + rotation, + transAxes.
    pub patch: NodeId,
}

#[derive(Default)]
pub struct Nodes {
    pub table: Vec<Node>,
}

impl Nodes {
    pub fn push(&mut self, node: Node) -> NodeId {
        self.table.push(node);
        (self.table.len() - 1) as NodeId
    }

    pub fn t(&mut self, node: TransformNode) -> NodeId { self.push(Node::Transform(node)) }
    pub fn b(&mut self, node: BboxNode) -> NodeId { self.push(Node::Bbox(node)) }

    pub fn composite(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.t(TransformNode::Composite { a, b })
    }

    pub fn blended(&mut self, x: NodeId, y: NodeId, x_eq_y: bool) -> NodeId {
        self.t(TransformNode::Blended { x, y, x_eq_y })
    }

    pub fn scaled_translation(&mut self, xt: f64, yt: f64, scale: NodeId) -> NodeId {
        self.t(TransformNode::ScaledTranslation { xt, yt, scale })
    }

    pub fn affine(&mut self, m: [f64; 6]) -> NodeId { self.t(TransformNode::Affine { m }) }

    pub fn literal(&mut self, p: [f64; 4]) -> NodeId { self.b(BboxNode::Literal { p }) }

    pub fn transformed(&mut self, bbox: NodeId, transform: NodeId) -> NodeId {
        self.b(BboxNode::Transformed { bbox, transform })
    }

    pub fn figure(&mut self) -> FigureNodes {
        let identity = self.t(TransformNode::Identity {});
        let dpi_scale = self.t(TransformNode::DpiScale {});
        let inches = self.b(BboxNode::FigureInches {});
        let bbox = self.transformed(inches, dpi_scale);
        let trans_figure = self.t(TransformNode::BboxTo { bbox });
        FigureNodes { identity, dpi_scale, bbox, trans_figure }
    }

    pub fn axes(&mut self, figure: &FigureNodes, index: usize) -> AxesNodes { self.axes_with(figure, index, false) }

    /// `data_is_axes`: transAxes equals transData by value (a unit view).
    /// Matplotlib's `x == y` of the y-axis blend (transAxes, transData) asks
    /// the affine transAxes, which compares matrices; the x-axis blend asks
    /// the composite transData, which is never equal to a non-composite.
    pub fn axes_with(&mut self, figure: &FigureNodes, index: usize, data_is_axes: bool) -> AxesNodes {
        self.axes_scaled(figure, index, data_is_axes, ScaleNode::Linear, ScaleNode::Linear)
    }

    /// The axes nodes with each axis' scale transform.
    pub fn axes_scaled(&mut self, figure: &FigureNodes, index: usize, data_is_axes: bool, x: ScaleNode<'_>,
                       y: ScaleNode<'_>) -> AxesNodes {
        let position = self.b(BboxNode::AxesPosition { axes: index });
        let bbox = self.transformed(position, figure.trans_figure);
        let trans_axes = self.t(TransformNode::BboxTo { bbox });
        let mut scale = |scale: ScaleNode<'_>| self.t(match scale {
            ScaleNode::Linear => TransformNode::Identity {},
            ScaleNode::Log => TransformNode::Log { base: 10.0, clip: true },
            ScaleNode::Lattice(coordinates) => TransformNode::Piecewise { coordinates: coordinates.to_vec() },
        });
        let x_scale = scale(x);
        let y_scale = scale(y);
        // blended_transform_factory's `x == y`: two IdentityTransforms are
        // equal; a LogTransform or a FuncScale's transform equals only itself.
        let linear = matches!((x, y), (ScaleNode::Linear, ScaleNode::Linear));
        let trans_scale = self.blended(x_scale, y_scale, linear);
        let view = self.b(BboxNode::ViewLim { axes: index });
        let scaled_view = self.transformed(view, trans_scale);
        let trans_limits = self.t(TransformNode::BboxFrom { bbox: scaled_view });
        let limits_axes = self.composite(trans_limits, trans_axes);
        let trans_data = self.composite(trans_scale, limits_axes);
        let xaxis = self.blended(trans_data, trans_axes, false);
        let yaxis = self.blended(trans_axes, trans_data, data_is_axes);
        let unit = self.literal([0.0, 0.0, 1.0, 1.0]);
        let unit_to = self.t(TransformNode::BboxTo { bbox: unit });
        let rotation = self.affine([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let patch_transform = self.composite(unit_to, rotation);
        let patch = self.composite(patch_transform, trans_axes);
        let clip_unit = self.literal([0.0, 0.0, 1.0, 1.0]);
        let patch_clip = self.transformed(clip_unit, patch);
        AxesNodes { bbox, trans_axes, x_scale, y_scale, trans_scale, limits_axes, data_is_axes, trans_data, xaxis, yaxis, patch_clip, patch }
    }

    /// `transAxes + ScaledTranslation(dx, dy, dpi_scale_trans)` (an
    /// `offset_copy` in points, or a title's offset).
    pub fn offset(&mut self, base: NodeId, figure: &FigureNodes, dx_inches: f64, dy_inches: f64) -> NodeId {
        let shift = self.scaled_translation(dx_inches, dy_inches, figure.dpi_scale);
        self.composite(base, shift)
    }
}
