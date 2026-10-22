//! The renderer's matplotlib `Figure`: its size and dpi, axes in figure
//! order, its own artists and texts, and its IR.
use zlc_figure::ir::{self, Artist, HAlign, VAlign};
use zlc_figure::transform::{Evaluator, M3};

use super::artist;
use super::axes::{AxesModel, Child, EmitContext};
use super::color;
use super::nodes::{AxesNodes, Nodes, ScaleNode};
use super::rc::Rc;

/// Where a figure text stands.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FigureFrame {
    /// `transFigure` (`figure.text`).
    Figure,
    /// An axes' title frame (a `Text` that copied a cell title with `update_from`).
    Title(usize),
}

/// A `Text` child of the figure.
#[derive(Clone, Debug)]
pub struct FigureText {
    pub visible: bool,
    pub zorder: f64,
    pub x: f64,
    pub y: f64,
    pub text: String,
    pub size: f64,
    pub color: String,
    pub ha: HAlign,
    pub va: VAlign,
    pub rotation: f64,
    pub frame: FigureFrame,
}

/// One label a grid's label carrier paints: a cell's own tick label (the
/// `index`-th major tick's `label1`/`label2`) or its offset text.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LabelRef {
    Tick { axes: usize, x: bool, index: usize, side: u8 },
    Offset { axes: usize, x: bool },
}

/// An artist given to the figure with `add_artist`.
#[derive(Clone, Debug)]
pub enum FigureArtist {
    Text(FigureText),
    /// `_FacetChromeLabels`: the grid's boundary tick labels.
    Labels(Vec<LabelRef>),
}

#[derive(Clone, Debug)]
pub struct FigureModel {
    pub size_inches: [f64; 2],
    pub dpi: f64,
    pub axes: Vec<AxesModel>,
    /// `figure.artists`, in order.
    pub artists: Vec<(u64, FigureArtist)>,
    /// `figure.texts`, in order.
    pub texts: Vec<(u64, FigureText)>,
    next_child: u64,
}

impl FigureModel {
    pub fn new(size_inches: [f64; 2], dpi: f64) -> FigureModel {
        FigureModel { size_inches, dpi, axes: Vec::new(), artists: Vec::new(), texts: Vec::new(), next_child: 0 }
    }

    /// `figure.bbox` width and height in pixels.
    pub fn size_px(&self) -> (f64, f64) { (self.size_inches[0] * self.dpi, self.size_inches[1] * self.dpi) }

    /// `figure.add_artist(artist)`.
    pub fn add_artist(&mut self, artist: FigureArtist) -> u64 {
        self.next_child += 1;
        self.artists.push((self.next_child, artist));
        self.next_child
    }

    /// `figure.text(...)`.
    pub fn add_text(&mut self, text: FigureText) -> u64 {
        self.next_child += 1;
        self.texts.push((self.next_child, text));
        self.next_child
    }

    pub fn text_mut(&mut self, id: u64) -> Option<&mut FigureText> {
        self.texts.iter_mut().find(|(k, _)| *k == id).map(|(_, t)| t)
            .or_else(|| self.artists.iter_mut().find(|(k, _)| *k == id).and_then(|(_, a)| match a { FigureArtist::Text(t) => Some(t), _ => None }))
    }

    /// The position of a figure child in the IR's `children` (artists, axes, texts).
    pub fn child_position(&self, id: u64) -> Option<usize> {
        if let Some(at) = self.artists.iter().position(|(k, _)| *k == id) { return Some(at); }
        self.texts.iter().position(|(k, _)| *k == id).map(|at| self.artists.len() + self.axes.len() + at)
    }

    /// The resolved artist tree (image planes left to the kernel).
    pub fn to_ir(&self, rc: &Rc) -> ir::Figure { self.to_ir_with(rc, false) }

    /// The resolved artist tree; `materialize` puts the image planes in it.
    pub fn to_ir_with(&self, rc: &Rc, materialize: bool) -> ir::Figure {
        let mut nodes = Nodes::default();
        let figure_nodes = nodes.figure();
        let ctx = EmitContext { rc, materialize, figure: &figure_nodes, dpi: self.dpi, size_inches: self.size_inches };
        // Two passes: the error-bar glyphs are display-space paths, made with
        // each axes' transData as matplotlib evaluates it.
        let placeholder = M3::IDENTITY;
        let mut axes = Vec::with_capacity(self.axes.len());
        let mut frames: Vec<(AxesNodes, ir::NodeId)> = Vec::with_capacity(self.axes.len());
        for (index, model) in self.axes.iter().enumerate() {
            let (ir_axes, an, title) = model.to_ir(&ctx, &mut nodes, index, &placeholder);
            axes.push(ir_axes);
            frames.push((an, title));
        }
        let mut patch = artist::rectangle(color::rgba(&rc.figure_facecolor), color::rgba(&rc.figure_edgecolor), 0.0, false,
                                          figure_nodes.trans_figure);
        patch.artist.zorder = 1.0;
        let text_ir = |t: &FigureText| -> ir::Text {
            let transform = match t.frame { FigureFrame::Figure => figure_nodes.trans_figure, FigureFrame::Title(k) => frames[k].1 };
            let mut text = artist::text(rc, t.x, t.y, &t.text, t.size, transform);
            text.color = color::rgba(&t.color);
            text.ha = t.ha;
            text.va = t.va;
            text.rotation = t.rotation;
            text.artist.visible = t.visible;
            text.artist.zorder = t.zorder;
            text
        };
        let mut children: Vec<Artist> = Vec::with_capacity(self.artists.len() + self.axes.len() + self.texts.len());
        for (_, figure_artist) in &self.artists {
            children.push(match figure_artist {
                FigureArtist::Text(t) => Artist::Text(text_ir(t)),
                FigureArtist::Labels(labels) => {
                    let texts = labels.iter().map(|label| match *label {
                        LabelRef::Tick { axes: k, x, index, side } => {
                            let axis = self.axes[k].axis(x);
                            let (loc, label) = &axis.tick_pool[index];
                            AxesModel::tick_label(&ctx, &mut nodes, &frames[k].0, axis, true, Some(index), *loc, label, side)
                        }
                        LabelRef::Offset { axes: k, x } => AxesModel::offset_text_ir(&ctx, &mut nodes, &frames[k].0, self.axes[k].axis(x)),
                    }).collect();
                    let mut props = artist::props(3.0, Some(figure_nodes.trans_figure));
                    props.clip_on = true;
                    Artist::Group(ir::Group { artist: props, cls: "_FacetChromeLabels".into(), texts })
                }
            });
        }
        children.extend((0..self.axes.len()).map(|index| Artist::Axes { index }));
        for (_, t) in &self.texts { children.push(Artist::Text(text_ir(t))); }
        let mut figure = ir::Figure {
            version: 1,
            dpi: self.dpi,
            size_inches: self.size_inches,
            visible: true,
            suppress_composite: None,
            rc: rc.ir.clone(),
            patch,
            axes: Vec::new(),
            children,
            trans_figure: figure_nodes.trans_figure,
            figure_bbox: figure_nodes.bbox,
            nodes: Vec::new(),
        };
        // Only an axes holding segments needs its transData: the matrix of an
        // affine one, else `get_affine()` -- the part past its scales (a grid
        // cell keeps the FuncScale its image cell set when it turns curve).
        let segmented: Vec<usize> = (0..self.axes.len())
            .filter(|&index| self.axes[index].children.iter().any(|(_, c)| matches!(c, Child::Segments(_)))).collect();
        if !segmented.is_empty() {
            figure.nodes = nodes.table.clone();
            figure.axes = axes.clone();
            let matrices: Vec<M3> = {
                let mut eval = Evaluator::new(&figure);
                segmented.iter().map(|&index| {
                    let model = &self.axes[index];
                    let linear = matches!((model.xaxis.scale_node(), model.yaxis.scale_node()), (ScaleNode::Linear, ScaleNode::Linear));
                    eval.matrix(if linear { axes[index].trans_data } else { frames[index].0.limits_axes }).expect("transData")
                }).collect()
            };
            for (&index, matrix) in segmented.iter().zip(&matrices) {
                axes[index].children = self.axes[index].to_ir(&ctx, &mut nodes, index, matrix).0.children;
            }
        }
        figure.nodes = nodes.table;
        figure.axes = axes;
        figure
    }
}
