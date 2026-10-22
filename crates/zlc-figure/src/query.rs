//! Geometry queries on an IR, computed by the same functions the draw uses
//! (one source of truth): an axes' box after `apply_aspect`, its transforms,
//! every text's window extent at the position the draw puts it, tick
//! placement, axis-label and offset-text positions.
use crate::backend::{Backend, Gc, GrayImage, PathRef, RgbaImage};
use crate::draw::Ctx;
use crate::fonts::{Fonts, MathText};
use crate::ir::{self, Artist, Dash};
use crate::transform::{M3, Points};
use crate::{invalid, Result};

/// A backend that draws nothing (queries measure, never paint).
pub struct NullBackend;

impl Backend for NullBackend {
    fn draw_path(&mut self, _: &Gc, _: PathRef<'_>, _: &M3, _: Option<[f64; 4]>) -> Result<()> { Ok(()) }
    fn draw_markers(&mut self, _: &Gc, _: PathRef<'_>, _: &M3, _: PathRef<'_>, _: &M3, _: Option<[f64; 4]>) -> Result<()> { Ok(()) }
    fn draw_text_image(&mut self, _: &dyn GrayImage, _: i32, _: i32, _: f64, _: &Gc) -> Result<()> { Ok(()) }
    fn draw_image(&mut self, _: &Gc, _: f64, _: f64, _: &RgbaImage) -> Result<()> { Ok(()) }
    fn draw_path_collection(&mut self, _: &Gc, _: &M3, _: &[PathRef<'_>], _: &[[[f64; 3]; 3]], _: &[[f64; 2]], _: &M3,
                            _: &[[f64; 4]], _: &[[f64; 4]], _: &[f64], _: &[Dash], _: &[bool]) -> Result<()> { Ok(()) }
    fn draw_quad_mesh(&mut self, _: &Gc, _: &M3, _: usize, _: usize, _: &[[f64; 2]], _: &[[f64; 2]], _: &M3,
                      _: &[[f64; 4]], _: bool, _: &[[f64; 4]]) -> Result<()> { Ok(()) }
}

/// The figure as laid out for a draw: every axes at its `apply_aspect` box.
pub struct Scene<'a> { pub ctx: Ctx<'a> }

/// Which axis of an axes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Which { X, Y }

impl<'a> Scene<'a> {
    pub fn new(fig: &'a ir::Figure, null: &'a mut NullBackend, fonts: &'a mut dyn Fonts,
               math: &'a mut dyn MathText) -> Result<Scene<'a>> {
        let mut ctx = Ctx::new(fig, null, fonts, math)?;
        for i in 0..fig.axes.len() { crate::axes::apply_aspect(&mut ctx, i)?; }
        Ok(Scene { ctx })
    }

    fn axis(&self, axes: usize, which: Which) -> &'a ir::Axis {
        let ax = &self.ctx.fig.axes[axes];
        match which { Which::X => &ax.xaxis, Which::Y => &ax.yaxis }
    }

    /// `axes.bbox` (display box) after apply_aspect.
    pub fn axes_bbox(&mut self, axes: usize) -> Result<Points> {
        let id = self.ctx.fig.axes[axes].bbox;
        self.ctx.eval.points(id)
    }

    /// The active position (figure fraction box).
    pub fn axes_position(&self, axes: usize) -> Points { self.ctx.eval.positions[axes] }

    /// `axes.transData` (data -> display) as an affine matrix.
    pub fn trans_data(&mut self, axes: usize) -> Result<M3> {
        let id = self.ctx.fig.axes[axes].trans_data;
        self.ctx.eval.affine(id)
    }

    pub fn trans_axes(&mut self, axes: usize) -> Result<M3> {
        let id = self.ctx.fig.axes[axes].trans_axes;
        self.ctx.eval.affine(id)
    }

    /// `text.get_window_extent()` of a text drawn at its own position.
    pub fn text_extent(&mut self, t: &ir::Text) -> Result<Points> { crate::text::window_extent(&mut self.ctx, t, None) }

    /// The ticks an axis draws: location and its label1/label2 extents.
    pub fn tick_label_extents(&mut self, axes: usize, which: Which) -> Result<Vec<(f64, Option<Points>, Option<Points>)>> {
        let axis = self.axis(axes, which);
        crate::axis::tick_label_extents(&mut self.ctx, axes, axis)
    }

    fn label_boxes(&mut self, axes: usize, axis: &ir::Axis) -> Result<(Vec<Points>, Vec<Points>)> {
        let ticks = crate::axis::ticks_to_draw(&mut self.ctx, axes, axis)?;
        crate::axis::ticklabel_bboxes(&mut self.ctx, &ticks, axis.name == "x")
    }

    /// Where the draw puts the axis label (`_update_label_position`).
    pub fn axis_label_position(&mut self, axes: usize, which: Which) -> Result<(f64, f64)> {
        let axis = self.axis(axes, which);
        let tlb = self.label_boxes(axes, axis)?;
        crate::axis::label_position(&mut self.ctx, axes, axis, &tlb)
    }

    /// The axis label's extent at that position.
    pub fn axis_label_extent(&mut self, axes: usize, which: Which) -> Result<Points> {
        let axis = self.axis(axes, which);
        let xy = self.axis_label_position(axes, which)?;
        crate::text::window_extent(&mut self.ctx, &axis.label, Some(xy))
    }

    /// Where the draw puts the offset text (`_update_offset_text_position`).
    pub fn offset_text_position(&mut self, axes: usize, which: Which) -> Result<(f64, f64)> {
        let axis = self.axis(axes, which);
        let tlb = self.label_boxes(axes, axis)?;
        crate::axis::offset_text_position(&mut self.ctx, axes, axis, &tlb)
    }

    /// The offset text's extent there, with the formatter's offset string.
    pub fn offset_text_extent(&mut self, axes: usize, which: Which) -> Result<Points> {
        let axis = self.axis(axes, which);
        let xy = self.offset_text_position(axes, which)?;
        let mut t = axis.offset_text.clone();
        t.text = match &axis.offset_string {
            Some(s) => s.clone(),
            None => return invalid("the offset text of an axis the draw did not capture"),
        };
        crate::text::window_extent(&mut self.ctx, &t, Some(xy))
    }

    /// A text by the oracle's path (`figure/j`, `figure/j/texts/k`,
    /// `axes/i/children/j`, `axes/i/titles/k`, `axes/i/xaxis/label`,
    /// `axes/i/yaxis/offset_text`, `axes/i/xaxis/major/k/label1`).
    pub fn text_extent_at(&mut self, path: &str) -> Result<Points> {
        let fig = self.ctx.fig;
        let parts: Vec<&str> = path.split('/').collect();
        let num = |s: &str| s.parse::<usize>().map_err(|_| crate::Error::Invalid(format!("text path {path}")));
        let which = |s: &str| match s {
            "xaxis" => Ok(Which::X),
            "yaxis" => Ok(Which::Y),
            _ => Err(crate::Error::Invalid(format!("text path {path}"))),
        };
        let axes = |s: &str| match num(s) {
            Ok(i) if i < fig.axes.len() => Ok(i),
            _ => Err(crate::Error::Invalid(format!("text path {path}"))),
        };
        match parts.as_slice() {
            ["figure", j] => match fig.children.get(num(j)?) {
                Some(Artist::Text(t)) => self.text_extent(t),
                _ => invalid(format!("no text at {path}")),
            },
            ["figure", j, "texts", k] => match fig.children.get(num(j)?) {
                Some(Artist::Group(g)) => match g.texts.get(num(k)?) { Some(t) => self.text_extent(t), None => invalid(path.to_owned()) },
                _ => invalid(format!("no group at {path}")),
            },
            ["axes", i, "children", j] => match fig.axes[axes(i)?].children.get(num(j)?) {
                Some(Artist::Text(t)) => self.text_extent(t),
                _ => invalid(format!("no text at {path}")),
            },
            ["axes", i, "titles", k] => {
                // The draw places the titles first (or refuses to).
                let ax = &fig.axes[axes(i)?];
                crate::axes::update_title_position(ax)?;
                match ax.titles.get(num(k)?) {
                    Some(t) => self.text_extent(t),
                    None => invalid(path.to_owned()),
                }
            }
            ["axes", i, axis, what] => {
                let (index, which) = (axes(i)?, which(axis)?);
                match *what {
                    "label" => self.axis_label_extent(index, which),
                    "offset_text" => self.offset_text_extent(index, which),
                    _ => invalid(path.to_owned()),
                }
            }
            ["axes", i, axis, "major", k, label] => {
                let (index, which) = (axes(i)?, which(axis)?);
                let ax_axis = self.axis(index, which);
                if !ax_axis.drawn { return invalid(format!("{path}: the draw did not capture this axis")); }
                let Some(major) = &ax_axis.major else { return invalid(path.to_owned()) };
                let k = num(k)?;
                if k >= major.locs.len() || k >= major.labels.len() { return invalid(path.to_owned()); }
                let Some(template) = &major.tick else { return invalid(path.to_owned()) };
                let template = match &major.per_tick {
                    None => template,
                    Some(p) => match p.get(k) { Some(t) => t, None => return invalid(path.to_owned()) },
                };
                let tick = crate::axis::DrawnTick { template, loc: major.locs[k], label: &major.labels[k], major: true };
                let a = crate::axis::tick_artists(&tick, which == Which::X);
                let t = match *label {
                    "label1" => a.label1,
                    "label2" => a.label2,
                    _ => return invalid(path.to_owned()),
                };
                self.text_extent(&t)
            }
            _ => invalid(format!("text path {path}")),
        }
    }
}
