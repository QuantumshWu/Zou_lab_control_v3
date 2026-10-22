//! `matplotlib.axis.Axis.draw`: tick selection (`_update_ticks`), Tick.draw,
//! axis-label placement from the tick-label extents and the spine extent
//! (`_update_label_position`), the offset text (`_update_offset_text_position`).
use crate::draw::Ctx;
use crate::ir::{Axis, Line2D, NamedSpine, Text, TickTemplate, Ticks};
use crate::text::{draw_text, window_extent};
use crate::transform::{union, Points};
use crate::{unsupported, Result};

const OFFSETTEXTPAD: f64 = 3.0;

/// `transforms._interval_contains_close(interval, val)`.
pub fn interval_contains_close(interval: (f64, f64), val: f64) -> bool {
    let (mut a, mut b) = interval;
    if a > b { std::mem::swap(&mut a, &mut b); }
    let rtol = (b - a) * 1e-10;
    a - rtol <= val && val <= b + rtol
}

/// One tick to draw: the template it is drawn as, its location and label.
#[derive(Clone, Copy)]
pub struct DrawnTick<'a> { pub template: &'a TickTemplate, pub loc: f64, pub label: &'a str, pub major: bool }

/// `Axis._update_ticks()`: the ticks that fall in the view interval.
pub fn ticks_to_draw<'a>(ctx: &mut Ctx<'_>, axes: usize, axis: &'a Axis) -> Result<Vec<DrawnTick<'a>>> {
    // The ticks are what this axis' _update_ticks produced in the draw.
    if !axis.drawn { return crate::invalid(format!("ticks of axis {} the draw did not update", axis.name)); }
    let vl = ctx.eval.view_lims[axes];
    let (mut low, mut high) = if axis.name == "x" { (vl[0], vl[2]) } else { (vl[1], vl[3]) };
    if low > high { std::mem::swap(&mut low, &mut high); }
    // interval_t and loc_t go through the axis' scale transform.
    let low = ctx.eval.non_affine_1d(axis.transform, low)?;
    let high = ctx.eval.non_affine_1d(axis.transform, high)?;
    let mut out = Vec::new();
    for (ticks, major) in [(&axis.major, true), (&axis.minor, false)] {
        let Some(ticks) = ticks else { return crate::invalid("an updated axis without its ticks") };
        if ticks.locs.is_empty() { continue; }
        let Some(template) = &ticks.tick else { return crate::invalid("ticks without a template") };
        if ticks.labels.len() != ticks.locs.len() { return crate::invalid("tick labels and locations differ in count"); }
        if ticks.per_tick.as_ref().is_some_and(|p| p.len() != ticks.locs.len()) {
            return crate::invalid("per-tick templates and locations differ in count");
        }
        for (k, (loc, label)) in ticks.locs.iter().zip(&ticks.labels).enumerate() {
            if interval_contains_close((low, high), ctx.eval.non_affine_1d(axis.transform, *loc)?) {
                let template = ticks.per_tick.as_ref().map_or(template, |p| &p[k]);
                // XTick/YTick: update_position moves the axis' own coordinate.
                if template.cls != if axis.name == "x" { "XTick" } else { "YTick" } {
                    return crate::invalid(format!("{} on axis {}", template.cls, axis.name));
                }
                out.push(DrawnTick { template, loc: *loc, label, major });
            }
        }
    }
    Ok(out)
}

/// A tick line at `loc` (`update_position`): the axis coordinate of every
/// vertex becomes `loc` (the other keeps the template's, as broadcasting does).
fn line_at(template: &Line2D, loc: f64, x_axis: bool) -> Line2D {
    let mut line = template.clone();
    let k = if x_axis { 0 } else { 1 };
    for p in &mut line.xy { p[k] = loc; }
    line
}

fn label_at(template: &Text, loc: f64, x_axis: bool, label: &str) -> Text {
    let mut text = template.clone();
    if x_axis { text.x = loc; } else { text.y = loc; }
    text.text = label.to_owned();
    text
}

/// The artists of one tick (gridline, tick1line, tick2line, label1, label2).
pub struct TickArtists { pub gridline: Line2D, pub tick1line: Line2D, pub tick2line: Line2D, pub label1: Text, pub label2: Text }

pub fn tick_artists(tick: &DrawnTick<'_>, x_axis: bool) -> TickArtists {
    let t = tick.template;
    TickArtists {
        gridline: line_at(&t.gridline, tick.loc, x_axis),
        tick1line: line_at(&t.tick1line, tick.loc, x_axis),
        tick2line: line_at(&t.tick2line, tick.loc, x_axis),
        label1: label_at(&t.label1, tick.loc, x_axis, tick.label),
        label2: label_at(&t.label2, tick.loc, x_axis, tick.label),
    }
}

/// `Axis._get_ticklabel_bboxes(ticks)`.
pub fn ticklabel_bboxes(ctx: &mut Ctx<'_>, ticks: &[DrawnTick<'_>], x_axis: bool) -> Result<(Vec<Points>, Vec<Points>)> {
    let mut b1 = Vec::new();
    let mut b2 = Vec::new();
    for tick in ticks {
        let a = tick_artists(tick, x_axis);
        if a.label1.artist.visible { b1.push(window_extent(ctx, &a.label1, None)?); }
        if a.label2.artist.visible { b2.push(window_extent(ctx, &a.label2, None)?); }
    }
    Ok((b1, b2))
}

/// `Spine.get_window_extent()` for an axes spine (ticks pad it).
pub fn spine_window_extent(ctx: &mut Ctx<'_>, axes: usize, spine: &NamedSpine) -> Result<Points> {
    let p = &spine.patch;
    let bb = crate::patches::patch_window_extent(ctx, p, Some(axes))?;
    let crate::ir::PatchShape::Spine { spine_type, axis: axis_name, .. } = &p.shape else {
        // The colorbar outline: Patch.get_window_extent.
        return Ok(bb);
    };
    let Some(axis_name) = axis_name else { return Ok(bb) };
    let ax = &ctx.fig.axes[axes];
    let axis = if axis_name == "x" { &ax.xaxis } else { &ax.yaxis };
    if !axis.visible { return Ok(bb); }
    let drawn = ticks_to_draw(ctx, axes, axis)?;
    let mut boxes = vec![bb];
    for (major, pick) in [(true, axis.spine_ticks[0]), (false, axis.spine_ticks[1])] {
        // The tick matplotlib's set intersection yields first, not the first drawn.
        let tick = match pick {
            Some(i) => match drawn.get(i) {
                Some(t) if t.major == major => t,
                _ => return crate::invalid(format!("spine tick {i} is not a drawn {} tick", if major { "major" } else { "minor" })),
            },
            None if drawn.iter().any(|t| t.major == major) => return crate::invalid("drawn ticks but no spine tick"),
            None => continue,
        };
        let mut b = bb;
        let tickl = tick.template.size;
        let (padout, padin) = match tick.template.tickdir.as_str() {
            "out" => (1.0, 0.0),
            "in" => (0.0, 1.0),
            _ => (0.5, 0.5),
        };
        let dpi = ctx.dpi;
        let padout = padout * tickl / 72.0 * dpi;
        let padin = padin * tickl / 72.0 * dpi;
        if tick.template.tick1line.artist.visible {
            match spine_type.as_str() {
                "left" => { b[0] -= padout; b[2] += padin; }
                "bottom" => { b[1] -= padout; b[3] += padin; }
                _ => {}
            }
        }
        if tick.template.tick2line.artist.visible {
            match spine_type.as_str() {
                "right" => { b[2] += padout; b[0] -= padin; }
                "top" => { b[3] += padout; b[1] -= padout; }
                _ => {}
            }
        }
        boxes.push(b);
    }
    Ok(union(&boxes))
}

/// `axes.spines.get(name, axes).get_window_extent()`.
fn spine_or_axes_extent(ctx: &mut Ctx<'_>, axes: usize, name: &str) -> Result<Points> {
    let ax = &ctx.fig.axes[axes];
    match ax.spines.iter().find(|s| s.name == name) {
        Some(spine) => spine_window_extent(ctx, axes, spine),
        None => ctx.eval.points(ax.bbox),
    }
}

/// Where the axis label goes (`_update_label_position`): its new (x, y).
pub fn label_position(ctx: &mut Ctx<'_>, axes: usize, axis: &Axis, tlb: &(Vec<Points>, Vec<Points>)) -> Result<(f64, f64)> {
    let (x, y) = (axis.label.x, axis.label.y);
    if !axis.autolabelpos { return Ok((x, y)); }
    let pad = axis.labelpad * ctx.dpi / 72.0;
    let (bboxes, bboxes2) = tlb;
    Ok(match (axis.name.as_str(), axis.label_position.as_str()) {
        ("x", "bottom") => {
            let mut all = bboxes.clone();
            all.push(spine_or_axes_extent(ctx, axes, "bottom")?);
            (x, union(&all)[1] - pad)
        }
        ("x", _) => {
            let mut all = bboxes2.clone();
            all.push(spine_or_axes_extent(ctx, axes, "top")?);
            (x, union(&all)[3] + pad)
        }
        ("y", "left") => {
            let mut all = bboxes.clone();
            all.push(spine_or_axes_extent(ctx, axes, "left")?);
            (union(&all)[0] - pad, y)
        }
        ("y", _) => {
            let mut all = bboxes2.clone();
            all.push(spine_or_axes_extent(ctx, axes, "right")?);
            (union(&all)[2] + pad, y)
        }
        (other, _) => return unsupported(format!("axis {other}")),
    })
}

/// `_update_offset_text_position(bboxes, bboxes2)`: the offset text's (x, y).
pub fn offset_text_position(ctx: &mut Ctx<'_>, axes: usize, axis: &Axis, tlb: &(Vec<Points>, Vec<Points>)) -> Result<(f64, f64)> {
    if axis.offset_text_pinned { return Ok((axis.offset_text.x, axis.offset_text.y)); }
    let (x, _) = (axis.offset_text.x, axis.offset_text.y);
    let pad = OFFSETTEXTPAD * ctx.dpi / 72.0;
    let ax = &ctx.fig.axes[axes];
    if axis.name == "x" {
        let bottom_ticks = match axis.tick_position.as_deref() {
            Some(position) => position == "bottom",
            None => return crate::invalid("an x axis without its tick position"),
        };
        if bottom_ticks {
            let bottom = if tlb.0.is_empty() {
                let b = ctx.eval.points(ax.bbox)?;
                b[1].min(b[3])
            } else {
                union(&tlb.0)[1]
            };
            Ok((x, bottom - pad))
        } else {
            let top = if tlb.1.is_empty() {
                let b = ctx.eval.points(ax.bbox)?;
                b[1].max(b[3])
            } else {
                union(&tlb.1)[3]
            };
            Ok((x, top + pad))
        }
    } else {
        let bbox = match ax.spines.iter().find(|s| s.name == "outline") {
            Some(outline) => spine_window_extent(ctx, axes, outline)?,
            None => ctx.eval.points(ax.bbox)?,
        };
        let top = bbox[1].max(bbox[3]);
        Ok((x, top + pad))
    }
}

/// `Axis.draw(renderer)`.
pub fn draw_axis(ctx: &mut Ctx<'_>, axes: usize, axis: &Axis) -> Result<()> {
    if !axis.visible { return Ok(()); }
    let x_axis = match axis.name.as_str() { "x" => true, "y" => false, other => return unsupported(format!("axis {other}")) };
    let ticks = ticks_to_draw(ctx, axes, axis)?;
    let tlb = ticklabel_bboxes(ctx, &ticks, x_axis)?;
    for tick in &ticks {
        if !tick.template.visible { continue; }
        let a = tick_artists(tick, x_axis);
        crate::lines::draw_line(ctx, &a.gridline)?;
        crate::lines::draw_line(ctx, &a.tick1line)?;
        crate::lines::draw_line(ctx, &a.tick2line)?;
        draw_text(ctx, &a.label1, None)?;
        draw_text(ctx, &a.label2, None)?;
    }
    let label_xy = label_position(ctx, axes, axis, &tlb)?;
    draw_text(ctx, &axis.label, Some(label_xy))?;
    let offset_xy = if axis.offset_text_pinned { (axis.offset_text.x, axis.offset_text.y) } else {
        offset_text_position(ctx, axes, axis, &tlb)?
    };
    let mut offset = axis.offset_text.clone();
    offset.text = match &axis.offset_string {
        Some(s) => s.clone(),
        None => return crate::invalid("drawn axis without its offset string"),
    };
    draw_text(ctx, &offset, Some(offset_xy))
}

/// The tick placement matplotlib's draw would produce, for queries: per
/// drawn tick its label1/label2 window extents.
pub fn tick_label_extents(ctx: &mut Ctx<'_>, axes: usize, axis: &Axis) -> Result<Vec<(f64, Option<Points>, Option<Points>)>> {
    let x_axis = axis.name == "x";
    let ticks = ticks_to_draw(ctx, axes, axis)?;
    let mut out = Vec::new();
    for tick in &ticks {
        let a = tick_artists(tick, x_axis);
        let e1 = if a.label1.artist.visible && !a.label1.text.is_empty() { Some(window_extent(ctx, &a.label1, None)?) } else { None };
        let e2 = if a.label2.artist.visible && !a.label2.text.is_empty() { Some(window_extent(ctx, &a.label2, None)?) } else { None };
        out.push((tick.loc, e1, e2));
    }
    Ok(out)
}

pub fn has_ticks(t: &Option<Ticks>) -> bool { t.as_ref().is_some_and(|t| !t.locs.is_empty()) }
