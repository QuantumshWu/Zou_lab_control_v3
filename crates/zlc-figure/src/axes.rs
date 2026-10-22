//! `matplotlib.axes.Axes.draw` and `apply_aspect`.
use crate::draw::{draw_artist, sort_by_zorder, Ctx};
use crate::ir::{Anchor, Aspect, Axes};
use crate::transform::{bbox_transformed, Points};
use crate::{unsupported, Result};

/// `Bbox.coefs`.
fn anchor_coefs(anchor: &Anchor) -> Result<(f64, f64)> {
    Ok(match anchor {
        Anchor::Point(p) => (p[0], p[1]),
        Anchor::Named(n) => match n.as_str() {
            "C" => (0.5, 0.5),
            "SW" => (0.0, 0.0),
            "S" => (0.5, 0.0),
            "SE" => (1.0, 0.0),
            "E" => (1.0, 0.5),
            "NE" => (1.0, 1.0),
            "N" => (0.5, 1.0),
            "NW" => (0.0, 1.0),
            "W" => (0.0, 0.5),
            other => return unsupported(format!("anchor {other}")),
        },
    })
}

/// `Bbox.shrunk_to_aspect(box_aspect, container=self, fig_aspect)`.
fn shrunk_to_aspect(p: &Points, box_aspect: f64, fig_aspect: f64) -> Result<Points> {
    if box_aspect <= 0.0 || fig_aspect <= 0.0 {
        return unsupported("'box_aspect' and 'fig_aspect' must be positive");
    }
    let (w, h) = (p[2] - p[0], p[3] - p[1]);
    let mut big_h = w * box_aspect / fig_aspect;
    let big_w = if big_h <= h { w } else {
        let w2 = h * fig_aspect / box_aspect;
        big_h = h;
        w2
    };
    Ok([p[0], p[1], p[0] + big_w, p[1] + big_h])
}

/// `Bbox.anchored(c, container)`.
fn anchored(p: &Points, c: (f64, f64), container: &Points) -> Points {
    let (l, b, w, h) = crate::transform::bounds(container);
    let (pl, pb, pw, ph) = crate::transform::bounds(p);
    let dx = (l + c.0 * (w - pw)) - pl;
    let dy = (b + c.1 * (h - ph)) - pb;
    [p[0] + dx, p[1] + dy, p[2] + dx, p[3] + dy]
}

/// `Axes.apply_aspect(position)` (or zlc's override), at draw time.
pub fn apply_aspect(ctx: &mut Ctx<'_>, index: usize) -> Result<()> {
    let ax: &Axes = &ctx.fig.axes[index];
    match ax.axes_locator.as_deref() {
        None | Some("colorbar_neither") => {}
        Some(other) => return unsupported(format!("axes locator {other}")),
    }
    // The locator returns the original position, which is also the default.
    let position = ax.position_original;
    if ax.apply_aspect == "position" {
        ctx.eval.set_position(index, position);
        return Ok(());
    }
    if ax.apply_aspect != "matplotlib" {
        return unsupported(format!("apply_aspect {}", ax.apply_aspect));
    }
    let aspect = match &ax.aspect {
        Aspect::Auto(s) if s == "auto" => None,
        Aspect::Auto(s) => return unsupported(format!("aspect {s}")),
        Aspect::Value(v) => Some(*v),
    };
    if aspect.is_none() && ax.box_aspect.is_none() {
        ctx.eval.set_position(index, position);
        return Ok(());
    }
    if ax.adjustable != "box" {
        return unsupported(format!("adjustable {}", ax.adjustable));
    }
    let Some(aspect) = aspect else { return unsupported("box aspect with auto data aspect") };
    if ax.box_aspect.is_some() { return unsupported("box_aspect with a data aspect"); }
    let trans = ctx.eval.matrix(ctx.fig.trans_figure)?;
    let bb = bbox_transformed(&trans, &[0.0, 0.0, 1.0, 1.0]);
    let fig_aspect = (bb[3] - bb[1]) / (bb[2] - bb[0]);
    let vl = ctx.eval.view_lims[index];
    let (x0, x1) = if vl[0] < vl[2] { (vl[0], vl[2]) } else { (vl[2], vl[0]) };
    let (y0, y1) = if vl[1] < vl[3] { (vl[1], vl[3]) } else { (vl[3], vl[1]) };
    // get_data_ratio: the bounds through each axis' scale transform.
    let (xt, yt) = (ctx.fig.axes[index].xaxis.transform, ctx.fig.axes[index].yaxis.transform);
    let (x0, x1) = (ctx.eval.non_affine_1d(xt, x0)?, ctx.eval.non_affine_1d(xt, x1)?);
    let (y0, y1) = (ctx.eval.non_affine_1d(yt, y0)?, ctx.eval.non_affine_1d(yt, y1)?);
    let xsize = (x1 - x0).abs().max(1e-30);
    let ysize = (y1 - y0).abs().max(1e-30);
    let data_ratio = ysize / xsize;
    let box_aspect = aspect * data_ratio;
    let pb1 = shrunk_to_aspect(&position, box_aspect, fig_aspect)?;
    let placed = anchored(&pb1, anchor_coefs(&ax.anchor)?, &position);
    ctx.eval.set_position(index, placed);
    Ok(())
}

enum Child<'a> {
    Artist(&'a crate::ir::Artist),
    Spine(&'a crate::ir::NamedSpine),
    XAxis,
    YAxis,
    Title(&'a crate::ir::Text),
}

/// `Axes._update_title_position(renderer)`: only its early exits are ported.
pub(crate) fn update_title_position(ax: &Axes) -> Result<()> {
    if ax.autotitlepos == Some(false) { return Ok(()); }
    if ax.titles.iter().all(|t| t.text.is_empty()) { return Ok(()); }
    unsupported("automatic title placement")
}

/// `Axes.draw(renderer)`.
pub fn draw_axes(ctx: &mut Ctx<'_>, index: usize) -> Result<()> {
    let fig = ctx.fig;
    let ax = &fig.axes[index];
    if !ax.visible { return Ok(()); }
    apply_aspect(ctx, index)?;
    let mut children: Vec<(f64, Child<'_>)> = Vec::new();
    for a in &ax.children {
        children.push((crate::draw::artist_zorder(fig, a), Child::Artist(a)));
    }
    if ax.axison && ax.frameon {
        for s in &ax.spines {
            children.push((s.patch.artist.zorder, Child::Spine(s)));
        }
    }
    update_title_position(ax)?;
    if ax.axison {
        children.push((ax.xaxis.zorder, Child::XAxis));
        children.push((ax.yaxis.zorder, Child::YAxis));
    }
    for t in &ax.titles {
        children.push((t.artist.zorder, Child::Title(t)));
    }
    sort_by_zorder(&mut children)?;
    if ax.axison && ax.frameon {
        crate::patches::draw_patch(ctx, &ax.patch)?;
    }
    for (_, child) in children {
        match child {
            Child::Artist(a) => draw_artist(ctx, a)?,
            Child::Spine(s) => crate::patches::draw_patch_in(ctx, &s.patch, Some(index))?,
            Child::XAxis => crate::axis::draw_axis(ctx, index, &ax.xaxis)?,
            Child::YAxis => crate::axis::draw_axis(ctx, index, &ax.yaxis)?,
            Child::Title(t) => crate::text::draw_text(ctx, t, None)?,
        }
    }
    Ok(())
}
