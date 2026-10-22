//! `matplotlib.lines.Line2D.draw`.
use crate::backend::PathRef;
use crate::draw::Ctx;
use crate::ir::{Color, Line2D, MarkEvery, Rc, Simplify};
use crate::transform::M3;
use crate::{unsupported, Result};

/// `mcolors.to_rgba(c, alpha)` for a resolved colour.
pub fn to_rgba_alpha(c: &Color, alpha: Option<f64>) -> [f64; 4] {
    if c.none {
        [0.0, 0.0, 0.0, 0.0]
    } else if let Some(a) = alpha {
        [c.rgba[0], c.rgba[1], c.rgba[2], a]
    } else {
        c.rgba
    }
}

/// `Path.__init__`'s `_update_values` for a code-less path of `n` vertices.
pub fn should_simplify(rc: &Rc, n: usize, codes: Option<&[u8]>) -> bool {
    rc.path_simplify_threshold > 0.0 && rc.path_simplify && n >= 128
        && codes.is_none_or(|c| c.iter().all(|&code| code <= 2))
}

/// numpy `searchsorted` on a sorted array of `n` values `x(i)` (no NaN).
fn searchsorted(n: usize, x: impl Fn(usize) -> f64, v: f64, right: bool) -> usize {
    let (mut lo, mut hi) = (0usize, n);
    while lo < hi {
        let mid = (lo + hi) / 2;
        let go_right = if right { x(mid) <= v } else { x(mid) < v };
        if go_right { lo = mid + 1; } else { hi = mid; }
    }
    lo
}

fn is_dashed(linestyle: &str) -> bool { matches!(linestyle, "--" | "-." | ":") }

fn draws_nothing(linestyle: &str) -> Result<bool> {
    Ok(match linestyle {
        "-" | "--" | "-." | ":" => false,
        "None" | " " | "" => true,
        other => return unsupported(format!("linestyle {other:?}")),
    })
}

/// `Line2D.draw(renderer)`.
pub fn draw_line(ctx: &mut Ctx<'_>, l: &Line2D) -> Result<()> {
    if !l.artist.visible { return Ok(()); }
    if let Some(b) = &l.bracket {
        let placed = bracket_line(ctx, l, b)?;
        let mut plain = placed;
        plain.bracket = None;
        return draw_line(ctx, &plain);
    }
    let rc = &ctx.fig.rc;
    // The recache's `_subslice` (as the last recache decided it, stale or
    // not): the draw strokes only the view's span, searched in `_x_filled`.
    let (mut start, mut end) = (0usize, l.xy.len());
    if l.subslice {
        let Some(axes) = l.axes else { return crate::invalid("subsliced line without an axes") };
        let n = l.xy.len();
        let x_filled = |i: usize| match &l.x_filled { Some(x) => x[i], None => l.xy[i][0] };
        if l.x_filled.as_ref().is_some_and(|x| x.len() != n) { return crate::invalid("x_filled and xy differ in length"); }
        let vl = match ctx.eval.view_lims.get(axes) { Some(vl) => *vl, None => return crate::invalid("line axes out of range") };
        let (x0, x1) = if vl[0] < vl[2] { (vl[0], vl[2]) } else { (vl[2], vl[0]) };
        let i0 = searchsorted(n, x_filled, x0, false);
        let i1 = searchsorted(n, x_filled, x1, true);
        start = i0.saturating_sub(1).min(n);
        end = (i1 + 1).min(n);
        if start > end { start = end; }
    }
    let data = &l.xy[start..end];
    // TransformedPath: the non-affine part applied to the vertices, the affine passed on.
    let id = ctx.transform_id(&l.artist)?;
    let (moved, affine) = ctx.eval.path_parts(id, data)?;
    let vertices: &[[f64; 2]] = moved.as_deref().unwrap_or(data);
    // The simplification the renderer is handed: a subsliced line's paths
    // are new each draw (rcParams'), any other's are the cached
    // TransformedPath's.
    let (stroke, points) = if l.subslice {
        let fresh = Simplify { should_simplify: should_simplify(rc, data.len(), None), threshold: rc.path_simplify_threshold };
        (fresh, fresh)
    } else {
        match (l.stroke_simplify, l.points_simplify) {
            (Some(s), Some(p)) => (s, p),
            _ => return crate::invalid("a line's transformed-path simplification missing"),
        }
    };
    let (simplify, threshold) = (stroke.should_simplify, stroke.threshold);

    if !draws_nothing(&l.linestyle)? && !vertices.is_empty() {
        let mut gc = ctx.new_gc();
        ctx.set_gc_clip(&mut gc, &l.artist)?;
        gc.antialiased = l.antialiased;
        gc.set_linewidth(l.linewidth);
        let (cap, join) = if is_dashed(&l.linestyle) {
            (l.dash_capstyle, l.dash_joinstyle)
        } else {
            (l.solid_capstyle, l.solid_joinstyle)
        };
        gc.joinstyle = join;
        gc.capstyle = cap;
        gc.snap = l.artist.snap;
        let lc = to_rgba_alpha(&l.color, l.artist.alpha);
        gc.set_foreground_rgba(lc);
        gc.set_dashes(&l.dash_pattern)?;
        let path = PathRef { vertices, codes: None, should_simplify: simplify, simplify_threshold: threshold };
        ctx.draw_path(&gc, path, &affine, None)?;
    }

    if let Some(marker) = &l.marker {
        if l.markersize > 0.0 && !marker.path.vertices.is_empty() {
            let mut gc = ctx.new_gc();
            ctx.set_gc_clip(&mut gc, &l.artist)?;
            gc.set_linewidth(l.markeredgewidth);
            gc.antialiased = l.antialiased;
            let mut ec = to_rgba_alpha(&l.markeredgecolor, l.artist.alpha);
            let fc = to_rgba_alpha(&l.markerfacecolor, l.artist.alpha);
            if l.mec_auto && !l.mfc_none {
                ec = [ec[0], ec[1], ec[2], fc[3]];
            }
            gc.set_foreground_rgba(ec);
            if !vertices.is_empty() {
                // `_transformed_points`; `_mark_every_path` makes a new Path.
                let picked: Vec<[f64; 2]>;
                let (points, points_simplify, points_threshold) = match &l.markevery {
                    None => (vertices, points.should_simplify, points.threshold),
                    Some(index) => {
                        // numpy raises (and so matplotlib) for an index out of
                        // range or a mask of another length.
                        picked = match index {
                            MarkEvery::Indices(indices) => indices.iter().map(|&i| {
                                let n = vertices.len() as i64;
                                let k = if i < 0 { i + n } else { i };
                                if k < 0 || k >= n { Err(crate::Error::Invalid(format!("markevery index {i}"))) }
                                else { Ok(vertices[k as usize]) }
                            }).collect::<Result<_>>()?,
                            MarkEvery::Mask(mask) if mask.len() == vertices.len() => {
                                vertices.iter().zip(mask).filter(|&(_, &m)| m).map(|(v, _)| *v).collect()
                            }
                            MarkEvery::Mask(mask) => {
                                return crate::invalid(format!("markevery mask of {} for {} points", mask.len(), vertices.len()));
                            }
                        };
                        let s = should_simplify(rc, picked.len(), None);
                        (&picked[..], s, rc.path_simplify_threshold)
                    }
                };
                gc.snap = match marker.snap_threshold {
                    Some(t) => Some(ctx.points_to_pixels(l.markersize) >= t),
                    None => None,
                };
                gc.joinstyle = marker.joinstyle;
                gc.capstyle = marker.capstyle;
                let w = ctx.points_to_pixels(l.markersize);
                let is_pixel = matches!(&marker.marker, crate::ir::MarkerId::Name(n) if n == ",");
                let mut marker_trans = M3::from6(marker.transform);
                if is_pixel { gc.set_linewidth(0.0); } else { marker_trans = marker_trans.scale(w, w); }
                let marker_path = PathRef {
                    vertices: &marker.path.vertices,
                    codes: marker.path.codes.as_deref(),
                    should_simplify: marker.path.should_simplify,
                    simplify_threshold: marker.path.simplify_threshold,
                };
                let path = PathRef { vertices: points, codes: None, should_simplify: points_simplify,
                                     simplify_threshold: points_threshold };
                ctx.backend.draw_markers(&gc, marker_path, &marker_trans, path, &affine, Some(fc))?;
                if marker.alt { return unsupported("marker alt path"); }
            }
        }
    }
    Ok(())
}

/// `RepeatBracketSide.draw`: the side's data from the axes' pixels.
fn bracket_line(ctx: &mut Ctx<'_>, l: &Line2D, b: &crate::ir::Bracket) -> Result<Line2D> {
    let Some(axes) = l.axes else { return unsupported("bracket side without an axes") };
    let td = ctx.fig.axes[axes].trans_data;
    let p0 = ctx.eval.transform(td, b.start, 0.0)?.0;
    let p1 = ctx.eval.transform(td, b.stop, 0.0)?.0;
    let span_pixels = (p1 - p0).abs();
    let bbox = ctx.eval.points(ctx.fig.axes[axes].bbox)?;
    let a = (bbox[2] - bbox[0]) * b.foot_fraction;
    let c = span_pixels * b.max_foot_fraction;
    let foot_pixels = if c < a { c } else { a };
    let span = b.stop - b.start;
    let foot = if span_pixels > 0.0 { span * foot_pixels / span_pixels } else { 0.0 };
    let x = if b.at_start { b.start } else { b.stop };
    let toe = if b.at_start { x + foot } else { x - foot };
    let mut top = b.y_high;
    if b.lift_pt != 0.0 {
        let u0 = ctx.eval.transform(td, 0.0, 0.0)?.1;
        let u1 = ctx.eval.transform(td, 0.0, 1.0)?.1;
        let pixels_per_unit = (u1 - u0).abs();
        if pixels_per_unit > 0.0 {
            top += b.lift_pt * ctx.points_to_pixels(1.0) / pixels_per_unit;
        }
    }
    // set_data, then the draw's recache: four vertices, a fresh short path.
    let mut line = l.clone();
    line.xy = vec![[toe, top], [x, top], [x, b.y_low], [toe, b.y_low]];
    let rc = &ctx.fig.rc;
    let fresh = Simplify { should_simplify: should_simplify(rc, line.xy.len(), None), threshold: rc.path_simplify_threshold };
    line.subslice = false;
    line.x_filled = None;
    line.stroke_simplify = Some(fresh);
    line.points_simplify = Some(fresh);
    Ok(line)
}
