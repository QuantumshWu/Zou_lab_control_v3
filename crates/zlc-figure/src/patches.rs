//! `matplotlib.patches.Patch.draw` for Rectangle, PathPatch/Polygon, line
//! Spines (with `_adjust_location`) and the colorbar outline, plus the
//! window extents axis-label placement reads.
use crate::backend::PathRef;
use crate::draw::Ctx;
use crate::ir::{Patch, PatchShape, RotationPoint};
use crate::transform::{M3, Points};
use crate::{unsupported, Result};

/// `Path.unit_rectangle()`.
pub const UNIT_RECTANGLE: [[f64; 2]; 5] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]];
pub const UNIT_RECTANGLE_CODES: [u8; 5] = [1, 2, 2, 2, 79];

/// `Path.unit_circle()` (`Path.circle(center=(0, 0), radius=1)`): 8 cubic Béziers.
pub fn unit_circle() -> (Vec<[f64; 2]>, Vec<u8>) {
    const MAGIC: f64 = 0.2652031;
    let sqrthalf = 0.5f64.sqrt();
    let magic45 = sqrthalf * MAGIC;
    let v = [
        [0.0, -1.0], [MAGIC, -1.0], [sqrthalf - magic45, -sqrthalf - magic45], [sqrthalf, -sqrthalf],
        [sqrthalf + magic45, -sqrthalf + magic45], [1.0, -MAGIC], [1.0, 0.0],
        [1.0, MAGIC], [sqrthalf + magic45, sqrthalf - magic45], [sqrthalf, sqrthalf],
        [sqrthalf - magic45, sqrthalf + magic45], [MAGIC, 1.0], [0.0, 1.0],
        [-MAGIC, 1.0], [-sqrthalf + magic45, sqrthalf + magic45], [-sqrthalf, sqrthalf],
        [-sqrthalf - magic45, sqrthalf - magic45], [-1.0, MAGIC], [-1.0, 0.0],
        [-1.0, -MAGIC], [-sqrthalf - magic45, -sqrthalf + magic45], [-sqrthalf, -sqrthalf],
        [-sqrthalf + magic45, -sqrthalf - magic45], [-MAGIC, -1.0], [0.0, -1.0],
        [0.0, -1.0],
    ];
    // vertices * radius + center, radius 1 and center (0, 0).
    let vertices = v.iter().map(|p: &[f64; 2]| [p[0] * 1.0 + 0.0, p[1] * 1.0 + 0.0]).collect();
    let mut codes = vec![4u8; 26];
    codes[0] = 1;
    codes[25] = 79;
    (vertices, codes)
}

/// `Ellipse._recompute_transform()`.
pub fn ellipse_transform(center: [f64; 2], width: f64, height: f64, angle: f64, arc: f64) -> M3 {
    M3::IDENTITY.scale(width * 0.5, height * 0.5 * arc).rotate_deg(angle).scale(1.0, 1.0 / arc)
        .translate(center[0], center[1])
}

/// `Rectangle.get_patch_transform()`.
pub fn rectangle_transform(x0: f64, y0: f64, width: f64, height: f64, angle: f64, rp: &RotationPoint) -> Result<M3> {
    let (bx0, by0, bx1, by1) = (x0, y0, x0 + width, y0 + height);
    let (rx, ry) = match rp {
        RotationPoint::Named(n) if n == "xy" => (bx0, by0),
        RotationPoint::Named(n) if n == "center" => {
            let (w, h) = (bx1 - bx0, by1 - by0);
            (bx0 + w / 2.0, by0 + h / 2.0)
        }
        RotationPoint::Point(p) => (p[0], p[1]),
        RotationPoint::Named(n) => return unsupported(format!("rotation point {n}")),
    };
    let to = M3([[bx1 - bx0, 0.0, bx0], [0.0, by1 - by0, by0], [0.0, 0.0, 1.0]]);
    let rot = M3::IDENTITY.translate(-rx, -ry).rotate_deg(angle).translate(rx, ry);
    Ok(rot.dot(&to))
}

/// The path a patch draws, with its vertices (spines adjusted to the view).
pub struct PatchPath { pub vertices: Vec<[f64; 2]>, pub codes: Option<Vec<u8>>, pub should_simplify: bool, pub simplify_threshold: f64 }

/// `patch.get_path()` and `patch.get_transform()` (the patch transform
/// composed with the artist transform), as the draw reads them.
pub fn path_and_transform(ctx: &mut Ctx<'_>, p: &Patch, axes: Option<usize>) -> Result<(PatchPath, M3)> {
    let (mut path, patch) = shape_path(ctx, p, axes)?;
    let id = ctx.transform_id(&p.artist)?;
    if ctx.eval.is_affine(id)? {
        let artist = ctx.eval.affine(id)?;
        return Ok((path, match patch { Some(m) => artist.dot(&m), None => artist }));
    }
    // (patch + artist).transform_path_non_affine: the patch transform is
    // affine, so it applies in full before the artist's non-affine part.
    for v in &mut path.vertices {
        let (x, y) = match &patch { Some(m) => m.apply(v[0], v[1]), None => (v[0], v[1]) };
        let (x, y) = ctx.eval.non_affine(id, x, y)?;
        *v = [x, y];
    }
    // A patch transform makes that a composite whose second part is
    // non-affine: a new Path, as is the artist transform's own when it says so.
    if patch.is_some() || ctx.eval.path_non_affine_is_new(id)? {
        let rc = &ctx.fig.rc;
        path.should_simplify = crate::lines::should_simplify(rc, path.vertices.len(), path.codes.as_deref());
        path.simplify_threshold = rc.path_simplify_threshold;
    }
    Ok((path, ctx.eval.affine(id)?))
}

/// A patch's path and its patch transform (None for the identity).
fn shape_path(ctx: &mut Ctx<'_>, p: &Patch, axes: Option<usize>) -> Result<(PatchPath, Option<M3>)> {
    Ok(match &p.shape {
        PatchShape::Rectangle { x0, y0, width, height, angle, rotation_point, path_simplify } => {
            let patch = rectangle_transform(*x0, *y0, *width, *height, *angle, rotation_point)?;
            (PatchPath { vertices: UNIT_RECTANGLE.to_vec(), codes: Some(UNIT_RECTANGLE_CODES.to_vec()),
                         should_simplify: path_simplify.should_simplify, simplify_threshold: path_simplify.threshold },
             Some(patch))
        }
        PatchShape::FancyBbox { .. } => return unsupported("a text box drawn outside its text"),
        PatchShape::Ellipse { center, width, height, angle, aspect_ratio_correction, path_simplify } => {
            let patch = ellipse_transform(*center, *width, *height, *angle, *aspect_ratio_correction);
            let (vertices, codes) = unit_circle();
            (PatchPath { vertices, codes: Some(codes), should_simplify: path_simplify.should_simplify,
                         simplify_threshold: path_simplify.threshold }, Some(patch))
        }
        PatchShape::Path { path } | PatchShape::ColorbarSpine { path } => (
            PatchPath { vertices: path.vertices.clone(), codes: path.codes.clone(),
                        should_simplify: path.should_simplify, simplify_threshold: path.simplify_threshold },
            None,
        ),
        PatchShape::Spine { spine_type, path, bounds, .. } => {
            // `Spine._adjust_location` of a 'line' spine: any type but the four
            // raises (or, 'circle', leaves the path as it is), bounds or not.
            let vertical = match spine_type.as_str() {
                "left" | "right" => true,
                "top" | "bottom" => false,
                other => return unsupported(format!("spine type {other}")),
            };
            let mut v = path.vertices.clone();
            if v.len() != 2 { return unsupported("spine path is not two vertices"); }
            let (low, high) = match bounds {
                Some(b) => (b[0], b[1]),
                None => {
                    let axes = match axes { Some(a) => a, None => return unsupported("spine outside an axes") };
                    let vl = ctx.eval.view_lims[axes];
                    if vertical { (vl[1], vl[3]) } else { (vl[0], vl[2]) }
                }
            };
            if vertical {
                v[0][1] = low;
                v[1][1] = high;
            } else {
                v[0][0] = low;
                v[1][0] = high;
            }
            (PatchPath { vertices: v, codes: path.codes.clone(), should_simplify: path.should_simplify,
                         simplify_threshold: path.simplify_threshold }, None)
        }
    })
}

/// `Patch.draw(renderer)` (for a spine, `axes` is its axes).
pub fn draw_patch_in(ctx: &mut Ctx<'_>, p: &Patch, axes: Option<usize>) -> Result<()> {
    if !p.artist.visible { return Ok(()); }
    let (path, affine) = path_and_transform(ctx, p, axes)?;
    let face = if p.facecolor[3] != 0.0 { Some(p.facecolor) } else { None };
    let mut gc = ctx.new_gc();
    gc.set_foreground_rgba(p.edgecolor);
    let lw = if p.edgecolor[3] == 0.0 || p.linestyle == "None" { 0.0 } else { p.linewidth };
    gc.set_linewidth(lw);
    gc.set_dashes(&p.dash_pattern)?;
    gc.capstyle = p.capstyle;
    gc.joinstyle = p.joinstyle;
    gc.antialiased = p.antialiased;
    ctx.set_gc_clip(&mut gc, &p.artist)?;
    gc.snap = p.artist.snap;
    gc.set_alpha(p.artist.alpha);
    if let Some(h) = &p.hatch {
        gc.hatch = Some(h.pattern.clone());
        gc.hatch_path = Some(crate::backend::hatch_path(&h.pattern, h.path_simplify)?);
        gc.hatch_color = p.hatch_color;
        gc.hatch_linewidth = p.hatch_linewidth;
    }
    let path_ref = PathRef { vertices: &path.vertices, codes: path.codes.as_deref(),
                             should_simplify: path.should_simplify, simplify_threshold: path.simplify_threshold };
    ctx.draw_path(&gc, path_ref, &affine, face)
}

pub fn draw_patch(ctx: &mut Ctx<'_>, p: &Patch) -> Result<()> { draw_patch_in(ctx, p, None) }

/// `Path.get_extents(transform)` for straight-segment paths.
pub fn path_extents(vertices: &[[f64; 2]], codes: Option<&[u8]>, m: &M3) -> Result<Points> {
    let mut xmin = f64::NAN;
    let mut ymin = f64::NAN;
    let mut xmax = f64::NAN;
    let mut ymax = f64::NAN;
    let mut first = true;
    for (i, v) in vertices.iter().enumerate() {
        if let Some(c) = codes {
            match c[i] {
                1 | 2 => {}
                3 | 4 => return unsupported("extents of curved paths"),
                _ => continue,
            }
        }
        let (x, y) = m.apply(v[0], v[1]);
        if first {
            (xmin, xmax, ymin, ymax) = (x, x, y, y);
            first = false;
        } else {
            // numpy min/max reductions propagate NaN.
            xmin = if x.is_nan() || xmin.is_nan() { f64::NAN } else if x < xmin { x } else { xmin };
            xmax = if x.is_nan() || xmax.is_nan() { f64::NAN } else if x > xmax { x } else { xmax };
            ymin = if y.is_nan() || ymin.is_nan() { f64::NAN } else if y < ymin { y } else { ymin };
            ymax = if y.is_nan() || ymax.is_nan() { f64::NAN } else if y > ymax { y } else { ymax };
        }
    }
    if first {
        // Bbox.null()
        return Ok([f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY]);
    }
    Ok([xmin, ymin, xmax, ymax])
}

/// `Patch.get_window_extent()`.
pub fn patch_window_extent(ctx: &mut Ctx<'_>, p: &Patch, axes: Option<usize>) -> Result<Points> {
    let (path, m) = path_and_transform(ctx, p, axes)?;
    path_extents(&path.vertices, path.codes.as_deref(), &m)
}

/// `Path.circle(center, radius)`: the unit-circle template scaled and moved.
pub fn circle(center: (f64, f64), radius: f64) -> (Vec<[f64; 2]>, Vec<u8>) {
    let (unit, codes) = unit_circle();
    (unit.iter().map(|p| [p[0] * radius + center.0, p[1] * radius + center.1]).collect(), codes)
}

/// `FancyBboxPatch.get_path()` for a box at (x, y, w, h) and its mutation scale.
pub fn fancy_bbox_path(style: &crate::ir::BoxStyle, mutation_aspect: f64, x: f64, y: f64, w: f64, h: f64,
                       scale: f64) -> Result<(Vec<[f64; 2]>, Vec<u8>)> {
    let (y0, h0) = (y / mutation_aspect, h / mutation_aspect);
    let pad = scale * style.pad;
    let (vertices, codes) = match style.name.as_str() {
        "Circle" => {
            let (width, height) = (w + 2.0 * pad, h0 + 2.0 * pad);
            let (bx, by) = (x - pad, y0 - pad);
            let radius = (if height > width { height } else { width }) / 2.0;
            circle((bx + width / 2.0, by + height / 2.0), radius)
        }
        "Round" => {
            let dr = match style.rounding_size { Some(r) if r != 0.0 => scale * r, _ => pad };
            let (width, height) = (w + 2.0 * pad, h0 + 2.0 * pad);
            let (x0, y0) = (x - pad, y0 - pad);
            let (x1, y1) = (x0 + width, y0 + height);
            let cp = vec![[x0 + dr, y0], [x1 - dr, y0], [x1, y0], [x1, y0 + dr], [x1, y1 - dr], [x1, y1],
                          [x1 - dr, y1], [x0 + dr, y1], [x0, y1], [x0, y1 - dr], [x0, y0 + dr], [x0, y0],
                          [x0 + dr, y0], [x0 + dr, y0]];
            (cp, vec![1, 2, 3, 3, 2, 3, 3, 2, 3, 3, 2, 3, 3, 79])
        }
        other => return unsupported(format!("boxstyle {other}")),
    };
    Ok((vertices.into_iter().map(|p| [p[0] * 1.0, p[1] * mutation_aspect]).collect(), codes))
}

/// A text's bbox patch drawn at the box its text placed it in.
pub fn draw_fancy_bbox(ctx: &mut Ctx<'_>, p: &Patch, w: f64, h: f64, transform: &M3, scale: f64) -> Result<()> {
    if !p.artist.visible { return Ok(()); }
    let PatchShape::FancyBbox { boxstyle, mutation_aspect } = &p.shape else { return crate::invalid("not a fancy box") };
    let (vertices, codes) = fancy_bbox_path(boxstyle, *mutation_aspect, 0.0, 0.0, w, h, scale)?;
    let face = if p.facecolor[3] != 0.0 { Some(p.facecolor) } else { None };
    let mut gc = ctx.new_gc();
    gc.set_foreground_rgba(p.edgecolor);
    let lw = if p.edgecolor[3] == 0.0 || p.linestyle == "None" { 0.0 } else { p.linewidth };
    gc.set_linewidth(lw);
    gc.set_dashes(&p.dash_pattern)?;
    gc.capstyle = p.capstyle;
    gc.joinstyle = p.joinstyle;
    gc.antialiased = p.antialiased;
    ctx.set_gc_clip(&mut gc, &p.artist)?;
    gc.snap = p.artist.snap;
    gc.set_alpha(p.artist.alpha);
    if let Some(h) = &p.hatch {
        gc.hatch = Some(h.pattern.clone());
        gc.hatch_path = Some(crate::backend::hatch_path(&h.pattern, h.path_simplify)?);
        gc.hatch_color = p.hatch_color;
        gc.hatch_linewidth = p.hatch_linewidth;
    }
    let simplify = crate::lines::should_simplify(&ctx.fig.rc, vertices.len(), Some(&codes));
    let path = PathRef { vertices: &vertices, codes: Some(&codes), should_simplify: simplify,
                         simplify_threshold: ctx.fig.rc.path_simplify_threshold };
    ctx.draw_path(&gc, path, transform, face)
}

/// Agg's `calc_intersection` of lines a-b and c-d.
fn calc_intersection(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> Option<[f64; 2]> {
    let num = (a[1] - c[1]) * (d[0] - c[0]) - (a[0] - c[0]) * (d[1] - c[1]);
    let den = (b[0] - a[0]) * (d[1] - c[1]) - (b[1] - a[1]) * (d[0] - c[0]);
    if den.abs() < 1.0e-30 { return None; }
    let r = num / den;
    Some([a[0] + r * (b[0] - a[0]), a[1] + r * (b[1] - a[1])])
}

/// `vcgen_contour` (miter joins) of a closed convex polygon at `width`:
/// one join vertex per source vertex, as Agg emits them for outer corners.
fn contour(poly: &[[f64; 2]], width: f64) -> Result<Vec<[f64; 2]>> {
    let n = poly.len();
    let w = width * 0.5;
    let mut out = Vec::with_capacity(n);
    for k in 0..n {
        let (v0, v1, v2) = (poly[(k + n - 1) % n], poly[k], poly[(k + 1) % n]);
        let len1 = ((v1[0] - v0[0]).powi(2) + (v1[1] - v0[1]).powi(2)).sqrt();
        let len2 = ((v2[0] - v1[0]).powi(2) + (v2[1] - v1[1]).powi(2)).sqrt();
        let (dx1, dy1) = (w * (v1[1] - v0[1]) / len1, w * (v1[0] - v0[0]) / len1);
        let (dx2, dy2) = (w * (v2[1] - v1[1]) / len2, w * (v2[0] - v1[0]) / len2);
        let cp = (v2[0] - v1[0]) * (v1[1] - v0[1]) - (v2[1] - v1[1]) * (v1[0] - v0[0]);
        if cp != 0.0 && (cp > 0.0) == (w > 0.0) {
            return unsupported("contour inner join (non-convex or clockwise patch)");
        }
        match calc_intersection([v0[0] + dx1, v0[1] - dy1], [v1[0] + dx1, v1[1] - dy1],
                                [v1[0] + dx2, v1[1] - dy2], [v2[0] + dx2, v2[1] - dy2]) {
            Some(p) => out.push(p),
            None => return unsupported("contour of collinear edges"),
        }
    }
    Ok(out)
}

/// `point_in_path_impl` (crossing test) for one closed polygon.
fn point_in_polygon(poly: &[[f64; 2]], tx: f64, ty: f64) -> bool {
    if !(tx.is_finite() && ty.is_finite()) { return false; }
    let (sx, sy) = (poly[0][0], poly[0][1]);
    let (mut vtx0, mut vty0, mut vtx1, mut vty1) = (sx, sy, sx, sy);
    let mut yflag0 = vty0 >= ty;
    let mut inside = false;
    let step = |vtx0: f64, vty0: f64, vtx1: f64, vty1: f64, yflag0: &mut bool, inside: &mut bool| {
        let yflag1 = vty1 >= ty;
        if *yflag0 != yflag1 && (((vty1 - ty) * (vtx0 - vtx1) >= (vtx1 - tx) * (vty0 - vty1)) == yflag1) {
            *inside ^= true;
        }
        *yflag0 = yflag1;
    };
    for p in poly[1..].iter().chain(std::iter::once(&[sx, sy])) {
        step(vtx0, vty0, vtx1, vty1, &mut yflag0, &mut inside);
        vtx0 = vtx1;
        vty0 = vty1;
        vtx1 = p[0];
        vty1 = p[1];
    }
    step(vtx0, vty0, vtx1, vty1, &mut yflag0, &mut inside);
    inside
}

/// `Patch.contains_point(point, radius)` for a Rectangle (as `Axes.contains_point`
/// asks of the axes patch with radius 1).
pub fn rectangle_contains_point(ctx: &mut Ctx<'_>, p: &Patch, point: (f64, f64), radius: f64) -> Result<bool> {
    let (path, m) = path_and_transform(ctx, p, None)?;
    if path.vertices.len() != 5 { return unsupported("contains_point of a non-rectangle"); }
    let poly: Vec<[f64; 2]> = path.vertices[..4].iter().map(|v| { let (x, y) = m.apply(v[0], v[1]); [x, y] }).collect();
    let shape = if radius != 0.0 { contour(&poly, radius)? } else { poly };
    Ok(point_in_polygon(&shape, point.0, point.1))
}
