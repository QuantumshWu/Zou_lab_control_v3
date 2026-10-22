//! `matplotlib.collections`: `Collection.draw` (with `_CollectionWithSizes`'
//! sizes at the figure dpi and the single-path `draw_markers` shortcut) and
//! `QuadMesh.draw` with its scalar mapping.
use crate::backend::PathRef;
use crate::colors::{map_scalars, ScalarInput};
use crate::draw::Ctx;
use crate::ir::{Collection, QuadMesh};
use crate::transform::M3;
use crate::{unsupported, Result};

/// `Collection.draw(renderer)`.
pub fn draw_collection(ctx: &mut Ctx<'_>, c: &Collection) -> Result<()> {
    // The classes whose draw is Collection.draw (with sizes or ellipses).
    match c.cls.as_str() {
        "PolyCollection" | "PathCollection" | "LineCollection" | "_SegmentBuffered" | "_BarBuffered"
        | "EllipseCollection" => {}
        other => return unsupported(format!("collection {other}")),
    }
    if c.mapped { return unsupported("scalar-mapped collection"); }
    if !c.artist.visible { return Ok(()); }
    // _CollectionWithSizes.draw: set_sizes(self._sizes, figure.dpi).
    let sized: Option<Vec<[[f64; 3]; 3]>> = match (&c.sizes, c.factor) {
        (Some(sizes), Some(factor)) => Some(sizes.iter().map(|&s| {
            let scale = s.sqrt() * ctx.dpi / 72.0 * factor;
            [[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]]
        }).collect()),
        (None, None) => None,
        _ => return crate::invalid("collection sizes without their factor (or the reverse)"),
    };
    // EllipseCollection._set_transforms (angles are zero: see the extractor).
    let mut master = None;
    let ellipse_transforms: Option<Vec<[[f64; 3]; 3]>> = match &c.ellipses {
        None => None,
        Some(e) => {
            let sc = match e.units.as_str() {
                "xy" => 1.0,
                "points" => ctx.dpi / 72.0,
                "inches" => ctx.dpi,
                "dots" => 1.0,
                other => return unsupported(format!("EllipseCollection units {other}")),
            };
            if e.units == "xy" {
                let td = ctx.eval.matrix(ctx.fig.axes[e.axes].trans_data)?;
                let mut m = td;
                m.0[0][2] = 0.0;
                m.0[1][2] = 0.0;
                master = Some(m);
            }
            Some(e.widths.iter().zip(&e.heights).zip(&e.angles).map(|((&w, &h), &a)| {
                let (w, h) = (w * sc, h * sc);
                let (s, c) = (a.sin(), a.cos());
                [[w * c, h * -s, 0.0], [w * s, h * c, 0.0], [0.0, 0.0, 1.0]]
            }).collect())
        }
    };
    let transforms: &[[[f64; 3]; 3]] = match (&sized, &ellipse_transforms, &c.transforms) {
        (Some(t), _, _) => t,
        (None, Some(t), _) => t,
        (None, None, Some(t)) => t,
        _ => return crate::invalid("collection without sizes or transforms"),
    };
    // _prepare_points: a non-affine transform moves the paths (and a
    // non-affine offset transform the offsets); their affines go on.
    let mut moved_paths: Option<Vec<Vec<[f64; 2]>>> = None;
    // Whether those moved paths are new Paths (rcParams' simplification).
    let mut new_paths = false;
    let transform = match master {
        Some(m) => m,
        None => {
            let id = ctx.transform_id(&c.artist)?;
            if !ctx.eval.is_affine(id)? {
                let mut all = Vec::with_capacity(c.paths.len());
                for p in &c.paths {
                    match ctx.eval.path_parts(id, &p.vertices)?.0 {
                        Some(moved) => all.push(moved),
                        None => return crate::invalid("a non-affine transform left a path unmoved"),
                    }
                }
                moved_paths = Some(all);
                new_paths = ctx.eval.path_non_affine_is_new(id)?;
            }
            ctx.eval.affine(id)?
        }
    };
    let rc = &ctx.fig.rc;
    let moved_offsets = ctx.eval.path_parts(c.offset_transform, &c.offsets)?;
    let offset_trf = moved_offsets.1;
    let offsets: &[[f64; 2]] = moved_offsets.0.as_deref().unwrap_or(&c.offsets);
    let mut gc = ctx.new_gc();
    ctx.set_gc_clip(&mut gc, &c.artist)?;
    gc.snap = c.artist.snap;
    let paths: Vec<PathRef<'_>> = c.paths.iter().enumerate().map(|(k, p)| {
        let vertices = match &moved_paths { Some(m) => &m[k], None => &p.vertices };
        let (should_simplify, simplify_threshold) = if new_paths {
            (crate::lines::should_simplify(rc, vertices.len(), p.codes.as_deref()), rc.path_simplify_threshold)
        } else {
            (p.should_simplify, p.simplify_threshold)
        };
        PathRef { vertices, codes: p.codes.as_deref(), should_simplify, simplify_threshold }
    }).collect();
    let single = paths.len() == 1 && transforms.len() <= 1 && c.facecolors.len() == 1 && c.edgecolors.len() == 1
        && c.linewidths.len() == 1 && c.linestyles.iter().all(|d| d.1.is_none()) && c.antialiaseds.len() == 1;
    if single {
        let combined = if transforms.is_empty() { transform } else { transform.dot(&M3(transforms[0])) };
        let extents = crate::patches::path_extents(paths[0].vertices, paths[0].codes, &combined)?;
        let fb = ctx.figure_bbox()?;
        if extents[2] - extents[0] < fb[2] - fb[0] && extents[3] - extents[1] < fb[3] - fb[1] {
            return unsupported("collection draw_markers shortcut");
        }
    }
    if let Some(j) = c.joinstyle { gc.joinstyle = j; }
    if let Some(cap) = c.capstyle { gc.capstyle = cap; }
    ctx.backend.draw_path_collection(&gc, &transform, &paths, transforms, offsets, &offset_trf, &c.facecolors,
                                     &c.edgecolors, &c.linewidths, &c.linestyles, &c.antialiaseds)
}

/// `QuadMesh.draw(renderer)` (flat shading).
pub fn draw_quad_mesh(ctx: &mut Ctx<'_>, q: &QuadMesh) -> Result<()> {
    if q.cls != "QuadMesh" { return unsupported(format!("quad mesh class {}", q.cls)); }
    if !q.artist.visible { return Ok(()); }
    let id = ctx.transform_id(&q.artist)?;
    let affine_mesh = ctx.eval.is_affine(id)?;
    let transform = if affine_mesh { ctx.eval.affine(id)? } else { M3::IDENTITY };
    let moved_offsets = ctx.eval.path_parts(q.offset_transform, &q.offsets)?;
    let offset_trf = moved_offsets.1;
    let offsets: &[[f64; 2]] = moved_offsets.0.as_deref().unwrap_or(&q.offsets);
    // update_scalarmappable: the mapped colours replace face and/or edge.
    let mapped = match &q.array {
        Some(a) => Some(map_scalars(&ScalarInput { data: a, mask: q.array_mask.as_ref() }, &q.cmap, &q.norm, q.artist.alpha)?),
        None => None,
    };
    let facecolors: &[[f64; 4]] = if q.face_is_mapped {
        match &mapped { Some(m) => m, None => return crate::invalid("mapped face without an array") }
    } else {
        match &q.facecolors { Some(f) => f, None => return crate::invalid("quad mesh without facecolors") }
    };
    let edgecolors: &[[f64; 4]] = if q.edge_is_mapped {
        match &mapped { Some(m) => m, None => return crate::invalid("mapped edge without an array") }
    } else {
        match &q.edgecolors { Some(e) => e, None => return crate::invalid("quad mesh without edgecolors") }
    };
    let shape = &q.coordinates.shape;
    if shape.len() != 3 || shape[2] != 2 { return crate::invalid("quad mesh coordinates shape"); }
    let flat = q.coordinates.to_f64();
    let mut coordinates: Vec<[f64; 2]> = flat.chunks_exact(2).map(|c| [c[0], c[1]]).collect();
    if !affine_mesh {
        // A non-affine transform is applied to the coordinates in full.
        for c in &mut coordinates {
            let (x, y) = ctx.eval.transform(id, c[0], c[1])?;
            *c = [x, y];
        }
    }
    let mut gc = ctx.new_gc();
    gc.snap = q.artist.snap;
    ctx.set_gc_clip(&mut gc, &q.artist)?;
    gc.set_linewidth(match q.linewidths.first() { Some(w) => *w, None => return crate::invalid("quad mesh linewidths") });
    ctx.backend.draw_quad_mesh(&gc, &transform, shape[1] - 1, shape[0] - 1, &coordinates, offsets, &offset_trf,
                               facecolors, q.antialiased, edgecolors)
}
