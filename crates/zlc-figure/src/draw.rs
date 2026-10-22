//! The draw context and the figure-level draw order.
use crate::backend::{Backend, Gc, PathRef};
use crate::fonts::{hinting_flag, Fonts, MathText};
use crate::ir::{self, Artist, ArtistProps};
use crate::transform::{Evaluator, M3, Points};
use crate::{unsupported, Result};

/// Everything a draw reads and writes.
pub struct Ctx<'a> {
    pub fig: &'a ir::Figure,
    pub eval: Evaluator<'a>,
    pub backend: &'a mut dyn Backend,
    pub fonts: &'a mut dyn Fonts,
    pub math: &'a mut dyn MathText,
    /// `RendererAgg.width/height` (the figure bbox size, floats) and dpi.
    pub width: f64,
    pub height: f64,
    pub dpi: f64,
    pub hinting: i32,
    /// A vector renderer ([`Backend::vector`]).
    pub vector: bool,
}

impl<'a> Ctx<'a> {
    pub fn new(fig: &'a ir::Figure, backend: &'a mut dyn Backend, fonts: &'a mut dyn Fonts,
               math: &'a mut dyn MathText) -> Result<Ctx<'a>> {
        let mut eval = Evaluator::new(fig);
        let bbox = eval.points(fig.figure_bbox)?;
        let vector = backend.vector();
        Ok(Ctx {
            fig,
            eval,
            backend,
            fonts,
            math,
            width: bbox[2] - bbox[0],
            height: bbox[3] - bbox[1],
            dpi: fig.dpi,
            hinting: hinting_flag(&fig.rc.text_hinting)?,
            vector,
        })
    }

    pub fn new_gc(&self) -> Gc { Gc::new(self.fig.rc.hatch_color, self.fig.rc.hatch_linewidth) }

    /// `renderer.points_to_pixels(points)`.
    pub fn points_to_pixels(&self, points: f64) -> f64 { points * self.dpi / 72.0 }

    /// `Artist._set_gc_clip(gc)` (clip paths are refused at extraction).
    pub fn set_gc_clip(&mut self, gc: &mut Gc, props: &ArtistProps) -> Result<()> {
        if props.clip_on {
            if let Some(clip) = props.clip_box {
                gc.cliprect = Some(self.eval.points(clip)?);
            }
        } else {
            gc.cliprect = None;
        }
        Ok(())
    }

    /// An artist's transform when it must be affine.
    pub fn transform(&mut self, props: &ArtistProps) -> Result<M3> {
        let id = self.transform_id(props)?;
        self.eval.matrix(id)
    }

    pub fn transform_id(&self, props: &ArtistProps) -> Result<crate::ir::NodeId> {
        match props.transform {
            Some(t) => Ok(t),
            None => unsupported("artist without a transform"),
        }
    }

    /// `artist.get_transform().transform((x, y))`.
    pub fn point(&mut self, props: &ArtistProps, x: f64, y: f64) -> Result<(f64, f64)> {
        let id = self.transform_id(props)?;
        self.eval.transform(id, x, y)
    }

    pub fn figure_bbox(&mut self) -> Result<Points> { self.eval.points(self.fig.figure_bbox) }

    /// `backend_agg.RendererAgg.draw_path`: a simplifiable, unfilled,
    /// unhatched path longer than `agg.path.chunksize` (> 100) reaches the
    /// C++ renderer as chunks of `ceil(npts / ceil(npts / nmax))` vertices,
    /// each but the last one vertex short (`i1 = next i0 - 1`), every chunk
    /// a new `Path` (its `should_simplify` from rcParams, the original's
    /// threshold) whose first code is MOVETO.
    pub fn draw_path(&mut self, gc: &Gc, path: PathRef<'_>, trans: &M3, face: Option<[f64; 4]>) -> Result<()> {
        let nmax = self.fig.rc.agg_path_chunksize;
        let npts = path.vertices.len();
        if self.vector || !((npts as i64) > nmax && nmax > 100 && path.should_simplify && face.is_none() && gc.hatch.is_none()) {
            return self.backend.draw_path(gc, path, trans, face);
        }
        let nch = (npts as f64 / nmax as f64).ceil();
        let chsize = (npts as f64 / nch).ceil() as usize;
        let starts: Vec<usize> = (0..npts).step_by(chsize).collect();
        for (k, &i0) in starts.iter().enumerate() {
            let i1 = match starts.get(k + 1) { Some(&next) => next - 1, None => npts };
            let vertices = &path.vertices[i0..i1];
            let codes: Option<Vec<u8>> = path.codes.map(|c| {
                let mut c = c[i0..i1].to_vec();
                c[0] = MOVETO;
                c
            });
            let chunk = PathRef {
                vertices,
                codes: codes.as_deref(),
                should_simplify: crate::lines::should_simplify(&self.fig.rc, vertices.len(), codes.as_deref()),
                simplify_threshold: path.simplify_threshold,
            };
            self.backend.draw_path(gc, chunk, trans, face)?;
        }
        Ok(())
    }
}

/// `Path.MOVETO`.
pub const MOVETO: u8 = 1;

/// Matplotlib's `sorted(artists, key=zorder)`: stable. A NaN zorder is
/// refused: `<` is then no order, and Python's sort and this one place the
/// artists differently.
pub fn sort_by_zorder<T>(items: &mut [(f64, T)]) -> Result<()> {
    if items.iter().any(|item| item.0.is_nan()) { return unsupported("a NaN zorder"); }
    // (partial_cmp, not total_cmp: -0.0 and 0.0 tie, as with Python's `<`.)
    items.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    Ok(())
}

pub fn artist_zorder(fig: &ir::Figure, artist: &Artist) -> f64 {
    match artist {
        Artist::Axes { index } => fig.axes[*index].zorder,
        other => other.props().map(|p| p.zorder).unwrap_or(0.0),
    }
}

/// `Figure.draw(renderer)`.
pub fn draw_figure(ctx: &mut Ctx<'_>) -> Result<()> {
    let fig = ctx.fig;
    if !fig.visible { return Ok(()); }
    if fig.suppress_composite == Some(false) {
        return unsupported("image compositing (suppressComposite False)");
    }
    // _get_draw_artists: sorted children (patch excluded), then every axes'
    // apply_aspect before anything draws.
    let mut artists: Vec<(f64, &Artist)> = fig.children.iter().map(|a| (artist_zorder(fig, a), a)).collect();
    sort_by_zorder(&mut artists)?;
    for index in 0..fig.axes.len() {
        crate::axes::apply_aspect(ctx, index)?;
    }
    crate::patches::draw_patch(ctx, &fig.patch)?;
    for (_, artist) in artists {
        draw_artist(ctx, artist)?;
    }
    Ok(())
}

/// One artist's `draw(renderer)`.
pub fn draw_artist(ctx: &mut Ctx<'_>, artist: &Artist) -> Result<()> {
    match artist {
        Artist::Axes { index } => crate::axes::draw_axes(ctx, *index),
        Artist::Text(t) => crate::text::draw_text(ctx, t, None),
        Artist::Line(l) => crate::lines::draw_line(ctx, l),
        Artist::Patch(p) => crate::patches::draw_patch(ctx, p),
        Artist::Collection(c) => crate::collections::draw_collection(ctx, c),
        Artist::QuadMesh(q) => crate::collections::draw_quad_mesh(ctx, q),
        Artist::Image(i) => crate::image::draw_image(ctx, i),
        Artist::Group(g) => {
            if g.cls != "_FacetChromeLabels" { return unsupported(format!("artist group {}", g.cls)); }
            if !g.artist.visible { return Ok(()); }
            for t in &g.texts {
                crate::text::draw_text(ctx, t, None)?;
            }
            Ok(())
        }
    }
}

/// Draw a whole figure through `backend`, measuring text with `fonts`/`math`.
pub fn draw(fig: &ir::Figure, backend: &mut dyn Backend, fonts: &mut dyn Fonts, math: &mut dyn MathText) -> Result<()> {
    if fig.version != 1 {
        return unsupported(format!("IR version {}", fig.version));
    }
    let mut ctx = Ctx::new(fig, backend, fonts, math)?;
    draw_figure(&mut ctx)
}
