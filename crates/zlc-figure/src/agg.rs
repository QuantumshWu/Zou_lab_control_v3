//! [`Backend`] over `zlc-agg`'s RendererAgg, converting arguments as
//! matplotlib's pybind11 wrapper and converters do (the same conversions as
//! `zlc_python.RendererAgg`): the `GCAgg` fields, odd dash lists repeated
//! twice, the face-alpha rule, `mpl_round` and `alpha = 1` for images.
use zlc_agg::basics::{mpl_round, RectD};
use zlc_agg::color::Rgba;
use zlc_agg::path::{PathData, SnapMode};
use zlc_agg::trans_affine::TransAffine;
use zlc_agg::vcgen::{LineCap, LineJoin};
use zlc_agg::{AggError, ClipPath, Collection, Dashes, Gc as AggGc, RendererAgg, SketchParams};

use crate::backend::{Backend, Gc, GrayImage, PathRef, RgbaImage};
use crate::ir::{CapStyle, Dash, JoinStyle};
use crate::transform::M3;
use crate::{Error, Result};

fn agg_err(e: AggError) -> Error { Error::Backend(format!("agg: {e:?}")) }

fn affine(m: &M3) -> TransAffine {
    let m = m.0;
    TransAffine { sx: m[0][0], shx: m[0][1], tx: m[0][2], shy: m[1][0], sy: m[1][1], ty: m[1][2] }
}

fn path(p: PathRef<'_>) -> PathData<'_> {
    PathData { vertices: p.vertices, codes: p.codes, should_simplify: p.should_simplify,
               simplify_threshold: p.simplify_threshold, id: p.vertices.as_ptr() as usize }
}

/// The `Dashes` caster: odd on-off lists repeat twice.
fn dashes(offset: Option<f64>, list: Option<&[f64]>) -> Dashes {
    let mut d = Dashes::default();
    let Some(list) = list else { return d };
    let n = list.len();
    let length = if n % 2 == 1 { 2 * n } else { n };
    for i in (0..length).step_by(2) {
        d.pairs.push((list[i % n], list[(i + 1) % n]));
    }
    d.offset = offset.unwrap_or(0.0);
    d
}

fn rgba(c: [f64; 4]) -> Rgba { Rgba::new(c[0], c[1], c[2], c[3]) }

fn gc(g: &Gc) -> Result<AggGc<'_>> {
    let hatchpath = match &g.hatch_path {
        Some(h) => PathData { vertices: &h.vertices, codes: Some(&h.codes), should_simplify: h.should_simplify,
                              simplify_threshold: h.simplify_threshold, id: h.vertices.as_ptr() as usize },
        None => PathData::EMPTY,
    };
    Ok(AggGc {
        linewidth: g.linewidth,
        alpha: g.alpha,
        forced_alpha: g.forced_alpha,
        color: rgba(g.rgb),
        isaa: g.antialiased,
        cap: match g.capstyle { CapStyle::Butt => LineCap::Butt, CapStyle::Round => LineCap::Round, CapStyle::Projecting => LineCap::Square },
        join: match g.joinstyle { JoinStyle::Miter => LineJoin::MiterRevert, JoinStyle::Round => LineJoin::Round, JoinStyle::Bevel => LineJoin::Bevel },
        cliprect: match g.cliprect { Some(c) => RectD::new(c[0], c[1], c[2], c[3]), None => RectD::default() },
        clippath: ClipPath { path: PathData::EMPTY, trans: TransAffine::IDENTITY },
        dashes: dashes(g.dashes.0, g.dashes.1.as_deref()),
        snap_mode: match g.snap { None => SnapMode::Auto, Some(true) => SnapMode::True, Some(false) => SnapMode::False },
        hatchpath,
        hatch_color: rgba(g.hatch_color),
        hatch_linewidth: g.hatch_linewidth,
        sketch: SketchParams::default(),
    })
}

/// The wrapper's face: an RGBA face takes the context's alpha under forced alpha.
fn face(face: Option<[f64; 4]>, g: &AggGc<'_>) -> Rgba {
    match face {
        None => Rgba::new(0.0, 0.0, 0.0, 0.0),
        Some(f) => {
            let mut c = rgba(f);
            if g.forced_alpha { c.a = g.alpha; }
            c
        }
    }
}

/// A figure canvas: matplotlib's `RendererAgg(int(width), int(height), dpi)`.
pub struct AggBackend { pub renderer: RendererAgg }

impl AggBackend {
    pub fn new(width: f64, height: f64, dpi: f64) -> Result<AggBackend> {
        Ok(AggBackend { renderer: RendererAgg::new(width as u32, height as u32, dpi).map_err(agg_err)? })
    }

    /// A canvas over a recycled allocation that keeps its contents.
    pub fn over_pixels(width: f64, height: f64, dpi: f64, pixels: Vec<u8>) -> Result<AggBackend> {
        Ok(AggBackend { renderer: RendererAgg::over_pixels(width as u32, height as u32, dpi, pixels).map_err(agg_err)? })
    }

    pub fn pixels(&self) -> &[u8] { &self.renderer.pixels }
}

impl Backend for AggBackend {
    fn draw_path(&mut self, g: &Gc, p: PathRef<'_>, trans: &M3, f: Option<[f64; 4]>) -> Result<()> {
        let g = gc(g)?;
        let color = face(f, &g);
        self.renderer.draw_path(&g, path(p), affine(trans), color).map_err(agg_err)
    }

    fn draw_markers(&mut self, g: &Gc, marker_path: PathRef<'_>, marker_trans: &M3, p: PathRef<'_>, trans: &M3,
                    f: Option<[f64; 4]>) -> Result<()> {
        let g = gc(g)?;
        let color = face(f, &g);
        self.renderer.draw_markers(&g, path(marker_path), affine(marker_trans), path(p), affine(trans), color)
            .map_err(agg_err)
    }

    fn draw_text_image(&mut self, image: &dyn GrayImage, x: i32, y: i32, angle: f64, g: &Gc) -> Result<()> {
        let g = gc(g)?;
        self.renderer.draw_text_image(&g, image.pixels(), image.rows(), image.cols(), x, y, angle).map_err(agg_err)
    }

    fn draw_image(&mut self, g: &Gc, x: f64, y: f64, image: &RgbaImage) -> Result<()> {
        let mut g = gc(g)?;
        g.alpha = 1.0;
        self.renderer.draw_image(&g, mpl_round(x), mpl_round(y), &image.data, image.rows, image.cols).map_err(agg_err)
    }

    fn draw_path_collection(&mut self, g: &Gc, master_transform: &M3, paths: &[PathRef<'_>],
                            transforms: &[[[f64; 3]; 3]], offsets: &[[f64; 2]], offset_trans: &M3,
                            facecolors: &[[f64; 4]], edgecolors: &[[f64; 4]], linewidths: &[f64],
                            dash_list: &[Dash], antialiaseds: &[bool]) -> Result<()> {
        let g = gc(g)?;
        let data: Vec<PathData<'_>> = paths.iter().map(|p| path(*p)).collect();
        let linestyles: Vec<Dashes> = dash_list.iter().map(|d| dashes(Some(d.0), d.1.as_deref())).collect();
        let aa: Vec<u8> = antialiaseds.iter().map(|&b| u8::from(b)).collect();
        let items = Collection { transforms, offsets, offset_trans: affine(offset_trans), facecolors, edgecolors,
                                 linewidths, linestyles: &linestyles, antialiaseds: &aa };
        self.renderer.draw_path_collection(&g, affine(master_transform), &data, data.len(), &items, usize::MAX)
            .map_err(agg_err)
    }

    fn draw_quad_mesh(&mut self, g: &Gc, master_transform: &M3, mesh_width: usize, mesh_height: usize,
                      coordinates: &[[f64; 2]], offsets: &[[f64; 2]], offset_trans: &M3, facecolors: &[[f64; 4]],
                      antialiased: bool, edgecolors: &[[f64; 4]]) -> Result<()> {
        let g = gc(g)?;
        self.renderer.draw_quad_mesh(&g, affine(master_transform), mesh_width, mesh_height, coordinates, offsets,
                                     affine(offset_trans), facecolors, antialiased, edgecolors).map_err(agg_err)
    }
}
