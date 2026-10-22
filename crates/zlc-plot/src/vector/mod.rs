//! PDF and SVG export of a figure: zlc-figure draws the materialized IR, as
//! for PNG, through [`VectorBackend`] -- matplotlib's vector renderer seams
//! (text as placed glyphs, measured as `RendererSVG` measures it; images at
//! their own resolution) -- into a [`Page`]; [`svg`] (glyph outlines) and
//! [`pdf`] (embedded TrueType text) write the page. Layout is decided once,
//! by the draw.
//!
//! Like `print_svg`/`print_pdf` after `savefig(dpi=export dpi)`, the figure is
//! materialized at the export dpi and drawn at 72 dpi, so one display pixel
//! is one point; images keep the resolution their data has, up to the export's.
use std::cell::Cell;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use zlc_agg::basics::*;
use zlc_agg::path::{ConvTransform, PathClipper, PathData, PathIterator, PathNanRemover, PathSimplifier};
use zlc_agg::trans_affine::TransAffine;
use zlc_figure::backend::{Backend, Gc, GrayImage, PathRef, RgbaImage, TextRun};
use zlc_figure::ir::{self, CapStyle, Dash, JoinStyle};
use zlc_figure::transform::M3;
use zlc_figure::{Error, Result};
use zlc_text::FT2Font;

pub(crate) mod pdf;
pub(crate) mod sfnt;
pub(crate) mod svg;

/// The vector formats.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format { Svg, Pdf }

impl Format {
    pub fn parse(format: &str) -> Option<Format> {
        match format { "svg" => Some(Format::Svg), "pdf" => Some(Format::Pdf), _ => None }
    }
}

/// A box `[x0, y0, x1, y1]` in page space (points, y up).
pub(crate) type Rect = [f64; 4];

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Seg { Move(f64, f64), Line(f64, f64), Quad(f64, f64, f64, f64), Cubic(f64, f64, f64, f64, f64, f64), Close }

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Stroke { pub color: [f64; 4], pub width: f64, pub cap: CapStyle, pub join: JoinStyle, pub dashes: Option<(f64, Vec<f64>)> }

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Paint { pub fill: Option<[f64; 4]>, pub stroke: Option<Stroke> }

/// A marker shape drawn at many places (in points around its place).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Symbol { pub segs: Vec<Seg>, pub paint: Paint }

/// One placed glyph of a text item: in the item's frame (baseline start, y up).
#[derive(Clone, Debug)]
pub(crate) struct Glyph { pub face: usize, pub size: f64, pub gid: u16, pub x: f64, pub y: f64 }

pub(crate) enum Item {
    Path { segs: Vec<Seg>, paint: Paint, clip: Option<Rect> },
    /// A symbol at each point.
    Uses { symbol: usize, at: Vec<(f64, f64)>, clip: Option<Rect> },
    /// RGBA rows (row 0 first) and the map of their pixel grid to the page.
    Image { cols: usize, rows: usize, rgba: Vec<u8>, matrix: M3, opacity: f64, clip: Option<Rect> },
    /// A line of text at (x, y), rotated `angle` degrees: its glyphs and rules.
    Text { x: f64, y: f64, angle: f64, color: [f64; 4], glyphs: Vec<Glyph>, rules: Vec<Rect>, clip: Option<Rect> },
}

/// A font a page uses: its file, a face to lay text out (and take glyph
/// outlines) with, and the glyphs drawn from it with the characters each
/// stands for.
pub(crate) struct Face {
    pub data: Arc<Vec<u8>>,
    pub font: FT2Font,
    pub postscript: String,
    pub italic: bool,
    pub used: BTreeMap<u16, String>,
}

/// What a draw leaves for a writer.
pub(crate) struct Page {
    pub width: f64,
    pub height: f64,
    pub items: Vec<Item>,
    pub symbols: Vec<Symbol>,
    pub faces: Vec<Face>,
}

/// `FT_KERNING_UNFITTED`: what `_text_helpers.layout` kerns with.
const KERNING_UNFITTED: u32 = 1;

use zlc_figure::text::FONT_SCALE;

/// matplotlib's vector renderer seams over a [`Page`].
pub(crate) struct VectorBackend {
    page: Page,
    /// Points per display unit (1 at the 72 dpi the page is drawn at).
    scale: f64,
    /// Image pixels per display unit images are sampled at (the export dpi's).
    magnification: f64,
    hinting_factor: i64,
    kerning_factor: i64,
    faces: HashMap<String, usize>,
    symbols: HashMap<String, usize>,
}

fn err(what: impl std::fmt::Display) -> Error { Error::Backend(format!("vector export: {what}")) }

fn affine(m: &M3) -> TransAffine {
    let m = m.0;
    TransAffine { sx: m[0][0], shx: m[0][1], tx: m[0][2], shy: m[1][0], sy: m[1][1], ty: m[1][2] }
}

impl VectorBackend {
    pub(crate) fn new(width: f64, height: f64, dpi: f64, image_dpi: f64, rc: &ir::Rc) -> VectorBackend {
        VectorBackend {
            page: Page { width, height, items: Vec::new(), symbols: Vec::new(), faces: Vec::new() },
            scale: 72.0 / dpi, magnification: image_dpi / dpi, hinting_factor: rc.text_hinting_factor, kerning_factor: rc.text_kerning_factor,
            faces: HashMap::new(), symbols: HashMap::new(),
        }
    }

    pub(crate) fn finish(self) -> Page { self.page }

    /// A path through matplotlib's `convert_to_string` stages: transformed,
    /// NaNs removed, clipped to the page when `clip`, simplified when asked.
    fn segments(&self, path: PathRef<'_>, trans: &M3, clip: bool, simplify: bool) -> Vec<Seg> {
        let data = PathData { vertices: path.vertices, codes: path.codes, should_simplify: path.should_simplify,
                              simplify_threshold: path.simplify_threshold, id: 0 };
        let t = Cell::new(affine(trans));
        let transformed = ConvTransform::new(PathIterator::new(data), &t);
        let nan_removed = PathNanRemover::new(transformed, true, data.has_codes());
        let clipped = PathClipper::new(nan_removed, clip, self.page.width / self.scale, self.page.height / self.scale);
        let mut source = PathSimplifier::new(clipped, simplify, data.simplify_threshold);
        source.rewind(0);
        let s = self.scale;
        let mut segs = Vec::new();
        let (mut x, mut y) = (0.0, 0.0);
        loop {
            let code = source.vertex(&mut x, &mut y);
            match code & PATH_CMD_MASK {
                PATH_CMD_STOP => break,
                PATH_CMD_MOVE_TO => segs.push(Seg::Move(x * s, y * s)),
                PATH_CMD_LINE_TO => segs.push(Seg::Line(x * s, y * s)),
                PATH_CMD_CURVE3 => {
                    let (cx, cy) = (x, y);
                    if source.vertex(&mut x, &mut y) == PATH_CMD_STOP { break; }
                    segs.push(Seg::Quad(cx * s, cy * s, x * s, y * s));
                }
                PATH_CMD_CURVE4 => {
                    let (c1x, c1y) = (x, y);
                    if source.vertex(&mut x, &mut y) == PATH_CMD_STOP { break; }
                    let (c2x, c2y) = (x, y);
                    if source.vertex(&mut x, &mut y) == PATH_CMD_STOP { break; }
                    segs.push(Seg::Cubic(c1x * s, c1y * s, c2x * s, c2y * s, x * s, y * s));
                }
                PATH_CMD_END_POLY => if is_closed(code) { segs.push(Seg::Close) },
                _ => {}
            }
        }
        segs
    }

    fn clip(&self, gc: &Gc) -> Option<Rect> { gc.cliprect.map(|c| c.map(|v| v * self.scale)) }

    /// The context's stroke (its width and dashes are points already).
    fn stroke(&self, gc: &Gc) -> Option<Stroke> {
        if gc.linewidth == 0.0 || gc.rgb[3] == 0.0 { return None; }
        let dashes = match &gc.dashes {
            (offset, Some(list)) if !list.is_empty() => Some((offset.unwrap_or(0.0), list.clone())),
            _ => None,
        };
        Some(Stroke { color: gc.rgb, width: gc.linewidth, cap: gc.capstyle, join: gc.joinstyle, dashes })
    }

    /// The face of one font file, opened once per page (a file is known by
    /// its name: text names it so, mathtext by its full path).
    fn face(&mut self, file: &str, open: impl FnOnce() -> std::result::Result<Arc<Vec<u8>>, String>) -> Result<usize> {
        let key = std::path::Path::new(file).file_name().map_or_else(|| file.to_lowercase(), |name| name.to_string_lossy().to_lowercase());
        if let Some(&id) = self.faces.get(&key) { return Ok(id); }
        let data = open().map_err(err)?;
        let font = FT2Font::new(data.to_vec(), self.hinting_factor as _, Vec::new(), self.kerning_factor as _).map_err(err)?;
        let text = |name: Option<&std::ffi::CStr>| name.map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let postscript = text(font.postscript_name());
        let face = Face { data, postscript, italic: font.style_flags() & 1 != 0, font, used: BTreeMap::new() };
        self.page.faces.push(face);
        let id = self.page.faces.len() - 1;
        self.faces.insert(key, id);
        Ok(id)
    }

    /// `TextToPath.get_glyphs_with_font`: `_text_helpers.layout(s, font)` at
    /// `FONT_SCALE` and 72 dpi, scaled to the size -- each character in the
    /// first of the font's files that has it, kerned with the previous glyph
    /// of the same file, advanced by its unhinted linear advance.
    fn layout(&mut self, text: &str, font: &ir::Font) -> Result<Vec<Glyph>> {
        let mut files = Vec::with_capacity(font.files.len());
        for name in &font.files {
            files.push(self.face(name, || crate::render::fonts::font_data(name))?);
        }
        let Some(&base) = files.first() else { return Err(err("a font without files")) };
        for &face in &files {
            let font_face = &mut self.page.faces[face].font;
            font_face.clear().map_err(err)?;
            font_face.set_size(FONT_SCALE, 72.0).map_err(err)?;
        }
        let scale = font.size / FONT_SCALE;
        let mut glyphs = Vec::new();
        let mut x = 0.0;
        let mut previous: Option<(usize, u32)> = None;
        for ch in text.chars() {
            let code = u32::from(ch);
            let mut owner = base;
            for &face in &files {
                if self.page.faces[face].font.get_char_index(code.into(), false).map_err(err)? != 0 { owner = face; break; }
            }
            let face = &mut self.page.faces[owner];
            let gid = face.font.get_char_index(code.into(), false).map_err(err)?;
            if let Some((previous_face, previous_gid)) = previous && previous_face == owner {
                x += f64::from(face.font.get_kerning(previous_gid, gid, KERNING_UNFITTED, false).map_err(err)?) / 64.0;
            }
            let loaded = face.font.load_glyph(gid, zlc_figure::fonts::load_flags::NO_HINTING, false).map_err(err)?;
            let gid16 = u16::try_from(gid).map_err(|_| err("a glyph index beyond 65535"))?;
            face.used.entry(gid16).or_insert_with(|| ch.to_string());
            glyphs.push(Glyph { face: owner, size: font.size, gid: gid16, x: x * scale, y: 0.0 });
            x += loaded.linear_hori_advance as f64 / 65536.0;
            previous = Some((owner, gid));
        }
        Ok(glyphs)
    }

    /// A vector parse's glyphs, each from its own math font file.
    fn math_glyphs(&mut self, parse: &zlc_figure::fonts::VectorParse) -> Result<Vec<Glyph>> {
        let mut glyphs = Vec::with_capacity(parse.glyphs.len());
        for g in &parse.glyphs {
            let path = g.font_file.clone();
            let face = self.face(&path.to_string_lossy(), || std::fs::read(&path).map(Arc::new).map_err(|e| format!("{}: {e}", path.display())))?;
            let gid = u16::try_from(g.glyph_id).map_err(|_| err("a glyph index beyond 65535"))?;
            let text = char::from_u32(g.num).map(String::from).unwrap_or_default();
            self.page.faces[face].used.entry(gid).or_insert(text);
            glyphs.push(Glyph { face, size: g.fontsize, gid, x: g.ox, y: g.oy });
        }
        Ok(glyphs)
    }

    fn symbol(&mut self, segs: Vec<Seg>, paint: Paint) -> usize {
        let key = format!("{segs:?}{paint:?}");
        if let Some(&id) = self.symbols.get(&key) { return id; }
        self.page.symbols.push(Symbol { segs, paint });
        let id = self.page.symbols.len() - 1;
        self.symbols.insert(key, id);
        id
    }

    fn image(&mut self, gc: &Gc, image: &RgbaImage, pixel_to_display: &M3) {
        let matrix = M3::IDENTITY.scale(self.scale, self.scale).dot(pixel_to_display);
        let opacity = if gc.forced_alpha { gc.alpha } else { 1.0 };
        self.page.items.push(Item::Image { cols: image.cols, rows: image.rows, rgba: image.data.clone(), matrix, opacity, clip: self.clip(gc) });
    }
}

/// The face colour Agg fills with: under a forced alpha, the context's alpha.
fn face_colour(face: Option<[f64; 4]>, gc: &Gc) -> Option<[f64; 4]> {
    let mut colour = face?;
    if gc.forced_alpha { colour[3] = gc.alpha; }
    (colour[3] != 0.0).then_some(colour)
}

impl Backend for VectorBackend {
    /// Both formats measure as `RendererSVG` does. (`RendererPdf` measures
    /// TrueType text hinted at its size and mathtext with whole-point
    /// metrics, which spaces a readout's lines unlike its PNG.)
    fn vector(&self) -> bool { true }

    fn draw_path(&mut self, gc: &Gc, path: PathRef<'_>, trans: &M3, face: Option<[f64; 4]>) -> Result<()> {
        if gc.hatch_path.is_some() { return Err(err("hatched paths")); }
        let fill = face_colour(face, gc);
        let clip = fill.is_none();
        let segs = self.segments(path, trans, clip, path.should_simplify && clip);
        let paint = Paint { fill, stroke: self.stroke(gc) };
        if segs.is_empty() || (paint.fill.is_none() && paint.stroke.is_none()) { return Ok(()); }
        let clip = self.clip(gc);
        self.page.items.push(Item::Path { segs, paint, clip });
        Ok(())
    }

    fn draw_markers(&mut self, gc: &Gc, marker_path: PathRef<'_>, marker_trans: &M3, path: PathRef<'_>, trans: &M3,
                    face: Option<[f64; 4]>) -> Result<()> {
        if gc.hatch_path.is_some() { return Err(err("hatched markers")); }
        let shape = self.segments(marker_path, marker_trans, false, false);
        let paint = Paint { fill: face_colour(face, gc), stroke: self.stroke(gc) };
        if shape.is_empty() || (paint.fill.is_none() && paint.stroke.is_none()) { return Ok(()); }
        // Places outside the clip (or the page), farther than the marker reaches, draw nothing.
        let reach = shape.iter().map(|seg| match *seg {
            Seg::Move(x, y) | Seg::Line(x, y) => x.abs().max(y.abs()),
            Seg::Quad(a, b, x, y) => a.abs().max(b.abs()).max(x.abs()).max(y.abs()),
            Seg::Cubic(a, b, c, d, x, y) => [a, b, c, d, x, y].iter().fold(0.0f64, |m, v| m.max(v.abs())),
            Seg::Close => 0.0,
        }).fold(0.0f64, f64::max) + paint.stroke.as_ref().map_or(0.0, |s| s.width);
        let clip = self.clip(gc);
        let bounds = clip.unwrap_or([0.0, 0.0, self.page.width, self.page.height]);
        let at: Vec<(f64, f64)> = self.segments(path, trans, false, false).into_iter().filter_map(|seg| match seg {
            Seg::Move(x, y) | Seg::Line(x, y) => Some((x, y)),
            _ => None,
        }).filter(|&(x, y)| x >= bounds[0] - reach && x <= bounds[2] + reach && y >= bounds[1] - reach && y <= bounds[3] + reach).collect();
        if at.is_empty() { return Ok(()); }
        let symbol = self.symbol(shape, paint);
        self.page.items.push(Item::Uses { symbol, at, clip });
        Ok(())
    }

    fn draw_text_image(&mut self, _image: &dyn GrayImage, _x: i32, _y: i32, _angle: f64, _gc: &Gc) -> Result<()> {
        Err(err("a glyph image reached a vector renderer"))
    }

    fn draw_text(&mut self, gc: &Gc, x: f64, y: f64, run: TextRun<'_>, angle: f64) -> Result<()> {
        let (glyphs, rules) = match run {
            TextRun::Plain { text, .. } if text.is_empty() => return Ok(()),
            TextRun::Plain { text, font } => (self.layout(text, font)?, Vec::new()),
            TextRun::Math { parse, .. } => (self.math_glyphs(parse)?, parse.rects.iter().map(|r| [r[0], r[1], r[0] + r[2], r[1] + r[3]]).collect()),
        };
        let clip = self.clip(gc);
        self.page.items.push(Item::Text { x: x * self.scale, y: y * self.scale, angle, color: gc.rgb, glyphs, rules, clip });
        Ok(())
    }

    fn image_magnification(&self) -> f64 { self.magnification }

    fn draw_image(&mut self, gc: &Gc, x: f64, y: f64, image: &RgbaImage) -> Result<()> {
        // A sampled image: its rows from the bottom, `magnification` per display unit.
        let k = 1.0 / self.magnification;
        self.image(gc, image, &M3::IDENTITY.scale(k, k).translate(x, y));
        Ok(())
    }

    fn draw_image_transformed(&mut self, gc: &Gc, image: &RgbaImage, pixel_to_display: &M3) -> Result<()> {
        self.image(gc, image, pixel_to_display);
        Ok(())
    }

    fn draw_path_collection(&mut self, gc: &Gc, master_transform: &M3, paths: &[PathRef<'_>],
                            transforms: &[[[f64; 3]; 3]], offsets: &[[f64; 2]], offset_trans: &M3,
                            facecolors: &[[f64; 4]], edgecolors: &[[f64; 4]], linewidths: &[f64],
                            dashes: &[Dash], _antialiaseds: &[bool]) -> Result<()> {
        if gc.hatch_path.is_some() { return Err(err("hatched collections")); }
        // RendererAgg._draw_path_collection_generic's loop.
        let n = paths.len().max(offsets.len());
        if (facecolors.is_empty() && edgecolors.is_empty()) || paths.is_empty() { return Ok(()); }
        let clip = self.clip(gc);
        for i in 0..n {
            let mut trans = if transforms.is_empty() { *master_transform } else { master_transform.dot(&M3(transforms[i % transforms.len()])) };
            if !offsets.is_empty() {
                let [xo, yo] = offsets[i % offsets.len()];
                let (xo, yo) = offset_trans.apply(xo, yo);
                trans = trans.translate(xo, yo);
            }
            let mut item = gc.clone();
            item.linewidth = 0.0;
            if !edgecolors.is_empty() {
                item.rgb = edgecolors[i % edgecolors.len()];
                item.linewidth = if linewidths.is_empty() { 1.0 } else { linewidths[i % linewidths.len()] };
                let styles = dashes.len().min(n);
                if styles != 0 {
                    let dash = &dashes[i % styles];
                    item.dashes = (Some(dash.0), dash.1.clone());
                }
            }
            let fill = (!facecolors.is_empty()).then(|| facecolors[i % facecolors.len()]).filter(|c| c[3] != 0.0);
            let segs = self.segments(paths[i % paths.len()], &trans, fill.is_none(), false);
            let paint = Paint { fill, stroke: self.stroke(&item) };
            if segs.is_empty() || (paint.fill.is_none() && paint.stroke.is_none()) { continue; }
            self.page.items.push(Item::Path { segs, paint, clip });
        }
        Ok(())
    }

    fn draw_quad_mesh(&mut self, gc: &Gc, master_transform: &M3, mesh_width: usize, mesh_height: usize,
                      coordinates: &[[f64; 2]], offsets: &[[f64; 2]], offset_trans: &M3, facecolors: &[[f64; 4]],
                      _antialiased: bool, edgecolors: &[[f64; 4]]) -> Result<()> {
        let mut trans = *master_transform;
        if let Some(&[xo, yo]) = offsets.first() {
            let (xo, yo) = offset_trans.apply(xo, yo);
            trans = trans.translate(xo, yo);
        }
        let columns = mesh_width + 1;
        let at = |i: usize, j: usize| { let [x, y] = coordinates[i * columns + j]; trans.apply(x, y) };
        if facecolors.is_empty() { return Ok(()); }
        let colour = |k: usize| facecolors[k % facecolors.len()];
        let stroked = !edgecolors.is_empty() && gc.linewidth != 0.0;
        if !stroked && mesh_width > 0 && mesh_height > 0 && offsets.len() <= 1 {
            // A rectilinear, evenly spaced mesh (a colorbar's ramp) is an image
            // of its cells, as matplotlib rasterizes a ramp of 50 or more.
            let (x0, y0) = at(0, 0);
            let (x1, _) = at(0, mesh_width);
            let (_, y1) = at(mesh_height, 0);
            let (dx, dy) = ((x1 - x0) / mesh_width as f64, (y1 - y0) / mesh_height as f64);
            let tolerance = 1e-9 * (dx.abs() + dy.abs() + x0.abs() + y0.abs());
            let even = (0..=mesh_height).all(|i| (0..=mesh_width).all(|j| {
                let (x, y) = at(i, j);
                (x - (x0 + j as f64 * dx)).abs() <= tolerance && (y - (y0 + i as f64 * dy)).abs() <= tolerance
            }));
            if even && dx != 0.0 && dy != 0.0 {
                let byte = |v: f64| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
                let rgba: Vec<u8> = (0..mesh_height * mesh_width).flat_map(|k| colour(k).map(byte)).collect();
                let image = RgbaImage { rows: mesh_height, cols: mesh_width, data: rgba };
                let matrix = M3::from6([dx, 0.0, x0, 0.0, dy, y0]);
                self.image(&Gc { forced_alpha: false, ..gc.clone() }, &image, &matrix);
                return Ok(());
            }
        }
        let clip = self.clip(gc);
        let s = self.scale;
        for i in 0..mesh_height {
            for j in 0..mesh_width {
                let k = i * mesh_width + j;
                let corners = [at(i, j), at(i, j + 1), at(i + 1, j + 1), at(i + 1, j)];
                let mut segs: Vec<Seg> = corners.iter().enumerate()
                    .map(|(n, &(x, y))| if n == 0 { Seg::Move(x * s, y * s) } else { Seg::Line(x * s, y * s) }).collect();
                segs.push(Seg::Close);
                let mut edge = gc.clone();
                if stroked { edge.rgb = edgecolors[k % edgecolors.len()]; } else { edge.linewidth = 0.0; }
                let paint = Paint { fill: Some(colour(k)).filter(|c| c[3] != 0.0), stroke: self.stroke(&edge) };
                if paint.fill.is_none() && paint.stroke.is_none() { continue; }
                self.page.items.push(Item::Path { segs, paint, clip });
            }
        }
        Ok(())
    }
}

/// A materialized figure (for a draw at 72 dpi) drawn into a page, written
/// as `format`; images finer than `image_dpi` are sampled at it.
pub fn export(figure: &ir::Figure, format: Format, image_dpi: f64, fonts: &mut dyn zlc_figure::fonts::Fonts,
              math: &mut dyn zlc_figure::fonts::MathText) -> std::result::Result<Vec<u8>, String> {
    let (width, height) = (figure.size_inches[0] * 72.0, figure.size_inches[1] * 72.0);
    let mut backend = VectorBackend::new(width, height, figure.dpi, image_dpi, &figure.rc);
    zlc_figure::draw(figure, &mut backend, fonts, math).map_err(|e| e.to_string())?;
    let mut page = backend.finish();
    match format {
        Format::Svg => svg::write(&mut page),
        Format::Pdf => pdf::write(&page),
    }
}

/// A number as the writers print it: at most `places` decimals, no trailing zeros.
pub(crate) fn num(value: f64, places: usize) -> String {
    let text = format!("{value:.places$}");
    let text = if text.contains('.') { text.trim_end_matches('0').trim_end_matches('.') } else { &text };
    match text { "-0" | "" => "0".to_owned(), other => other.to_owned() }
}

/// A colour channel as a byte (matplotlib's `rgb2hex`: `round(v * 255)`).
pub(crate) fn byte(value: f64) -> u8 { (value.clamp(0.0, 1.0) * 255.0).round() as u8 }

/// RGBA rows as a PNG (for SVG's `<image>`).
pub(crate) fn png(cols: usize, rows: usize, rgba: &[u8]) -> std::result::Result<Vec<u8>, String> {
    let mut out = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut out, cols as u32, rows as u32);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().map_err(|e| e.to_string())?;
        writer.write_image_data(rgba).map_err(|e| e.to_string())?;
    }
    Ok(out)
}
