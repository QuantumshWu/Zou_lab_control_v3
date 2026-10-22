//! Matplotlib's `RendererAgg` (`_backend_agg.h`/`.cpp`): one RGBA canvas,
//! its persistent rasterizer, clip-path alpha mask and hatch tile.
use std::cell::Cell;

use crate::basics::*;
use crate::color::{Rgba, Rgba8};
use crate::curves::ConvCurve;
use crate::path::*;
use crate::pixfmt::*;
use crate::rasterizer::{CellBlockLimit, RasterizerScanlineAa};
use crate::scanline::*;
use crate::span::*;
use crate::trans_affine::{AFFINE_EPSILON, TransAffine};
use crate::vcgen::{ConvDash, ConvStroke, LineCap, LineJoin};

#[derive(Debug, Clone, PartialEq)]
pub enum AggError {
    /// `std::overflow_error`.
    Overflow(String),
    /// `std::range_error`, `std::invalid_argument` or `py::value_error`.
    Value(String),
    /// `std::runtime_error`.
    Runtime(String),
    /// `std::bad_alloc`.
    Memory,
}

impl From<CellBlockLimit> for AggError {
    fn from(_: CellBlockLimit) -> Self { AggError::Overflow("Exceeded cell block limit".into()) }
}

#[derive(Clone, Debug, Default)]
pub struct Dashes { pub offset: f64, pub pairs: Vec<(f64, f64)> }

impl Dashes {
    pub fn is_empty(&self) -> bool { self.pairs.is_empty() }

    fn dash_to_stroke<S: VertexSource>(&self, dash: &mut ConvDash<S>, dpi: f64, isaa: bool) {
        let scaleddpi = dpi / 72.0;
        for &(length, skip) in &self.pairs {
            let mut val0 = length * scaleddpi;
            let mut val1 = skip * scaleddpi;
            if !isaa {
                val0 = f64::from(trunc_i32(val0)) + 0.5;
                val1 = f64::from(trunc_i32(val1)) + 0.5;
            }
            dash.generator.add_dash(val0, val1);
        }
        dash.generator.dash_start(self.offset * scaleddpi);
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SketchParams { pub scale: f64, pub length: f64, pub randomness: f64 }

#[derive(Clone, Copy)]
pub struct ClipPath<'a> { pub path: PathData<'a>, pub trans: TransAffine }

/// `GCAgg`, as Matplotlib's converter fills it from a GraphicsContext.
#[derive(Clone)]
pub struct Gc<'a> {
    pub linewidth: f64,
    pub alpha: f64,
    pub forced_alpha: bool,
    pub color: Rgba,
    pub isaa: bool,
    pub cap: LineCap,
    pub join: LineJoin,
    pub cliprect: RectD,
    pub clippath: ClipPath<'a>,
    pub dashes: Dashes,
    pub snap_mode: SnapMode,
    pub hatchpath: PathData<'a>,
    pub hatch_color: Rgba,
    pub hatch_linewidth: f64,
    pub sketch: SketchParams,
}

impl Gc<'_> {
    pub fn has_hatchpath(&self) -> bool { self.hatchpath.total_vertices() != 0 }
}

/// `BufferRegion`: a canvas rectangle saved row by row.
pub struct BufferRegion { pub data: Vec<u8>, pub rect: RectI, pub width: i32, pub height: i32 }

impl BufferRegion {
    pub fn stride(&self) -> i32 { self.width * 4 }
}

/// Paths of a collection, indexed modulo their count.
pub trait PathGenerator {
    type Iter<'s>: VertexSource where Self: 's;
    fn num_paths(&self) -> usize;
    fn path(&self, i: usize) -> (Self::Iter<'_>, u32);
}

/// `count` paths of which the first `paths.len()` are converted.
pub struct PathList<'a> { pub paths: &'a [PathData<'a>], pub count: usize }

impl<'a> PathGenerator for PathList<'a> {
    type Iter<'s> = PathIterator<'a> where Self: 's;
    fn num_paths(&self) -> usize { self.count }
    fn path(&self, i: usize) -> (PathIterator<'a>, u32) {
        let path = self.paths[i % self.count];
        (PathIterator::new(path), path.total_vertices())
    }
}

/// `QuadMeshGenerator` over a (mesh_height+1) × (mesh_width+1) grid.
pub struct QuadMeshGenerator<'a> { pub mesh_width: usize, pub mesh_height: usize, pub coordinates: &'a [[f64; 2]] }

impl<'a> PathGenerator for QuadMeshGenerator<'a> {
    type Iter<'s> = QuadMeshPathIterator<'a> where Self: 's;
    fn num_paths(&self) -> usize { self.mesh_width * self.mesh_height }
    fn path(&self, i: usize) -> (QuadMeshPathIterator<'a>, u32) {
        (QuadMeshPathIterator::new(i % self.mesh_width, i / self.mesh_width, self.coordinates, self.mesh_width + 1), 5)
    }
}

/// Per-item arrays of a path collection; empty slices mean "none".
pub struct Collection<'a> {
    pub transforms: &'a [[[f64; 3]; 3]],
    pub offsets: &'a [[f64; 2]],
    pub offset_trans: TransAffine,
    pub facecolors: &'a [[f64; 4]],
    pub edgecolors: &'a [[f64; 4]],
    pub linewidths: &'a [f64],
    pub linestyles: &'a [Dashes],
    pub antialiaseds: &'a [u8],
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Solid { AaP8, AaMask, BinP8, BinMask, BinBin }

pub struct RendererAgg {
    pub width: u32,
    pub height: u32,
    pub dpi: f64,
    pub pixels: Vec<u8>,
    alpha_buffer: Option<Vec<u8>>,
    hatch_size: u32,
    hatch_buffer: Vec<u8>,
    ras: RasterizerScanlineAa,
    sl_p8: ScanlineP8,
    sl_bin: ScanlineBin,
    sl_am: ScanlineU8,
    base_clip: RectI,
    last_clippath: usize,
    last_clippath_transform: TransAffine,
    fill_color: Rgba8,
    colors: Vec<Rgba8>,
    /// Rotated text draws' filtered coverage by everything it depends on
    /// (the bitmap, its placement, the clip, the canvas): a label drawn again
    /// in place is copied from its recording, the same values.
    text_filters: std::collections::HashMap<u128, TextFilter>,
    text_filter_bytes: usize,
}

/// What a rotated text draw's filtered coverage is a function of.
fn text_key(image: &[u8], rows: usize, columns: usize, x: i32, y: i32, angle: f64, clip: &RectD, canvas: (u32, u32)) -> u128 {
    let (mut a, mut b) = (0x243f_6a88_85a3_08d3u64, 0x1319_8a2e_0370_7344u64);
    let mut word = |w: u64| {
        a = (a ^ w).wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left(29);
        b = (b.rotate_left(17) ^ w).wrapping_mul(0xc2b2_ae3d_27d4_eb4f);
    };
    let mut chunks = image.chunks_exact(8);
    for chunk in &mut chunks { word(u64::from_le_bytes(chunk.try_into().unwrap())); }
    let mut last = [0u8; 8];
    last[..chunks.remainder().len()].copy_from_slice(chunks.remainder());
    word(u64::from_le_bytes(last));
    for v in [image.len() as u64, rows as u64, columns as u64, x as u64, y as u64, angle.to_bits(), clip.x1.to_bits(), clip.y1.to_bits(),
              clip.x2.to_bits(), clip.y2.to_bits(), u64::from(canvas.0), u64::from(canvas.1)] { word(v); }
    let mix = |mut h: u64| { h ^= h >> 33; h = h.wrapping_mul(0xff51_afd7_ed55_8ccd); h ^= h >> 33; h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53); h ^ (h >> 33) };
    (u128::from(mix(a)) << 64) | u128::from(mix(b))
}

/// The coverage the text recordings may hold before they are dropped.
const TEXT_FILTER_BUDGET: usize = 16 << 20;

const MASK_FILL: Rgba8 = Rgba8::new(255, 255, 255, 0);

/// A zeroed buffer, or `std::bad_alloc` where `new int8u[len]` would throw.
fn zeroed(len: u128) -> Result<Vec<u8>, AggError> {
    let len = usize::try_from(len).map_err(|_| AggError::Memory)?;
    let mut buffer = Vec::new();
    buffer.try_reserve_exact(len).map_err(|_| AggError::Memory)?;
    buffer.resize(len, 0);
    Ok(buffer)
}

/// Room for `len` bytes, allocated as `new int8u[len]` would be (failing
/// alike) but not yet written: a buffer every use clears first.
fn reserved(len: u128) -> Result<Vec<u8>, AggError> {
    let len = usize::try_from(len).map_err(|_| AggError::Memory)?;
    let mut buffer = Vec::new();
    buffer.try_reserve_exact(len).map_err(|_| AggError::Memory)?;
    Ok(buffer)
}

impl RendererAgg {
    pub fn new(width: u32, height: u32, dpi: f64) -> Result<Self, AggError> {
        Self::check(width, height, dpi)?;
        let mut renderer = Self::over_pixels(width, height, dpi, zeroed(u128::from(width) * u128::from(height) * 4)?)?;
        renderer.clear();
        Ok(renderer)
    }

    fn check(width: u32, height: u32, dpi: f64) -> Result<(), AggError> {
        if dpi <= 0.0 { return Err(AggError::Value("dpi must be positive".into())); }
        if width >= 1 << 23 || height >= 1 << 23 {
            return Err(AggError::Value(format!(
                "Image size of {width}x{height} pixels is too large. It must be less than 2^23 in each direction.")));
        }
        Ok(())
    }

    /// `new` over a canvas allocation of exactly `width x height` RGBA pixels
    /// that keeps its contents (a caller that overwrites them, or clears).
    pub fn over_pixels(width: u32, height: u32, dpi: f64, pixels: Vec<u8>) -> Result<Self, AggError> {
        Self::check(width, height, dpi)?;
        if pixels.len() as u128 != u128::from(width) * u128::from(height) * 4 {
            return Err(AggError::Value("canvas pixels differ from the canvas size".into()));
        }
        let hatch_size = trunc_i32(dpi) as u32;
        Ok(Self {
            width, height, dpi,
            pixels,
            alpha_buffer: None,
            hatch_size,
            hatch_buffer: reserved(u128::from(hatch_size) * u128::from(hatch_size) * 4)?,
            ras: RasterizerScanlineAa::new(32768),
            sl_p8: ScanlineP8::new(),
            sl_bin: ScanlineBin::new(),
            sl_am: ScanlineU8::new(),
            base_clip: RectI::new(0, 0, width as i32 - 1, height as i32 - 1),
            last_clippath: 0,
            last_clippath_transform: TransAffine::IDENTITY,
            fill_color: MASK_FILL,
            colors: Vec::new(),
            text_filters: Default::default(),
            text_filter_bytes: 0,
        })
    }

    #[inline]
    fn points_to_pixels(&self, points: f64) -> f64 { points * self.dpi / 72.0 }

    fn reset_base_clipping(&mut self) { self.base_clip = RectI::new(0, 0, self.width as i32 - 1, self.height as i32 - 1); }

    fn clip_values(&self, r: &RectD) -> (i32, i32, i32, i32) {
        if r.x1 != 0.0 || r.y1 != 0.0 || r.x2 != 0.0 || r.y2 != 0.0 {
            let h = f64::from(self.height);
            (trunc_i32((r.x1 + 0.5).floor()).max(0), trunc_i32((h - r.y1 + 0.5).floor()).max(0),
             trunc_i32((r.x2 + 0.5).floor()).min(self.width as i32), trunc_i32((h - r.y2 + 0.5).floor()).min(self.height as i32))
        } else {
            (0, 0, self.width as i32, self.height as i32)
        }
    }

    fn set_clipbox_ras(&mut self, r: &RectD) {
        let (x1, y1, x2, y2) = self.clip_values(r);
        self.ras.clip_box(f64::from(x1), f64::from(y1), f64::from(x2), f64::from(y2));
    }

    fn set_clipbox_base(&mut self, r: &RectD) {
        let (x1, y1, x2, y2) = self.clip_values(r);
        let mut base = RendererBase::with_clip(PixfmtRgba32::new(&mut [], self.width, self.height), self.base_clip);
        base.clip_box(x1, y1, x2, y2);
        self.base_clip = base.clip;
    }

    fn solid(&mut self, mode: Solid, color: &Rgba8) -> Result<(), CellBlockLimit> {
        let (w, h) = (self.width, self.height);
        match mode {
            Solid::AaP8 | Solid::BinP8 | Solid::BinBin => {
                let mut ren = RendererBase::with_clip(PixfmtRgba32::new(&mut self.pixels, w, h), self.base_clip);
                match mode {
                    Solid::AaP8 => render_scanlines_aa_solid(&mut self.ras, &mut self.sl_p8, &mut ren, color),
                    Solid::BinP8 => render_scanlines_bin_solid(&mut self.ras, &mut self.sl_p8, &mut ren, color),
                    _ => render_scanlines_bin_solid(&mut self.ras, &mut self.sl_bin, &mut ren, color),
                }
            }
            Solid::AaMask | Solid::BinMask => {
                let mask = self.alpha_buffer.as_deref().unwrap_or(&[]);
                let mut ren = RendererBase::new(AmaskAdaptor::new(PixfmtRgba32::new(&mut self.pixels, w, h), mask));
                if mode == Solid::AaMask {
                    render_scanlines_aa_solid(&mut self.ras, &mut self.sl_am, &mut ren, color)
                } else {
                    render_scanlines_bin_solid(&mut self.ras, &mut self.sl_am, &mut ren, color)
                }
            }
        }
    }

    /// Spans through `rendererBase` or the alpha-mask renderer, swept with
    /// `slineP8` or `scanlineAlphaMask`.
    fn spans<G: SpanGenerator>(&mut self, masked: bool, am_scanline: bool, span_gen: &mut G) -> Result<(), CellBlockLimit> {
        let (w, h) = (self.width, self.height);
        if masked {
            let mask = self.alpha_buffer.as_deref().unwrap_or(&[]);
            let mut ren = RendererBase::new(AmaskAdaptor::new(PixfmtRgba32::new(&mut self.pixels, w, h), mask));
            if am_scanline { render_scanlines_aa(&mut self.ras, &mut self.sl_am, &mut ren, &mut self.colors, span_gen) }
            else { render_scanlines_aa(&mut self.ras, &mut self.sl_p8, &mut ren, &mut self.colors, span_gen) }
        } else {
            let mut ren = RendererBase::with_clip(PixfmtRgba32::new(&mut self.pixels, w, h), self.base_clip);
            if am_scanline { render_scanlines_aa(&mut self.ras, &mut self.sl_am, &mut ren, &mut self.colors, span_gen) }
            else { render_scanlines_aa(&mut self.ras, &mut self.sl_p8, &mut ren, &mut self.colors, span_gen) }
        }
    }

    fn render_clippath(&mut self, clippath: &PathData<'_>, clippath_trans: &TransAffine, snap_mode: SnapMode)
        -> Result<bool, CellBlockLimit> {
        let has_clippath = clippath.total_vertices() != 0;
        if has_clippath && (clippath.id != self.last_clippath
                            || !clippath_trans.is_equal(&self.last_clippath_transform, AFFINE_EPSILON)) {
            let (w, h) = (self.width, self.height);
            let mask = self.alpha_buffer.get_or_insert_with(|| vec![0; w as usize * h as usize]);
            let mut trans = *clippath_trans;
            trans.multiply(&TransAffine::scaling(1.0, -1.0));
            trans.multiply(&TransAffine::translation(0.0, f64::from(h)));
            let mut base = RendererBase::new(PixfmtGray8 { buf: mask, width: w, height: h });
            base.clear(&Gray8 { v: 0, a: 0 });
            let trans = Cell::new(trans);
            let transformed = ConvTransform::new(PathIterator::new(*clippath), &trans);
            let nan_removed = PathNanRemover::new(transformed, true, clippath.has_codes());
            let snapped = PathSnapper::new(nan_removed, snap_mode, clippath.total_vertices(), 0.0);
            let simplified = PathSimplifier::new(snapped, clippath.should_simplify && !clippath.has_codes(),
                                                 clippath.simplify_threshold);
            let mut curved = ConvCurve::new(simplified);
            self.ras.add_path(&mut curved, 0)?;
            render_scanlines_aa_solid(&mut self.ras, &mut self.sl_am, &mut base, &Gray8 { v: 255, a: 255 })?;
            self.last_clippath = clippath.id;
            self.last_clippath_transform = *clippath_trans;
        }
        Ok(has_clippath)
    }

    fn draw_path_inner<P: VertexSource>(&mut self, path: &mut P, has_clippath: bool, face: (bool, Rgba), gc: &Gc<'_>)
        -> Result<(), AggError> {
        if face.0 {
            self.ras.add_path(path, 0)?;
            let color = Rgba8::from_rgba(&face.1);
            let mode = match (gc.isaa, has_clippath) {
                (true, true) => Solid::AaMask,
                (true, false) => Solid::AaP8,
                (false, true) => Solid::BinMask,
                (false, false) => Solid::BinP8,
            };
            self.solid(mode, &color)?;
        }

        if gc.has_hatchpath() {
            self.ras.reset_clipping();
            self.reset_base_clipping();
            let hs = f64::from(self.hatch_size);
            let mut hatch_trans = TransAffine::IDENTITY;
            hatch_trans.multiply(&TransAffine::scaling(1.0, -1.0));
            hatch_trans.multiply(&TransAffine::translation(0.0, 1.0));
            hatch_trans.multiply(&TransAffine::scaling(hs, hs));
            let hatch_trans = Cell::new(hatch_trans);
            let mut hatch_curve = ConvCurve::new(ConvTransform::new(PathIterator::new(gc.hatchpath), &hatch_trans));
            let hatch_color = Rgba8::from_rgba(&gc.hatch_color);
            let hatch_width = self.points_to_pixels(gc.hatch_linewidth);
            {
                // Sized at its first use (within the room reserved for it); cleared at every use.
                let len = self.hatch_size as usize * self.hatch_size as usize * 4;
                if self.hatch_buffer.len() != len { self.hatch_buffer.resize(len, 0); }
                let mut rb = RendererBase::new(PixfmtRgba32::new(&mut self.hatch_buffer, self.hatch_size, self.hatch_size));
                rb.clear(&self.fill_color);
                self.ras.add_path(&mut hatch_curve, 0)?;
                render_scanlines_aa_solid(&mut self.ras, &mut self.sl_p8, &mut rb, &hatch_color)?;
                let mut hatch_stroke = ConvStroke::new(&mut hatch_curve);
                hatch_stroke.width(hatch_width);
                hatch_stroke.line_cap(LineCap::Square);
                self.ras.add_path(&mut hatch_stroke, 0)?;
                render_scanlines_aa_solid(&mut self.ras, &mut self.sl_p8, &mut rb, &hatch_color)?;
            }
            self.set_clipbox_ras(&gc.cliprect);
            if has_clippath {
                self.render_clippath(&gc.clippath.path, &gc.clippath.trans, gc.snap_mode)?;
            }
            let hatch = std::mem::take(&mut self.hatch_buffer);
            let mut pattern = PatternSpans { pattern: &hatch, size: self.hatch_size };
            let result = self.ras.add_path(path, 0).and_then(|_| self.spans(has_clippath, false, &mut pattern));
            self.hatch_buffer = hatch;
            result?;
        }

        if gc.linewidth != 0.0 {
            let mut linewidth = self.points_to_pixels(gc.linewidth);
            if !gc.isaa { linewidth = if linewidth < 0.5 { 0.5 } else { mpl_round(linewidth) }; }
            if gc.dashes.is_empty() {
                let mut stroke = ConvStroke::new(&mut *path);
                stroke.width(self.points_to_pixels(gc.linewidth));
                stroke.line_cap(gc.cap);
                stroke.line_join(gc.join);
                stroke.miter_limit(self.points_to_pixels(gc.linewidth));
                self.ras.add_path(&mut stroke, 0)?;
            } else {
                let mut dash = ConvDash::new(&mut *path);
                gc.dashes.dash_to_stroke(&mut dash, self.dpi, gc.isaa);
                let mut stroke = ConvStroke::new(&mut dash);
                stroke.line_cap(gc.cap);
                stroke.line_join(gc.join);
                stroke.width(linewidth);
                stroke.miter_limit(self.points_to_pixels(gc.linewidth));
                self.ras.add_path(&mut stroke, 0)?;
                // Where C++ Agg would not stop, nothing is drawn: the dashes so far are dropped.
                if let Some(why) = dash.generator.refused {
                    self.ras.reset();
                    return Err(AggError::Value(why.into()));
                }
            }
            let color = Rgba8::from_rgba(&gc.color);
            let mode = match (gc.isaa, has_clippath) {
                (true, true) => Solid::AaMask,
                (true, false) => Solid::AaP8,
                (false, true) => Solid::BinMask,
                (false, false) => Solid::BinBin,
            };
            self.solid(mode, &color)?;
        }
        Ok(())
    }

    pub fn draw_path(&mut self, gc: &Gc<'_>, path: PathData<'_>, trans: TransAffine, color: Rgba) -> Result<(), AggError> {
        let face = (color.a != 0.0, color);
        self.ras.reset_clipping();
        self.reset_base_clipping();
        self.set_clipbox_ras(&gc.cliprect);
        let has_clippath = self.render_clippath(&gc.clippath.path, &gc.clippath.trans, gc.snap_mode)?;
        let mut trans = trans;
        trans.multiply(&TransAffine::scaling(1.0, -1.0));
        trans.multiply(&TransAffine::translation(0.0, f64::from(self.height)));
        let clip = !face.0 && !gc.has_hatchpath();
        let simplify = path.should_simplify && clip;
        let mut snapping_linewidth = self.points_to_pixels(gc.linewidth);
        if gc.color.a == 0.0 { snapping_linewidth = 0.0; }
        let trans = Cell::new(trans);
        let tpath = ConvTransform::new(PathIterator::new(path), &trans);
        let nan_removed = PathNanRemover::new(tpath, true, path.has_codes());
        let clipped = PathClipper::new(nan_removed, clip, f64::from(self.width), f64::from(self.height));
        let snapped = PathSnapper::new(clipped, gc.snap_mode, path.total_vertices(), snapping_linewidth);
        let simplified = PathSimplifier::new(snapped, simplify, path.simplify_threshold);
        let curve = ConvCurve::new(simplified);
        let mut sketch = Sketch::new(curve, gc.sketch.scale, gc.sketch.length, gc.sketch.randomness);
        self.draw_path_inner(&mut sketch, has_clippath, face, gc)?;
        Ok(())
    }

    pub fn draw_markers(&mut self, gc: &Gc<'_>, marker_path: PathData<'_>, marker_trans: TransAffine,
                        path: PathData<'_>, trans: TransAffine, color: Rgba) -> Result<(), AggError> {
        let result = self.draw_markers_inner(gc, marker_path, marker_trans, path, trans, color);
        if result.is_err() {
            self.ras.reset_clipping();
            self.reset_base_clipping();
        }
        result
    }

    fn draw_markers_inner(&mut self, gc: &Gc<'_>, marker_path: PathData<'_>, marker_trans: TransAffine,
                          path: PathData<'_>, trans: TransAffine, color: Rgba) -> Result<(), AggError> {
        let mut marker_trans = marker_trans;
        marker_trans.multiply(&TransAffine::scaling(1.0, -1.0));
        let mut trans = trans;
        trans.multiply(&TransAffine::scaling(1.0, -1.0));
        trans.multiply(&TransAffine::translation(0.5, f64::from(self.height) + 0.5));

        let marker_trans = Cell::new(marker_trans);
        let marker_transformed = ConvTransform::new(PathIterator::new(marker_path), &marker_trans);
        let marker_nan_removed = PathNanRemover::new(marker_transformed, true, marker_path.has_codes());
        let marker_snapped = PathSnapper::new(marker_nan_removed, gc.snap_mode, marker_path.total_vertices(),
                                              self.points_to_pixels(gc.linewidth));
        let snapping = marker_snapped.is_snapping();
        let mut marker_curve = ConvCurve::new(marker_snapped);
        if !snapping {
            let mut t = marker_trans.get();
            t.multiply(&TransAffine::translation(0.5, 0.5));
            marker_trans.set(t);
        }

        let trans = Cell::new(trans);
        let path_transformed = ConvTransform::new(PathIterator::new(path), &trans);
        let path_nan_removed = PathNanRemover::new(path_transformed, false, false);
        let path_snapped = PathSnapper::new(path_nan_removed, SnapMode::False, path.total_vertices(), 0.0);
        let mut path_curve = ConvCurve::new(path_snapped);
        path_curve.rewind(0);

        let face = (color.a != 0.0, color);
        let mut scanlines = ScanlineStorageAa8::new();
        self.ras.reset();
        self.ras.reset_clipping();
        self.reset_base_clipping();
        let mut marker_size = RectI::new(0x7FFFFFFF, 0x7FFFFFFF, -0x7FFFFFFF, -0x7FFFFFFF);

        let mut fill = None;
        if face.0 {
            self.ras.add_path(&mut marker_curve, 0)?;
            self.store_scanlines(&mut scanlines)?;
            marker_size = RectI::new(scanlines.min_x, scanlines.min_y, scanlines.max_x, scanlines.max_y);
            fill = Some(clone_storage(&scanlines));
        }
        {
            let mut stroke = ConvStroke::new(&mut marker_curve);
            stroke.width(self.points_to_pixels(gc.linewidth));
            stroke.line_cap(gc.cap);
            stroke.line_join(gc.join);
            stroke.miter_limit(self.points_to_pixels(gc.linewidth));
            self.ras.reset();
            self.ras.add_path(&mut stroke, 0)?;
        }
        self.store_scanlines(&mut scanlines)?;
        let stroke_cache = scanlines;
        marker_size = RectI::new(marker_size.x1.min(stroke_cache.min_x), marker_size.y1.min(stroke_cache.min_y),
                                 marker_size.x2.max(stroke_cache.max_x), marker_size.y2.max(stroke_cache.max_y));

        self.ras.reset_clipping();
        self.reset_base_clipping();
        self.set_clipbox_base(&gc.cliprect);
        let has_clippath = self.render_clippath(&gc.clippath.path, &gc.clippath.trans, gc.snap_mode)?;

        let clipping_rect = RectD::new(-1.0 - f64::from(marker_size.x2), -1.0 - f64::from(marker_size.y2),
                                       1.0 + f64::from(self.width) - f64::from(marker_size.x1),
                                       1.0 + f64::from(self.height) - f64::from(marker_size.y1));
        let face_color = Rgba8::from_rgba(&face.1);
        let stroke_color = Rgba8::from_rgba(&gc.color);
        let (w, h) = (self.width, self.height);
        let (mut x, mut y) = (0.0, 0.0);
        if has_clippath {
            let mask = self.alpha_buffer.as_deref().unwrap_or(&[]);
            let mut ren = RendererBase::new(AmaskAdaptor::new(PixfmtRgba32::new(&mut self.pixels, w, h), mask));
            while path_curve.vertex(&mut x, &mut y) != PATH_CMD_STOP {
                if !(x.is_finite() && y.is_finite()) { continue; }
                x = x.floor();
                y = y.floor();
                if !clipping_rect.hit_test(x, y) { continue; }
                let (dx, dy) = (iround(x), iround(y));
                if let Some(fill) = &fill { render_stored(fill, dx, dy, &mut ren, &face_color); }
                render_stored(&stroke_cache, dx, dy, &mut ren, &stroke_color);
            }
        } else {
            let mut ren = RendererBase::with_clip(PixfmtRgba32::new(&mut self.pixels, w, h), self.base_clip);
            while path_curve.vertex(&mut x, &mut y) != PATH_CMD_STOP {
                if !(x.is_finite() && y.is_finite()) { continue; }
                x = x.floor();
                y = y.floor();
                if !clipping_rect.hit_test(x, y) { continue; }
                let (dx, dy) = (iround(x), iround(y));
                if let Some(fill) = &fill { render_stored(fill, dx, dy, &mut ren, &face_color); }
                render_stored(&stroke_cache, dx, dy, &mut ren, &stroke_color);
            }
        }
        self.ras.reset_clipping();
        self.reset_base_clipping();
        Ok(())
    }

    /// `render_scanlines(theRasterizer, slineP8, storage)`: the storage is
    /// only prepared, and so only replaced, when the rasterizer has cells.
    fn store_scanlines(&mut self, storage: &mut ScanlineStorageAa8) -> Result<(), CellBlockLimit> {
        if self.ras.rewind_scanlines()? {
            use crate::rasterizer::ScanlineSink;
            self.sl_p8.reset(self.ras.min_x(), self.ras.max_x());
            storage.prepare();
            while self.ras.sweep_scanline(&mut self.sl_p8) { storage.render(&self.sl_p8); }
        }
        Ok(())
    }

    pub fn draw_text_image(&mut self, gc: &Gc<'_>, image: &[u8], rows: usize, columns: usize, x: i32, y: i32, angle: f64)
        -> Result<(), AggError> {
        self.ras.reset_clipping();
        self.reset_base_clipping();
        if angle != 0.0 {
            self.set_clipbox_ras(&gc.cliprect);
            let image_height = rows as f64;
            let image_width = columns as f64;
            let mut mtx = TransAffine::IDENTITY;
            mtx.multiply(&TransAffine::translation(0.0, -image_height));
            mtx.multiply(&TransAffine::rotation(-angle * (PI / 180.0)));
            mtx.multiply(&TransAffine::translation(f64::from(x), f64::from(y)));
            let mut rect = Transformed { source: RectPath::new(image_width, image_height), trans: mtx };
            let mut inv_mtx = mtx;
            inv_mtx.invert();
            let mut spans = TextSpans::new(image, columns as i32, rows as i32, InterpolatorLinear::new(inv_mtx),
                                           ImageFilterLut::spline36(), Rgba8::from_rgba(&gc.color));
            let key = text_key(image, rows, columns, x, y, angle, &gc.cliprect, (self.width, self.height));
            let size = |filter: &TextFilter| filter.coverage.len() + 16 * filter.spans.len();
            match self.text_filters.remove(&key) {
                Some(filter) => { self.text_filter_bytes -= size(&filter); spans.replay = Some((filter, 0, 0)); }
                None => spans.record = Some(TextFilter::default()),
            }
            self.ras.add_path(&mut rect, 0)?;
            self.spans(false, false, &mut spans)?;
            if let Some(filter) = spans.replay.take().map(|(filter, _, _)| filter).or(spans.record.take()) {
                if self.text_filter_bytes + size(&filter) > TEXT_FILTER_BUDGET { self.text_filters.clear(); self.text_filter_bytes = 0; }
                self.text_filter_bytes += size(&filter);
                self.text_filters.insert(key, filter);
            }
        } else {
            let deltay = y.wrapping_sub(rows as i32);
            let fig = RectI::new(0, 0, self.width as i32, self.height as i32);
            let mut text = RectI::new(x, deltay, x.wrapping_add(columns as i32), y);
            text.clip(&fig);
            let r = &gc.cliprect;
            if r.x1 != 0.0 || r.y1 != 0.0 || r.x2 != 0.0 || r.y2 != 0.0 {
                let h = f64::from(self.height);
                let clip = RectI::new(mpl_round_to_int(r.x1), mpl_round_to_int(h - r.y2),
                                      mpl_round_to_int(r.x2), mpl_round_to_int(h - r.y1));
                text.clip(&clip);
            }
            if text.x2 > text.x1 {
                let deltax = text.x2 - text.x1;
                let deltax2 = text.x1 - x;
                let color = Rgba8::from_rgba(&gc.color);
                let mut pixf = PixfmtRgba32::new(&mut self.pixels, self.width, self.height);
                for yi in text.y1..text.y2 {
                    let start = (yi - deltay) as usize * columns + deltax2 as usize;
                    pixf.blend_solid_hspan(text.x1, yi, deltax as u32, &color, &image[start..start + deltax as usize]);
                }
            }
        }
        Ok(())
    }

    /// `RendererAgg::draw_image`; Matplotlib's wrapper has rounded `x`/`y`
    /// and forced the alpha to 1.
    pub fn draw_image(&mut self, gc: &Gc<'_>, x: f64, y: f64, image: &[u8], rows: usize, columns: usize)
        -> Result<(), AggError> {
        let alpha = gc.alpha;
        self.ras.reset_clipping();
        self.reset_base_clipping();
        self.set_clipbox_ras(&gc.cliprect);
        let has_clippath = self.render_clippath(&gc.clippath.path, &gc.clippath.trans, gc.snap_mode)?;
        let source = ImageRgba { data: image, width: columns as u32, height: rows as u32, stride: -(columns as isize) * 4 };
        if has_clippath {
            let image_height = rows as f64;
            let image_width = columns as f64;
            let mut mtx = TransAffine::IDENTITY;
            mtx.multiply(&TransAffine::translation(f64::from(trunc_i32(x)),
                                                   f64::from(trunc_i32(f64::from(self.height) - (y + image_height)))));
            let mut rect = Transformed { source: RectPath::new(image_width, image_height), trans: mtx };
            let mut inv_mtx = mtx;
            inv_mtx.invert();
            let mut spans = ImageSpansNn { image: source, interpolator: InterpolatorLinear::new(inv_mtx), alpha };
            self.ras.add_path(&mut rect, 0)?;
            self.spans(true, true, &mut spans)?;
        } else {
            self.set_clipbox_base(&gc.cliprect);
            let mut ren = RendererBase::with_clip(PixfmtRgba32::new(&mut self.pixels, self.width, self.height), self.base_clip);
            ren.blend_from(&source, trunc_i32(x), trunc_i32(f64::from(self.height) - (y + rows as f64)),
                           trunc_i32(alpha * 255.0) as u8);
        }
        self.reset_base_clipping();
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_path_collection_generic<G: PathGenerator>(&mut self, gc: &Gc<'_>, master_transform: TransAffine,
                                                      paths: &G, items: &Collection<'_>, check_snap: bool,
                                                      has_codes: bool, stop: usize) -> Result<(), AggError> {
        let npaths = paths.num_paths();
        let noffsets = items.offsets.len();
        let n = npaths.max(noffsets);
        let ntransforms = items.transforms.len();
        let nfacecolors = items.facecolors.len();
        let nedgecolors = items.edgecolors.len();
        let nlinewidths = items.linewidths.len();
        let nlinestyles = items.linestyles.len().min(n);
        let naa = items.antialiaseds.len();
        if (nfacecolors == 0 && nedgecolors == 0) || npaths == 0 { return Ok(()); }

        self.ras.reset_clipping();
        self.reset_base_clipping();
        self.set_clipbox_ras(&gc.cliprect);
        let has_clippath = self.render_clippath(&gc.clippath.path, &gc.clippath.trans, gc.snap_mode)?;

        // Matplotlib divides by the antialiased count here and would crash.
        if naa == 0 && n.min(stop) > 0 { return Err(AggError::Value("antialiaseds must not be empty".into())); }
        let mut gc = gc.clone();
        gc.linewidth = 0.0;
        let mut face = (nfacecolors != 0, Rgba::default());
        let do_clip = !face.0 && !gc.has_hatchpath();
        for i in 0..n.min(stop) {
            let (path, total_vertices) = paths.path(i);
            let mut trans;
            if ntransforms != 0 {
                let t = &items.transforms[i % ntransforms];
                trans = TransAffine::new(t[0][0], t[1][0], t[0][1], t[1][1], t[0][2], t[1][2]);
                trans.multiply(&master_transform);
            } else {
                trans = master_transform;
            }
            if noffsets != 0 {
                let [mut xo, mut yo] = items.offsets[i % noffsets];
                items.offset_trans.transform(&mut xo, &mut yo);
                trans.multiply(&TransAffine::translation(xo, yo));
            }
            trans.multiply(&TransAffine::scaling(1.0, -1.0));
            trans.multiply(&TransAffine::translation(0.0, f64::from(self.height)));
            if nfacecolors != 0 {
                let c = items.facecolors[i % nfacecolors];
                face.1 = Rgba::new(c[0], c[1], c[2], c[3]);
            }
            if nedgecolors != 0 {
                let c = items.edgecolors[i % nedgecolors];
                gc.color = Rgba::new(c[0], c[1], c[2], c[3]);
                gc.linewidth = if nlinewidths != 0 { items.linewidths[i % nlinewidths] } else { 1.0 };
                if nlinestyles != 0 { gc.dashes = items.linestyles[i % nlinestyles].clone(); }
            }
            gc.isaa = items.antialiaseds[i % naa] != 0;
            let trans = Cell::new(trans);
            let tpath = ConvTransform::new(path, &trans);
            let nan_removed = PathNanRemover::new(tpath, true, has_codes);
            let clipped = PathClipper::new(nan_removed, do_clip, f64::from(self.width), f64::from(self.height));
            let s = &gc.sketch;
            if check_snap {
                let snapped = PathSnapper::new(clipped, gc.snap_mode, total_vertices, self.points_to_pixels(gc.linewidth));
                if has_codes {
                    let mut sketch = Sketch::new(ConvCurve::new(snapped), s.scale, s.length, s.randomness);
                    self.draw_path_inner(&mut sketch, has_clippath, face, &gc)?;
                } else {
                    let mut sketch = Sketch::new(snapped, s.scale, s.length, s.randomness);
                    self.draw_path_inner(&mut sketch, has_clippath, face, &gc)?;
                }
            } else if has_codes {
                let mut sketch = Sketch::new(ConvCurve::new(clipped), s.scale, s.length, s.randomness);
                self.draw_path_inner(&mut sketch, has_clippath, face, &gc)?;
            } else {
                let mut sketch = Sketch::new(clipped, s.scale, s.length, s.randomness);
                self.draw_path_inner(&mut sketch, has_clippath, face, &gc)?;
            }
        }
        Ok(())
    }

    /// `draw_path_collection` over `num_paths` paths, stopping before item `stop`;
    /// `paths` holds those that items below `stop` reach. `antialiaseds` must be
    /// nonempty when any item is drawn.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_path_collection(&mut self, gc: &Gc<'_>, master_transform: TransAffine, paths: &[PathData<'_>],
                                num_paths: usize, items: &Collection<'_>, stop: usize) -> Result<(), AggError> {
        self.draw_path_collection_generic(gc, master_transform, &PathList { paths, count: num_paths }, items, true, true, stop)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn draw_quad_mesh(&mut self, gc: &Gc<'_>, master_transform: TransAffine, mesh_width: usize, mesh_height: usize,
                          coordinates: &[[f64; 2]], offsets: &[[f64; 2]], offset_trans: TransAffine,
                          facecolors: &[[f64; 4]], antialiased: bool, edgecolors: &[[f64; 4]]) -> Result<(), AggError> {
        let linewidths = [gc.linewidth];
        let antialiaseds = [u8::from(antialiased)];
        let items = Collection { transforms: &[], offsets, offset_trans, facecolors, edgecolors, linewidths: &linewidths,
                                 linestyles: &[], antialiaseds: &antialiaseds };
        let generator = QuadMeshGenerator { mesh_width, mesh_height, coordinates };
        self.draw_path_collection_generic(gc, master_transform, &generator, &items, true, false, usize::MAX)
    }

    pub fn draw_gouraud_triangles(&mut self, gc: &Gc<'_>, points: &[[[f64; 2]; 3]], colors: &[[[f64; 4]; 3]],
                                  trans: TransAffine) -> Result<(), AggError> {
        self.ras.reset_clipping();
        self.reset_base_clipping();
        self.set_clipbox_ras(&gc.cliprect);
        let has_clippath = self.render_clippath(&gc.clippath.path, &gc.clippath.trans, gc.snap_mode)?;
        for (p, c) in points.iter().zip(colors) {
            self.draw_gouraud_triangle(p, c, trans, has_clippath)?;
        }
        Ok(())
    }

    fn draw_gouraud_triangle(&mut self, points: &[[f64; 2]; 3], colors: &[[f64; 4]; 3], trans: TransAffine,
                             has_clippath: bool) -> Result<(), CellBlockLimit> {
        let mut trans = trans;
        trans.multiply(&TransAffine::scaling(1.0, -1.0));
        trans.multiply(&TransAffine::translation(0.0, f64::from(self.height)));
        let mut tpoints = [[0.0; 2]; 3];
        for i in 0..3 {
            tpoints[i] = points[i];
            let [mut x, mut y] = tpoints[i];
            trans.transform(&mut x, &mut y);
            tpoints[i] = [x, y];
            if x.is_nan() || y.is_nan() { return Ok(()); }
        }
        let color = |c: &[f64; 4]| Rgba8::from_rgba(&Rgba::new(c[0], c[1], c[2], c[3]));
        let mut gouraud = GouraudSpans::new(color(&colors[0]), color(&colors[1]), color(&colors[2]),
                                        tpoints[0][0], tpoints[0][1], tpoints[1][0], tpoints[1][1],
                                        tpoints[2][0], tpoints[2][1], 0.5);
        self.ras.add_path(&mut gouraud, 0)?;
        self.spans(has_clippath, has_clippath, &mut gouraud)
    }

    pub fn clear(&mut self) {
        let fill = self.fill_color;
        let mut base = RendererBase::new(PixfmtRgba32::new(&mut self.pixels, self.width, self.height));
        base.clear(&fill);
    }

    pub fn copy_from_bbox(&self, r: RectD) -> Result<BufferRegion, AggError> {
        let h = self.height;
        let rect = RectI::new(trunc_i32(r.x1), h.wrapping_sub(trunc_i32(r.y2) as u32) as i32,
                              trunc_i32(r.x2), h.wrapping_sub(trunc_i32(r.y1) as u32) as i32);
        let width = rect.x2.wrapping_sub(rect.x1);
        let height = rect.y2.wrapping_sub(rect.y1);
        let size = i64::from(width.wrapping_mul(4)) * i64::from(height);
        if width < 0 || height < 0 || width.checked_mul(4).and_then(|s| s.checked_mul(height)).is_none() || size < 0 {
            return Err(AggError::Memory);
        }
        let mut region = BufferRegion { data: zeroed(size as u128)?, rect, width, height };
        let mut rb = RendererBase::new(PixfmtRgba32::new(&mut region.data, width as u32, height as u32));
        rb.copy_from(&self.pixels, self.width, self.height, Some(&rect), -rect.x1, -rect.y1);
        Ok(region)
    }

    pub fn restore_region(&mut self, region: &BufferRegion) {
        let mut ren = RendererBase::with_clip(PixfmtRgba32::new(&mut self.pixels, self.width, self.height), self.base_clip);
        ren.copy_from(&region.data, region.width as u32, region.height as u32, None, region.rect.x1, region.rect.y1);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn restore_region_part(&mut self, region: &BufferRegion, xx1: i32, yy1: i32, xx2: i32, yy2: i32, x: i32, y: i32) {
        let r = region.rect;
        let rect = RectI::new(xx1 - r.x1, yy1 - r.y1, xx2 - r.x1, yy2 - r.y1);
        let mut ren = RendererBase::with_clip(PixfmtRgba32::new(&mut self.pixels, self.width, self.height), self.base_clip);
        ren.copy_from(&region.data, region.width as u32, region.height as u32, Some(&rect), x, y);
    }
}

fn clone_storage(s: &ScanlineStorageAa8) -> ScanlineStorageAa8 {
    ScanlineStorageAa8 { covers: s.covers.clone(), spans: s.spans.clone(), scanlines: s.scanlines.clone(),
                         min_x: s.min_x, min_y: s.min_y, max_x: s.max_x, max_y: s.max_y }
}

/// One serialized marker stamp at integer offset (dx, dy).
#[inline]
fn render_stored<P: PixFmt<Color = Rgba8>>(storage: &ScanlineStorageAa8, dx: i32, dy: i32, ren: &mut RendererBase<P>,
                                           color: &Rgba8) {
    for line in &storage.scanlines {
        let y = line.y + dy;
        for span in &storage.spans[line.first_span..line.first_span + line.num_spans] {
            let x = span.x + dx;
            if span.len > 0 {
                ren.blend_solid_hspan(x, y, span.len, color, &storage.covers[span.cover..]);
            } else {
                ren.blend_hline(x, y, x - span.len - 1, color, storage.covers[span.cover]);
            }
        }
    }
}
