//! `_foreground_strokes`: how the compose paints recorded boundary chrome
//! (tick marks, gridlines, spines) while the kernels are engaged and no
//! gesture holds the frame. Each two-vertex stroke is rasterised white on a
//! scratch renderer the size of its own padded box, shifted onto it by whole
//! pixels, and its coverage is laid on the canvas in the stroke's colour.
//! The shift is folded into the stroke's transform (`trans + shift`), so a
//! mark whose position is a hair from a half pixel can round to the other
//! side of it than the same mark drawn on the canvas directly. What is
//! lowered is a `draw_markers` call (tick marks) and a `draw_path` call made
//! with a face argument (a patch: the spines); a Line2D's `draw_path`
//! (gridlines) passes no face and is drawn as is.
use std::cell::Cell;
use std::collections::HashMap;
use std::hash::Hasher;
use std::sync::Arc;

use zlc_figure::agg::AggBackend;
use zlc_figure::backend::{Backend, Gc, GrayImage, PathRef, RgbaImage};
use zlc_figure::ir::Dash;
use zlc_figure::transform::{Points, M3};
use zlc_figure::Result;
use zlc_render::composition::{replay_foreground_masks, Mask};
use zlc_render::Canvas;

use super::hashkey::Lanes;

/// What the artist being drawn may have lowered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lower { Nothing, Markers, Patch }

/// The canvas backend while dynamic artists are painted: a stroke of the
/// kind `active` names is lowered to a coverage mask where
/// `_foreground_strokes` lowers it; everything else reaches the canvas as
/// drawn.
pub struct Lowering<'b> {
    pub inner: &'b mut AggBackend,
    pub active: &'b Cell<Lower>,
    pub memo: &'b mut LoweredMemo,
}

/// The lowered strokes of the last compose, by everything a stroke's
/// coverage depends on: chrome strokes stand still from frame to frame.
#[derive(Default)]
pub struct LoweredMemo { entries: HashMap<u128, (u64, Option<Arc<Lowered>>)>, generation: u64 }

impl LoweredMemo {
    /// Forget what the compose just ended did not draw.
    pub fn sweep(&mut self) {
        let generation = self.generation;
        self.entries.retain(|_, (used, _)| *used == generation);
        self.generation += 1;
    }
}

#[derive(Debug)]
enum Stroke<'p> {
    Path { path: PathRef<'p>, trans: &'p M3, face: Option<[f64; 4]> },
    Markers { marker_path: PathRef<'p>, marker_trans: &'p M3, path: PathRef<'p>, trans: &'p M3, face: Option<[f64; 4]> },
}

fn path_key(k: &mut Lanes, path: &PathRef<'_>) {
    k.write_usize(path.vertices.len());
    for v in path.vertices { k.f64(v[0]); k.f64(v[1]); }
    match path.codes {
        None => k.write_u8(0),
        Some(codes) => { k.write_u8(1); k.write_usize(codes.len()); k.write(codes); }
    }
    k.write_u8(u8::from(path.should_simplify));
    k.f64(path.simplify_threshold);
}

fn matrix_key(k: &mut Lanes, m: &M3) { for row in &m.0 { k.f64s(row); } }

fn face_key(k: &mut Lanes, face: Option<[f64; 4]>) {
    match face { None => k.write_u8(0), Some(face) => { k.write_u8(1); k.f64s(&face); } }
}

/// Everything a lowered stroke's coverage depends on -- the gc, the stroke,
/// the canvas -- as raw numbers, not text.
fn stroke_key(gc: &Gc, stroke: &Stroke<'_>, width: u32, height: u32, dpi: f64) -> u128 {
    let mut k = Lanes::new();
    let Gc { linewidth, alpha, forced_alpha, rgb, antialiased, capstyle, joinstyle, dashes, cliprect, snap, hatch, hatch_path,
             hatch_color, hatch_linewidth } = gc;
    k.f64(*linewidth);
    k.f64(*alpha);
    k.write_u8(u8::from(*forced_alpha));
    k.f64s(rgb);
    k.write_u8(u8::from(*antialiased));
    k.write_u8(*capstyle as u8);
    k.write_u8(*joinstyle as u8);
    match dashes.0 { None => k.write_u8(0), Some(offset) => { k.write_u8(1); k.f64(offset); } }
    match &dashes.1 { None => k.write_u8(0), Some(list) => { k.write_u8(1); k.f64s(list); } }
    match cliprect { None => k.write_u8(0), Some(c) => { k.write_u8(1); k.f64s(c); } }
    k.write_u8(match snap { None => 0, Some(false) => 1, Some(true) => 2 });
    match hatch { None => k.write_u8(0), Some(h) => { k.write_u8(1); k.write_usize(h.len()); k.write(h.as_bytes()); } }
    match hatch_path {
        None => k.write_u8(0),
        Some(h) => {
            k.write_u8(1);
            k.write_usize(h.vertices.len());
            for v in &h.vertices { k.f64(v[0]); k.f64(v[1]); }
            k.write_usize(h.codes.len());
            k.write(&h.codes);
            k.write_u8(u8::from(h.should_simplify));
            k.f64(h.simplify_threshold);
        }
    }
    k.f64s(hatch_color);
    k.f64(*hatch_linewidth);
    match stroke {
        Stroke::Path { path, trans, face } => {
            k.write_u8(0);
            path_key(&mut k, path);
            matrix_key(&mut k, trans);
            face_key(&mut k, *face);
        }
        Stroke::Markers { marker_path, marker_trans, path, trans, face } => {
            k.write_u8(1);
            path_key(&mut k, marker_path);
            matrix_key(&mut k, marker_trans);
            path_key(&mut k, path);
            matrix_key(&mut k, trans);
            face_key(&mut k, *face);
        }
    }
    k.write_u32(width);
    k.write_u32(height);
    k.f64(dpi);
    k.finish128()
}

/// `Path.get_extents(transform)` of a path without curves; `None` with curves.
fn extents(path: &PathRef<'_>, trans: &M3) -> Option<Points> {
    let mut xs = Vec::with_capacity(path.vertices.len());
    for (index, vertex) in path.vertices.iter().enumerate() {
        match path.codes.map(|codes| codes[index]) {
            Some(3) | Some(4) => return None,
            Some(1) | Some(2) | None => xs.push(trans.apply(vertex[0], vertex[1])),
            Some(_) => {}
        }
    }
    let nan_min = |a: f64, b: f64| if a.is_nan() || b.is_nan() { f64::NAN } else { a.min(b) };
    let nan_max = |a: f64, b: f64| if a.is_nan() || b.is_nan() { f64::NAN } else { a.max(b) };
    let first = *xs.first()?;
    let (mut x0, mut y0, mut x1, mut y1) = (first.0, first.1, first.0, first.1);
    for &(x, y) in &xs[1..] { x0 = nan_min(x0, x); y0 = nan_min(y0, y); x1 = nan_max(x1, x); y1 = nan_max(y1, y); }
    Some([x0, y0, x1, y1])
}

/// One lowered stroke's coverage, placed in canvas rows.
struct Lowered { coverage: Vec<u8>, top: i64, left: i64, width: usize, height: usize, color: [u8; 4] }

impl Lowering<'_> {
    /// `None`: not a stroke `_foreground_strokes` lowers (drawn as is);
    /// `Some(None)`: lowered to nothing.
    fn lower(&self, gc: &Gc, stroke: &Stroke<'_>) -> Result<Option<Option<Lowered>>> {
        if gc.hatch.is_some() { return Ok(None); }
        let (width, height) = (i64::from(self.inner.renderer.width), i64::from(self.inner.renderer.height));
        let bounds = match stroke {
            Stroke::Path { path, trans, face } => {
                if face.is_some() || path.vertices.len() != 2 { return Ok(None); }
                match extents(path, trans) { Some(b) => b, None => return Ok(None) }
            }
            Stroke::Markers { marker_path, marker_trans, path, trans, .. } => {
                if marker_path.vertices.len() != 2 { return Ok(None); }
                let points: Vec<(f64, f64)> = path.vertices.iter().map(|v| trans.apply(v[0], v[1])).collect();
                if points.is_empty() { return Ok(Some(None)); }
                let Some(marker) = extents(marker_path, marker_trans) else { return Ok(None) };
                let fold = |f: fn(f64, f64) -> f64, pick: fn(&(f64, f64)) -> f64| {
                    points.iter().map(pick).fold(None, |m: Option<f64>, v| Some(m.map_or(v, |m| f(m, v)))).unwrap()
                };
                let min = |a: f64, b: f64| if a.is_nan() || b.is_nan() { f64::NAN } else { a.min(b) };
                let max = |a: f64, b: f64| if a.is_nan() || b.is_nan() { f64::NAN } else { a.max(b) };
                [fold(min, |p| p.0) + marker[0], fold(min, |p| p.1) + marker[1],
                 fold(max, |p| p.0) + marker[2], fold(max, |p| p.1) + marker[3]]
            }
        };
        if bounds.iter().any(|v| !v.is_finite()) { return Ok(None); }
        let dpi = self.inner.renderer.dpi;
        let pad = (gc.linewidth * dpi / 72.0).ceil() as i64 + 4;
        let left = (bounds[0].floor() as i64 - pad).max(0);
        let right = (bounds[2].ceil() as i64 + pad).min(width);
        let top = (height - bounds[3].ceil() as i64 - pad).max(0);
        let bottom = (height - bounds[1].floor() as i64 + pad).min(height);
        if right <= left || bottom <= top { return Ok(Some(None)); }
        let scratch_span = super::profile::span("lower.scratch");
        let mut scratch = AggBackend::new((right - left) as f64, (bottom - top) as f64, dpi)?;
        drop(scratch_span);
        let draw_span = super::profile::span("lower.draw");
        let (sx, sy) = (-(left as f64), -((height - bottom) as f64));
        let shift = M3::IDENTITY.translate(sx, sy);
        let mut white = gc.clone();
        white.alpha = 1.0;
        white.forced_alpha = true;
        white.rgb = [1.0, 1.0, 1.0, 1.0];
        white.cliprect = gc.cliprect.map(|c| [c[0] + sx, c[1] + sy, c[2] + sx, c[3] + sy]);
        match stroke {
            Stroke::Path { path, trans, .. } => scratch.draw_path(&white, *path, &shift.dot(trans), None)?,
            Stroke::Markers { marker_path, marker_trans, path, trans, face } => {
                scratch.draw_markers(&white, *marker_path, marker_trans, *path, &shift.dot(trans), face.map(|_| [1.0; 4]))?
            }
        }
        drop(draw_span);
        let _scan = super::profile::span("lower.scan");
        let (w, h) = ((right - left) as usize, (bottom - top) as usize);
        let alpha = |row: usize, col: usize| scratch.renderer.pixels[(row * w + col) * 4 + 3];
        let rows: Vec<usize> = (0..h).filter(|&r| (0..w).any(|c| alpha(r, c) != 0)).collect();
        let cols: Vec<usize> = (0..w).filter(|&c| (0..h).any(|r| alpha(r, c) != 0)).collect();
        let (Some(&r0), Some(&r1), Some(&c0), Some(&c1)) = (rows.first(), rows.last(), cols.first(), cols.last()) else {
            return Ok(Some(None));
        };
        let mut coverage = Vec::with_capacity((r1 - r0 + 1) * (c1 - c0 + 1));
        for r in r0..=r1 { for c in c0..=c1 { coverage.push(alpha(r, c)); } }
        let color = gc.rgb.map(|v| (v.clamp(0.0, 1.0) * 255.0 + 0.5).floor() as u8);
        Ok(Some(Some(Lowered { coverage, top: top + r0 as i64, left: left + c0 as i64, width: c1 - c0 + 1, height: r1 - r0 + 1, color })))
    }

    /// Lower `stroke` when active; true when it was (and is painted).
    fn painted(&mut self, gc: &Gc, stroke: Stroke<'_>) -> Result<bool> {
        let wanted = match stroke { Stroke::Path { .. } => Lower::Patch, Stroke::Markers { .. } => Lower::Markers };
        if self.active.get() != wanted { return Ok(false); }
        let r = &self.inner.renderer;
        let key = stroke_key(gc, &stroke, r.width, r.height, r.dpi);
        let generation = self.memo.generation;
        let lowered = match self.memo.entries.get_mut(&key) {
            Some((used, lowered)) => { *used = generation; Some(lowered.clone()) }
            None => {
                let lowered = self.lower(gc, &stroke)?.map(|m| m.map(Arc::new));
                if let Some(lowered) = &lowered { self.memo.entries.insert(key, (generation, lowered.clone())); }
                lowered
            }
        };
        match lowered {
            None => Ok(false),
            Some(None) => Ok(true),
            Some(Some(m)) => {
                let _replay = super::profile::span("lower.replay");
                let (w, h) = (self.inner.renderer.width as usize, self.inner.renderer.height as usize);
                let mut canvas = Canvas::new(&mut self.inner.renderer.pixels, w, h).map_err(zlc_figure::Error::Backend)?;
                let mask = Mask { coverage: &m.coverage, top: m.top, left: m.left, width: m.width, height: m.height, color: m.color };
                replay_foreground_masks(&mut canvas, &[mask]).map_err(zlc_figure::Error::Backend)?;
                Ok(true)
            }
        }
    }
}

impl Backend for Lowering<'_> {
    fn draw_path(&mut self, gc: &Gc, path: PathRef<'_>, trans: &M3, face: Option<[f64; 4]>) -> Result<()> {
        if self.painted(gc, Stroke::Path { path, trans, face })? { return Ok(()); }
        self.inner.draw_path(gc, path, trans, face)
    }

    fn draw_markers(&mut self, gc: &Gc, marker_path: PathRef<'_>, marker_trans: &M3, path: PathRef<'_>, trans: &M3,
                    face: Option<[f64; 4]>) -> Result<()> {
        if self.painted(gc, Stroke::Markers { marker_path, marker_trans, path, trans, face })? { return Ok(()); }
        self.inner.draw_markers(gc, marker_path, marker_trans, path, trans, face)
    }

    fn draw_text_image(&mut self, image: &dyn GrayImage, x: i32, y: i32, angle: f64, gc: &Gc) -> Result<()> {
        self.inner.draw_text_image(image, x, y, angle, gc)
    }

    fn draw_image(&mut self, gc: &Gc, x: f64, y: f64, image: &RgbaImage) -> Result<()> { self.inner.draw_image(gc, x, y, image) }

    fn draw_path_collection(&mut self, gc: &Gc, master_transform: &M3, paths: &[PathRef<'_>],
                            transforms: &[[[f64; 3]; 3]], offsets: &[[f64; 2]], offset_trans: &M3,
                            facecolors: &[[f64; 4]], edgecolors: &[[f64; 4]], linewidths: &[f64],
                            dashes: &[Dash], antialiaseds: &[bool]) -> Result<()> {
        self.inner.draw_path_collection(gc, master_transform, paths, transforms, offsets, offset_trans, facecolors, edgecolors,
                                        linewidths, dashes, antialiaseds)
    }

    fn draw_quad_mesh(&mut self, gc: &Gc, master_transform: &M3, mesh_width: usize, mesh_height: usize,
                      coordinates: &[[f64; 2]], offsets: &[[f64; 2]], offset_trans: &M3, facecolors: &[[f64; 4]],
                      antialiased: bool, edgecolors: &[[f64; 4]]) -> Result<()> {
        self.inner.draw_quad_mesh(gc, master_transform, mesh_width, mesh_height, coordinates, offsets, offset_trans, facecolors,
                                  antialiased, edgecolors)
    }
}
