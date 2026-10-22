//! What a compose keeps between frames besides the background: the canvas
//! it paints on, the buffer of the last front, and the backend calls of a
//! grid's chrome (cell strokes and titles), replayed while nothing they
//! depend on moved -- the same calls reach the canvas, so the same pixels.
use std::cell::RefCell;
use std::collections::HashMap;

use zlc_figure::backend::{Backend, Gc, GrayImage, PathRef, RgbaImage};
use zlc_figure::ir::Dash;
use zlc_figure::record::{Call, OwnedPath, TextImage};
use zlc_figure::transform::M3;
use zlc_figure::Result;

/// State a renderer keeps for itself only: a copy of the renderer starts without it.
pub struct Retained<T>(pub Option<T>);

impl<T> Clone for Retained<T> {
    fn clone(&self) -> Self { Retained(None) }
}

impl<T> Default for Retained<T> {
    fn default() -> Self { Retained(None) }
}

/// The chrome's recorded calls per (axes, stroke or figure position, figure?),
/// each under the key of what it draws: the last compose's, and this one's.
#[derive(Debug, Default)]
pub struct Recordings {
    pub previous: HashMap<(usize, usize, bool), (u64, Vec<Call>)>,
    pub next: HashMap<(usize, usize, bool), (u64, Vec<Call>)>,
}

/// A backend that forwards every call and, while `recording` holds a list, records it there too.
pub struct Tap<'b> {
    pub inner: &'b mut dyn Backend,
    pub recording: &'b RefCell<Option<Vec<Call>>>,
}

impl Tap<'_> {
    fn record(&self, call: impl FnOnce() -> Call) {
        if let Some(calls) = self.recording.borrow_mut().as_mut() { calls.push(call()); }
    }
}

impl Backend for Tap<'_> {
    fn draw_path(&mut self, gc: &Gc, path: PathRef<'_>, trans: &M3, face: Option<[f64; 4]>) -> Result<()> {
        self.record(|| Call::DrawPath { gc: gc.clone(), path: path.into(), trans: *trans, face });
        self.inner.draw_path(gc, path, trans, face)
    }

    fn draw_markers(&mut self, gc: &Gc, marker_path: PathRef<'_>, marker_trans: &M3, path: PathRef<'_>, trans: &M3,
                    face: Option<[f64; 4]>) -> Result<()> {
        self.record(|| Call::DrawMarkers { gc: gc.clone(), marker_path: marker_path.into(), marker_trans: *marker_trans,
                                           path: path.into(), trans: *trans, face });
        self.inner.draw_markers(gc, marker_path, marker_trans, path, trans, face)
    }

    fn draw_text_image(&mut self, image: &dyn GrayImage, x: i32, y: i32, angle: f64, gc: &Gc) -> Result<()> {
        self.record(|| Call::DrawTextImage {
            image: TextImage { rows: image.rows(), cols: image.cols(), id: image.oracle_id().map(str::to_owned), pixels: image.pixels().to_vec() },
            x, y, angle, gc: gc.clone(),
        });
        self.inner.draw_text_image(image, x, y, angle, gc)
    }

    fn draw_image(&mut self, gc: &Gc, x: f64, y: f64, image: &RgbaImage) -> Result<()> {
        self.record(|| Call::DrawImage { gc: gc.clone(), x, y, image: image.clone() });
        self.inner.draw_image(gc, x, y, image)
    }

    fn draw_path_collection(&mut self, gc: &Gc, master_transform: &M3, paths: &[PathRef<'_>],
                            transforms: &[[[f64; 3]; 3]], offsets: &[[f64; 2]], offset_trans: &M3,
                            facecolors: &[[f64; 4]], edgecolors: &[[f64; 4]], linewidths: &[f64],
                            dashes: &[Dash], antialiaseds: &[bool]) -> Result<()> {
        self.record(|| Call::DrawPathCollection {
            gc: gc.clone(), master_transform: *master_transform, paths: paths.iter().map(|p| (*p).into()).collect(),
            transforms: transforms.to_vec(), offsets: offsets.to_vec(), offset_trans: *offset_trans,
            facecolors: facecolors.to_vec(), edgecolors: edgecolors.to_vec(), linewidths: linewidths.to_vec(),
            dashes: dashes.to_vec(), antialiaseds: antialiaseds.to_vec(),
        });
        self.inner.draw_path_collection(gc, master_transform, paths, transforms, offsets, offset_trans, facecolors, edgecolors,
                                        linewidths, dashes, antialiaseds)
    }

    fn draw_quad_mesh(&mut self, gc: &Gc, master_transform: &M3, mesh_width: usize, mesh_height: usize,
                      coordinates: &[[f64; 2]], offsets: &[[f64; 2]], offset_trans: &M3, facecolors: &[[f64; 4]],
                      antialiased: bool, edgecolors: &[[f64; 4]]) -> Result<()> {
        self.record(|| Call::DrawQuadMesh {
            gc: gc.clone(), master_transform: *master_transform, mesh_width, mesh_height,
            coordinates: coordinates.to_vec(), offsets: offsets.to_vec(), offset_trans: *offset_trans,
            facecolors: facecolors.to_vec(), antialiased, edgecolors: edgecolors.to_vec(),
        });
        self.inner.draw_quad_mesh(gc, master_transform, mesh_width, mesh_height, coordinates, offsets, offset_trans,
                                  facecolors, antialiased, edgecolors)
    }
}

struct Image<'a>(&'a TextImage);

impl GrayImage for Image<'_> {
    fn rows(&self) -> usize { self.0.rows }
    fn cols(&self) -> usize { self.0.cols }
    fn pixels(&self) -> &[u8] { &self.0.pixels }
    fn oracle_id(&self) -> Option<&str> { self.0.id.as_deref() }
}

fn path(p: &OwnedPath) -> PathRef<'_> {
    PathRef { vertices: &p.vertices, codes: p.codes.as_deref(), should_simplify: p.should_simplify, simplify_threshold: p.simplify_threshold }
}

/// Make the recorded calls again, in order.
pub fn replay(backend: &mut dyn Backend, calls: &[Call]) -> Result<()> {
    for call in calls {
        match call {
            Call::DrawPath { gc, path: p, trans, face } => backend.draw_path(gc, path(p), trans, *face)?,
            Call::DrawMarkers { gc, marker_path, marker_trans, path: p, trans, face } =>
                backend.draw_markers(gc, path(marker_path), marker_trans, path(p), trans, *face)?,
            Call::DrawTextImage { image, x, y, angle, gc } => backend.draw_text_image(&Image(image), *x, *y, *angle, gc)?,
            Call::DrawImage { gc, x, y, image } => backend.draw_image(gc, *x, *y, image)?,
            Call::DrawPathCollection { gc, master_transform, paths, transforms, offsets, offset_trans, facecolors, edgecolors,
                                       linewidths, dashes, antialiaseds } => {
                let paths: Vec<PathRef<'_>> = paths.iter().map(path).collect();
                backend.draw_path_collection(gc, master_transform, &paths, transforms, offsets, offset_trans, facecolors, edgecolors,
                                             linewidths, dashes, antialiaseds)?
            }
            Call::DrawQuadMesh { gc, master_transform, mesh_width, mesh_height, coordinates, offsets, offset_trans, facecolors,
                                 antialiased, edgecolors } =>
                backend.draw_quad_mesh(gc, master_transform, *mesh_width, *mesh_height, coordinates, offsets, offset_trans, facecolors,
                                       *antialiased, edgecolors)?,
        }
    }
    Ok(())
}
