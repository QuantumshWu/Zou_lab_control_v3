//! A [`Backend`] that records every call with owned arguments (for tests and
//! for diffing a draw against matplotlib's recorded stream).
use crate::backend::{Backend, Gc, GrayImage, PathRef, RgbaImage};
use crate::ir::Dash;
use crate::transform::M3;
use crate::Result;

#[derive(Clone, Debug, PartialEq)]
pub struct OwnedPath {
    pub vertices: Vec<[f64; 2]>,
    pub codes: Option<Vec<u8>>,
    pub should_simplify: bool,
    pub simplify_threshold: f64,
}

impl From<PathRef<'_>> for OwnedPath {
    fn from(p: PathRef<'_>) -> Self {
        OwnedPath { vertices: p.vertices.to_vec(), codes: p.codes.map(<[u8]>::to_vec), should_simplify: p.should_simplify,
                    simplify_threshold: p.simplify_threshold }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TextImage { pub rows: usize, pub cols: usize, pub id: Option<String>, pub pixels: Vec<u8> }

#[derive(Clone, Debug, PartialEq)]
pub enum Call {
    DrawPath { gc: Gc, path: OwnedPath, trans: M3, face: Option<[f64; 4]> },
    DrawMarkers { gc: Gc, marker_path: OwnedPath, marker_trans: M3, path: OwnedPath, trans: M3, face: Option<[f64; 4]> },
    DrawTextImage { image: TextImage, x: i32, y: i32, angle: f64, gc: Gc },
    DrawImage { gc: Gc, x: f64, y: f64, image: RgbaImage },
    DrawPathCollection {
        gc: Gc, master_transform: M3, paths: Vec<OwnedPath>, transforms: Vec<[[f64; 3]; 3]>, offsets: Vec<[f64; 2]>,
        offset_trans: M3, facecolors: Vec<[f64; 4]>, edgecolors: Vec<[f64; 4]>, linewidths: Vec<f64>,
        dashes: Vec<Dash>, antialiaseds: Vec<bool>,
    },
    DrawQuadMesh {
        gc: Gc, master_transform: M3, mesh_width: usize, mesh_height: usize, coordinates: Vec<[f64; 2]>,
        offsets: Vec<[f64; 2]>, offset_trans: M3, facecolors: Vec<[f64; 4]>, antialiased: bool,
        edgecolors: Vec<[f64; 4]>,
    },
}

impl Call {
    pub fn name(&self) -> &'static str {
        match self {
            Call::DrawPath { .. } => "draw_path",
            Call::DrawMarkers { .. } => "draw_markers",
            Call::DrawTextImage { .. } => "draw_text_image",
            Call::DrawImage { .. } => "draw_image",
            Call::DrawPathCollection { .. } => "draw_path_collection",
            Call::DrawQuadMesh { .. } => "draw_quad_mesh",
        }
    }
}

#[derive(Default)]
pub struct Recorder { pub calls: Vec<Call>, pub keep_text_pixels: bool }

impl Backend for Recorder {
    fn draw_path(&mut self, gc: &Gc, path: PathRef<'_>, trans: &M3, face: Option<[f64; 4]>) -> Result<()> {
        self.calls.push(Call::DrawPath { gc: gc.clone(), path: path.into(), trans: *trans, face });
        Ok(())
    }

    fn draw_markers(&mut self, gc: &Gc, marker_path: PathRef<'_>, marker_trans: &M3, path: PathRef<'_>, trans: &M3,
                    face: Option<[f64; 4]>) -> Result<()> {
        self.calls.push(Call::DrawMarkers { gc: gc.clone(), marker_path: marker_path.into(), marker_trans: *marker_trans,
                                            path: path.into(), trans: *trans, face });
        Ok(())
    }

    fn draw_text_image(&mut self, image: &dyn GrayImage, x: i32, y: i32, angle: f64, gc: &Gc) -> Result<()> {
        let pixels = if self.keep_text_pixels { image.pixels().to_vec() } else { Vec::new() };
        self.calls.push(Call::DrawTextImage {
            image: TextImage { rows: image.rows(), cols: image.cols(), id: image.oracle_id().map(str::to_owned), pixels },
            x, y, angle, gc: gc.clone(),
        });
        Ok(())
    }

    fn draw_image(&mut self, gc: &Gc, x: f64, y: f64, image: &RgbaImage) -> Result<()> {
        self.calls.push(Call::DrawImage { gc: gc.clone(), x, y, image: image.clone() });
        Ok(())
    }

    fn draw_path_collection(&mut self, gc: &Gc, master_transform: &M3, paths: &[PathRef<'_>],
                            transforms: &[[[f64; 3]; 3]], offsets: &[[f64; 2]], offset_trans: &M3,
                            facecolors: &[[f64; 4]], edgecolors: &[[f64; 4]], linewidths: &[f64],
                            dashes: &[Dash], antialiaseds: &[bool]) -> Result<()> {
        self.calls.push(Call::DrawPathCollection {
            gc: gc.clone(), master_transform: *master_transform, paths: paths.iter().map(|p| (*p).into()).collect(),
            transforms: transforms.to_vec(), offsets: offsets.to_vec(), offset_trans: *offset_trans,
            facecolors: facecolors.to_vec(), edgecolors: edgecolors.to_vec(), linewidths: linewidths.to_vec(),
            dashes: dashes.to_vec(), antialiaseds: antialiaseds.to_vec(),
        });
        Ok(())
    }

    fn draw_quad_mesh(&mut self, gc: &Gc, master_transform: &M3, mesh_width: usize, mesh_height: usize,
                      coordinates: &[[f64; 2]], offsets: &[[f64; 2]], offset_trans: &M3, facecolors: &[[f64; 4]],
                      antialiased: bool, edgecolors: &[[f64; 4]]) -> Result<()> {
        self.calls.push(Call::DrawQuadMesh {
            gc: gc.clone(), master_transform: *master_transform, mesh_width, mesh_height,
            coordinates: coordinates.to_vec(), offsets: offsets.to_vec(), offset_trans: *offset_trans,
            facecolors: facecolors.to_vec(), antialiased, edgecolors: edgecolors.to_vec(),
        });
        Ok(())
    }
}
