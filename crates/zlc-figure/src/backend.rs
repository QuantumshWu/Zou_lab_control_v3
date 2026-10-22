//! The renderer boundary: what matplotlib's Python artists pass to the C++
//! `RendererAgg` (`matplotlib.backends._backend_agg`), argument for argument,
//! BEFORE pybind11's converters (an adapter to `zlc-agg` applies those: the
//! face-alpha rule of `draw_path`/`draw_markers`, `mpl_round` and `alpha = 1`
//! of `draw_image`, odd dash lists doubled, the GC fields the `GCAgg` caster
//! reads).
use crate::ir::{CapStyle, Dash, JoinStyle};
use crate::transform::{M3, Points};
use crate::Result;

/// `GraphicsContextBase`, as the `GCAgg` converter reads it.
#[derive(Clone, Debug, PartialEq)]
pub struct Gc {
    pub linewidth: f64,
    pub alpha: f64,
    pub forced_alpha: bool,
    /// `_rgb` (always four components here).
    pub rgb: [f64; 4],
    pub antialiased: bool,
    pub capstyle: CapStyle,
    pub joinstyle: JoinStyle,
    /// `get_dashes()`: (offset, on-off list); offset None only when a list is None.
    pub dashes: (Option<f64>, Option<Vec<f64>>),
    /// The clip rectangle's points at call time.
    pub cliprect: Option<Points>,
    pub snap: Option<bool>,
    pub hatch: Option<String>,
    /// `get_hatch_path()`: the hatch pattern in the unit square (density 6).
    pub hatch_path: Option<HatchPath>,
    pub hatch_color: [f64; 4],
    pub hatch_linewidth: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HatchPath { pub vertices: Vec<[f64; 2]>, pub codes: Vec<u8>, pub should_simplify: bool, pub simplify_threshold: f64 }

/// `numpy.linspace(start, stop, num)` (endpoint=True).
pub fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    if num == 0 { return Vec::new(); }
    if num == 1 { return vec![start]; }
    let div = (num - 1) as f64;
    let delta = stop - start;
    let step = delta / div;
    let mut y: Vec<f64> = (0..num).map(|i| {
        let v = i as f64;
        if step == 0.0 { (v / div) * delta } else { v * step }
    }).map(|v| v + start).collect();
    y[num - 1] = stop;
    y
}

/// `matplotlib.hatch.get_path(hatch, 6)` for patterns of '/' only
/// (NorthEastHatch), with the cached path's simplification.
pub fn hatch_path(hatch: &str, simplify: crate::ir::Simplify) -> Result<HatchPath> {
    if hatch.is_empty() || hatch.chars().any(|c| c != '/') {
        return crate::unsupported(format!("hatch {hatch:?}"));
    }
    let num_lines = hatch.chars().count() * 6;
    let steps = linspace(-0.5, 0.5, num_lines + 1);
    let mut vertices = Vec::with_capacity(2 * steps.len());
    let mut codes = Vec::with_capacity(2 * steps.len());
    for s in steps {
        vertices.push([0.0 + s, 0.0 - s]);
        vertices.push([1.0 + s, 1.0 - s]);
        codes.push(1);
        codes.push(2);
    }
    // Path._update_values simplifies only a path of 128 vertices or more, at a positive threshold.
    if simplify.should_simplify && !(vertices.len() >= 128 && simplify.threshold > 0.0) {
        return crate::invalid(format!("a hatch path of {} vertices that simplifies", vertices.len()));
    }
    Ok(HatchPath { vertices, codes, should_simplify: simplify.should_simplify, simplify_threshold: simplify.threshold })
}

impl Gc {
    /// `renderer.new_gc()`: `GraphicsContextBase.__init__` (hatch colour and
    /// width come from rcParams).
    pub fn new(hatch_color: [f64; 4], hatch_linewidth: f64) -> Gc {
        Gc {
            linewidth: 1.0,
            alpha: 1.0,
            forced_alpha: false,
            rgb: [0.0, 0.0, 0.0, 1.0],
            antialiased: true,
            capstyle: CapStyle::Butt,
            joinstyle: JoinStyle::Round,
            dashes: (Some(0.0), None),
            cliprect: None,
            snap: None,
            hatch: None,
            hatch_path: None,
            hatch_color,
            hatch_linewidth,
        }
    }

    /// `set_foreground(fg, isRGBA=True)` for an RGBA tuple.
    pub fn set_foreground_rgba(&mut self, fg: [f64; 4]) {
        self.rgb = if self.forced_alpha { [fg[0], fg[1], fg[2], self.alpha] } else { fg };
    }

    /// `set_foreground(fg)` where `fg` already went through `to_rgba` (not
    /// isRGBA): with a forced alpha that is `to_rgba(fg, alpha)`.
    pub fn set_foreground_color(&mut self, rgba: [f64; 4], none: bool) {
        self.rgb = if self.forced_alpha {
            if none { [0.0, 0.0, 0.0, 0.0] } else { [rgba[0], rgba[1], rgba[2], self.alpha] }
        } else {
            rgba
        };
    }

    /// `set_alpha(alpha)`.
    pub fn set_alpha(&mut self, alpha: Option<f64>) {
        match alpha {
            Some(a) => { self.alpha = a; self.forced_alpha = true; }
            None => { self.alpha = 1.0; self.forced_alpha = false; }
        }
        let rgb = self.rgb;
        self.set_foreground_rgba(rgb);
    }

    pub fn set_linewidth(&mut self, w: f64) { self.linewidth = w; }

    /// `set_dashes(offset, list)`, which raises ValueError for a list with a
    /// negative value or a non-empty one without a positive value.
    pub fn set_dashes(&mut self, dash: &Dash) -> Result<()> {
        if let Some(list) = &dash.1 {
            if list.iter().any(|&v| v < 0.0) {
                return crate::invalid(format!("dash list {list:?} has a negative value (set_dashes raises)"));
            }
            if !list.is_empty() && !list.iter().any(|&v| v > 0.0) {
                return crate::invalid(format!("dash list {list:?} has no positive value (set_dashes raises)"));
            }
        }
        self.dashes = (Some(dash.0), dash.1.clone());
        Ok(())
    }
}

/// A borrowed `Path` as the `PathIterator` converter reads it.
#[derive(Clone, Copy, Debug)]
pub struct PathRef<'a> {
    pub vertices: &'a [[f64; 2]],
    pub codes: Option<&'a [u8]>,
    pub should_simplify: bool,
    pub simplify_threshold: f64,
}

/// An 8-bit coverage image handed to `draw_text_image` (an FT2Font's buffer or
/// a mathtext FT2Image).
pub trait GrayImage {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    fn pixels(&self) -> &[u8];
    /// For a recorded font: the oracle's identity of this image (its hash).
    fn oracle_id(&self) -> Option<&str> { None }
}

/// An owned coverage bitmap.
#[derive(Clone, Debug, PartialEq)]
pub struct Bitmap { pub rows: usize, pub cols: usize, pub data: Vec<u8>, pub id: Option<String> }

impl GrayImage for Bitmap {
    fn rows(&self) -> usize { self.rows }
    fn cols(&self) -> usize { self.cols }
    fn pixels(&self) -> &[u8] { &self.data }
    fn oracle_id(&self) -> Option<&str> { self.id.as_deref() }
}

/// An (rows, cols, 4) uint8 image for `draw_image`.
#[derive(Clone, Debug, PartialEq)]
pub struct RgbaImage { pub rows: usize, pub cols: usize, pub data: Vec<u8> }

/// One line of text as a vector renderer's `draw_text` takes it: the string
/// and its font, or mathtext's vector parse (`MathTextParser("path")`).
#[derive(Clone, Copy, Debug)]
pub enum TextRun<'a> {
    Plain { text: &'a str, font: &'a crate::ir::Font },
    Math { parse: &'a crate::fonts::VectorParse, font: &'a crate::ir::Font },
}

/// The C++ `RendererAgg` methods matplotlib's artists call; a vector
/// renderer (`RendererSVG`, `RendererPdf`) also answers the seams below,
/// whose defaults are Agg's.
pub trait Backend {
    /// A vector renderer: text is measured as `TextToPath` measures it
    /// (`RendererSVG`'s rule: unhinted at 100 pt, scaled) and every line
    /// reaches [`Backend::draw_text`]; images reach
    /// [`Backend::draw_image_transformed`] at their own resolution
    /// (`option_scale_image`); paths are not chunked.
    fn vector(&self) -> bool { false }

    /// `draw_text(gc, x, y, s, prop, angle, ismath)`: the line's baseline
    /// start (x, y) in display space (y up), rotated `angle` degrees about it.
    fn draw_text(&mut self, _gc: &Gc, _x: f64, _y: f64, _run: TextRun<'_>, _angle: f64) -> Result<()> {
        crate::unsupported("draw_text on a raster backend")
    }

    /// `get_image_magnification()`: the resolution images are sampled at,
    /// in image pixels per display unit (a vector renderer's export dpi / 72).
    fn image_magnification(&self) -> f64 { 1.0 }

    /// `draw_image(gc, x, y, im, transform)`: `pixel_to_display` maps the
    /// image's pixel grid (column, row; row 0 first, (0, 0) its corner) to
    /// display space.
    fn draw_image_transformed(&mut self, _gc: &Gc, _image: &RgbaImage, _pixel_to_display: &M3) -> Result<()> {
        crate::unsupported("a transformed image on a raster backend")
    }

    fn draw_path(&mut self, gc: &Gc, path: PathRef<'_>, trans: &M3, face: Option<[f64; 4]>) -> Result<()>;

    fn draw_markers(&mut self, gc: &Gc, marker_path: PathRef<'_>, marker_trans: &M3, path: PathRef<'_>, trans: &M3,
                    face: Option<[f64; 4]>) -> Result<()>;

    fn draw_text_image(&mut self, image: &dyn GrayImage, x: i32, y: i32, angle: f64, gc: &Gc) -> Result<()>;

    fn draw_image(&mut self, gc: &Gc, x: f64, y: f64, image: &RgbaImage) -> Result<()>;

    #[allow(clippy::too_many_arguments)]
    fn draw_path_collection(&mut self, gc: &Gc, master_transform: &M3, paths: &[PathRef<'_>],
                            transforms: &[[[f64; 3]; 3]], offsets: &[[f64; 2]], offset_trans: &M3,
                            facecolors: &[[f64; 4]], edgecolors: &[[f64; 4]], linewidths: &[f64],
                            dashes: &[Dash], antialiaseds: &[bool]) -> Result<()>;

    #[allow(clippy::too_many_arguments)]
    fn draw_quad_mesh(&mut self, gc: &Gc, master_transform: &M3, mesh_width: usize, mesh_height: usize,
                      coordinates: &[[f64; 2]], offsets: &[[f64; 2]], offset_trans: &M3, facecolors: &[[f64; 4]],
                      antialiased: bool, edgecolors: &[[f64; 4]]) -> Result<()>;
}
