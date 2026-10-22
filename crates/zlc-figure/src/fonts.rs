//! Text measurement and rasterization boundaries.
//!
//! [`Fonts`] is the FT2Font surface `RendererAgg.draw_text`,
//! `get_text_width_height_descent` and `_prepare_font` use (implemented by
//! `zlc-text`'s FT2Font port, [`crate::ftfonts`]). [`MathText`] is
//! `MathTextParser("agg").parse` (implemented by `zlc-mathtext`,
//! [`crate::mathtext`]). Stream tests answer both from values recorded off
//! matplotlib.
use crate::backend::GrayImage;
use crate::ir;
use crate::Result;

/// FreeType load flags (`matplotlib.ft2font.LoadFlags`).
pub mod load_flags {
    pub const DEFAULT: i32 = 0;
    pub const NO_HINTING: i32 = 2;
    pub const FORCE_AUTOHINT: i32 = 32;
    pub const NO_AUTOHINT: i32 = 32768;
}

/// `backend_agg.get_hinting_flag()` for `rcParams['text.hinting']`.
pub fn hinting_flag(hinting: &str) -> Result<i32> {
    use load_flags::*;
    Ok(match hinting {
        "default" | "either" => DEFAULT,
        "no_autohint" | "native" => NO_AUTOHINT,
        "force_autohint" | "auto" => FORCE_AUTOHINT,
        "no_hinting" | "none" => NO_HINTING,
        other => return crate::unsupported(format!("text.hinting {other}")),
    })
}

/// A font handle returned by [`Fonts::get_font`].
pub type FontId = usize;

/// The FT2Font operations matplotlib's Agg text path performs.
pub trait Fonts {
    /// `get_font(files)`: the FT2Font of `files[0]` with `files[1..]` as
    /// its fallbacks, at rcParams' hinting and kerning factors. The same
    /// files give the same font object (matplotlib's cache), whose state
    /// persists between calls.
    fn get_font(&mut self, files: &[String], hinting_factor: i64, kerning_factor: i64) -> Result<FontId>;
    fn clear(&mut self, font: FontId) -> Result<()>;
    fn set_size(&mut self, font: FontId, ptsize: f64, dpi: f64) -> Result<()>;
    fn set_text(&mut self, font: FontId, text: &str, angle: f64, flags: i32) -> Result<()>;
    /// In 26.6 subpixels.
    fn get_width_height(&mut self, font: FontId) -> Result<(i64, i64)>;
    fn get_descent(&mut self, font: FontId) -> Result<i64>;
    fn get_bitmap_offset(&mut self, font: FontId) -> Result<(i64, i64)>;
    fn draw_glyphs_to_bitmap(&mut self, font: FontId, antialiased: bool) -> Result<()>;
    /// The font's image buffer, which `draw_text_image` reads.
    fn image(&mut self, font: FontId) -> Result<&dyn GrayImage>;
}

/// `MathTextParser("agg").parse(s, dpi, prop, antialiased=...)`.
pub struct MathParse {
    pub ox: f64,
    pub oy: f64,
    pub width: f64,
    pub height: f64,
    pub descent: f64,
    pub image: Box<dyn GrayImage>,
}

/// One glyph of a vector parse: its font file and size, the character code
/// it was loaded by (`FontInfo.num`) and its glyph index there, and its
/// origin from the line's baseline start (y up).
#[derive(Clone, Debug, PartialEq)]
pub struct VectorGlyph { pub font_file: std::path::PathBuf, pub fontsize: f64, pub num: u32, pub glyph_id: u32, pub ox: f64, pub oy: f64 }

/// `TextToPath`'s mathtext: `MathTextParser("path").parse(s, 72, prop)`
/// (`Output.to_vector`) at `FONT_SCALE` points, scaled to the font's size:
/// width, height (with depth) and depth, the glyphs, and the rules as
/// (x, y, width, height) from the baseline start (y up).
#[derive(Clone, Debug, PartialEq)]
pub struct VectorParse { pub width: f64, pub height: f64, pub descent: f64, pub glyphs: Vec<VectorGlyph>, pub rects: Vec<[f64; 4]> }

pub trait MathText {
    /// Under the figure's rcParams `rc` (the parser reads `text.hinting`,
    /// `text.antialiased`, `mathtext.*` and the font manager's hinting and
    /// kerning factors). `antialiased: None` resolves to
    /// `rcParams['text.antialiased']`, as the parser does.
    fn parse(&mut self, rc: &ir::Rc, s: &str, dpi: f64, font: &ir::Font, antialiased: Option<bool>) -> Result<MathParse>;

    /// The vector parse a vector renderer measures and draws mathtext with.
    fn vector(&mut self, _rc: &ir::Rc, s: &str, _font: &ir::Font) -> Result<VectorParse> {
        crate::unsupported(format!("vector mathtext without an implementation: {s:?}"))
    }
}

/// A MathText that refuses every string (for figures without mathtext).
pub struct NoMathText;

impl MathText for NoMathText {
    fn parse(&mut self, _rc: &ir::Rc, s: &str, _dpi: f64, _font: &ir::Font, _antialiased: Option<bool>) -> Result<MathParse> {
        crate::unsupported(format!("mathtext without a MathText implementation: {s:?}"))
    }
}
