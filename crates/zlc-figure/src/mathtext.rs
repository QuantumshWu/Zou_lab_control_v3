//! [`MathText`] over `zlc-mathtext`'s port of matplotlib's mathtext:
//! `MathTextParser("agg").parse(s, dpi, prop, antialiased=...)` with the
//! figure's rcParams. The fontset is `prop.get_math_fontfamily()` (the port
//! has `dejavusans`), the raster load flags `get_hinting_flag()` of
//! `text.hinting`, the font manager's `text.hinting_factor` /
//! `text.kerning_factor` open every face. The non-math font
//! (`findfont(prop)`, [`ir::Font::file`]) is resolved by file name through a
//! caller-supplied function, the math fonts come from matplotlib's
//! `mpl-data/fonts/ttf`.
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use zlc_mathtext::{Config, Raster, Request};

use crate::backend::GrayImage;
use crate::fonts::{hinting_flag, MathParse, MathText, VectorGlyph, VectorParse};
use crate::{ir, Error, Result};

/// A raster parse's coverage image (`image_height` x `image_width`).
struct MathImage(Arc<Raster>);

impl GrayImage for MathImage {
    fn rows(&self) -> usize { self.0.image_height }
    fn cols(&self) -> usize { self.0.image_width }
    fn pixels(&self) -> &[u8] { &self.0.image }
}

fn math_err(e: zlc_mathtext::Error) -> Error {
    match e {
        zlc_mathtext::Error::Unsupported(what) => Error::Unsupported(format!("mathtext: {what}")),
        other => Error::Backend(format!("mathtext: {other}")),
    }
}

pub struct NativeMathText {
    math_fonts_dir: PathBuf,
    resolve: Box<dyn Fn(&str) -> Option<PathBuf>>,
    files: HashMap<String, PathBuf>,
    /// One engine (faces and caches) per (hinting factor, kerning factor, load flags).
    engines: HashMap<(i64, i64, i32), zlc_mathtext::MathText>,
}

impl NativeMathText {
    /// `math_fonts_dir`: matplotlib's `mpl-data/fonts/ttf` (DejaVu, STIX, BaKoMa).
    pub fn new(math_fonts_dir: impl Into<PathBuf>, resolve: Box<dyn Fn(&str) -> Option<PathBuf>>) -> Self {
        NativeMathText { math_fonts_dir: math_fonts_dir.into(), resolve, files: HashMap::new(), engines: HashMap::new() }
    }

    /// Forgets cached layouts and rasters (faces and loaded glyphs stay).
    pub fn clear_cache(&mut self) {
        for engine in self.engines.values_mut() { engine.clear_cache(); }
    }
}

impl NativeMathText {
    /// The engine of `rc`'s configuration and the non-math font's file.
    ///
    /// The parser's Fonts class is the font's math_fontfamily's. The port
    /// has DejaVuSansFonts, whose rm/it/bf/sf/tt/bfit faces are fixed
    /// DejaVu files with STIX Sans behind them: it reads none of rcParams
    /// mathtext.{rm, it, bf, sf, tt, cal, bfit, fallback} (only 'custom',
    /// UnicodeFonts, reads them; refused with every other fontset).
    fn engine(&mut self, rc: &ir::Rc, font: &ir::Font) -> Result<(&mut zlc_mathtext::MathText, PathBuf)> {
        if font.math_fontfamily != "dejavusans" {
            return crate::unsupported(format!("mathtext fontset {} (rcParams mathtext.rm {:?}, fallback {:?})",
                                              font.math_fontfamily, rc.mathtext_rm, rc.mathtext_fallback));
        }
        if rc.mathtext_default != "it" {
            return crate::unsupported(format!("mathtext.default {}", rc.mathtext_default));
        }
        let flags = hinting_flag(&rc.text_hinting)?;
        let kerning = i32::try_from(rc.text_kerning_factor)
            .map_err(|_| Error::Invalid(format!("text.kerning_factor {}", rc.text_kerning_factor)))?;
        if !self.files.contains_key(&font.file) {
            let path = (self.resolve)(&font.file)
                .ok_or_else(|| Error::Backend(format!("font file {} not found", font.file)))?;
            self.files.insert(font.file.clone(), path);
        }
        let file = self.files[&font.file].clone();
        let dir = &self.math_fonts_dir;
        let engine = self.engines.entry((rc.text_hinting_factor, rc.text_kerning_factor, flags)).or_insert_with(|| {
            let mut config = Config::new(dir.clone());
            config.hinting_factor = rc.text_hinting_factor;
            config.kerning_factor = kerning;
            config.raster_load_flags = flags;
            zlc_mathtext::MathText::new(config)
        });
        Ok((engine, file))
    }
}

impl MathText for NativeMathText {
    /// `Output.to_vector()` of the unhinted parse at `FONT_SCALE` (ceiled box
    /// metrics, each glyph at `(ox, h - oy + offset)`, each rule at
    /// `(x1, h - y2, x2 - x1, y2 - y1)`), scaled to the font's size.
    fn vector(&mut self, rc: &ir::Rc, s: &str, font: &ir::Font) -> Result<VectorParse> {
        let (engine, file) = self.engine(rc, font)?;
        let request = Request { text: s, font_file: &file, size: crate::text::FONT_SCALE, dpi: 72.0 };
        let layout = engine.layout_with_flags(&request, zlc_mathtext::load_flags::NO_HINTING).map_err(math_err)?;
        let (w, h, d) = (layout.width.ceil(), layout.height.ceil(), layout.depth.ceil());
        let k = font.size / crate::text::FONT_SCALE;
        let glyphs = layout.glyphs.iter().map(|g| VectorGlyph {
            font_file: g.font_file.clone(), fontsize: g.fontsize * k, num: g.num, glyph_id: g.glyph_id, ox: g.ox * k,
            oy: (h - g.oy + g.offset) * k,
        }).collect();
        let rects = layout.rects.iter().map(|&[x1, y1, x2, y2]| [x1 * k, (h - y2) * k, (x2 - x1) * k, (y2 - y1) * k]).collect();
        Ok(VectorParse { width: w * k, height: (h + d) * k, descent: d * k, glyphs, rects })
    }

    fn parse(&mut self, rc: &ir::Rc, s: &str, dpi: f64, font: &ir::Font, antialiased: Option<bool>) -> Result<MathParse> {
        let (engine, file) = self.engine(rc, font)?;
        let request = Request { text: s, font_file: &file, size: font.size, dpi };
        let raster = engine.raster(&request, antialiased.unwrap_or(rc.text_antialiased)).map_err(math_err)?;
        Ok(MathParse { ox: raster.ox, oy: raster.oy, width: raster.width, height: raster.height, descent: raster.depth,
                       image: Box::new(MathImage(raster)) })
    }
}
