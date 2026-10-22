//! [`Fonts`] over `zlc-text`'s FT2Font port: `get_font(files)` opens the first
//! file with the rest as fallbacks (fresh per distinct file list, as
//! matplotlib's `_get_font` cache does), resolving file names through a
//! caller-supplied function.
use std::collections::HashMap;
use std::path::PathBuf;

use zlc_text::{FT2Font, FT2Image};

use crate::backend::GrayImage;
use crate::fonts::{FontId, Fonts};
use crate::{Error, Result};

fn text_err(e: zlc_text::Error) -> Error { Error::Backend(format!("ft2font: {e}")) }

impl GrayImage for FT2Image {
    fn rows(&self) -> usize { self.height() as usize }
    fn cols(&self) -> usize { self.width() as usize }
    fn pixels(&self) -> &[u8] { FT2Image::pixels(self) }
}

pub struct TextFonts {
    resolve: Box<dyn Fn(&str) -> Option<PathBuf>>,
    by_files: HashMap<(Vec<String>, i64, i64), FontId>,
    fonts: Vec<FT2Font>,
}

impl TextFonts {
    pub fn new(resolve: Box<dyn Fn(&str) -> Option<PathBuf>>) -> Self {
        TextFonts { resolve, by_files: HashMap::new(), fonts: Vec::new() }
    }

    fn open(&self, name: &str, hinting_factor: i64, fallbacks: Vec<zlc_text::SharedFont>, kerning_factor: i64)
            -> Result<FT2Font> {
        let path = (self.resolve)(name).ok_or_else(|| Error::Backend(format!("font file {name} not found")))?;
        let data = std::fs::read(&path).map_err(|e| Error::Backend(format!("{}: {e}", path.display())))?;
        FT2Font::new(data, hinting_factor as _, fallbacks, kerning_factor as _).map_err(text_err)
    }
}

impl Fonts for TextFonts {
    fn get_font(&mut self, files: &[String], hinting_factor: i64, kerning_factor: i64) -> Result<FontId> {
        let key = (files.to_vec(), hinting_factor, kerning_factor);
        if let Some(&id) = self.by_files.get(&key) { return Ok(id); }
        let mut fallbacks = Vec::new();
        for name in &files[1..] {
            let font = self.open(name, hinting_factor, Vec::new(), kerning_factor)?;
            fallbacks.push(std::sync::Arc::new(std::sync::Mutex::new(font)));
        }
        let font = self.open(&files[0], hinting_factor, fallbacks, kerning_factor)?;
        self.fonts.push(font);
        let id = self.fonts.len() - 1;
        self.by_files.insert(key, id);
        Ok(id)
    }
    fn clear(&mut self, font: FontId) -> Result<()> { self.fonts[font].clear().map_err(text_err) }
    fn set_size(&mut self, font: FontId, ptsize: f64, dpi: f64) -> Result<()> {
        self.fonts[font].set_size(ptsize, dpi).map_err(text_err)
    }
    fn set_text(&mut self, font: FontId, text: &str, angle: f64, flags: i32) -> Result<()> {
        let codes: Vec<u32> = text.chars().map(u32::from).collect();
        let mut xys = Vec::new();
        let mut warn = |_: std::ffi::c_ulong, _: &[u8]| -> std::result::Result<(), zlc_text::Error> { Ok(()) };
        self.fonts[font].set_text(&codes, angle, flags, &mut xys, &mut warn).map_err(text_err)
    }
    fn get_width_height(&mut self, font: FontId) -> Result<(i64, i64)> {
        let (w, h) = self.fonts[font].get_width_height();
        Ok((w as i64, h as i64))
    }
    fn get_descent(&mut self, font: FontId) -> Result<i64> { Ok(self.fonts[font].get_descent() as i64) }
    fn get_bitmap_offset(&mut self, font: FontId) -> Result<(i64, i64)> {
        let (x, y) = self.fonts[font].get_bitmap_offset();
        Ok((x as i64, y as i64))
    }
    fn draw_glyphs_to_bitmap(&mut self, font: FontId, antialiased: bool) -> Result<()> {
        self.fonts[font].draw_glyphs_to_bitmap(antialiased).map_err(text_err)
    }
    fn image(&mut self, font: FontId) -> Result<&dyn GrayImage> { Ok(self.fonts[font].image()) }
}
