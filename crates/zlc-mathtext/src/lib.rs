//! Byte-exact Rust port of Matplotlib 3.10.8's mathtext (`_mathtext.py`,
//! `mathtext.py`) for the `dejavusans` fontset, over `zlc-text`'s FT2Font.
//!
//! [`MathText`] answers the three questions a renderer asks of a mathtext string:
//!
//! * [`MathText::layout`]: the parsed box (width, height, depth) and the glyphs and
//!   rectangles `_mathtext.ship(box)` places, i.e. `Output.glyphs` / `Output.rects`.
//! * [`MathText::raster`]: `MathTextParser("agg").parse(s, dpi, prop, antialiased=...)`,
//!   i.e. `Output.to_raster`: (ox, oy, width, height, depth) and the coverage image.
//! * [`MathText::metrics`]: `RendererAgg.get_text_width_height_descent(s, prop,
//!   ismath=True)`, i.e. the raster parse's (width, height, depth).
//!
//! A request names the text, the file of the non-math font (what `findfont(prop)`
//! returns: it serves the `default`/`regular` faces and the text outside `$`), the
//! size in points and the dpi. Matplotlib's rc state is fixed by [`Config`]:
//! `mathtext.fontset` = `dejavusans` (whose fallback is STIX sans regardless of
//! `mathtext.fallback`), `mathtext.default` = `it`, the raster load flags of
//! `text.hinting` and the font manager's `text.hinting_factor` / `text.kerning_factor`.
//!
//! Text that Matplotlib rejects is [`Error::Syntax`] / [`Error::UnknownSymbol`] (it raises
//! ValueError); [`Error::MatplotlibRaises`] where Matplotlib fails with another
//! exception. Grammar this port does not implement (accents, `\operatorname`,
//! `\boldsymbol`, `\dfrac`, `\binom`, `\genfrac`, `\overset`, `\underset`, `\overline`,
//! `\text`, `\substack`, `\middle`) is [`Error::Unsupported`]: never approximated.
//!
//! Layouts and rasters are cached per (text, font file, size, dpi, flags[, antialiased]);
//! a repeat costs a hash lookup, a new string (a live readout) tens of microseconds.
//!
//! ```no_run
//! use std::path::Path;
//! use zlc_mathtext::{Config, MathText, Request};
//!
//! // Matplotlib's mpl-data/fonts/ttf; the non-math font is what findfont(prop) names.
//! let mut engine = MathText::new(Config::new("…/matplotlib/mpl-data/fonts/ttf"));
//! let request = Request { text: r"$\sigma$ = 17.14 ± 0.1447 pixel",
//!     font_file: Path::new("…/helvetica-light-587ebe5a59211.ttf"), size: 3.25, dpi: 525.0 };
//! let (width, height, descent) = engine.metrics(&request)?;
//! let raster = engine.raster(&request, true)?; // RasterParse: image is image_height × image_width
//! let layout = engine.layout(&request)?;       // ship(box): positioned glyphs and rects
//! # Ok::<(), zlc_mathtext::Error>(())
//! ```
//!
//! Not reproduced: Matplotlib's parser keeps `_in_subscript_or_superscript` set after a
//! parse that raised, which can space the next parse's first script; every parse here
//! starts clean, as Matplotlib's does after a successful one.

mod data;
mod fonts;
mod node;
mod parser;

use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub use fonts::{FontMetrics, MathFontFile};
use fonts::{FontCache, FontId, FontStore, Fontset};
use node::{ListNode, Output, pymax, pymin, ship};

/// FreeType load flags (`ft2font.LoadFlags`).
pub mod load_flags {
    pub const DEFAULT: i32 = 0;
    pub const NO_HINTING: i32 = 0x2;
    pub const NO_AUTOHINT: i32 = 0x8000;
    pub const FORCE_AUTOHINT: i32 = 0x20;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// Matplotlib's parser rejects the text (`MathTextParser.parse` raises ValueError).
    Syntax(String),
    /// A `\name` without a glyph mapping (Matplotlib raises ValueError "Unknown symbol").
    UnknownSymbol(String),
    /// Matplotlib itself fails here with another exception (an AttributeError for
    /// a script in a font the fontset lacks, ...); the text names it.
    MatplotlibRaises(String),
    /// Mathtext outside the ported subset; nothing is approximated.
    Unsupported(String),
    /// A font file could not be read or FreeType failed.
    Font(String),
}

impl Error {
    /// Whether Matplotlib itself raises ValueError for this text.
    pub fn is_matplotlib_value_error(&self) -> bool { matches!(self, Error::Syntax(_) | Error::UnknownSymbol(_)) }

    /// Whether Matplotlib itself raises (any exception) for this text, as a
    /// caller that falls back on failure (`except Exception`) must know.
    pub fn is_matplotlib_failure(&self) -> bool { self.is_matplotlib_value_error() || matches!(self, Error::MatplotlibRaises(_)) }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Syntax(message) => write!(formatter, "mathtext syntax: {message}"),
            Error::UnknownSymbol(symbol) => write!(formatter, "Unknown symbol: {symbol}"),
            Error::MatplotlibRaises(what) => write!(formatter, "Matplotlib raises {what}"),
            Error::Unsupported(what) => write!(formatter, "mathtext not supported by zlc-mathtext: {what}"),
            Error::Font(message) => write!(formatter, "mathtext font: {message}"),
        }
    }
}

impl std::error::Error for Error {}

/// The Matplotlib configuration the engine reproduces.
#[derive(Clone, Debug)]
pub struct Config {
    /// Matplotlib's `mpl-data/fonts/ttf` (DejaVu, STIX and BaKoMa files).
    pub math_fonts_dir: PathBuf,
    /// `rcParams['text.hinting_factor']` (8).
    pub hinting_factor: i64,
    /// `rcParams['text.kerning_factor']` (0).
    pub kerning_factor: i32,
    /// `backend_agg.get_hinting_flag()` for raster parses: FORCE_AUTOHINT for
    /// `text.hinting: force_autohint` (Matplotlib's default value, and ZLC's).
    pub raster_load_flags: i32,
    /// Layouts / rasters kept (each least recently used out).
    pub cache_entries: usize,
    /// Loaded glyphs (all faces) beyond which every face is cleared.
    pub glyph_limit: usize,
}

impl Config {
    pub fn new(math_fonts_dir: impl Into<PathBuf>) -> Self {
        Self { math_fonts_dir: math_fonts_dir.into(), hinting_factor: 8, kerning_factor: 0,
            raster_load_flags: load_flags::FORCE_AUTOHINT, cache_entries: 512, glyph_limit: 20_000 }
    }
}

/// One mathtext string to set.
#[derive(Clone, Copy, Debug)]
pub struct Request<'a> {
    pub text: &'a str,
    /// `findfont(prop)`: the non-math font file.
    pub font_file: &'a Path,
    /// `prop.get_size_in_points()`.
    pub size: f64,
    pub dpi: f64,
}

/// A glyph `ship` placed: `Output.glyphs`' (ox, oy, FontInfo), with y down from the
/// box top (oy is the baseline of the glyph).
#[derive(Clone, Debug, PartialEq)]
pub struct PlacedGlyph {
    pub ox: f64,
    pub oy: f64,
    /// The font file (canonical path) and its PostScript name.
    pub font_file: PathBuf,
    pub postscript_name: String,
    /// `FontInfo.fontsize` (points).
    pub fontsize: f64,
    /// `FontInfo.num`: the character code loaded with `load_char`.
    pub num: u32,
    /// The glyph index `num` maps to in that font.
    pub glyph_id: u32,
    pub metrics: FontMetrics,
    /// `FontInfo.offset` (non-zero only for cmex10).
    pub offset: f64,
}

/// `_mathtext.ship(box)` of the parse: the box metrics and what it places.
#[derive(Clone, Debug, PartialEq)]
pub struct Layout {
    pub width: f64,
    pub height: f64,
    pub depth: f64,
    pub glyphs: Vec<PlacedGlyph>,
    /// `Output.rects`: (x1, y1, x2, y2).
    pub rects: Vec<[f64; 4]>,
}

/// `RasterParse`: offsets (always 0), metrics and the coverage image (row-major,
/// `image_width` × `image_height` bytes).
#[derive(Clone, Debug, PartialEq)]
pub struct Raster {
    pub ox: f64,
    pub oy: f64,
    pub width: f64,
    pub height: f64,
    pub depth: f64,
    pub image_width: usize,
    pub image_height: usize,
    pub image: Vec<u8>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct Key { text: String, font: FontId, size: u64, dpi: u64, flags: i32 }

/// A parsed, once-shipped box (the second ship of `to_raster` runs on it).
struct Parsed {
    tree: ListNode,
    layout: Arc<Layout>,
    /// (w, h + d, d) of `to_raster` and the bbox origin it ships to.
    bounds: Bounds,
}

#[derive(Clone, Copy)]
struct Bounds { xmin: f64, ymin: f64, w: f64, h: f64, d: f64 }

/// A small least-recently-used map.
struct Lru<K, V> { entries: HashMap<K, (V, u64)>, clock: u64, capacity: usize }

impl<K: Hash + Eq + Clone, V> Lru<K, V> {
    fn new(capacity: usize) -> Self { Self { entries: HashMap::new(), clock: 0, capacity: capacity.max(1) } }

    fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        self.clock += 1;
        let clock = self.clock;
        self.entries.get_mut(key).map(|(value, stamp)| { *stamp = clock; value })
    }

    fn insert(&mut self, key: K, value: V) {
        if self.entries.len() >= self.capacity && !self.entries.contains_key(&key) {
            if let Some(oldest) = self.entries.iter().min_by_key(|(_, (_, stamp))| *stamp).map(|(key, _)| key.clone()) {
                self.entries.remove(&oldest);
            }
        }
        self.clock += 1;
        self.entries.insert(key, (value, self.clock));
    }

    fn clear(&mut self) { self.entries.clear() }
}

/// The mathtext engine: owns its FT2Font faces (one per file) and its caches.
/// Not shared between threads; give each rendering thread its own.
pub struct MathText {
    config: Config,
    fonts: FontCache,
    parsed: Lru<Key, Parsed>,
    rasters: Lru<(Key, bool), Arc<Raster>>,
}

impl MathText {
    pub fn new(config: Config) -> Self {
        let store = FontStore::new(config.math_fonts_dir.clone(), config.hinting_factor as std::ffi::c_long, config.kerning_factor);
        let entries = config.cache_entries;
        Self { config, fonts: FontCache::new(store), parsed: Lru::new(entries), rasters: Lru::new(entries) }
    }

    pub fn config(&self) -> &Config { &self.config }

    fn key(&mut self, request: &Request<'_>, flags: i32) -> Result<Key, Error> {
        let font = self.fonts.store.open(request.font_file)?;
        Ok(Key { text: request.text.to_string(), font, size: request.size.to_bits(), dpi: request.dpi.to_bits(), flags })
    }

    /// Parses and ships (once) into the cache.
    fn parsed(&mut self, key: &Key) -> Result<&mut Parsed, Error> {
        if self.parsed.get_mut(key).is_none() {
            // Cached parses address the loaded glyphs: they go with them.
            if self.fonts.trim(self.config.glyph_limit).inspect_err(|_| self.parsed.clear())? { self.parsed.clear() }
            let size = f64::from_bits(key.size);
            let dpi = f64::from_bits(key.dpi);
            let mut fontset = Fontset::new(&mut self.fonts, key.font, key.flags);
            let mut tree = parser::Parser::new(&mut fontset).parse(&key.text, size, dpi)?;
            let output = ship(&mut tree, (0.0, 0.0), &mut fontset)?;
            let bounds = bounds(&output, &tree, &self.fonts);
            let layout = Arc::new(public_layout(&tree, &output, &mut self.fonts)?);
            self.parsed.insert(key.clone(), Parsed { tree, layout, bounds });
        }
        Ok(self.parsed.get_mut(key).expect("just parsed"))
    }

    /// The parsed box and `ship(box)`'s glyphs and rectangles (raster load flags).
    pub fn layout(&mut self, request: &Request<'_>) -> Result<Arc<Layout>, Error> {
        let key = self.key(request, self.config.raster_load_flags)?;
        Ok(self.parsed(&key)?.layout.clone())
    }

    /// Like [`layout`](Self::layout) with other load flags (`NO_HINTING` is what
    /// `MathTextParser("path")` uses).
    pub fn layout_with_flags(&mut self, request: &Request<'_>, flags: i32) -> Result<Arc<Layout>, Error> {
        let key = self.key(request, flags)?;
        Ok(self.parsed(&key)?.layout.clone())
    }

    /// `RendererAgg.get_text_width_height_descent(s, prop, ismath=True)`.
    pub fn metrics(&mut self, request: &Request<'_>) -> Result<(f64, f64, f64), Error> {
        let key = self.key(request, self.config.raster_load_flags)?;
        let bounds = self.parsed(&key)?.bounds;
        Ok((bounds.w, bounds.h + bounds.d, bounds.d))
    }

    /// `MathTextParser("agg").parse(s, dpi, prop, antialiased=antialiased)`.
    pub fn raster(&mut self, request: &Request<'_>, antialiased: bool) -> Result<Arc<Raster>, Error> {
        let key = self.key(request, self.config.raster_load_flags)?;
        let raster_key = (key.clone(), antialiased);
        if let Some(raster) = self.rasters.get_mut(&raster_key) { return Ok(raster.clone()) }
        let flags = key.flags;
        self.parsed(&key)?;
        let fonts = &mut self.fonts;
        let parsed = self.parsed.get_mut(&key).expect("parsed");
        let raster = Arc::new(to_raster(parsed, fonts, key.font, flags, antialiased)?);
        self.rasters.insert(raster_key, raster.clone());
        Ok(raster)
    }

    /// Forgets cached layouts and rasters (faces and loaded glyph metrics stay).
    pub fn clear_cache(&mut self) {
        self.parsed.clear();
        self.rasters.clear();
    }

    /// Whether Matplotlib parses `text` (any size/dpi gives the same answer).
    pub fn check(&mut self, text: &str, font_file: &Path) -> Result<(), Error> {
        self.layout_with_flags(&Request { text, font_file, size: 12.0, dpi: 72.0 }, load_flags::NO_HINTING).map(|_| ())
    }
}

/// `to_raster`'s bounding box of the first ship.
fn bounds(output: &Output, tree: &ListNode, fonts: &FontCache) -> Bounds {
    let glyph = |index: usize| fonts.info(output.glyphs[index].info).metrics;
    let fold = |values: &mut dyn Iterator<Item = f64>, pick: fn(f64, f64) -> f64| {
        let mut current: Option<f64> = None;
        for value in values { current = Some(match current { None => value, Some(c) => pick(c, value) }) }
        current.expect("the trailing 0")
    };
    let n = output.glyphs.len();
    let xmin = fold(&mut (0..n).map(|i| output.glyphs[i].ox + glyph(i).xmin)
        .chain(output.rects.iter().map(|r| r[0])).chain(std::iter::once(0.0)), pymin) - 1.0;
    let ymin = fold(&mut (0..n).map(|i| output.glyphs[i].oy - glyph(i).ymax)
        .chain(output.rects.iter().map(|r| r[1])).chain(std::iter::once(0.0)), pymin) - 1.0;
    let xmax = fold(&mut (0..n).map(|i| output.glyphs[i].ox + glyph(i).xmax)
        .chain(output.rects.iter().map(|r| r[2])).chain(std::iter::once(0.0)), pymax) + 1.0;
    let ymax = fold(&mut (0..n).map(|i| output.glyphs[i].oy - glyph(i).ymin)
        .chain(output.rects.iter().map(|r| r[3])).chain(std::iter::once(0.0)), pymax) + 1.0;
    let w = xmax - xmin;
    let h = ymax - ymin - tree.depth;
    let d = ymax - ymin - tree.height;
    Bounds { xmin, ymin, w, h, d }
}

fn public_layout(tree: &ListNode, output: &Output, fonts: &mut FontCache) -> Result<Layout, Error> {
    let mut glyphs = Vec::with_capacity(output.glyphs.len());
    for placed in &output.glyphs {
        let info = fonts.info(placed.info);
        let (font, num, fontsize, metrics, offset) = (info.font, info.num, info.fontsize, info.metrics, info.offset);
        let stored = &fonts.store.fonts[font];
        let glyph_id = stored.font.get_char_index(num as std::ffi::c_ulong, false)?;
        glyphs.push(PlacedGlyph { ox: placed.ox, oy: placed.oy, font_file: stored.path.clone(),
            postscript_name: stored.postscript_name.clone(), fontsize, num, glyph_id, metrics, offset });
    }
    Ok(Layout { width: tree.width, height: tree.height, depth: tree.depth, glyphs, rects: output.rects.clone() })
}

/// Python's `int(x)` of a float.
fn trunc(value: f64) -> i64 { value.trunc() as i64 }

/// `Output.to_raster(antialiased=...)`.
fn to_raster(parsed: &mut Parsed, fonts: &mut FontCache, default_font: FontId, flags: i32, antialiased: bool)
    -> Result<Raster, Error> {
    let Bounds { xmin, ymin, w, h, d } = parsed.bounds;
    let mut image = zlc_text::FT2Image::new(w.ceil() as std::ffi::c_long, (h + pymax(d, 0.0)).ceil() as std::ffi::c_long)?;
    let shifted = {
        let mut fontset = Fontset::new(fonts, default_font, flags);
        ship(&mut parsed.tree, (-xmin, -ymin), &mut fontset)?
    };
    for placed in &shifted.glyphs {
        let iceberg = fonts.info(placed.info).metrics.iceberg;
        fonts.draw(placed.info, &mut image, trunc(placed.ox) as i32, trunc(placed.oy - iceberg) as i32, antialiased, flags)?;
    }
    for &[x1, y1, x2, y2] in &shifted.rects {
        let height = (trunc(y2 - y1) - 1).max(0);
        let y = if height == 0 {
            let center = (y2 + y1) / 2.0;
            trunc(center - (height + 1) as f64 / 2.0)
        } else {
            trunc(y1)
        };
        image.draw_rect_filled(trunc(x1) as std::ffi::c_ulong, y as std::ffi::c_ulong, trunc(x2.ceil()) as std::ffi::c_ulong,
            (y + height) as std::ffi::c_ulong);
    }
    let (image_width, image_height) = (image.width() as usize, image.height() as usize);
    Ok(Raster { ox: 0.0, oy: 0.0, width: w, height: h + d, depth: d, image_width, image_height, image: image.pixels().to_vec() })
}
