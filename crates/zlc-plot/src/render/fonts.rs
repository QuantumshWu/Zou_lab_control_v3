//! Fonts for the renderer: matplotlib's font resolution for the families the
//! style names (`font_manager._find_fonts_by_props`), zlc-figure's [`Fonts`]
//! over zlc-text's FT2Font (what `RendererAgg` measures and rasterizes with),
//! and `TextToPath.get_text_width_height_descent` -- the ruler the tick
//! ladder and the readout pin price text with.
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

use zlc_figure::backend::GrayImage;
use zlc_figure::fonts::{FontId, Fonts};
use zlc_text::FT2Font;

/// The font file matplotlib's font manager finds for one family name (the
/// families zlc's style and matplotlib's defaults name).
pub fn family_file(family: &str) -> Option<&'static str> {
    Some(match family {
        "Helvetica Light" => "helvetica-light-587ebe5a59211.ttf",
        "Arial" => "arial.ttf",
        "DejaVu Sans" => "DejaVuSans.ttf",
        _ => return None,
    })
}

/// `_find_fonts_by_props(prop)` for a family list: the file of every family
/// found, in order (the first is the FT2Font, the rest its fallbacks).
pub fn font_files(families: &[String]) -> Vec<String> {
    families.iter().filter_map(|family| family_file(family)).map(str::to_owned).collect()
}

/// The fonts this crate ships (`fonts/`: matplotlib's DejaVu, STIX and
/// BaKoMa files and the style's face; see `fonts/NOTICE`): beside the running
/// executable, where the desktop build copies them, else the checkout's.
fn shipped_fonts_dir() -> &'static PathBuf {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        std::env::current_exe().ok().and_then(|exe| exe.parent().map(|dir| dir.join("fonts")))
            .filter(|dir| dir.join("DejaVuSans.ttf").is_file())
            .unwrap_or_else(|| zlc_core::checkout::path("crates/zlc-plot/fonts"))
    })
}

/// Where font files are looked up: `ZLC_FONT_DIRS` (`;`-separated, for
/// tests), else the shipped fonts and the Windows font directory (Arial).
pub fn font_dirs() -> &'static [PathBuf] {
    static DIRS: OnceLock<Vec<PathBuf>> = OnceLock::new();
    DIRS.get_or_init(|| {
        if let Ok(list) = std::env::var("ZLC_FONT_DIRS") {
            return list.split(';').filter(|s| !s.is_empty()).map(PathBuf::from).collect();
        }
        let mut dirs = vec![shipped_fonts_dir().clone()];
        if let Ok(windir) = std::env::var("WINDIR") { dirs.push(PathBuf::from(windir).join("Fonts")); }
        dirs.push(PathBuf::from(r"C:\Windows\Fonts"));
        dirs
    })
}

/// The math fonts mathtext and the DejaVu families open: `ZLC_MPL_FONTS`
/// (for tests), else the shipped fonts.
pub fn math_fonts_dir() -> Option<&'static PathBuf> {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    Some(DIR.get_or_init(|| std::env::var_os("ZLC_MPL_FONTS").map(PathBuf::from).unwrap_or_else(|| shipped_fonts_dir().clone())))
}

thread_local! {
    static MATH: std::cell::RefCell<Option<zlc_figure::mathtext::NativeMathText>> = const { std::cell::RefCell::new(None) };
    static CHECK: std::cell::RefCell<Option<zlc_mathtext::MathText>> = const { std::cell::RefCell::new(None) };
}

/// `_drawable_text(text)`: the text, unless Matplotlib's mathtext parser
/// refuses one of its lines -- then it is shown as the characters it is
/// made of (its dollars escaped). The parse uses the style's font.
pub fn drawable_text(text: &str) -> String {
    if !text.contains('$') { return text.to_owned(); }
    let (Some(dir), Some(file)) = (math_fonts_dir(), family_file("Helvetica Light")) else { return text.to_owned() };
    let Some(font) = font_dirs().iter().map(|d| d.join(file)).find(|p| p.is_file()) else { return text.to_owned() };
    let refused = CHECK.with(|cell| {
        let mut slot = cell.borrow_mut();
        let engine = slot.get_or_insert_with(|| zlc_mathtext::MathText::new(zlc_mathtext::Config::new(dir.clone())));
        text.split('\n').any(|line| engine.check(line, &font).is_err_and(|e| e.is_matplotlib_failure()))
    });
    if refused { super::labels::literal_text(text) } else { text.to_owned() }
}

/// Mathtext for the renderer's draws: zlc-figure's [`NativeMathText`]
/// (one per thread; its faces and layouts are cached there). Without
/// matplotlib's math fonts every `$...$` string is refused, as
/// [`NoMathText`] does.
///
/// [`NativeMathText`]: zlc_figure::mathtext::NativeMathText
/// [`NoMathText`]: zlc_figure::fonts::NoMathText
pub struct Math;

impl Math {
    fn with<T>(&self, s: &str, run: impl FnOnce(&mut zlc_figure::mathtext::NativeMathText) -> zlc_figure::Result<T>) -> zlc_figure::Result<T> {
        let Some(dir) = math_fonts_dir() else {
            return Err(zlc_figure::Error::Unsupported(format!("mathtext without matplotlib's math fonts: {s:?}")));
        };
        MATH.with(|cell| {
            let mut slot = cell.borrow_mut();
            let math = slot.get_or_insert_with(|| zlc_figure::mathtext::NativeMathText::new(dir.clone(), Box::new(|name: &str| {
                font_dirs().iter().map(|d| d.join(name)).find(|p| p.is_file())
            })));
            run(math)
        })
    }
}

impl zlc_figure::fonts::MathText for Math {
    fn parse(&mut self, rc: &zlc_figure::ir::Rc, s: &str, dpi: f64, font: &zlc_figure::ir::Font,
             antialiased: Option<bool>) -> zlc_figure::Result<zlc_figure::fonts::MathParse> {
        self.with(s, |math| math.parse(rc, s, dpi, font, antialiased))
    }

    fn vector(&mut self, rc: &zlc_figure::ir::Rc, s: &str, font: &zlc_figure::ir::Font)
              -> zlc_figure::Result<zlc_figure::fonts::VectorParse> {
        self.with(s, |math| math.vector(rc, s, font))
    }
}

/// The bytes of one font file, read once per process.
pub fn font_data(name: &str) -> Result<Arc<Vec<u8>>, String> {
    static CACHE: OnceLock<Mutex<HashMap<String, Arc<Vec<u8>>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);
    if let Some(data) = cache.lock().unwrap().get(name) { return Ok(data.clone()); }
    for dir in font_dirs() {
        let path = dir.join(name);
        if let Ok(bytes) = std::fs::read(&path) {
            let data = Arc::new(bytes);
            cache.lock().unwrap().insert(name.to_owned(), data.clone());
            return Ok(data);
        }
    }
    Err(format!("font file {name} not found in {:?}", font_dirs()))
}

fn new_font(files: &[String], hinting_factor: i64, kerning_factor: i64) -> Result<FT2Font, String> {
    let (first, rest) = files.split_first().ok_or("an empty font file list")?;
    let mut fallbacks = Vec::with_capacity(rest.len());
    for file in rest {
        let data = font_data(file)?;
        let font = FT2Font::new((*data).clone(), hinting_factor as _, Vec::new(), kerning_factor as _)
            .map_err(|e| e.to_string())?;
        fallbacks.push(Arc::new(Mutex::new(font)));
    }
    let data = font_data(first)?;
    FT2Font::new((*data).clone(), hinting_factor as _, fallbacks, kerning_factor as _).map_err(|e| e.to_string())
}

fn codepoints(text: &str) -> Vec<u32> { text.chars().map(|c| c as u32).collect() }

fn ignore_missing(_charcode: std::ffi::c_ulong, _fonts: &[u8]) -> Result<(), zlc_text::Error> { Ok(()) }

/// One FT2Font and the call sequence it was last given: `clear`,
/// `set_size`, `set_text` are applied lazily, so a text laid out before is
/// answered from [`FontSet`]'s layouts without FreeType.
struct Slot {
    font: FT2Font,
    size: Option<(u64, u64)>,
    text: Option<(String, u64, i32)>,
    /// The FT2Font holds exactly `size` and `text`.
    synced: bool,
    /// `size` then `text` were set in that order: the layout is a pure function of them.
    cacheable: bool,
    /// The layout whose bitmap `image` returns, when drawn from the cache.
    drawn: Option<LayoutKey>,
}

type LayoutKey = (FontId, (u64, u64), (String, u64, i32));

/// What FreeType answered for one layout, and its bitmap once drawn.
#[derive(Default)]
struct Layout {
    width_height: Option<(i64, i64)>,
    descent: Option<i64>,
    offset: Option<(i64, i64)>,
    bitmap: Option<(bool, zlc_figure::backend::Bitmap)>,
}

/// [`Fonts`] over zlc-text: one FT2Font per file list (matplotlib's
/// `get_font` cache), each with its own fallback fonts; a text laid out at a
/// size before is measured and rasterized from its recorded layout.
#[derive(Default)]
pub struct FontSet {
    slots: Vec<Slot>,
    by_files: HashMap<(Vec<String>, i64, i64), FontId>,
    layouts: HashMap<LayoutKey, Layout>,
}

impl FontSet {
    pub fn new() -> FontSet { FontSet::default() }

    fn slot(&mut self, font: FontId) -> zlc_figure::Result<&mut Slot> {
        self.slots.get_mut(font).ok_or_else(|| zlc_figure::Error::Backend(format!("no font {font}")))
    }

    /// Give the FT2Font the state the calls so far describe.
    fn sync(&mut self, font: FontId) -> zlc_figure::Result<&mut FT2Font> {
        let slot = self.slot(font)?;
        if !slot.synced {
            slot.font.clear().map_err(backend_error)?;
            if let Some((ptsize, dpi)) = slot.size { slot.font.set_size(f64::from_bits(ptsize), f64::from_bits(dpi)).map_err(backend_error)?; }
            if let Some((text, angle, flags)) = &slot.text {
                let points = codepoints(text);
                let (mut xys, mut warn) = (Vec::new(), ignore_missing);
                slot.font.set_text::<zlc_text::Error>(&points, f64::from_bits(*angle), *flags, &mut xys, &mut warn).map_err(backend_error)?;
            }
            slot.synced = true;
        }
        Ok(&mut slot.font)
    }

    fn layout_key(&mut self, font: FontId) -> zlc_figure::Result<Option<LayoutKey>> {
        let slot = self.slot(font)?;
        Ok(match (slot.cacheable, slot.size, &slot.text) {
            (true, Some(size), Some(text)) => Some((font, size, text.clone())),
            _ => None,
        })
    }

    /// One answer about the current layout: recorded, or asked of FreeType and recorded.
    fn answer<T: Copy>(&mut self, font: FontId, field: fn(&mut Layout) -> &mut Option<T>,
                       ask: impl FnOnce(&mut FT2Font) -> zlc_figure::Result<T>) -> zlc_figure::Result<T> {
        let Some(key) = self.layout_key(font)? else { return ask(self.sync(font)?) };
        if let Some(value) = self.layouts.get_mut(&key).and_then(|layout| *field(layout)) { return Ok(value); }
        let value = ask(self.sync(font)?)?;
        if self.layouts.len() >= 8192 { self.layouts.clear(); }
        *field(self.layouts.entry(key).or_default()) = Some(value);
        Ok(value)
    }
}

fn backend_error(error: impl ToString) -> zlc_figure::Error { zlc_figure::Error::Backend(error.to_string()) }

impl Fonts for FontSet {
    fn get_font(&mut self, files: &[String], hinting_factor: i64, kerning_factor: i64) -> zlc_figure::Result<FontId> {
        let key = (files.to_vec(), hinting_factor, kerning_factor);
        if let Some(&id) = self.by_files.get(&key) { return Ok(id); }
        let font = new_font(files, hinting_factor, kerning_factor).map_err(backend_error)?;
        self.slots.push(Slot { font, size: None, text: None, synced: true, cacheable: false, drawn: None });
        let id = self.slots.len() - 1;
        self.by_files.insert(key, id);
        Ok(id)
    }

    fn clear(&mut self, font: FontId) -> zlc_figure::Result<()> {
        let slot = self.slot(font)?;
        slot.text = None;
        slot.synced = false;
        slot.cacheable = false;
        slot.drawn = None;
        Ok(())
    }

    fn set_size(&mut self, font: FontId, ptsize: f64, dpi: f64) -> zlc_figure::Result<()> {
        let slot = self.slot(font)?;
        if slot.text.is_some() {
            // A size after the text: FreeType keeps the old layout; answer from it as it does.
            self.sync(font)?.set_size(ptsize, dpi).map_err(backend_error)?;
            let slot = self.slot(font)?;
            slot.size = Some((ptsize.to_bits(), dpi.to_bits()));
            slot.cacheable = false;
            slot.drawn = None;
            return Ok(());
        }
        slot.size = Some((ptsize.to_bits(), dpi.to_bits()));
        slot.synced = false;
        Ok(())
    }

    fn set_text(&mut self, font: FontId, text: &str, angle: f64, flags: i32) -> zlc_figure::Result<()> {
        let slot = self.slot(font)?;
        slot.text = Some((text.to_owned(), angle.to_bits(), flags));
        slot.cacheable = slot.size.is_some();
        slot.synced = false;
        slot.drawn = None;
        Ok(())
    }

    fn get_width_height(&mut self, font: FontId) -> zlc_figure::Result<(i64, i64)> {
        self.answer(font, |l| &mut l.width_height, |f| { let (w, h) = f.get_width_height(); Ok((w as i64, h as i64)) })
    }

    fn get_descent(&mut self, font: FontId) -> zlc_figure::Result<i64> {
        self.answer(font, |l| &mut l.descent, |f| Ok(f.get_descent() as i64))
    }

    fn get_bitmap_offset(&mut self, font: FontId) -> zlc_figure::Result<(i64, i64)> {
        self.answer(font, |l| &mut l.offset, |f| { let (x, y) = f.get_bitmap_offset(); Ok((x as i64, y as i64)) })
    }

    fn draw_glyphs_to_bitmap(&mut self, font: FontId, antialiased: bool) -> zlc_figure::Result<()> {
        let Some(key) = self.layout_key(font)? else {
            return self.sync(font)?.draw_glyphs_to_bitmap(antialiased).map_err(backend_error);
        };
        let recorded = self.layouts.get(&key).and_then(|l| l.bitmap.as_ref()).is_some_and(|(aa, _)| *aa == antialiased);
        if !recorded {
            let real = self.sync(font)?;
            real.draw_glyphs_to_bitmap(antialiased).map_err(backend_error)?;
            let image = real.image();
            let bitmap = zlc_figure::backend::Bitmap { rows: image.rows(), cols: image.cols(), data: image.pixels().to_vec(),
                                                       id: image.oracle_id().map(str::to_owned) };
            if self.layouts.len() >= 8192 { self.layouts.clear(); }
            self.layouts.entry(key.clone()).or_default().bitmap = Some((antialiased, bitmap));
        }
        self.slot(font)?.drawn = Some(key);
        Ok(())
    }

    fn image(&mut self, font: FontId) -> zlc_figure::Result<&dyn GrayImage> {
        let drawn = self.slot(font)?.drawn.clone().filter(|key| self.layouts.get(key).is_some_and(|l| l.bitmap.is_some()));
        if let Some(key) = drawn { return Ok(&self.layouts[&key].bitmap.as_ref().unwrap().1); }
        Ok(self.sync(font)?.image())
    }
}

/// `TextToPath().get_text_width_height_descent(s, FontProperties(family, size), False)`:
/// FT2Font at 100 pt / 72 dpi, unhinted, scaled to the size; in points.
pub struct TextToPath {
    families: Vec<String>,
    fonts: Mutex<HashMap<Vec<String>, FT2Font>>,
    memo: Mutex<HashMap<(String, u64), (f64, f64, f64)>>,
}

const FONT_SCALE: f64 = 100.0;
const NO_HINTING: i32 = 2;

impl TextToPath {
    pub fn new(families: Vec<String>) -> TextToPath {
        TextToPath { families, fonts: Mutex::new(HashMap::new()), memo: Mutex::new(HashMap::new()) }
    }

    /// (width, height, descent) in points of one line at `size_pt`.
    pub fn width_height_descent(&self, text: &str, size_pt: f64) -> (f64, f64, f64) {
        let key = (text.to_owned(), size_pt.to_bits());
        if let Some(&answer) = self.memo.lock().unwrap().get(&key) { return answer; }
        let files = font_files(&self.families);
        let mut fonts = self.fonts.lock().unwrap();
        if !fonts.contains_key(&files) {
            // get_font(...) with rcParams' hinting factor 8 and kerning factor 0.
            let font = new_font(&files, 8, 0).unwrap_or_else(|e| panic!("TextToPath font: {e}"));
            fonts.insert(files.clone(), font);
        }
        let font = fonts.get_mut(&files).unwrap();
        font.clear().expect("FT2Font.clear");
        font.set_size(FONT_SCALE, 72.0).expect("FT2Font.set_size");
        let mut xys = Vec::new();
        let mut warn = ignore_missing;
        font.set_text::<zlc_text::Error>(&codepoints(text), 0.0, NO_HINTING, &mut xys, &mut warn).expect("FT2Font.set_text");
        let (w, h) = font.get_width_height();
        let d = font.get_descent();
        let scale = size_pt / FONT_SCALE;
        let answer = (w as f64 / 64.0 * scale, h as f64 / 64.0 * scale, d as f64 / 64.0 * scale);
        drop(fonts);
        self.memo.lock().unwrap().insert(key, answer);
        answer
    }

    /// `layout._text_width_pt`.
    pub fn width(&self, text: &str, size_pt: f64) -> f64 { self.width_height_descent(text, size_pt).0 }
}

impl crate::ticks::Measure for TextToPath {
    fn size(&self, text: &str, size_pt: f64) -> (f64, f64) {
        let (w, h, _) = self.width_height_descent(text, size_pt);
        (w, h)
    }

    fn line(&self, size_pt: f64) -> f64 {
        let (_, h, d) = self.width_height_descent("lp", size_pt);
        h + d
    }
}
