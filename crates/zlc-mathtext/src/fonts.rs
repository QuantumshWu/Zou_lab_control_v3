//! Matplotlib's mathtext font classes for the `dejavusans` fontset: `DejaVuSansFonts`
//! (a `UnicodeFonts`) with its `StixSansFonts` fallback, the `BakomaFonts` it borrows
//! `\prime` from and that class's own `StixFonts` fallback. `_get_glyph`, `_get_info`,
//! `get_kern`, `get_xheight` and the sized alternatives follow `_mathtext.py` line by
//! line, including which call sets a font's size: kerning reads the size a face was
//! last set to, so the order of `set_size` calls is part of the result.

use std::collections::HashMap;
use std::ffi::c_long;
use std::path::{Path, PathBuf};

use zlc_text::{FT2Font, SfntTable};

use crate::{Error, data};

/// A mathtext font name: the grammar's (`rm`, `it`, ...), the size-alternative
/// indices `0..=5`, the STIX private-use variants and the BaKoMa basenames.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum FontName {
    Default, Regular, Rm, It, Bf, Bfit, Sf, Tt, Cal, Bb, Frak, Scr, Ex,
    NonUniRm, NonUniIt, NonUniBf, NonUniBfit,
    Size(u8),
    Cm(Cm),
}

/// The BaKoMa Computer Modern files by basename.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Cm { Sy, R, Tt, Mi, B, Ss, Ex }

impl FontName {
    /// The grammar's font names (`Parser._fontnames`).
    pub(crate) fn from_grammar(name: &str) -> Option<Self> {
        Some(match name {
            "default" => Self::Default, "regular" => Self::Regular, "rm" => Self::Rm, "it" => Self::It,
            "bf" => Self::Bf, "bfit" => Self::Bfit, "sf" => Self::Sf, "tt" => Self::Tt, "cal" => Self::Cal,
            "bb" => Self::Bb, "frak" => Self::Frak, "scr" => Self::Scr,
            _ => return None,
        })
    }

    /// Target names of `stix_virtual_fonts` ranges.
    fn from_virtual(name: &str) -> Self {
        match name {
            "rm" => Self::Rm, "it" => Self::It, "bf" => Self::Bf, "bfit" => Self::Bfit,
            other => unreachable!("stix_virtual_fonts target {other}"),
        }
    }

    /// The key of `stix_virtual_fonts` this name looks up (`dict.get(fontname)`).
    fn virtual_key(self) -> Option<&'static str> {
        Some(match self {
            Self::Bb => "bb", Self::Cal => "cal", Self::Frak => "frak", Self::Scr => "scr",
            Self::Sf => "sf", Self::Tt => "tt",
            _ => return None,
        })
    }

    fn class_key(self) -> &'static str {
        match self {
            Self::Rm => "rm", Self::It => "it", Self::Bf => "bf", Self::Bfit => "bfit",
            Self::Default => "default", Self::Regular => "regular", Self::Sf => "sf", Self::Tt => "tt",
            Self::Cal => "cal", Self::Bb => "bb", Self::Frak => "frak", Self::Scr => "scr", Self::Ex => "ex",
            _ => "",
        }
    }

    /// `ParserState.font`'s setter also sets the font class for these.
    pub(crate) fn sets_class(self) -> bool { matches!(self, Self::Rm | Self::It | Self::Bf | Self::Bfit) }
}

/// Matplotlib's bundled math font files (`mpl-data/fonts/ttf`), as `findfont`
/// resolves the names the font classes ask for.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MathFontFile {
    DejaVuSans, DejaVuSansOblique, DejaVuSansBold, DejaVuSansBoldOblique, DejaVuSansMono, DejaVuSansDisplay,
    StixGeneral, StixGeneralItalic, StixGeneralBold, StixGeneralBoldItalic,
    StixNonUni, StixNonUniItalic, StixNonUniBold,
    StixSize1, StixSize2, StixSize3, StixSize4, StixSize5,
    Cmsy10, Cmr10, Cmtt10, Cmmi10, Cmb10, Cmss10, Cmex10,
}

impl MathFontFile {
    pub fn file_name(self) -> &'static str {
        match self {
            Self::DejaVuSans => "DejaVuSans.ttf",
            Self::DejaVuSansOblique => "DejaVuSans-Oblique.ttf",
            Self::DejaVuSansBold => "DejaVuSans-Bold.ttf",
            Self::DejaVuSansBoldOblique => "DejaVuSans-BoldOblique.ttf",
            Self::DejaVuSansMono => "DejaVuSansMono.ttf",
            Self::DejaVuSansDisplay => "DejaVuSansDisplay.ttf",
            Self::StixGeneral => "STIXGeneral.ttf",
            Self::StixGeneralItalic => "STIXGeneralItalic.ttf",
            Self::StixGeneralBold => "STIXGeneralBol.ttf",
            Self::StixGeneralBoldItalic => "STIXGeneralBolIta.ttf",
            Self::StixNonUni => "STIXNonUni.ttf",
            Self::StixNonUniItalic => "STIXNonUniIta.ttf",
            Self::StixNonUniBold => "STIXNonUniBol.ttf",
            Self::StixSize1 => "STIXSizOneSymReg.ttf",
            Self::StixSize2 => "STIXSizTwoSymReg.ttf",
            Self::StixSize3 => "STIXSizThreeSymReg.ttf",
            Self::StixSize4 => "STIXSizFourSymReg.ttf",
            Self::StixSize5 => "STIXSizFiveSymReg.ttf",
            Self::Cmsy10 => "cmsy10.ttf",
            Self::Cmr10 => "cmr10.ttf",
            Self::Cmtt10 => "cmtt10.ttf",
            Self::Cmmi10 => "cmmi10.ttf",
            Self::Cmb10 => "cmb10.ttf",
            Self::Cmss10 => "cmss10.ttf",
            Self::Cmex10 => "cmex10.ttf",
        }
    }

    fn stix_size(size: u8) -> Self {
        [Self::StixGeneral, Self::StixSize1, Self::StixSize2, Self::StixSize3, Self::StixSize4, Self::StixSize5][size as usize]
    }

    fn cm(face: Cm) -> Self {
        match face {
            Cm::Sy => Self::Cmsy10, Cm::R => Self::Cmr10, Cm::Tt => Self::Cmtt10, Cm::Mi => Self::Cmmi10,
            Cm::B => Self::Cmb10, Cm::Ss => Self::Cmss10, Cm::Ex => Self::Cmex10,
        }
    }
}

/// Which of Matplotlib's font classes a lookup runs in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Kind {
    /// `DejaVuSansFonts`, the parser's fontset.
    DejaVuSans,
    /// `StixSansFonts`, DejaVu's fallback.
    StixSans,
    /// `StixFonts`, BaKoMa's fallback.
    Stix,
    /// `BakomaFonts`, which DejaVu asks for `\prime`.
    Bakoma,
}

impl Kind {
    fn is_stix(self) -> bool { matches!(self, Self::StixSans | Self::Stix) }
}

/// A font in the store; ids are stable for the store's life.
pub(crate) type FontId = usize;

pub(crate) struct StoredFont {
    pub(crate) font: FT2Font,
    pub(crate) path: PathBuf,
    pub(crate) family_name: String,
    pub(crate) postscript_name: String,
    pclt_x_height: Option<u16>,
    /// Glyphs loaded into `font` (each a slot `draw_glyph_to_bitmap` addresses).
    pub(crate) slots: usize,
}

/// One FT2Font per file, as `font_manager.get_font` shares them per thread.
pub(crate) struct FontStore {
    math_dir: PathBuf,
    hinting_factor: c_long,
    kerning_factor: i32,
    pub(crate) fonts: Vec<StoredFont>,
    by_path: HashMap<PathBuf, FontId>,
    /// Paths as callers spelled them (canonicalizing touches the file system).
    by_request: HashMap<PathBuf, FontId>,
    math: HashMap<MathFontFile, FontId>,
}

impl FontStore {
    pub(crate) fn new(math_dir: PathBuf, hinting_factor: c_long, kerning_factor: i32) -> Self {
        Self { math_dir, hinting_factor, kerning_factor, fonts: Vec::new(), by_path: HashMap::new(), by_request: HashMap::new(), math: HashMap::new() }
    }

    pub(crate) fn open(&mut self, path: &Path) -> Result<FontId, Error> {
        if let Some(&id) = self.by_request.get(path) { return Ok(id) }
        let key = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        if let Some(&id) = self.by_path.get(&key) {
            self.by_request.insert(path.to_path_buf(), id);
            return Ok(id);
        }
        let data = std::fs::read(path).map_err(|error| Error::Font(format!("{}: {error}", path.display())))?;
        let font = FT2Font::new(data, self.hinting_factor, Vec::new(), self.kerning_factor)
            .map_err(|error| Error::Font(format!("{}: {error}", path.display())))?;
        let family_name = font.family_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_default();
        let postscript_name = font.postscript_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_default();
        let pclt_x_height = match font.get_sfnt_table("pclt") { Some(SfntTable::Pclt(table)) => Some(table.xHeight), _ => None };
        let id = self.fonts.len();
        self.fonts.push(StoredFont { font, path: key.clone(), family_name, postscript_name, pclt_x_height, slots: 0 });
        self.by_path.insert(key, id);
        self.by_request.insert(path.to_path_buf(), id);
        Ok(id)
    }

    pub(crate) fn math(&mut self, file: MathFontFile) -> Result<FontId, Error> {
        if let Some(&id) = self.math.get(&file) { return Ok(id) }
        let path = self.math_dir.join(file.file_name());
        let id = self.open(&path)?;
        self.math.insert(file, id);
        Ok(id)
    }

    /// Drops every loaded glyph of every font (their slots become invalid).
    pub(crate) fn clear_glyphs(&mut self) -> Result<(), Error> {
        for stored in &mut self.fonts {
            stored.font.clear().map_err(text_error)?;
            stored.slots = 0;
        }
        Ok(())
    }
}

pub(crate) fn text_error(error: zlc_text::Error) -> Error { Error::Font(error.message) }

/// `FontMetrics`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FontMetrics {
    pub advance: f64,
    pub height: f64,
    pub width: f64,
    pub xmin: f64,
    pub xmax: f64,
    pub ymin: f64,
    pub ymax: f64,
    pub iceberg: f64,
    pub slanted: bool,
}

/// `FontInfo`, plus the glyph slots it has drawn from.
pub(crate) struct Info {
    pub(crate) font: FontId,
    pub(crate) fontsize: f64,
    pub(crate) dpi: f64,
    pub(crate) num: u32,
    pub(crate) metrics: FontMetrics,
    pub(crate) offset: f64,
    /// The outline slot `load_char` filled and nothing has drawn yet.
    fresh: Option<usize>,
    /// Slots already converted to a bitmap by an antialiased / monochrome draw.
    rendered: [Option<usize>; 2],
}

pub(crate) type InfoId = usize;

/// `_get_info`'s cache key: (fontname, font_class, sym, fontsize, dpi).
#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct InfoKey { name: FontName, class: FontName, sym: String, fontsize: u64, dpi: u64 }

/// What a glyph's metrics depend on (one load per key and load flags).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct LoadKey { font: FontId, num: u32, fontsize: u64, dpi: u64, flags: i32 }

/// Fonts shared by every parse of an engine: the store and the loaded glyph metrics.
pub(crate) struct FontCache {
    pub(crate) store: FontStore,
    pub(crate) infos: Vec<Info>,
    loads: HashMap<LoadKey, InfoId>,
    /// Sized alternatives per (fontname, symbol) (`functools.cache` in StixFonts).
    alternatives: HashMap<(FontName, String), Vec<(FontName, String)>>,
}

impl FontCache {
    pub(crate) fn new(store: FontStore) -> Self {
        Self { store, infos: Vec::new(), loads: HashMap::new(), alternatives: HashMap::new() }
    }

    /// Forgets every glyph (slots and metrics) once the faces hold too many.
    pub(crate) fn trim(&mut self, limit: usize) -> Result<bool, Error> {
        if self.store.fonts.iter().map(|font| font.slots).sum::<usize>() <= limit { return Ok(false) }
        self.infos.clear();
        self.loads.clear();
        self.store.clear_glyphs()?;
        Ok(true)
    }

    pub(crate) fn info(&self, id: InfoId) -> &Info { &self.infos[id] }

    /// Draws `info`'s glyph into `image` with its top-left at (x, y) the way
    /// `FT2Font.draw_glyph_to_bitmap` draws the glyph `load_char` returned.
    pub(crate) fn draw(&mut self, id: InfoId, image: &mut zlc_text::FT2Image, x: i32, y: i32, antialiased: bool,
        flags: i32) -> Result<(), Error> {
        let mode = antialiased as usize;
        let slot = match self.infos[id].rendered[mode] {
            Some(slot) => slot,
            None => {
                let slot = match self.infos[id].fresh.take() {
                    Some(slot) => slot,
                    None => {
                        // The other mode converted the outline: load it again at its size.
                        let info = &self.infos[id];
                        let (font, num, fontsize, dpi) = (info.font, info.num, info.fontsize, info.dpi);
                        let stored = &mut self.store.fonts[font];
                        stored.font.set_size(fontsize, dpi).map_err(text_error)?;
                        let glyph = stored.font.load_char(num as c_long, flags, true, &mut ignore_missing)?;
                        stored.slots += 1;
                        glyph.glyph_index
                    }
                };
                self.infos[id].rendered[mode] = Some(slot);
                slot
            }
        };
        let font = self.infos[id].font;
        self.store.fonts[font].font.draw_glyph_to_bitmap(image, x, y, slot, antialiased).map_err(text_error)
    }
}

impl From<zlc_text::Error> for Error {
    fn from(error: zlc_text::Error) -> Self { text_error(error) }
}

fn ignore_missing(_charcode: std::ffi::c_ulong, _families: &[u8]) -> Result<(), Error> { Ok(()) }

/// `get_unicode_index`.
pub(crate) fn unicode_index(sym: &str) -> Option<u32> {
    let mut chars = sym.chars();
    if let (Some(single), None) = (chars.next(), chars.next()) { return Some(single as u32) }
    tex2uni(sym.trim_matches('\\'))
}

/// `tex2uni[name]`.
pub(crate) fn tex2uni(name: &str) -> Option<u32> {
    static TABLE: std::sync::OnceLock<HashMap<&'static str, u32>> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| data::TEX2UNI.iter().copied().collect()).get(name).copied()
}

fn bakoma(sym: &str) -> Option<(Cm, u32)> {
    data::LATEX_TO_BAKOMA.iter().find(|(key, _, _)| *key == sym).map(|&(_, base, num)| {
        let face = match base {
            "cmsy10" => Cm::Sy, "cmr10" => Cm::R, "cmtt10" => Cm::Tt, "cmmi10" => Cm::Mi,
            "cmb10" => Cm::B, "cmss10" => Cm::Ss, "cmex10" => Cm::Ex,
            other => unreachable!("latex_to_bakoma basename {other}"),
        };
        (face, num)
    })
}

fn it_letter(code: u32) -> bool { data::IT_LETTER[(code >> 3) as usize] & (1 << (code & 7)) != 0 }

/// The per-parse view of the fonts: Matplotlib builds a fresh fontset (with a fresh
/// `_get_info` cache) for each parse.
pub(crate) struct Fontset<'a> {
    pub(crate) cache: &'a mut FontCache,
    pub(crate) default_font: FontId,
    pub(crate) flags: i32,
    /// `_get_info` keys this parse has computed (each computes with `set_size`).
    seen: HashMap<InfoKey, InfoId>,
}

const SLANTED_SYMBOLS: [&str; 2] = ["\\int", "\\oint"];

impl<'a> Fontset<'a> {
    pub(crate) fn new(cache: &'a mut FontCache, default_font: FontId, flags: i32) -> Self {
        Self { cache, default_font, flags, seen: HashMap::new() }
    }

    /// `_get_font(name)` of the given font class (None where Matplotlib finds nothing).
    pub(crate) fn get_font(&mut self, kind: Kind, name: FontName) -> Result<Option<FontId>, Error> {
        use FontName::*;
        let file = match (kind, name) {
            (_, Default | Regular) => return Ok(Some(self.default_font)),
            (Kind::DejaVuSans, Rm | Sf | Size(0)) => MathFontFile::DejaVuSans,
            (Kind::DejaVuSans, It) => MathFontFile::DejaVuSansOblique,
            (Kind::DejaVuSans, Bf) => MathFontFile::DejaVuSansBold,
            (Kind::DejaVuSans, Bfit) => MathFontFile::DejaVuSansBoldOblique,
            (Kind::DejaVuSans, Tt) => MathFontFile::DejaVuSansMono,
            (Kind::DejaVuSans, Ex) => MathFontFile::DejaVuSansDisplay,
            (Kind::DejaVuSans | Kind::StixSans | Kind::Stix, Size(size @ 1..=5)) => MathFontFile::stix_size(size),
            (Kind::StixSans | Kind::Stix, Rm | Size(0)) => MathFontFile::StixGeneral,
            (Kind::StixSans | Kind::Stix, It) => MathFontFile::StixGeneralItalic,
            (Kind::StixSans | Kind::Stix, Bf) => MathFontFile::StixGeneralBold,
            (Kind::StixSans | Kind::Stix, Bfit) => MathFontFile::StixGeneralBoldItalic,
            (Kind::StixSans | Kind::Stix, NonUniRm) => MathFontFile::StixNonUni,
            (Kind::StixSans | Kind::Stix, NonUniIt) => MathFontFile::StixNonUniItalic,
            (Kind::StixSans | Kind::Stix, NonUniBf) => MathFontFile::StixNonUniBold,
            (Kind::Bakoma, Cal) => MathFontFile::Cmsy10,
            (Kind::Bakoma, Rm) => MathFontFile::Cmr10,
            (Kind::Bakoma, Tt) => MathFontFile::Cmtt10,
            (Kind::Bakoma, It) => MathFontFile::Cmmi10,
            (Kind::Bakoma, Bf) => MathFontFile::Cmb10,
            (Kind::Bakoma, Sf) => MathFontFile::Cmss10,
            (Kind::Bakoma, Ex) => MathFontFile::Cmex10,
            (Kind::Bakoma, Cm(face)) => MathFontFile::cm(face),
            _ => return Ok(None),
        };
        self.cache.store.math(file).map(Some)
    }

    fn char_index(&mut self, font: FontId, code: u32) -> Result<u32, Error> {
        Ok(self.cache.store.fonts[font].font.get_char_index(code as std::ffi::c_ulong, true)?)
    }

    /// `_get_glyph` of the given font class: (font, character code, slanted).
    pub(crate) fn get_glyph(&mut self, kind: Kind, name: FontName, class: FontName, sym: &str)
        -> Result<(FontId, u32, bool), Error> {
        match kind {
            Kind::DejaVuSans => {
                // DejaVuFonts._get_glyph
                if sym == "\\prime" { return self.get_glyph(Kind::Bakoma, name, class, sym) }
                let code = unicode_index(sym).ok_or_else(|| Error::UnknownSymbol(sym.to_string()))?;
                if let Some(ex) = self.get_font(kind, FontName::Ex)? && self.char_index(ex, code)? != 0 {
                    return self.unicode_glyph(kind, FontName::Ex, class, sym);
                }
                self.unicode_glyph(kind, name, class, sym)
            }
            Kind::StixSans | Kind::Stix => self.unicode_glyph(kind, name, class, sym),
            Kind::Bakoma => {
                let in_fontmap = matches!(name, FontName::Cal | FontName::Rm | FontName::Tt | FontName::It
                    | FontName::Bf | FontName::Sf | FontName::Ex | FontName::Cm(_));
                let mut found = None;
                if let (true, Some((face, num))) = (in_fontmap, bakoma(sym)) {
                    let slanted = face == Cm::Mi || SLANTED_SYMBOLS.contains(&sym);
                    found = self.get_font(kind, FontName::Cm(face))?.map(|font| (font, num, slanted));
                } else {
                    let mut chars = sym.chars();
                    if let (Some(single), None) = (chars.next(), chars.next()) {
                        let slanted = name == FontName::It;
                        found = self.get_font(kind, name)?.map(|font| (font, single as u32, slanted));
                    }
                }
                if let Some((font, num, slanted)) = found && self.char_index(font, num)? != 0 {
                    return Ok((font, num, slanted));
                }
                self.get_glyph(Kind::Stix, name, class, sym)
            }
        }
    }

    /// `UnicodeFonts._get_glyph` for `kind` (DejaVu or a STIX class).
    fn unicode_glyph(&mut self, kind: Kind, name: FontName, class: FontName, sym: &str)
        -> Result<(FontId, u32, bool), Error> {
        let (mut found_symbol, code) = match unicode_index(sym) {
            Some(code) => (true, code),
            None => (false, '?' as u32),
        };
        let (fontname, mut code) = if kind.is_stix() { self.map_virtual_font(kind, name, class, code) } else { (name, code) };
        let mut new_fontname = fontname;
        let mut slanted = false;
        let mut font = None;
        if found_symbol {
            if fontname == FontName::It && code < 0x10000 && !it_letter(code) { new_fontname = FontName::Rm }
            slanted = new_fontname == FontName::It || SLANTED_SYMBOLS.contains(&sym);
            found_symbol = false;
            font = self.get_font(kind, new_fontname)?;
            if let Some(mut id) = font {
                let substitute = match code { 0x00D7 => Some(0x00A3), 0x2212 => Some(0x00A1), _ => None };
                if let Some(substitute) = substitute && self.cache.store.fonts[id].family_name == "cmr10" {
                    id = self.cache.store.math(MathFontFile::Cmsy10)?;
                    font = Some(id);
                    code = substitute;
                }
                if self.char_index(id, code)? != 0 { found_symbol = true }
            }
        }
        if !found_symbol {
            let fallback = match kind {
                Kind::DejaVuSans => Some(Kind::StixSans),
                _ => None,
            };
            if let Some(fallback) = fallback {
                let fontname = if matches!(fontname, FontName::It | FontName::Regular) { FontName::Rm } else { fontname };
                return self.get_glyph(fallback, fontname, class, sym);
            }
            if matches!(fontname, FontName::It | FontName::Regular) && kind.is_stix() {
                return self.get_glyph(kind, FontName::Rm, class, sym);
            }
            font = self.get_font(kind, FontName::Rm)?;
            code = 0xA4;
            slanted = false;
        }
        let font = font.ok_or_else(|| Error::Unsupported(format!("no font for {sym:?} in {fontname:?}")))?;
        Ok((font, code, slanted))
    }

    /// `StixFonts._map_virtual_font`.
    fn map_virtual_font(&self, kind: Kind, name: FontName, class: FontName, code: u32) -> (FontName, u32) {
        let mut fontname = name;
        let mut code = code;
        let mut key = name.virtual_key();
        let doing_sans_conversion = kind == Kind::StixSans && key.is_none()
            && !matches!(name, FontName::Regular | FontName::Default);
        if doing_sans_conversion { key = Some("sf") }
        let mapping = key.and_then(|key| {
            let entries: Vec<_> = data::STIX_VIRTUAL_FONTS.iter().filter(|(font, _, _)| *font == key).collect();
            if entries.len() == 1 && entries[0].1.is_empty() { return Some(entries[0].2) }
            entries.iter().find(|(_, cls, _)| *cls == class.class_key())
                .or_else(|| entries.iter().find(|(_, cls, _)| *cls == "rm"))
                .map(|(_, _, ranges)| *ranges)
        });
        if let Some(mapping) = mapping {
            let (mut lo, mut hi) = (0, mapping.len());
            let mut range = mapping[0];
            while lo < hi {
                let mid = (lo + hi) / 2;
                range = mapping[mid];
                if code < range.0 { hi = mid } else if code <= range.1 { break } else { lo = mid + 1 }
            }
            if range.0 <= code && code <= range.1 {
                code = code - range.0 + range.3;
                fontname = FontName::from_virtual(range.2);
            } else if !doing_sans_conversion {
                code = 0x1;
                fontname = FontName::It; // mathtext.default
            }
        }
        if matches!(fontname, FontName::Rm | FontName::It)
            && let Some(&(_, fixed)) = data::STIX_GLYPH_FIXES.iter().find(|(from, _)| *from == code) {
            code = fixed;
        }
        if (0xe000..=0xf8ff).contains(&code) {
            fontname = match fontname {
                FontName::It => FontName::NonUniIt, FontName::Rm => FontName::NonUniRm,
                FontName::Bf => FontName::NonUniBf, FontName::Bfit => FontName::NonUniBfit,
                other => other,
            };
        }
        (fontname, code)
    }

    /// `_get_info` (through `get_metrics`): the first call for a key in this parse
    /// sets the font's size, as Matplotlib's per-fontset cache does.
    pub(crate) fn get_info(&mut self, name: FontName, class: FontName, sym: &str, fontsize: f64, dpi: f64)
        -> Result<InfoId, Error> {
        let key = InfoKey { name, class, sym: sym.to_string(), fontsize: fontsize.to_bits(), dpi: dpi.to_bits() };
        if let Some(&id) = self.seen.get(&key) { return Ok(id) }
        let (font, num, slanted) = self.get_glyph(Kind::DejaVuSans, name, class, sym)?;
        let stored = &mut self.cache.store.fonts[font];
        stored.font.set_size(fontsize, dpi)?;
        let load = LoadKey { font, num, fontsize: fontsize.to_bits(), dpi: dpi.to_bits(), flags: self.flags };
        let id = match self.cache.loads.get(&load) {
            Some(&id) if self.cache.infos[id].metrics.slanted == slanted => id,
            _ => {
                let glyph = stored.font.load_char(num as c_long, self.flags, true, &mut ignore_missing)?;
                stored.slots += 1;
                let offset = if stored.postscript_name == "Cmex10" {
                    (glyph.height as f64 / 64.0 / 2.0) + (fontsize / 3.0 * dpi / 72.0)
                } else { 0.0 };
                let [xmin, ymin, xmax, ymax] = glyph.bbox.map(|value| value as f64 / 64.0);
                let metrics = FontMetrics {
                    advance: glyph.linear_hori_advance as f64 / 65536.0,
                    height: glyph.height as f64 / 64.0,
                    width: glyph.width as f64 / 64.0,
                    xmin, xmax,
                    ymin: ymin + offset,
                    ymax: ymax + offset,
                    iceberg: glyph.hori_bearing_y as f64 / 64.0 + offset,
                    slanted,
                };
                let id = self.cache.infos.len();
                self.cache.infos.push(Info { font, fontsize, dpi, num, metrics, offset,
                    fresh: Some(glyph.glyph_index), rendered: [None; 2] });
                self.cache.loads.insert(load, id);
                id
            }
        };
        self.seen.insert(key, id);
        Ok(id)
    }

    pub(crate) fn metrics(&mut self, name: FontName, class: FontName, sym: &str, fontsize: f64, dpi: f64)
        -> Result<FontMetrics, Error> {
        let id = self.get_info(name, class, sym, fontsize, dpi)?;
        Ok(self.cache.infos[id].metrics)
    }

    /// `TruetypeFonts.get_xheight`.
    pub(crate) fn get_xheight(&mut self, name: FontName, fontsize: f64, dpi: f64) -> Result<f64, Error> {
        let font = self.get_font(Kind::DejaVuSans, name)?
            .ok_or_else(|| Error::MatplotlibRaises(format!("AttributeError: get_xheight of font {name:?}, which the fontset lacks")))?;
        let stored = &mut self.cache.store.fonts[font];
        stored.font.set_size(fontsize, dpi)?;
        match stored.pclt_x_height {
            None => Ok(self.metrics(name, FontName::It, "x", fontsize, dpi)?.iceberg),
            Some(x_height) => Ok((x_height as f64 / 64.0) * (fontsize / 12.0) * (dpi / 100.0)),
        }
    }

    /// `TruetypeFonts.get_underline_thickness`.
    pub(crate) fn underline_thickness(fontsize: f64, dpi: f64) -> f64 { ((0.75 / 12.0) * fontsize * dpi) / 72.0 }

    /// `TruetypeFonts.get_kern`: kerning of the two character codes (as glyph
    /// indices, like Matplotlib) at the size the first glyph's face was last set to.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn get_kern(&mut self, font1: FontName, class1: FontName, sym1: &str, fontsize1: f64,
        font2: FontName, class2: FontName, sym2: &str, fontsize2: f64, dpi: f64) -> Result<f64, Error> {
        if font1 == font2 && fontsize1 == fontsize2 {
            let info1 = self.get_info(font1, class1, sym1, fontsize1, dpi)?;
            let info2 = self.get_info(font2, class2, sym2, fontsize2, dpi)?;
            let (font, left, right) = (self.cache.infos[info1].font, self.cache.infos[info1].num, self.cache.infos[info2].num);
            let kern = self.cache.store.fonts[font].font.get_kerning(left, right, zlc_text::ffi::FT_KERNING_DEFAULT, true)?;
            return Ok(kern as f64 / 64.0);
        }
        Ok(0.0)
    }

    /// The family name `_get_font_constant_set` keys on.
    pub(crate) fn family_name(&mut self, name: FontName) -> Result<String, Error> {
        let font = self.get_font(Kind::DejaVuSans, name)?
            .ok_or_else(|| Error::MatplotlibRaises(format!("AttributeError: font constants of font {name:?}, which the fontset lacks")))?;
        Ok(self.cache.store.fonts[font].family_name.clone())
    }

    /// `get_sized_alternatives_for_symbol` (DejaVu defers to its StixSans fallback).
    pub(crate) fn sized_alternatives(&mut self, name: FontName, sym: &str) -> Result<Vec<(FontName, String)>, Error> {
        let key = (name, sym.to_string());
        if let Some(found) = self.cache.alternatives.get(&key) { return Ok(found.clone()) }
        let fixed = match sym {
            "\\{" => "{", "\\}" => "}", "\\[" => "[", "\\]" => "]",
            "<" => "\u{27E8}", ">" => "\u{27E9}",
            other => other,
        };
        let alternatives = match unicode_index(fixed) {
            None => vec![(name, fixed.to_string())],
            Some(code) => {
                let mut found = Vec::new();
                for size in 0..6u8 {
                    let font = self.get_font(Kind::StixSans, FontName::Size(size))?.expect("STIX size fonts");
                    if self.char_index(font, code)? != 0 {
                        found.push((FontName::Size(size), char::from_u32(code).expect("code point").to_string()));
                    }
                }
                if fixed == "\\__sqrt__" { found.pop(); }
                found
            }
        };
        self.cache.alternatives.insert(key, alternatives.clone());
        Ok(alternatives)
    }
}
