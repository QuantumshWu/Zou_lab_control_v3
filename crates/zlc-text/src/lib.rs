//! Byte-exact Rust port of Matplotlib 3.10.8's `FT2Image` and `FT2Font`
//! (`src/ft2font.cpp`) over the same static FreeType 2.6.1 build. Integers keep
//! the C++ `long`/`int` widths of the target and float expressions keep the C++
//! operation order, so bitmaps, metrics, positions and outlines equal Matplotlib's.
//!
//! A font owns its face exclusively. Fallback fonts are shared (`SharedFont`)
//! because their owners also use them directly; a busy font is reported, never
//! waited for. Faces are opened and closed under one library lock.

pub mod ffi;

use std::collections::{BTreeMap, HashMap};
use std::ffi::{CStr, c_char, c_int, c_long, c_uchar, c_uint, c_ulong, c_void};
use std::fmt;
use std::ptr::{self, null_mut};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, TryLockError};

use ffi::{FT_BBox, FT_Error, FT_Face, FT_Glyph, FT_Int, FT_Int32, FT_Matrix, FT_Pos, FT_UInt, FT_ULong, FT_Vector};

/// Which Python exception Matplotlib's binding raises for this failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorKind { Runtime, Value, Memory }

#[derive(Clone, Debug)]
pub struct Error { pub kind: ErrorKind, pub message: String }

impl Error {
    fn runtime(message: impl Into<String>) -> Self { Self { kind: ErrorKind::Runtime, message: message.into() } }
    fn value(message: impl Into<String>) -> Self { Self { kind: ErrorKind::Value, message: message.into() } }

    /// Matplotlib's `throw_ft_error` text.
    fn freetype(message: &str, error: FT_Error) -> Self {
        Self::runtime(match ffi::error_string(error) {
            Some(text) => format!("{message} ({text}; error code 0x{error:x})"),
            None => format!("{message} (error code 0x{error:x})"),
        })
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result { formatter.write_str(&self.message) }
}

impl std::error::Error for Error {}

struct Library(ffi::FT_Library);
// SAFETY: the handle is created once and never destroyed; operations that mutate
// library-wide state (face creation/destruction) are serialized by FACES.
unsafe impl Send for Library {}
unsafe impl Sync for Library {}

static LIBRARY: OnceLock<Result<Library, Error>> = OnceLock::new();
static FACES: Mutex<()> = Mutex::new(());

fn library() -> Result<ffi::FT_Library, Error> {
    LIBRARY.get_or_init(|| {
        let mut library = null_mut();
        if unsafe { ffi::FT_Init_FreeType(&mut library) } != 0 {
            return Err(Error::runtime("Could not initialize the freetype2 library"));
        }
        Ok(Library(library))
    }).as_ref().map(|library| library.0).map_err(Clone::clone)
}

fn faces() -> MutexGuard<'static, ()> { FACES.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) }

/// The linked FreeType version, formatted as Matplotlib's `__freetype_version__`.
pub fn freetype_version() -> Result<String, Error> {
    let (mut major, mut minor, mut patch) = (0, 0, 0);
    unsafe { ffi::FT_Library_Version(library()?, &mut major, &mut minor, &mut patch) };
    Ok(format!("{major}.{minor}.{patch}"))
}

/// Matplotlib's stream: a read callback with an unknown (0x7fffffff) size,
/// answering like Python's `seek(offset); read(count)` over the font bytes.
struct Source { stream: Box<ffi::FT_StreamRec>, _data: Box<Vec<u8>> }

unsafe extern "C" fn read_source(stream: *mut ffi::FT_StreamRec, offset: c_ulong, buffer: *mut c_uchar, count: c_ulong) -> c_ulong {
    // SAFETY: descriptor points at the immutable Vec owned by this stream's Source.
    let data = unsafe { &*((*stream).descriptor.pointer as *const Vec<u8>) };
    let start = (offset as usize).min(data.len());
    let length = (count as usize).min(data.len() - start);
    if length != 0 { unsafe { ptr::copy_nonoverlapping(data.as_ptr().add(start), buffer, length) } }
    length as c_ulong
}

impl Source {
    fn new(data: Vec<u8>) -> Self {
        let data = Box::new(data);
        let stream = Box::new(ffi::FT_StreamRec {
            base: null_mut(), size: 0x7fffffff, pos: 0,
            descriptor: ffi::FT_StreamDesc { pointer: (&*data as *const Vec<u8>).cast_mut().cast() },
            pathname: ffi::FT_StreamDesc { pointer: null_mut() },
            read: Some(read_source), close: None,
            memory: null_mut(), cursor: null_mut(), limit: null_mut(),
        });
        Self { stream, _data: data }
    }
}

/// The FreeType string rendered into a width×height coverage buffer.
#[derive(Default)]
pub struct FT2Image { buffer: Vec<u8>, width: c_ulong, height: c_ulong }

impl FT2Image {
    pub fn new(width: c_long, height: c_long) -> Result<Self, Error> {
        let mut image = Self::default();
        image.resize(width, height)?;
        Ok(image)
    }

    pub fn width(&self) -> c_ulong { self.width }
    pub fn height(&self) -> c_ulong { self.height }

    /// The visible height×width bytes, row-major.
    pub fn pixels(&self) -> &[u8] { &self.buffer[..self.width as usize * self.height as usize] }
    pub fn pixels_mut(&mut self) -> &mut [u8] {
        let length = self.width as usize * self.height as usize;
        &mut self.buffer[..length]
    }

    /// Clears to at least 1×1; reallocates only beyond the previous extent.
    pub fn resize(&mut self, width: c_long, height: c_long) -> Result<(), Error> {
        let width = if width <= 0 { 1 } else { width };
        let height = if height <= 0 { 1 } else { height };
        let bytes = width.checked_mul(height)
            .ok_or(Error { kind: ErrorKind::Memory, message: "std::bad_alloc".into() })? as usize;
        if width as c_ulong != self.width || height as c_ulong != self.height {
            if bytes > self.width.wrapping_mul(self.height) as usize {
                self.buffer = vec![0; bytes];
            }
            self.width = width as c_ulong;
            self.height = height as c_ulong;
        }
        if bytes != 0 && !self.buffer.is_empty() { self.buffer[..bytes].fill(0) }
        Ok(())
    }

    /// ORs gray coverage (or sets mono ink to 255) with the bitmap's top-left at (x, y).
    ///
    /// # Safety
    /// `bitmap` must describe a live FreeType bitmap of `rows`×|`pitch`| bytes.
    unsafe fn draw_bitmap(&mut self, bitmap: &ffi::FT_Bitmap, x: FT_Int, y: FT_Int) -> Result<(), Error> {
        let image_width = self.width as FT_Int;
        let image_height = self.height as FT_Int;
        let char_width = bitmap.width as FT_Int;
        let char_height = bitmap.rows as FT_Int;
        let x1 = x.max(0).min(image_width);
        let y1 = y.max(0).min(image_height);
        let x2 = x.wrapping_add(char_width).max(0).min(image_width);
        let y2 = y.wrapping_add(char_height).max(0).min(image_height);
        let x_start = 0.max(x.wrapping_neg());
        let y_offset = y1 - 0.max(y.wrapping_neg());
        let gray = match bitmap.pixel_mode {
            ffi::FT_PIXEL_MODE_GRAY => true,
            ffi::FT_PIXEL_MODE_MONO => false,
            _ => return Err(Error::runtime("Unknown pixel mode")),
        };
        let span = (x2 - x1).max(0) as usize;
        if span == 0 { return Ok(()) }
        if gray {
            for i in y1..y2 {
                let start = (i * image_width + x1) as usize;
                let source = unsafe { std::slice::from_raw_parts(
                    bitmap.buffer.offset(((i - y_offset) * bitmap.pitch + x_start) as isize), span) };
                for (target, value) in self.buffer[start..start + span].iter_mut().zip(source) { *target |= value }
            }
        } else {
            for i in y1..y2 {
                let start = (i * image_width + x1) as usize;
                let source = unsafe { bitmap.buffer.offset(((i - y_offset) * bitmap.pitch) as isize) };
                for (column, target) in self.buffer[start..start + span].iter_mut().enumerate() {
                    let x = column as c_int + x_start;
                    if unsafe { *source.offset((x >> 3) as isize) } & (1 << (7 - (x & 0x7))) != 0 { *target = 255 }
                }
            }
        }
        Ok(())
    }

    /// Sets the inclusive rectangle to 255, clipped to the image.
    pub fn draw_rect_filled(&mut self, x0: c_ulong, y0: c_ulong, x1: c_ulong, y1: c_ulong) {
        let x0 = x0.min(self.width) as usize;
        let y0 = y0.min(self.height) as usize;
        let x1 = x1.wrapping_add(1).min(self.width) as usize;
        let y1 = y1.wrapping_add(1).min(self.height) as usize;
        let width = self.width as usize;
        for row in y0..y1 {
            if x0 < x1 { self.buffer[row * width + x0..row * width + x1].fill(255) }
        }
    }
}

pub type SharedFont = Arc<Mutex<FT2Font>>;

/// Exclusive access to a shared font; a font already in use is an error.
pub fn lock(font: &SharedFont) -> Result<MutexGuard<'_, FT2Font>, Error> {
    match font.try_lock() {
        Ok(guard) => Ok(guard),
        Err(TryLockError::Poisoned(poisoned)) => Ok(poisoned.into_inner()),
        Err(TryLockError::WouldBlock) => Err(Error::runtime("FT2Font is already in use")),
    }
}

/// The font holding a glyph or character, as seen from the font whose tables
/// record it (C++ `FT2Font*`, compared by identity).
#[derive(Clone)]
enum Owner { Itself, Fallback(SharedFont) }

impl Owner {
    fn same(&self, other: &Owner) -> bool {
        match (self, other) {
            (Owner::Itself, Owner::Itself) => true,
            (Owner::Fallback(left), Owner::Fallback(right)) => Arc::ptr_eq(left, right),
            _ => false,
        }
    }
}

struct Core { face: FT_Face, fallbacks: Vec<SharedFont>, hinting_factor: c_long, kerning_factor: c_int }

#[derive(Default)]
struct Tables { glyphs: Vec<FT_Glyph>, glyph_to_font: HashMap<FT_UInt, Owner>, char_to_font: HashMap<c_long, Owner> }

/// `Glyph` metrics as Matplotlib's `PyGlyph_from_FT2Font` reads them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Glyph {
    /// Index into the loaded glyphs of the font that loaded it.
    pub glyph_index: usize,
    pub width: c_long,
    pub height: c_long,
    pub hori_bearing_x: c_long,
    pub hori_bearing_y: c_long,
    pub hori_advance: c_long,
    pub linear_hori_advance: c_long,
    pub vert_bearing_x: c_long,
    pub vert_bearing_y: c_long,
    pub vert_advance: c_long,
    /// Control box (xMin, yMin, xMax, yMax) in 26.6 subpixels.
    pub bbox: [FT_Pos; 4],
}

pub struct SfntName { pub platform_id: u16, pub encoding_id: u16, pub language_id: u16, pub name_id: u16, pub string: Vec<u8> }

pub struct PsFontInfo {
    pub version: Vec<u8>,
    pub notice: Vec<u8>,
    pub full_name: Vec<u8>,
    pub family_name: Vec<u8>,
    pub weight: Vec<u8>,
    pub italic_angle: c_long,
    pub is_fixed_pitch: u8,
    pub underline_position: i16,
    pub underline_thickness: u16,
}

pub enum SfntTable {
    Head(ffi::TT_Header),
    Maxp(ffi::TT_MaxProfile),
    Os2(ffi::TT_OS2),
    Hhea(ffi::TT_HoriHeader),
    Vhea(ffi::TT_VertHeader),
    Post(ffi::TT_Postscript),
    Pclt(ffi::TT_PCLT),
}

/// Matplotlib's path codes.
pub const MOVETO: u8 = 1;
pub const LINETO: u8 = 2;
pub const CURVE3: u8 = 3;
pub const CURVE4: u8 = 4;
pub const CLOSEPOLY: u8 = 0x4f;

/// The family-name pointers a missing glyph was looked up in, in pointer order
/// (C++ `std::set<FT_String*>`), joined like Matplotlib's `ft_glyph_warn`.
#[derive(Default)]
struct SeenFonts(BTreeMap<usize, *const c_char>);

impl SeenFonts {
    fn insert(&mut self, name: *const c_char) { self.0.insert(name as usize, name); }

    fn joined(&self) -> Vec<u8> {
        let mut text = Vec::new();
        for (index, &name) in self.0.values().enumerate() {
            // Streaming a null char* sets badbit; nothing further is written.
            if name.is_null() { break }
            if index != 0 { text.extend_from_slice(b", ") }
            text.extend_from_slice(unsafe { CStr::from_ptr(name) }.to_bytes());
        }
        text
    }
}

/// A missing-glyph report: the character code and the families searched.
pub type Warn<'a, E> = &'a mut dyn FnMut(c_ulong, &[u8]) -> Result<(), E>;

pub struct FT2Font {
    core: Core,
    tables: Tables,
    image: FT2Image,
    pen: FT_Vector,
    bbox: FT_BBox,
    advance: FT_Pos,
    // Dropped after the face (see Drop), which reads through this stream.
    _source: Source,
}

// SAFETY: the face, its glyphs and its stream are owned exclusively by this
// value; FreeType face operations need no other synchronization than `&mut`
// access, and library-wide operations are serialized by FACES.
unsafe impl Send for FT2Font {}

impl Drop for FT2Font {
    fn drop(&mut self) {
        for &glyph in &self.tables.glyphs { unsafe { ffi::FT_Done_Glyph(glyph) } }
        if !self.core.face.is_null() {
            let _faces = faces();
            unsafe { ffi::FT_Done_Face(self.core.face) };
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn load_char_with_fallback(
    me: &Owner, core: &Core, parent: &mut Tables, owner_with_glyph: &mut Owner, final_glyph_index: &mut FT_UInt,
    charcode: c_long, flags: FT_Int32, charcode_error: &mut FT_Error, glyph_error: &mut FT_Error,
    seen: &mut SeenFonts, override_missing: bool,
) -> Result<bool, Error> {
    let face = core.face;
    let glyph_index = unsafe { ffi::FT_Get_Char_Index(face, charcode as FT_ULong) };
    seen.insert(unsafe { (*face).family_name });
    if glyph_index != 0 || override_missing {
        *charcode_error = unsafe { ffi::FT_Load_Glyph(face, glyph_index, flags) };
        if *charcode_error != 0 { return Ok(false) }
        let mut glyph = null_mut();
        *glyph_error = unsafe { ffi::FT_Get_Glyph((*face).glyph, &mut glyph) };
        if *glyph_error != 0 { return Ok(false) }
        *final_glyph_index = glyph_index;
        *owner_with_glyph = me.clone();
        parent.glyph_to_font.insert(glyph_index, me.clone());
        parent.char_to_font.insert(charcode, me.clone());
        parent.glyphs.push(glyph);
        return Ok(true);
    }
    for fallback in &core.fallbacks {
        let font = lock(fallback)?;
        if load_char_with_fallback(&Owner::Fallback(fallback.clone()), &font.core, parent, owner_with_glyph,
            final_glyph_index, charcode, flags, charcode_error, glyph_error, seen, override_missing)? {
            return Ok(true);
        }
    }
    Ok(false)
}

impl FT2Font {
    /// Opens the first face of `data` (Matplotlib's constructor followed by its
    /// binding's `set_kerning_factor`, which also reaches the fallbacks).
    pub fn new(data: Vec<u8>, hinting_factor: c_long, fallbacks: Vec<SharedFont>, kerning_factor: c_int) -> Result<Self, Error> {
        if hinting_factor <= 0 { return Err(Error::value("hinting_factor must be greater than 0")) }
        let library = library()?;
        let mut source = Source::new(data);
        let args = ffi::FT_Open_Args {
            flags: ffi::FT_OPEN_STREAM, memory_base: ptr::null(), memory_size: 0, pathname: null_mut(),
            stream: &mut *source.stream, driver: null_mut(), num_params: 0, params: null_mut(),
        };
        let mut face = null_mut();
        let error = { let _faces = faces(); unsafe { ffi::FT_Open_Face(library, &args, 0, &mut face) } };
        if error != 0 { return Err(Error::freetype("Can not load face", error)) }
        let mut font = Self {
            core: Core { face, fallbacks: Vec::new(), hinting_factor, kerning_factor: 0 },
            tables: Tables::default(), image: FT2Image::default(), pen: FT_Vector::default(),
            bbox: FT_BBox::default(), advance: 0, _source: source,
        };
        let error = unsafe { ffi::FT_Set_Char_Size(face, 12 * 64, 0, 72u32.wrapping_mul(hinting_factor as c_uint), 72) };
        if error != 0 { return Err(Error::freetype("Could not set the fontsize", error)) }
        unsafe { (*face).face_flags |= ffi::FT_FACE_FLAG_EXTERNAL_STREAM };
        font.set_transform();
        font.core.fallbacks = fallbacks;
        font.set_kerning_factor(kerning_factor)?;
        Ok(font)
    }

    fn set_transform(&mut self) {
        let mut transform = FT_Matrix { xx: 65536 / self.core.hinting_factor, xy: 0, yx: 0, yy: 65536 };
        unsafe { ffi::FT_Set_Transform(self.core.face, &mut transform, null_mut()) };
    }

    fn with_owner<R>(&mut self, owner: &Owner, action: impl FnOnce(&mut FT2Font) -> Result<R, Error>) -> Result<R, Error> {
        match owner {
            Owner::Itself => action(self),
            Owner::Fallback(font) => action(&mut *lock(font)?),
        }
    }

    fn read_owner<R>(&self, owner: &Owner, action: impl FnOnce(&FT2Font) -> Result<R, Error>) -> Result<R, Error> {
        match owner {
            Owner::Itself => action(self),
            Owner::Fallback(font) => action(&*lock(font)?),
        }
    }

    fn face(&self) -> &ffi::FT_FaceRec { unsafe { &*self.core.face } }

    /// Drops every loaded glyph and the string layout, here and in the fallbacks.
    pub fn clear(&mut self) -> Result<(), Error> {
        self.pen = FT_Vector::default();
        self.bbox = FT_BBox::default();
        self.advance = 0;
        for glyph in self.tables.glyphs.drain(..) { unsafe { ffi::FT_Done_Glyph(glyph) } }
        self.tables.glyph_to_font.clear();
        self.tables.char_to_font.clear();
        for fallback in &self.core.fallbacks { lock(fallback)?.clear()? }
        Ok(())
    }

    pub fn set_size(&mut self, ptsize: f64, dpi: f64) -> Result<(), Error> {
        let error = unsafe { ffi::FT_Set_Char_Size(self.core.face, (ptsize * 64.0) as ffi::FT_F26Dot6, 0,
            (dpi * self.core.hinting_factor as f64) as FT_UInt, dpi as FT_UInt) };
        if error != 0 { return Err(Error::freetype("Could not set the fontsize", error)) }
        self.set_transform();
        for fallback in &self.core.fallbacks { lock(fallback)?.set_size(ptsize, dpi)? }
        Ok(())
    }

    pub fn set_charmap(&mut self, index: c_int) -> Result<(), Error> {
        let face = self.face();
        if index >= face.num_charmaps || index < 0 {
            return Err(Error::runtime("i exceeds the available number of char maps"));
        }
        let charmap = unsafe { *face.charmaps.add(index as usize) };
        let error = unsafe { ffi::FT_Set_Charmap(self.core.face, charmap) };
        if error != 0 { return Err(Error::freetype("Could not set the charmap", error)) }
        Ok(())
    }

    pub fn select_charmap(&mut self, encoding: c_ulong) -> Result<(), Error> {
        let error = unsafe { ffi::FT_Select_Charmap(self.core.face, encoding as c_uint) };
        if error != 0 { return Err(Error::freetype("Could not set the charmap", error)) }
        Ok(())
    }

    pub fn has_kerning(&self) -> bool { self.face().face_flags & ffi::FT_FACE_FLAG_KERNING != 0 }

    /// Kerning in 26.6 subpixels, divided by `hinting_factor << kerning_factor`.
    pub fn get_kerning(&self, left: FT_UInt, right: FT_UInt, mode: FT_UInt, fallback: bool) -> Result<c_int, Error> {
        if fallback && let (Some(left_owner), Some(right_owner)) =
            (self.tables.glyph_to_font.get(&left), self.tables.glyph_to_font.get(&right)) {
            if !left_owner.same(right_owner) { return Ok(0) }
            return self.read_owner(right_owner, |font| Ok(font.kerning_delta(left, right, mode)));
        }
        Ok(self.kerning_delta(left, right, mode))
    }

    fn kerning_delta(&self, left: FT_UInt, right: FT_UInt, mode: FT_UInt) -> c_int {
        if !self.has_kerning() { return 0 }
        let mut delta = FT_Vector::default();
        if unsafe { ffi::FT_Get_Kerning(self.core.face, left, right, mode, &mut delta) } != 0 { return 0 }
        let divisor = self.core.hinting_factor.wrapping_shl(self.core.kerning_factor as u32);
        if divisor == 0 { 0 } else { (delta.x as c_int as c_long).wrapping_div(divisor) as c_int }
    }

    pub fn set_kerning_factor(&mut self, factor: c_int) -> Result<(), Error> {
        self.core.kerning_factor = factor;
        for fallback in &self.core.fallbacks { lock(fallback)?.set_kerning_factor(factor)? }
        Ok(())
    }

    /// Lays out `text` (code points): loads each glyph with fallback, kerns,
    /// translates by the pen and rotates by `angle` degrees. `xys` receives each
    /// glyph's pen position in 26.6 subpixels.
    pub fn set_text<E: From<Error>>(&mut self, text: &[u32], angle: f64, flags: FT_Int32, xys: &mut Vec<f64>,
        warn: Warn<'_, E>) -> Result<(), E> {
        let angle = angle * (2.0 * std::f64::consts::PI / 360.0);
        let cosangle = angle.cos() * 0x10000 as f64;
        let sinangle = angle.sin() * 0x10000 as f64;
        let mut matrix = FT_Matrix {
            xx: cosangle as ffi::FT_Fixed, xy: -sinangle as ffi::FT_Fixed,
            yx: sinangle as ffi::FT_Fixed, yy: cosangle as ffi::FT_Fixed,
        };
        self.clear()?;
        self.bbox = FT_BBox { xMin: 32000, yMin: 32000, xMax: -32000, yMax: -32000 };
        let mut previous: FT_UInt = 0;
        let mut previous_owner: Option<Owner> = None;
        for &codepoint in text {
            let mut glyph_index: FT_UInt = 0;
            let (mut charcode_error, mut glyph_error) = (0, 0);
            let mut seen = SeenFonts::default();
            let mut owner = Owner::Itself;
            let charcode = codepoint as c_long;
            let found = load_char_with_fallback(&Owner::Itself, &self.core, &mut self.tables, &mut owner,
                &mut glyph_index, charcode, flags, &mut charcode_error, &mut glyph_error, &mut seen, false)?;
            if !found {
                warn(codepoint as c_ulong, &seen.joined())?;
                // Render the missing-glyph tofu from the top-most font.
                owner = Owner::Itself;
                self.tables.char_to_font.insert(charcode, Owner::Itself);
                self.tables.glyph_to_font.insert(glyph_index, Owner::Itself);
                self.load_glyph_here(glyph_index, flags)?;
            }
            let kern = previous_owner.as_ref().is_some_and(|previous_owner| previous_owner.same(&owner))
                && previous != 0 && glyph_index != 0;
            let (kerning, last_advance) = self.read_owner(&owner, |font| Ok((
                if kern && font.has_kerning() { font.kerning_delta(previous, glyph_index, ffi::FT_KERNING_DEFAULT) } else { 0 },
                unsafe { (*font.face().glyph).advance.x },
            )))?;
            self.pen.x = self.pen.x.wrapping_add(kerning as FT_Pos);
            let glyph = *self.tables.glyphs.last().ok_or_else(|| Error::runtime("No glyph loaded"))?;
            let mut glyph_bbox = FT_BBox::default();
            unsafe {
                ffi::FT_Glyph_Transform(glyph, null_mut(), &mut self.pen);
                ffi::FT_Glyph_Transform(glyph, &mut matrix, null_mut());
            }
            xys.push(self.pen.x as f64);
            xys.push(self.pen.y as f64);
            unsafe { ffi::FT_Glyph_Get_CBox(glyph, ffi::FT_GLYPH_BBOX_SUBPIXELS, &mut glyph_bbox) };
            self.bbox.xMin = self.bbox.xMin.min(glyph_bbox.xMin);
            self.bbox.xMax = self.bbox.xMax.max(glyph_bbox.xMax);
            self.bbox.yMin = self.bbox.yMin.min(glyph_bbox.yMin);
            self.bbox.yMax = self.bbox.yMax.max(glyph_bbox.yMax);
            self.pen.x = self.pen.x.wrapping_add(last_advance);
            previous = glyph_index;
            previous_owner = Some(owner);
        }
        unsafe { ffi::FT_Vector_Transform(&mut self.pen, &matrix) };
        self.advance = self.pen.x;
        if self.bbox.xMin > self.bbox.xMax { self.bbox = FT_BBox::default() }
        Ok(())
    }

    /// Loads one character; with `fallback`, through the fonts chosen by the
    /// last `set_text`, else from this font (missing glyphs load `.notdef`).
    pub fn load_char<E: From<Error>>(&mut self, charcode: c_long, flags: FT_Int32, fallback: bool,
        warn: Warn<'_, E>) -> Result<Glyph, E> {
        let mut seen = SeenFonts::default();
        if fallback && let Some(owner) = self.tables.char_to_font.get(&charcode).cloned() {
            return match owner {
                Owner::Itself => self.load_char(charcode, flags, false, warn),
                Owner::Fallback(font) => lock(&font)?.load_char(charcode, flags, false, warn),
            };
        }
        if fallback {
            let mut glyph_index = 0;
            let (mut charcode_error, mut glyph_error) = (0, 0);
            let mut owner = Owner::Itself;
            let found = load_char_with_fallback(&Owner::Itself, &self.core, &mut self.tables, &mut owner,
                &mut glyph_index, charcode, flags, &mut charcode_error, &mut glyph_error, &mut seen, true)?;
            if !found {
                warn(charcode as c_ulong, &seen.joined())?;
                if charcode_error != 0 { return Err(Error::freetype("Could not load charcode", charcode_error).into()) }
                if glyph_error != 0 { return Err(Error::freetype("Could not load charcode", glyph_error).into()) }
            }
            return Ok(self.read_owner(&owner, |font| font.last_glyph())?);
        }
        let face = self.core.face;
        let glyph_index = unsafe { ffi::FT_Get_Char_Index(face, charcode as FT_ULong) };
        if glyph_index == 0 {
            seen.insert(unsafe { (*face).family_name });
            warn(charcode as c_ulong, &seen.joined())?;
        }
        let error = unsafe { ffi::FT_Load_Glyph(face, glyph_index, flags) };
        if error != 0 { return Err(Error::freetype("Could not load charcode", error).into()) }
        let mut glyph = null_mut();
        let error = unsafe { ffi::FT_Get_Glyph((*face).glyph, &mut glyph) };
        if error != 0 { return Err(Error::freetype("Could not get glyph", error).into()) }
        self.tables.glyphs.push(glyph);
        Ok(self.last_glyph()?)
    }

    /// Loads one glyph index; with `fallback`, from the font that last served it.
    pub fn load_glyph(&mut self, glyph_index: FT_UInt, flags: FT_Int32, fallback: bool) -> Result<Glyph, Error> {
        let owner = match self.tables.glyph_to_font.get(&glyph_index) {
            Some(owner) if fallback => owner.clone(),
            _ => Owner::Itself,
        };
        self.with_owner(&owner, |font| {
            font.load_glyph_here(glyph_index, flags)?;
            font.last_glyph()
        })
    }

    fn load_glyph_here(&mut self, glyph_index: FT_UInt, flags: FT_Int32) -> Result<(), Error> {
        let face = self.core.face;
        let error = unsafe { ffi::FT_Load_Glyph(face, glyph_index, flags) };
        if error != 0 { return Err(Error::freetype("Could not load glyph", error)) }
        let mut glyph = null_mut();
        let error = unsafe { ffi::FT_Get_Glyph((*face).glyph, &mut glyph) };
        if error != 0 { return Err(Error::freetype("Could not get glyph", error)) }
        self.tables.glyphs.push(glyph);
        Ok(())
    }

    /// The most recently loaded glyph with this face's current slot metrics.
    fn last_glyph(&self) -> Result<Glyph, Error> {
        let glyph = *self.tables.glyphs.last().ok_or_else(|| Error::runtime("No glyph loaded"))?;
        let mut bbox = FT_BBox::default();
        unsafe { ffi::FT_Glyph_Get_CBox(glyph, ffi::FT_GLYPH_BBOX_SUBPIXELS, &mut bbox) };
        let slot = unsafe { &*self.face().glyph };
        let metrics = &slot.metrics;
        let hinting_factor = self.core.hinting_factor;
        Ok(Glyph {
            glyph_index: self.tables.glyphs.len() - 1,
            width: metrics.width / hinting_factor,
            height: metrics.height,
            hori_bearing_x: metrics.horiBearingX / hinting_factor,
            hori_bearing_y: metrics.horiBearingY,
            hori_advance: metrics.horiAdvance,
            linear_hori_advance: slot.linearHoriAdvance / hinting_factor,
            vert_bearing_x: metrics.vertBearingX,
            vert_bearing_y: metrics.vertBearingY,
            vert_advance: metrics.vertAdvance,
            bbox: [bbox.xMin, bbox.yMin, bbox.xMax, bbox.yMax],
        })
    }

    /// (advance, height) of the current string in 26.6 subpixels.
    pub fn get_width_height(&self) -> (c_long, c_long) { (self.advance, self.bbox.yMax - self.bbox.yMin) }
    pub fn get_descent(&self) -> c_long { -self.bbox.yMin }
    pub fn get_bitmap_offset(&self) -> (c_long, c_long) { (self.bbox.xMin, 0) }
    pub fn get_num_glyphs(&self) -> usize { self.tables.glyphs.len() }
    pub fn hinting_factor(&self) -> c_long { self.core.hinting_factor }
    pub fn image(&self) -> &FT2Image { &self.image }

    /// Renders every glyph of the current string into the font's own image.
    pub fn draw_glyphs_to_bitmap(&mut self, antialiased: bool) -> Result<(), Error> {
        let width = (self.bbox.xMax - self.bbox.xMin) / 64 + 2;
        let height = (self.bbox.yMax - self.bbox.yMin) / 64 + 2;
        self.image.resize(width, height)?;
        let mode = if antialiased { ffi::FT_RENDER_MODE_NORMAL } else { ffi::FT_RENDER_MODE_MONO };
        for index in 0..self.tables.glyphs.len() {
            let error = unsafe { ffi::FT_Glyph_To_Bitmap(&mut self.tables.glyphs[index], mode, null_mut(), 1) };
            if error != 0 { return Err(Error::freetype("Could not convert glyph to bitmap", error)) }
            let bitmap = unsafe { &*(self.tables.glyphs[index] as *const ffi::FT_BitmapGlyphRec) };
            // Bitmap left and top are in pixels, the string bbox in subpixels.
            let x = (bitmap.left as f64 - (self.bbox.xMin as f64 * (1.0 / 64.0))) as FT_Int;
            let y = ((self.bbox.yMax as f64 * (1.0 / 64.0)) - bitmap.top as f64 + 1.0) as FT_Int;
            unsafe { self.image.draw_bitmap(&bitmap.bitmap, x, y)? };
        }
        Ok(())
    }

    /// Renders loaded glyph `glyph_index` into `image` at pixel (x, y).
    pub fn draw_glyph_to_bitmap(&mut self, image: &mut FT2Image, x: c_int, y: c_int, glyph_index: usize,
        antialiased: bool) -> Result<(), Error> {
        let mut sub_offset = FT_Vector::default();
        if glyph_index >= self.tables.glyphs.len() { return Err(Error::runtime("glyph num is out of range")) }
        let mode = if antialiased { ffi::FT_RENDER_MODE_NORMAL } else { ffi::FT_RENDER_MODE_MONO };
        let error = unsafe { ffi::FT_Glyph_To_Bitmap(&mut self.tables.glyphs[glyph_index], mode, &mut sub_offset, 1) };
        if error != 0 { return Err(Error::freetype("Could not convert glyph to bitmap", error)) }
        let bitmap = unsafe { &*(self.tables.glyphs[glyph_index] as *const ffi::FT_BitmapGlyphRec) };
        unsafe { image.draw_bitmap(&bitmap.bitmap, x.wrapping_add(bitmap.left), y) }
    }

    /// The glyph's name, or Matplotlib's `uni%08x` for fonts without names.
    pub fn get_glyph_name(&self, glyph_number: c_uint, fallback: bool) -> Result<Vec<u8>, Error> {
        if fallback && let Some(owner) = self.tables.glyph_to_font.get(&glyph_number) {
            return self.read_owner(owner, |font| font.get_glyph_name(glyph_number, false));
        }
        if self.face().face_flags & ffi::FT_FACE_FLAG_GLYPH_NAMES == 0 {
            return Ok(format!("uni{glyph_number:08x}").into_bytes());
        }
        let mut buffer = [0u8; 128];
        let error = unsafe { ffi::FT_Get_Glyph_Name(self.core.face, glyph_number, buffer.as_mut_ptr().cast(), 128) };
        if error != 0 { return Err(Error::freetype("Could not get glyph names", error)) }
        let length = buffer.iter().position(|&byte| byte == 0).unwrap_or(buffer.len());
        Ok(buffer[..length].to_vec())
    }

    pub fn get_name_index(&self, name: &CStr) -> FT_UInt {
        unsafe { ffi::FT_Get_Name_Index(self.core.face, name.as_ptr().cast_mut()) }
    }

    /// Glyph index of `charcode`; with `fallback`, in the font that last served it.
    pub fn get_char_index(&self, charcode: FT_ULong, fallback: bool) -> Result<FT_UInt, Error> {
        let owner = match self.tables.char_to_font.get(&(charcode as c_long)) {
            Some(owner) if fallback => owner.clone(),
            _ => Owner::Itself,
        };
        self.read_owner(&owner, |font| Ok(unsafe { ffi::FT_Get_Char_Index(font.core.face, charcode) }))
    }

    /// `Some(-1)` if this font maps `charcode`, else the first fallback that does.
    pub fn get_char_fallback_index(&self, charcode: FT_ULong) -> Result<Option<c_int>, Error> {
        if unsafe { ffi::FT_Get_Char_Index(self.core.face, charcode) } != 0 { return Ok(Some(-1)) }
        for (index, fallback) in self.core.fallbacks.iter().enumerate() {
            if lock(fallback)?.get_char_fallback_index(charcode)?.is_some() { return Ok(Some(index as c_int)) }
        }
        Ok(None)
    }

    /// The current slot's outline as Matplotlib path vertices (x, y pairs) and codes.
    pub fn get_path(&self) -> Result<(Vec<f64>, Vec<u8>), Error> {
        let slot = self.face().glyph;
        if slot.is_null() { return Err(Error::runtime("No glyph loaded")) }
        let outline = unsafe { ptr::addr_of_mut!((*slot).outline) };
        let estimated = unsafe { (*outline).n_contours as usize + (*outline).n_points as usize };
        let mut path = Decomposer { vertices: Vec::with_capacity(2 * estimated), codes: Vec::with_capacity(estimated) };
        let error = unsafe { ffi::FT_Outline_Decompose(outline, &OUTLINE_FUNCS, (&mut path as *mut Decomposer).cast()) };
        if error != 0 { return Err(Error::runtime(format!("FT_Outline_Decompose failed with error {error}"))) }
        if !path.vertices.is_empty() {
            path.vertices.extend([0.0, 0.0]);
            path.codes.push(CLOSEPOLY);
        }
        Ok((path.vertices, path.codes))
    }

    /// Character code → glyph index of the selected charmap, in FreeType's order.
    pub fn get_charmap(&self) -> Vec<(FT_ULong, FT_UInt)> {
        let mut charmap = Vec::new();
        let mut index = 0;
        let mut code = unsafe { ffi::FT_Get_First_Char(self.core.face, &mut index) };
        while index != 0 {
            charmap.push((code, index));
            code = unsafe { ffi::FT_Get_Next_Char(self.core.face, code, &mut index) };
        }
        charmap
    }

    pub fn get_sfnt(&self) -> Result<Vec<SfntName>, Error> {
        if self.face().face_flags & ffi::FT_FACE_FLAG_SFNT == 0 { return Err(Error::value("No SFNT name table")) }
        let count = unsafe { ffi::FT_Get_Sfnt_Name_Count(self.core.face) };
        (0..count).map(|index| {
            let mut name = std::mem::MaybeUninit::<ffi::FT_SfntName>::zeroed();
            if unsafe { ffi::FT_Get_Sfnt_Name(self.core.face, index, name.as_mut_ptr()) } != 0 {
                return Err(Error::value("Could not get SFNT name"));
            }
            let name = unsafe { name.assume_init() };
            let string = if name.string_len == 0 { Vec::new() }
                else { unsafe { std::slice::from_raw_parts(name.string, name.string_len as usize) }.to_vec() };
            Ok(SfntName { platform_id: name.platform_id, encoding_id: name.encoding_id,
                language_id: name.language_id, name_id: name.name_id, string })
        }).collect()
    }

    pub fn get_ps_font_info(&self) -> Result<PsFontInfo, Error> {
        let mut info = std::mem::MaybeUninit::<ffi::PS_FontInfoRec>::zeroed();
        if unsafe { ffi::FT_Get_PS_Font_Info(self.core.face, info.as_mut_ptr()) } != 0 {
            return Err(Error::value("Could not get PS font info"));
        }
        let info = unsafe { info.assume_init() };
        let text = |value: *mut c_char| if value.is_null() { Vec::new() }
            else { unsafe { CStr::from_ptr(value) }.to_bytes().to_vec() };
        Ok(PsFontInfo {
            version: text(info.version), notice: text(info.notice), full_name: text(info.full_name),
            family_name: text(info.family_name), weight: text(info.weight), italic_angle: info.italic_angle,
            is_fixed_pitch: info.is_fixed_pitch, underline_position: info.underline_position,
            underline_thickness: info.underline_thickness,
        })
    }

    /// One SFNT table by Matplotlib's name ("head", "maxp", "OS/2", "hhea", "vhea", "post", "pclt").
    pub fn get_sfnt_table(&self, name: &str) -> Option<SfntTable> {
        let tag = match name {
            "head" => ffi::FT_SFNT_HEAD, "maxp" => ffi::FT_SFNT_MAXP, "OS/2" => ffi::FT_SFNT_OS2,
            "hhea" => ffi::FT_SFNT_HHEA, "vhea" => ffi::FT_SFNT_VHEA, "post" => ffi::FT_SFNT_POST,
            "pclt" => ffi::FT_SFNT_PCLT, _ => return None,
        };
        let table = unsafe { ffi::FT_Get_Sfnt_Table(self.core.face, tag) };
        if table.is_null() { return None }
        // SAFETY: FreeType returns the record type documented for each tag; only
        // the declared leading fields are read.
        unsafe {
            Some(match tag {
                ffi::FT_SFNT_HEAD => SfntTable::Head(*table.cast()),
                ffi::FT_SFNT_MAXP => SfntTable::Maxp(*table.cast()),
                ffi::FT_SFNT_OS2 => SfntTable::Os2(*table.cast()),
                ffi::FT_SFNT_HHEA => SfntTable::Hhea(*table.cast()),
                ffi::FT_SFNT_VHEA => SfntTable::Vhea(*table.cast()),
                ffi::FT_SFNT_POST => SfntTable::Post(*table.cast()),
                _ => SfntTable::Pclt(*table.cast()),
            })
        }
    }

    pub fn postscript_name(&self) -> Option<&CStr> {
        let name = unsafe { ffi::FT_Get_Postscript_Name(self.core.face) };
        (!name.is_null()).then(|| unsafe { CStr::from_ptr(name) })
    }
    pub fn family_name(&self) -> Option<&CStr> {
        let name = self.face().family_name;
        (!name.is_null()).then(|| unsafe { CStr::from_ptr(name) })
    }
    pub fn style_name(&self) -> Option<&CStr> {
        let name = self.face().style_name;
        (!name.is_null()).then(|| unsafe { CStr::from_ptr(name) })
    }
    pub fn num_faces(&self) -> c_long { self.face().num_faces }
    pub fn face_flags(&self) -> c_long { self.face().face_flags }
    pub fn style_flags(&self) -> c_long { self.face().style_flags }
    pub fn num_glyphs(&self) -> c_long { self.face().num_glyphs }
    pub fn num_fixed_sizes(&self) -> c_int { self.face().num_fixed_sizes }
    pub fn num_charmaps(&self) -> c_int { self.face().num_charmaps }
    pub fn scalable(&self) -> bool { self.face().face_flags & ffi::FT_FACE_FLAG_SCALABLE != 0 }
    pub fn units_per_em(&self) -> u16 { self.face().units_per_EM }
    pub fn bbox(&self) -> [c_long; 4] { let b = self.face().bbox; [b.xMin, b.yMin, b.xMax, b.yMax] }
    pub fn ascender(&self) -> i16 { self.face().ascender }
    pub fn descender(&self) -> i16 { self.face().descender }
    pub fn height(&self) -> i16 { self.face().height }
    pub fn max_advance_width(&self) -> i16 { self.face().max_advance_width }
    pub fn max_advance_height(&self) -> i16 { self.face().max_advance_height }
    pub fn underline_position(&self) -> i16 { self.face().underline_position }
    pub fn underline_thickness(&self) -> i16 { self.face().underline_thickness }
}

struct Decomposer { vertices: Vec<f64>, codes: Vec<u8> }

unsafe fn decomposer<'a>(user: *mut c_void) -> &'a mut Decomposer { unsafe { &mut *user.cast::<Decomposer>() } }

unsafe extern "C" fn move_to(to: *const FT_Vector, user: *mut c_void) -> c_int {
    let path = unsafe { decomposer(user) };
    let to = unsafe { *to };
    if !path.vertices.is_empty() {
        // Appending CLOSEPOLY is important to make patheffects work.
        path.vertices.extend([0.0, 0.0]);
        path.codes.push(CLOSEPOLY);
    }
    path.vertices.extend([to.x as f64 * (1.0 / 64.0), to.y as f64 * (1.0 / 64.0)]);
    path.codes.push(MOVETO);
    0
}

unsafe extern "C" fn line_to(to: *const FT_Vector, user: *mut c_void) -> c_int {
    let path = unsafe { decomposer(user) };
    let to = unsafe { *to };
    path.vertices.extend([to.x as f64 * (1.0 / 64.0), to.y as f64 * (1.0 / 64.0)]);
    path.codes.push(LINETO);
    0
}

unsafe extern "C" fn conic_to(control: *const FT_Vector, to: *const FT_Vector, user: *mut c_void) -> c_int {
    let path = unsafe { decomposer(user) };
    let (control, to) = unsafe { (*control, *to) };
    path.vertices.extend([control.x as f64 * (1.0 / 64.0), control.y as f64 * (1.0 / 64.0),
        to.x as f64 * (1.0 / 64.0), to.y as f64 * (1.0 / 64.0)]);
    path.codes.extend([CURVE3, CURVE3]);
    0
}

unsafe extern "C" fn cubic_to(first: *const FT_Vector, second: *const FT_Vector, to: *const FT_Vector, user: *mut c_void) -> c_int {
    let path = unsafe { decomposer(user) };
    let (first, second, to) = unsafe { (*first, *second, *to) };
    path.vertices.extend([first.x as f64 * (1.0 / 64.0), first.y as f64 * (1.0 / 64.0),
        second.x as f64 * (1.0 / 64.0), second.y as f64 * (1.0 / 64.0),
        to.x as f64 * (1.0 / 64.0), to.y as f64 * (1.0 / 64.0)]);
    path.codes.extend([CURVE4, CURVE4, CURVE4]);
    0
}

static OUTLINE_FUNCS: ffi::FT_Outline_Funcs =
    ffi::FT_Outline_Funcs { move_to, line_to, conic_to, cubic_to, shift: 0, delta: 0 };
