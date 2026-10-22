//! Hand-written FreeType 2.6.1 declarations, checked against its headers for
//! x86_64-pc-windows-msvc (LLP64: `FT_Long`, `FT_ULong`, `FT_Pos` and
//! `FT_Fixed` are 32-bit). FreeType-owned records declare their leading
//! fields only; they are read through FreeType's pointers, never constructed.
#![allow(non_camel_case_types, non_snake_case, dead_code)]

use std::ffi::{c_char, c_int, c_long, c_short, c_uchar, c_uint, c_ulong, c_ushort, c_void};

pub type FT_Long = c_long;
pub type FT_ULong = c_ulong;
pub type FT_Pos = c_long;
pub type FT_Fixed = c_long;
pub type FT_F26Dot6 = c_long;
pub type FT_Int = c_int;
pub type FT_UInt = c_uint;
pub type FT_Int32 = i32;
pub type FT_Short = c_short;
pub type FT_UShort = c_ushort;
pub type FT_Error = c_int;
pub type FT_Bool = c_uchar;
pub type FT_Byte = u8;
pub type FT_Char = i8;

pub type FT_Library = *mut c_void;
pub type FT_Face = *mut FT_FaceRec;
pub type FT_GlyphSlot = *mut FT_GlyphSlotRec;
pub type FT_Glyph = *mut FT_GlyphRec;
pub type FT_CharMap = *mut c_void;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FT_Vector { pub x: FT_Pos, pub y: FT_Pos }

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FT_BBox { pub xMin: FT_Pos, pub yMin: FT_Pos, pub xMax: FT_Pos, pub yMax: FT_Pos }

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct FT_Matrix { pub xx: FT_Fixed, pub xy: FT_Fixed, pub yx: FT_Fixed, pub yy: FT_Fixed }

#[repr(C)]
pub struct FT_Generic { pub data: *mut c_void, pub finalizer: Option<unsafe extern "C" fn(*mut c_void)> }

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct FT_Glyph_Metrics {
    pub width: FT_Pos, pub height: FT_Pos,
    pub horiBearingX: FT_Pos, pub horiBearingY: FT_Pos, pub horiAdvance: FT_Pos,
    pub vertBearingX: FT_Pos, pub vertBearingY: FT_Pos, pub vertAdvance: FT_Pos,
}

#[repr(C)]
pub struct FT_Bitmap {
    pub rows: c_uint,
    pub width: c_uint,
    pub pitch: c_int,
    pub buffer: *mut c_uchar,
    pub num_grays: c_ushort,
    pub pixel_mode: c_uchar,
    pub palette_mode: c_uchar,
    pub palette: *mut c_void,
}

#[repr(C)]
pub struct FT_Outline {
    pub n_contours: c_short,
    pub n_points: c_short,
    pub points: *mut FT_Vector,
    pub tags: *mut c_char,
    pub contours: *mut c_short,
    pub flags: c_int,
}

#[repr(C)]
pub struct FT_FaceRec {
    pub num_faces: FT_Long,
    pub face_index: FT_Long,
    pub face_flags: FT_Long,
    pub style_flags: FT_Long,
    pub num_glyphs: FT_Long,
    pub family_name: *mut c_char,
    pub style_name: *mut c_char,
    pub num_fixed_sizes: FT_Int,
    pub available_sizes: *mut c_void,
    pub num_charmaps: FT_Int,
    pub charmaps: *mut FT_CharMap,
    pub generic: FT_Generic,
    pub bbox: FT_BBox,
    pub units_per_EM: FT_UShort,
    pub ascender: FT_Short,
    pub descender: FT_Short,
    pub height: FT_Short,
    pub max_advance_width: FT_Short,
    pub max_advance_height: FT_Short,
    pub underline_position: FT_Short,
    pub underline_thickness: FT_Short,
    pub glyph: FT_GlyphSlot,
    pub size: *mut c_void,
    pub charmap: FT_CharMap,
}

#[repr(C)]
pub struct FT_GlyphSlotRec {
    pub library: FT_Library,
    pub face: FT_Face,
    pub next: FT_GlyphSlot,
    pub reserved: FT_UInt,
    pub generic: FT_Generic,
    pub metrics: FT_Glyph_Metrics,
    pub linearHoriAdvance: FT_Fixed,
    pub linearVertAdvance: FT_Fixed,
    pub advance: FT_Vector,
    pub format: c_int,
    pub bitmap: FT_Bitmap,
    pub bitmap_left: FT_Int,
    pub bitmap_top: FT_Int,
    pub outline: FT_Outline,
}

#[repr(C)]
pub struct FT_GlyphRec {
    pub library: FT_Library,
    pub clazz: *const c_void,
    pub format: c_int,
    pub advance: FT_Vector,
}

#[repr(C)]
pub struct FT_BitmapGlyphRec {
    pub root: FT_GlyphRec,
    pub left: FT_Int,
    pub top: FT_Int,
    pub bitmap: FT_Bitmap,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub union FT_StreamDesc { pub value: c_long, pub pointer: *mut c_void }

pub type FT_Stream_IoFunc = unsafe extern "C" fn(*mut FT_StreamRec, c_ulong, *mut c_uchar, c_ulong) -> c_ulong;
pub type FT_Stream_CloseFunc = unsafe extern "C" fn(*mut FT_StreamRec);

#[repr(C)]
pub struct FT_StreamRec {
    pub base: *mut c_uchar,
    pub size: c_ulong,
    pub pos: c_ulong,
    pub descriptor: FT_StreamDesc,
    pub pathname: FT_StreamDesc,
    pub read: Option<FT_Stream_IoFunc>,
    pub close: Option<FT_Stream_CloseFunc>,
    pub memory: *mut c_void,
    pub cursor: *mut c_uchar,
    pub limit: *mut c_uchar,
}

#[repr(C)]
pub struct FT_Open_Args {
    pub flags: FT_UInt,
    pub memory_base: *const FT_Byte,
    pub memory_size: FT_Long,
    pub pathname: *mut c_char,
    pub stream: *mut FT_StreamRec,
    pub driver: *mut c_void,
    pub num_params: FT_Int,
    pub params: *mut c_void,
}

pub type FT_Outline_MoveToFunc = unsafe extern "C" fn(*const FT_Vector, *mut c_void) -> c_int;
pub type FT_Outline_ConicToFunc = unsafe extern "C" fn(*const FT_Vector, *const FT_Vector, *mut c_void) -> c_int;
pub type FT_Outline_CubicToFunc =
    unsafe extern "C" fn(*const FT_Vector, *const FT_Vector, *const FT_Vector, *mut c_void) -> c_int;

#[repr(C)]
pub struct FT_Outline_Funcs {
    pub move_to: FT_Outline_MoveToFunc,
    pub line_to: FT_Outline_MoveToFunc,
    pub conic_to: FT_Outline_ConicToFunc,
    pub cubic_to: FT_Outline_CubicToFunc,
    pub shift: c_int,
    pub delta: FT_Pos,
}

#[repr(C)]
pub struct FT_SfntName {
    pub platform_id: FT_UShort,
    pub encoding_id: FT_UShort,
    pub language_id: FT_UShort,
    pub name_id: FT_UShort,
    pub string: *mut FT_Byte,
    pub string_len: FT_UInt,
}

#[repr(C)]
pub struct PS_FontInfoRec {
    pub version: *mut c_char,
    pub notice: *mut c_char,
    pub full_name: *mut c_char,
    pub family_name: *mut c_char,
    pub weight: *mut c_char,
    pub italic_angle: FT_Long,
    pub is_fixed_pitch: FT_Bool,
    pub underline_position: FT_Short,
    pub underline_thickness: FT_UShort,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TT_Header {
    pub Table_Version: FT_Fixed,
    pub Font_Revision: FT_Fixed,
    pub CheckSum_Adjust: FT_Long,
    pub Magic_Number: FT_Long,
    pub Flags: FT_UShort,
    pub Units_Per_EM: FT_UShort,
    pub Created: [FT_Long; 2],
    pub Modified: [FT_Long; 2],
    pub xMin: FT_Short,
    pub yMin: FT_Short,
    pub xMax: FT_Short,
    pub yMax: FT_Short,
    pub Mac_Style: FT_UShort,
    pub Lowest_Rec_PPEM: FT_UShort,
    pub Font_Direction: FT_Short,
    pub Index_To_Loc_Format: FT_Short,
    pub Glyph_Data_Format: FT_Short,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TT_MaxProfile {
    pub version: FT_Fixed,
    pub numGlyphs: FT_UShort,
    pub maxPoints: FT_UShort,
    pub maxContours: FT_UShort,
    pub maxCompositePoints: FT_UShort,
    pub maxCompositeContours: FT_UShort,
    pub maxZones: FT_UShort,
    pub maxTwilightPoints: FT_UShort,
    pub maxStorage: FT_UShort,
    pub maxFunctionDefs: FT_UShort,
    pub maxInstructionDefs: FT_UShort,
    pub maxStackElements: FT_UShort,
    pub maxSizeOfInstructions: FT_UShort,
    pub maxComponentElements: FT_UShort,
    pub maxComponentDepth: FT_UShort,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TT_OS2 {
    pub version: FT_UShort,
    pub xAvgCharWidth: FT_Short,
    pub usWeightClass: FT_UShort,
    pub usWidthClass: FT_UShort,
    pub fsType: FT_UShort,
    pub ySubscriptXSize: FT_Short,
    pub ySubscriptYSize: FT_Short,
    pub ySubscriptXOffset: FT_Short,
    pub ySubscriptYOffset: FT_Short,
    pub ySuperscriptXSize: FT_Short,
    pub ySuperscriptYSize: FT_Short,
    pub ySuperscriptXOffset: FT_Short,
    pub ySuperscriptYOffset: FT_Short,
    pub yStrikeoutSize: FT_Short,
    pub yStrikeoutPosition: FT_Short,
    pub sFamilyClass: FT_Short,
    pub panose: [FT_Byte; 10],
    pub ulUnicodeRange1: FT_ULong,
    pub ulUnicodeRange2: FT_ULong,
    pub ulUnicodeRange3: FT_ULong,
    pub ulUnicodeRange4: FT_ULong,
    pub achVendID: [FT_Char; 4],
    pub fsSelection: FT_UShort,
    pub usFirstCharIndex: FT_UShort,
    pub usLastCharIndex: FT_UShort,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TT_HoriHeader {
    pub Version: FT_Fixed,
    pub Ascender: FT_Short,
    pub Descender: FT_Short,
    pub Line_Gap: FT_Short,
    pub advance_Width_Max: FT_UShort,
    pub min_Left_Side_Bearing: FT_Short,
    pub min_Right_Side_Bearing: FT_Short,
    pub xMax_Extent: FT_Short,
    pub caret_Slope_Rise: FT_Short,
    pub caret_Slope_Run: FT_Short,
    pub caret_Offset: FT_Short,
    pub Reserved: [FT_Short; 4],
    pub metric_Data_Format: FT_Short,
    pub number_Of_HMetrics: FT_UShort,
}

/// `TT_VertHeader` shares `TT_HoriHeader`'s layout field for field.
pub type TT_VertHeader = TT_HoriHeader;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TT_Postscript {
    pub FormatType: FT_Fixed,
    pub italicAngle: FT_Fixed,
    pub underlinePosition: FT_Short,
    pub underlineThickness: FT_Short,
    pub isFixedPitch: FT_ULong,
    pub minMemType42: FT_ULong,
    pub maxMemType42: FT_ULong,
    pub minMemType1: FT_ULong,
    pub maxMemType1: FT_ULong,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TT_PCLT {
    pub Version: FT_Fixed,
    pub FontNumber: FT_ULong,
    pub Pitch: FT_UShort,
    pub xHeight: FT_UShort,
    pub Style: FT_UShort,
    pub TypeFamily: FT_UShort,
    pub CapHeight: FT_UShort,
    pub SymbolSet: FT_UShort,
    pub TypeFace: [FT_Char; 16],
    pub CharacterComplement: [FT_Char; 8],
    pub FileName: [FT_Char; 6],
    pub StrokeWeight: FT_Char,
    pub WidthType: FT_Char,
    pub SerifStyle: FT_Byte,
    pub Reserved: FT_Byte,
}

pub const FT_OPEN_STREAM: FT_UInt = 0x2;
pub const FT_FACE_FLAG_SFNT: FT_Long = 1 << 3;
pub const FT_FACE_FLAG_KERNING: FT_Long = 1 << 6;
pub const FT_FACE_FLAG_GLYPH_NAMES: FT_Long = 1 << 9;
pub const FT_FACE_FLAG_EXTERNAL_STREAM: FT_Long = 1 << 10;
pub const FT_FACE_FLAG_SCALABLE: FT_Long = 1 << 0;
pub const FT_KERNING_DEFAULT: FT_UInt = 0;
pub const FT_GLYPH_BBOX_SUBPIXELS: FT_UInt = 0;
pub const FT_RENDER_MODE_NORMAL: c_int = 0;
pub const FT_RENDER_MODE_MONO: c_int = 2;
pub const FT_PIXEL_MODE_MONO: c_uchar = 1;
pub const FT_PIXEL_MODE_GRAY: c_uchar = 2;
pub const FT_SFNT_HEAD: c_int = 0;
pub const FT_SFNT_MAXP: c_int = 1;
pub const FT_SFNT_OS2: c_int = 2;
pub const FT_SFNT_HHEA: c_int = 3;
pub const FT_SFNT_VHEA: c_int = 4;
pub const FT_SFNT_POST: c_int = 5;
pub const FT_SFNT_PCLT: c_int = 6;

unsafe extern "C" {
    pub fn FT_Init_FreeType(alibrary: *mut FT_Library) -> FT_Error;
    pub fn FT_Library_Version(library: FT_Library, amajor: *mut FT_Int, aminor: *mut FT_Int, apatch: *mut FT_Int);
    pub fn FT_Open_Face(library: FT_Library, args: *const FT_Open_Args, face_index: FT_Long, aface: *mut FT_Face) -> FT_Error;
    pub fn FT_Done_Face(face: FT_Face) -> FT_Error;
    pub fn FT_Set_Char_Size(face: FT_Face, char_width: FT_F26Dot6, char_height: FT_F26Dot6,
        horz_resolution: FT_UInt, vert_resolution: FT_UInt) -> FT_Error;
    pub fn FT_Set_Transform(face: FT_Face, matrix: *mut FT_Matrix, delta: *mut FT_Vector);
    pub fn FT_Set_Charmap(face: FT_Face, charmap: FT_CharMap) -> FT_Error;
    pub fn FT_Select_Charmap(face: FT_Face, encoding: c_uint) -> FT_Error;
    pub fn FT_Get_Kerning(face: FT_Face, left_glyph: FT_UInt, right_glyph: FT_UInt, kern_mode: FT_UInt,
        akerning: *mut FT_Vector) -> FT_Error;
    pub fn FT_Get_Char_Index(face: FT_Face, charcode: FT_ULong) -> FT_UInt;
    pub fn FT_Load_Glyph(face: FT_Face, glyph_index: FT_UInt, load_flags: FT_Int32) -> FT_Error;
    pub fn FT_Get_Glyph(slot: FT_GlyphSlot, aglyph: *mut FT_Glyph) -> FT_Error;
    pub fn FT_Done_Glyph(glyph: FT_Glyph);
    pub fn FT_Glyph_Transform(glyph: FT_Glyph, matrix: *mut FT_Matrix, delta: *mut FT_Vector) -> FT_Error;
    pub fn FT_Glyph_Get_CBox(glyph: FT_Glyph, bbox_mode: FT_UInt, acbox: *mut FT_BBox);
    pub fn FT_Vector_Transform(vec: *mut FT_Vector, matrix: *const FT_Matrix);
    pub fn FT_Glyph_To_Bitmap(the_glyph: *mut FT_Glyph, render_mode: c_int, origin: *mut FT_Vector,
        destroy: FT_Bool) -> FT_Error;
    pub fn FT_Outline_Decompose(outline: *mut FT_Outline, func_interface: *const FT_Outline_Funcs,
        user: *mut c_void) -> FT_Error;
    pub fn FT_Get_Glyph_Name(face: FT_Face, glyph_index: FT_UInt, buffer: *mut c_void, buffer_max: FT_UInt) -> FT_Error;
    pub fn FT_Get_Name_Index(face: FT_Face, glyph_name: *mut c_char) -> FT_UInt;
    pub fn FT_Get_First_Char(face: FT_Face, agindex: *mut FT_UInt) -> FT_ULong;
    pub fn FT_Get_Next_Char(face: FT_Face, char_code: FT_ULong, agindex: *mut FT_UInt) -> FT_ULong;
    pub fn FT_Get_Sfnt_Name_Count(face: FT_Face) -> FT_UInt;
    pub fn FT_Get_Sfnt_Name(face: FT_Face, idx: FT_UInt, aname: *mut FT_SfntName) -> FT_Error;
    pub fn FT_Get_PS_Font_Info(face: FT_Face, afont_info: *mut PS_FontInfoRec) -> FT_Error;
    pub fn FT_Get_Sfnt_Table(face: FT_Face, tag: c_int) -> *mut c_void;
    pub fn FT_Get_Postscript_Name(face: FT_Face) -> *const c_char;
}

/// FreeType 2.6.1's `fterrdef.h` strings, as Matplotlib's `ft_error_string`.
pub fn error_string(error: FT_Error) -> Option<&'static str> {
    Some(match error {
        0x00 => "no error",
        0x01 => "cannot open resource",
        0x02 => "unknown file format",
        0x03 => "broken file",
        0x04 => "invalid FreeType version",
        0x05 => "module version is too low",
        0x06 => "invalid argument",
        0x07 => "unimplemented feature",
        0x08 => "broken table",
        0x09 => "broken offset within table",
        0x0A => "array allocation size too large",
        0x0B => "missing module",
        0x0C => "missing property",
        0x10 => "invalid glyph index",
        0x11 => "invalid character code",
        0x12 => "unsupported glyph image format",
        0x13 => "cannot render this glyph format",
        0x14 => "invalid outline",
        0x15 => "invalid composite glyph",
        0x16 => "too many hints",
        0x17 => "invalid pixel size",
        0x20 => "invalid object handle",
        0x21 => "invalid library handle",
        0x22 => "invalid module handle",
        0x23 => "invalid face handle",
        0x24 => "invalid size handle",
        0x25 => "invalid glyph slot handle",
        0x26 => "invalid charmap handle",
        0x27 => "invalid cache manager handle",
        0x28 => "invalid stream handle",
        0x30 => "too many modules",
        0x31 => "too many extensions",
        0x40 => "out of memory",
        0x41 => "unlisted object",
        0x51 => "cannot open stream",
        0x52 => "invalid stream seek",
        0x53 => "invalid stream skip",
        0x54 => "invalid stream read",
        0x55 => "invalid stream operation",
        0x56 => "invalid frame operation",
        0x57 => "nested frame access",
        0x58 => "invalid frame read",
        0x60 => "raster uninitialized",
        0x61 => "raster corrupted",
        0x62 => "raster overflow",
        0x63 => "negative height while rastering",
        0x70 => "too many registered caches",
        0x80 => "invalid opcode",
        0x81 => "too few arguments",
        0x82 => "stack overflow",
        0x83 => "code overflow",
        0x84 => "bad argument",
        0x85 => "division by zero",
        0x86 => "invalid reference",
        0x87 => "found debug opcode",
        0x88 => "found ENDF opcode in execution stream",
        0x89 => "nested DEFS",
        0x8A => "invalid code range",
        0x8B => "execution context too long",
        0x8C => "too many function definitions",
        0x8D => "too many instruction definitions",
        0x8E => "SFNT font table missing",
        0x8F => "horizontal header (hhea) table missing",
        0x90 => "locations (loca) table missing",
        0x91 => "name table missing",
        0x92 => "character map (cmap) table missing",
        0x93 => "horizontal metrics (hmtx) table missing",
        0x94 => "PostScript (post) table missing",
        0x95 => "invalid horizontal metrics",
        0x96 => "invalid character map (cmap) format",
        0x97 => "invalid ppem value",
        0x98 => "invalid vertical metrics",
        0x99 => "could not find context",
        0x9A => "invalid PostScript (post) table format",
        0x9B => "invalid PostScript (post) table",
        0xA0 => "opcode syntax error",
        0xA1 => "argument stack underflow",
        0xA2 => "ignore",
        0xA3 => "no Unicode glyph name found",
        0xA4 => "glyph to big for hinting",
        0xB0 => "`STARTFONT' field missing",
        0xB1 => "`FONT' field missing",
        0xB2 => "`SIZE' field missing",
        0xB3 => "`FONTBOUNDINGBOX' field missing",
        0xB4 => "`CHARS' field missing",
        0xB5 => "`STARTCHAR' field missing",
        0xB6 => "`ENCODING' field missing",
        0xB7 => "`BBX' field missing",
        0xB8 => "`BBX' too big",
        0xB9 => "Font header corrupted or missing fields",
        0xBA => "Font glyphs corrupted or missing fields",
        _ => return None,
    })
}

#[cfg(test)]
mod layout {
    use super::*;
    use std::mem::{offset_of, size_of};

    /// Offsets printed by a C program compiled against the same 2.6.1 headers.
    #[test]
    fn matches_the_c_headers() {
        assert_eq!(offset_of!(FT_FaceRec, face_flags), 8);
        assert_eq!(offset_of!(FT_FaceRec, num_glyphs), 16);
        assert_eq!(offset_of!(FT_FaceRec, family_name), 24);
        assert_eq!(offset_of!(FT_FaceRec, num_fixed_sizes), 40);
        assert_eq!(offset_of!(FT_FaceRec, num_charmaps), 56);
        assert_eq!(offset_of!(FT_FaceRec, charmaps), 64);
        assert_eq!(offset_of!(FT_FaceRec, bbox), 88);
        assert_eq!(offset_of!(FT_FaceRec, units_per_EM), 104);
        assert_eq!(offset_of!(FT_FaceRec, underline_thickness), 118);
        assert_eq!(offset_of!(FT_FaceRec, glyph), 120);
        assert_eq!(offset_of!(FT_FaceRec, charmap), 136);
        assert_eq!(offset_of!(FT_GlyphSlotRec, reserved), 24);
        assert_eq!(offset_of!(FT_GlyphSlotRec, metrics), 48);
        assert_eq!(offset_of!(FT_GlyphSlotRec, linearHoriAdvance), 80);
        assert_eq!(offset_of!(FT_GlyphSlotRec, advance), 88);
        assert_eq!(offset_of!(FT_GlyphSlotRec, format), 96);
        assert_eq!(offset_of!(FT_GlyphSlotRec, bitmap), 104);
        assert_eq!(offset_of!(FT_GlyphSlotRec, bitmap_left), 144);
        assert_eq!(offset_of!(FT_GlyphSlotRec, outline), 152);
        assert_eq!((size_of::<FT_Bitmap>(), offset_of!(FT_Bitmap, buffer), offset_of!(FT_Bitmap, pixel_mode)), (40, 16, 26));
        assert_eq!((size_of::<FT_Outline>(), offset_of!(FT_Outline, points), offset_of!(FT_Outline, flags)), (40, 8, 32));
        assert_eq!((size_of::<FT_Outline_Funcs>(), offset_of!(FT_Outline_Funcs, delta)), (40, 36));
        assert_eq!((size_of::<FT_GlyphRec>(), offset_of!(FT_GlyphRec, advance)), (32, 20));
        assert_eq!((size_of::<FT_BitmapGlyphRec>(), offset_of!(FT_BitmapGlyphRec, bitmap)), (80, 40));
        assert_eq!((size_of::<FT_StreamRec>(), offset_of!(FT_StreamRec, read), offset_of!(FT_StreamRec, limit)), (72, 32, 64));
        assert_eq!((size_of::<FT_Open_Args>(), offset_of!(FT_Open_Args, stream), offset_of!(FT_Open_Args, params)), (64, 32, 56));
        assert_eq!((size_of::<FT_SfntName>(), offset_of!(FT_SfntName, string_len)), (24, 16));
        assert_eq!((size_of::<PS_FontInfoRec>(), offset_of!(PS_FontInfoRec, underline_thickness)), (56, 48));
        assert_eq!((size_of::<TT_Header>(), offset_of!(TT_Header, Created), offset_of!(TT_Header, Glyph_Data_Format)), (56, 20, 52));
        assert_eq!((size_of::<TT_MaxProfile>(), offset_of!(TT_MaxProfile, maxComponentDepth)), (32, 30));
        assert_eq!((offset_of!(TT_OS2, panose), offset_of!(TT_OS2, achVendID), offset_of!(TT_OS2, usLastCharIndex)), (32, 60, 68));
        assert_eq!((offset_of!(TT_HoriHeader, Reserved), offset_of!(TT_HoriHeader, number_Of_HMetrics)), (24, 34));
        assert_eq!((size_of::<TT_Postscript>(), offset_of!(TT_Postscript, isFixedPitch)), (32, 12));
        assert_eq!((offset_of!(TT_PCLT, TypeFace), offset_of!(TT_PCLT, StrokeWeight), offset_of!(TT_PCLT, SerifStyle)), (20, 50, 52));
    }
}
