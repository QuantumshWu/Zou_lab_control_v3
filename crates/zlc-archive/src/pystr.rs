//! Python text that Rust strings cannot hold as is.
//!
//! A Python `str` may carry lone UTF-16 surrogates (`"\ud800"` in JSON, a
//! surrogate code unit in a NumPy `U` array). Archive strings keep them in
//! a Rust `String` by one reversible mapping: surrogate U+D800 + k is the
//! private-use character U+10F800 + k, a genuine character at or above
//! U+10F7FF is written as U+10F7FF followed by itself, and a UTF-32 unit
//! beyond U+10FFFF (CPython keeps one from a NumPy string of two or more
//! characters) as U+10F7FF and three characters U+100000 + its 12, 16 and 4
//! bits. [`code_points`] undoes the mapping wherever Python's code points
//! matter (UTF-32 planes, JSON escapes, `repr`, key order).

use crate::error::{ArchiveError, ErrorKind, Result};

const ESCAPE: u32 = 0x10_F7FF;
const SURROGATES: u32 = 0x10_F800;

/// Append one Python code point (0..=0x10FFFF, surrogates included).
pub fn push(text: &mut String, code: u32) {
    let private = |bits: u32| char::from_u32(0x10_0000 + bits).unwrap();
    match code {
        0xD800..=0xDFFF => text.push(char::from_u32(SURROGATES + (code - 0xD800)).unwrap()),
        ESCAPE..=0x10_FFFF => { text.push(char::from_u32(ESCAPE).unwrap()); text.push(char::from_u32(code).unwrap()); }
        0x11_0000.. => {
            text.push(char::from_u32(ESCAPE).unwrap());
            text.push(private(code >> 20));
            text.push(private((code >> 4) & 0xFFFF));
            text.push(private(code & 0xF));
        }
        _ => text.push(char::from_u32(code).unwrap()),
    }
}

/// A Rust string as archive text (its own characters, escaped where needed).
pub fn from_rust(text: &str) -> String {
    if !text.chars().any(|c| c as u32 >= ESCAPE) { return text.to_owned(); }
    let mut out = String::with_capacity(text.len() + 4);
    for c in text.chars() { push(&mut out, c as u32); }
    out
}

/// The Python code points of archive text.
pub fn code_points(text: &str) -> impl Iterator<Item = u32> + '_ {
    let mut chars = text.chars();
    std::iter::from_fn(move || {
        let c = chars.next()? as u32;
        Some(match c {
            ESCAPE => match chars.next().map(|next| next as u32) {
                Some(high) if high < ESCAPE => {
                    let middle = chars.next().map_or(0, |next| next as u32 - 0x10_0000);
                    let low = chars.next().map_or(0, |next| next as u32 - 0x10_0000);
                    ((high - 0x10_0000) << 20) | (middle << 4) | low
                }
                Some(next) => next,
                None => ESCAPE,
            },
            SURROGATES..=0x10_FFFF => 0xD800 + (c - SURROGATES),
            other => other,
        })
    })
}

/// `len(text)` in Python.
pub fn len(text: &str) -> usize { code_points(text).count() }

/// Whether `text` holds a lone surrogate (it is not valid UTF-8 text).
pub fn has_surrogates(text: &str) -> bool { code_points(text).any(|code| (0xD800..=0xDFFF).contains(&code)) }

/// `bytes.decode("utf-8")` with Python's `UnicodeDecodeError` message.
pub fn decode_utf8(bytes: &[u8]) -> Result<String> {
    match std::str::from_utf8(bytes) {
        Ok(text) => Ok(from_rust(text)),
        Err(error) => {
            let start = error.valid_up_to();
            let lead = bytes[start];
            let (end, reason) = match error.error_len() {
                Some(width) if width == 1 && !matches!(lead, 0xC2..=0xF4) => (start + 1, "invalid start byte"),
                Some(width) => (start + width, "invalid continuation byte"),
                None => (bytes.len(), "unexpected end of data"),
            };
            let place = if end - start == 1 { format!("byte 0x{lead:02x} in position {start}") } else { format!("bytes in position {start}-{}", end - 1) };
            Err(ArchiveError::new(ErrorKind::UnicodeDecode, format!("'utf-8' codec can't decode {place}: {reason}")))
        }
    }
}
