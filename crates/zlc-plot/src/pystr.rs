//! Python's `str` semantics where ported code must agree with it exactly:
//! `str.isspace`, `str.strip`, `repr(str)` and `float(str)`, from CPython's
//! own character tables (`pystr_tables.rs`, written by
//! `tests/gen/pystr_tables.py`).

#[path = "pystr_tables.rs"]
mod tables;

fn in_ranges(table: &[(u32, u32)], code: u32) -> bool {
    let index = table.partition_point(|&(_, last)| last < code);
    table.get(index).is_some_and(|&(first, _)| first <= code)
}

/// `str.isspace` for one character.
pub fn isspace(c: char) -> bool { in_ranges(tables::SPACE, c as u32) }

/// `str.isprintable` for one character.
pub fn isprintable(c: char) -> bool { !in_ranges(tables::UNPRINTABLE, c as u32) }

/// A non-ASCII decimal digit's value (`Py_UNICODE_TODECIMAL`).
fn decimal(c: char) -> Option<u8> {
    let code = c as u32;
    let index = tables::DECIMAL.partition_point(|&(_, last, _)| last < code);
    let &(first, _, value) = tables::DECIMAL.get(index).filter(|&&(first, _, _)| first <= code)?;
    Some(value + (code - first) as u8)
}

/// `text.strip()`.
pub fn strip(text: &str) -> &str { text.trim_matches(isspace) }

/// `repr(text)`.
pub fn repr(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') { '"' } else { '\'' };
    let mut out = String::with_capacity(text.len() + 2);
    out.push(quote);
    for c in text.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if isprintable(c) => out.push(c),
            c => {
                let code = c as u32;
                out.push_str(&if code <= 0xff {
                    format!("\\x{code:02x}")
                } else if code <= 0xffff {
                    format!("\\u{code:04x}")
                } else {
                    format!("\\U{code:08x}")
                });
            }
        }
    }
    out.push(quote);
    out
}

/// `float(text)` for a `str`: CPython's `PyFloat_FromString`.  Non-ASCII
/// whitespace becomes a space and any Unicode decimal digit its ASCII digit
/// (the first other character ends the text with an invalid mark); ASCII
/// whitespace is stripped at both ends; an underscore must sit between two
/// digits; what remains is `strtod`'s grammar (with `inf`, `infinity` and
/// `nan` in any case), correctly rounded.
pub fn float(text: &str) -> Option<f64> {
    let mut ascii = String::with_capacity(text.len());
    for c in text.chars() {
        if (c as u32) < 127 {
            ascii.push(c);
        } else if isspace(c) {
            ascii.push(' ');
        } else if let Some(value) = decimal(c) {
            ascii.push((b'0' + value) as char);
        } else {
            ascii.push('?');
            break;
        }
    }
    let body = ascii.trim_matches(|c: char| matches!(c, ' ' | '\t' | '\n' | '\x0b' | '\x0c' | '\r'));
    if body.is_empty() {
        return None;
    }
    let mut cleaned = String::with_capacity(body.len());
    let mut previous = '\0';
    for c in body.chars() {
        if c == '_' {
            if !previous.is_ascii_digit() {
                return None;
            }
        } else {
            if previous == '_' && !c.is_ascii_digit() {
                return None;
            }
            cleaned.push(c);
        }
        previous = c;
    }
    if previous == '_' {
        return None;
    }
    // strtod's grammar: a sign, then digits with at most one point and an
    // exponent, or a special value; Rust's parser takes exactly this.
    if !cleaned.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'+' | b'-' | b'.')) {
        return None;
    }
    let unsigned = cleaned.strip_prefix(['+', '-']).unwrap_or(&cleaned);
    let special = ["inf", "infinity", "nan"].iter().any(|s| unsigned.eq_ignore_ascii_case(s));
    if !special && unsigned.bytes().any(|b| b.is_ascii_alphabetic() && !matches!(b, b'e' | b'E')) {
        return None;
    }
    cleaned.parse::<f64>().ok()
}
