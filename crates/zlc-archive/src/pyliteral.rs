//! What numpy's `_read_array_header` does with a header's text: CPython
//! 3.13's `ast.literal_eval` (the C tokenizer, the expression grammar, the
//! literal conversion) and, for a Python 2 header, `_filter_header`'s
//! `tokenize` round trip. Outcomes are the Python ones: a value, a
//! `SyntaxError`, a malformed-node `ValueError`, an unhashable `TypeError`,
//! a `RecursionError`, or a `TokenError`/`IndentationError` from tokenize.

use crate::pystr;

/// A Python int: exact while it fits, its decimal digits otherwise.
#[derive(Clone, Debug, PartialEq)]
pub enum Int { Small(i128), Big { negative: bool, digits: String } }

impl Int {
    fn negate(self) -> Int {
        match self {
            Int::Small(value) => value.checked_neg().map(Int::Small).unwrap_or(Int::Big { negative: false, digits: value.unsigned_abs().to_string() }),
            Int::Big { negative, digits } => Int::Big { negative: !negative, digits },
        }
    }
    pub fn as_i128(&self) -> Option<i128> { match self { Int::Small(value) => Some(*value), Int::Big { .. } => None } }
    pub fn negative(&self) -> bool { match self { Int::Small(value) => *value < 0, Int::Big { negative, .. } => *negative } }
    pub fn repr(&self) -> String {
        match self { Int::Small(value) => value.to_string(), Int::Big { negative, digits } => format!("{}{digits}", if *negative { "-" } else { "" }) }
    }
    fn to_f64(&self) -> f64 { match self { Int::Small(value) => *value as f64, Int::Big { negative, digits } => { let v: f64 = digits.parse().unwrap_or(f64::INFINITY); if *negative { -v } else { v } } } }
}

/// The digits of `text` (no underscores) in `radix`, exactly.
fn parse_int(text: &str, radix: u32) -> Int {
    if let Ok(value) = i128::from_str_radix(text, radix) { return Int::Small(value); }
    // Base 10^9 limbs, least significant first.
    let mut limbs: Vec<u64> = vec![0];
    for c in text.chars() {
        let mut carry = u64::from(c.to_digit(radix).unwrap());
        for limb in limbs.iter_mut() {
            let value = *limb * u64::from(radix) + carry;
            *limb = value % 1_000_000_000;
            carry = value / 1_000_000_000;
        }
        while carry > 0 { limbs.push(carry % 1_000_000_000); carry /= 1_000_000_000; }
    }
    let mut digits = limbs.last().unwrap().to_string();
    for limb in limbs.iter().rev().skip(1) { digits.push_str(&format!("{limb:09}")); }
    Int::Big { negative: false, digits }
}

/// A Python value a literal can produce.
#[derive(Clone, Debug)]
pub enum Obj { None, Ellipsis, Bool(bool), Int(Int), Float(f64), Complex(f64, f64), Str(String), Bytes(Vec<u8>), Tuple(Vec<Obj>), List(Vec<Obj>), Set(Vec<Obj>), Dict(Vec<(Obj, Obj)>) }

impl Obj {
    pub fn type_name(&self) -> &'static str {
        match self {
            Obj::None => "NoneType", Obj::Ellipsis => "ellipsis", Obj::Bool(_) => "bool", Obj::Int(_) => "int", Obj::Float(_) => "float",
            Obj::Complex(..) => "complex", Obj::Str(_) => "str", Obj::Bytes(_) => "bytes", Obj::Tuple(_) => "tuple", Obj::List(_) => "list",
            Obj::Set(_) => "set", Obj::Dict(_) => "dict",
        }
    }

    /// The first unhashable value inside (`hash(value)`), if any.
    pub fn unhashable(&self) -> Option<&'static str> {
        match self {
            Obj::List(_) | Obj::Set(_) | Obj::Dict(_) => Some(self.type_name()),
            Obj::Tuple(items) => items.iter().find_map(Obj::unhashable),
            _ => None,
        }
    }

    fn real(&self) -> Option<f64> {
        match self { Obj::Bool(flag) => Some(f64::from(u8::from(*flag))), Obj::Int(value) => Some(value.to_f64()), Obj::Float(value) => Some(*value), _ => None }
    }

    /// Python `==` between hashable values.
    pub fn equals(&self, other: &Obj) -> bool {
        match (self, other) {
            (Obj::None, Obj::None) | (Obj::Ellipsis, Obj::Ellipsis) => true,
            (Obj::Str(a), Obj::Str(b)) => a == b,
            (Obj::Bytes(a), Obj::Bytes(b)) => a == b,
            (Obj::Tuple(a), Obj::Tuple(b)) => a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.equals(y)),
            (Obj::Int(a), Obj::Int(b)) => a == b,
            (Obj::Bool(a), Obj::Bool(b)) => a == b,
            (Obj::Bool(a), Obj::Int(Int::Small(b))) | (Obj::Int(Int::Small(b)), Obj::Bool(a)) => i128::from(*a) == *b,
            (Obj::Complex(ar, ai), Obj::Complex(br, bi)) => ar == br && ai == bi,
            (Obj::Complex(re, im), other) | (other, Obj::Complex(re, im)) => *im == 0.0 && other.real() == Some(*re),
            (a, b) => match (a, b) {
                (Obj::Float(x), y) | (y, Obj::Float(x)) => match y { Obj::Int(Int::Small(v)) => (*v as f64) == *x && (*x as i128) == *v, other => other.real() == Some(*x) },
                _ => false,
            },
        }
    }

    pub fn repr(&self) -> String {
        match self {
            Obj::None => "None".into(),
            Obj::Ellipsis => "Ellipsis".into(),
            Obj::Bool(flag) => if *flag { "True" } else { "False" }.into(),
            Obj::Int(value) => value.repr(),
            Obj::Float(value) => zlc_core::pyfmt::repr(*value),
            Obj::Complex(re, im) => complex_repr(*re, *im),
            Obj::Str(text) => crate::json::py_repr_str(text),
            Obj::Bytes(bytes) => crate::json::py_repr_bytes(bytes),
            Obj::Tuple(items) if items.len() == 1 => format!("({},)", items[0].repr()),
            Obj::Tuple(items) => format!("({})", items.iter().map(Obj::repr).collect::<Vec<_>>().join(", ")),
            Obj::List(items) => format!("[{}]", items.iter().map(Obj::repr).collect::<Vec<_>>().join(", ")),
            Obj::Set(items) if items.is_empty() => "set()".into(),
            Obj::Set(items) => format!("{{{}}}", items.iter().map(Obj::repr).collect::<Vec<_>>().join(", ")),
            Obj::Dict(pairs) => format!("{{{}}}", pairs.iter().map(|(k, v)| format!("{}: {}", k.repr(), v.repr())).collect::<Vec<_>>().join(", ")),
        }
    }
}

/// `repr(complex(re, im))`: parts without a forced `.0`.
fn complex_repr(re: f64, im: f64) -> String {
    let part = |value: f64| { let text = zlc_core::pyfmt::repr(value); if text.ends_with(".0") && !text.contains('e') { text[..text.len() - 2].to_owned() } else { text } };
    if re == 0.0 && re.is_sign_positive() { return format!("{}j", part(im)); }
    let imag = part(im);
    let sign = if imag.starts_with('-') { "" } else { "+" };
    format!("({}{sign}{imag}j)", part(re))
}

/// Python's `<` between two literal values, or the TypeError message.
pub fn less(a: &Obj, b: &Obj) -> Result<bool, String> {
    let refuse = || format!("'<' not supported between instances of '{}' and '{}'", a.type_name(), b.type_name());
    match (a, b) {
        (Obj::Str(x), Obj::Str(y)) => Ok(pystr::code_points(x).lt(pystr::code_points(y))),
        (Obj::Bytes(x), Obj::Bytes(y)) => Ok(x < y),
        (Obj::Tuple(x), Obj::Tuple(y)) | (Obj::List(x), Obj::List(y)) => {
            for (p, q) in x.iter().zip(y) {
                if !p.equals(q) { return less(p, q); }
            }
            Ok(x.len() < y.len())
        }
        (Obj::Int(Int::Small(x)), Obj::Int(Int::Small(y))) => Ok(x < y),
        _ => match (a.real(), b.real()) {
            (Some(x), Some(y)) if !matches!(a, Obj::Complex(..)) && !matches!(b, Obj::Complex(..)) => Ok(x < y),
            _ => Err(refuse()),
        },
    }
}

/// `sorted(items)` (CPython 3.13 `list.sort` for short lists: the first
/// run, then binary insertion), or the comparison's TypeError.
pub fn sorted(mut items: Vec<Obj>) -> Result<Vec<Obj>, String> {
    let count = items.len();
    if count < 2 { return Ok(items); }
    let mut n = 1;
    while n < count && !less(&items[n], &items[n - 1])? { n += 1; }
    if n < count {
        let descending_from_start = n == 1 || {
            if less(&items[0], &items[n - 1])? { false } else { items[..n].reverse(); true }
        };
        if descending_from_start {
            let mut equal = 0;
            n += 1;
            while n < count {
                if less(&items[n], &items[n - 1])? {
                    if equal > 0 { let from = n - equal - 1; items[from..n].reverse(); equal = 0; }
                } else if less(&items[n - 1], &items[n])? {
                    break;
                } else {
                    equal += 1;
                }
                n += 1;
            }
            if equal > 0 { let from = n - equal - 1; items[from..n].reverse(); }
            items[..n].reverse();
            while n < count && !less(&items[n], &items[n - 1])? { n += 1; }
        }
    }
    for ok in n..count {
        let (mut low, mut high) = (0, ok);
        while low < high {
            let middle = (low + high) >> 1;
            if less(&items[ok], &items[middle])? { high = middle; } else { low = middle + 1; }
        }
        let pivot = items.remove(ok);
        items.insert(low, pivot);
    }
    Ok(items)
}

// ------------------------------------------------------------- tokenizer

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind { Name, Number, Str, Op, Comment, Newline, Nl, Indent, Dedent, End }

#[derive(Clone, Debug)]
struct Token { kind: Kind, text: String, start: (usize, usize), end: (usize, usize) }

/// A tokenizer refusal: `TokenError(msg, (line, col))`, or an indentation
/// error tokenize re-raises as itself.
#[derive(Clone, Debug, PartialEq)]
pub enum TokenFailure { Token(String, usize, usize), Indentation(String, usize), Tab(usize) }

const MAXLEVEL: usize = 200;

fn identifier_start(c: char) -> bool { c.is_ascii_alphabetic() || c == '_' || c as u32 >= 128 }
fn identifier_char(c: char) -> bool { identifier_start(c) || c.is_ascii_digit() }

/// `str.isidentifier()` for one character after the first (`first` for the start).
fn valid_identifier_char(c: char, first: bool) -> bool {
    if c.is_ascii() { return c.is_ascii_alphabetic() || c == '_' || (!first && c.is_ascii_digit()); }
    match c as u32 {
        0xAA | 0xB5 | 0xBA | 0xC0..=0xD6 | 0xD8..=0xF6 | 0xF8..=0xFF => true,
        0xB7 => !first,
        0x80..=0xFF => false,
        _ => c.is_alphabetic() || (!first && (c.is_alphanumeric() || c as u32 == 0x200C || c as u32 == 0x200D)),
    }
}

/// The C tokenizer in `tokenize` mode (`extra_tokens=True`).
fn tokenize(source: &[char]) -> Result<Vec<Token>, TokenFailure> {
    let mut tokens = Vec::new();
    let mut line_starts = vec![0usize];
    for (index, c) in source.iter().enumerate() { if *c == '\n' { line_starts.push(index + 1); } }
    let line_count = if source.last() == Some(&'\n') { line_starts.len() - 1 } else { line_starts.len() }.max(1);
    // Each line is checked for NUL as the tokenizer reads it, after the
    // tokens of the lines before it.
    let mut checked = 0usize;
    let mut read_through = |index: usize| -> Result<(), TokenFailure> {
        let line = line_starts.iter().rposition(|start| *start <= index).unwrap();
        while checked <= line {
            let end = line_starts.get(checked + 1).copied().unwrap_or(source.len());
            if source[line_starts[checked]..end].contains(&'\0') {
                return Err(TokenFailure::Token("source code cannot contain null bytes".into(), checked + 1, 0));
            }
            checked += 1;
        }
        Ok(())
    };
    let position = |index: usize| -> (usize, usize) {
        let line = line_starts.iter().rposition(|start| *start <= index).unwrap();
        (line + 1, index - line_starts[line])
    };
    let line_length = |index: usize| -> usize {
        let line = line_starts.iter().rposition(|start| *start <= index).unwrap();
        let end = source[line_starts[line]..].iter().position(|c| *c == '\n').map_or(source.len(), |p| line_starts[line] + p + 1);
        end - line_starts[line] + usize::from(end == source.len() && source.last() != Some(&'\n'))
    };
    let mut at = 0usize;
    let mut levels: Vec<char> = Vec::new();
    let mut indents: Vec<(usize, usize)> = vec![(0, 0)];
    let mut line_begin = true;
    let mut continued = false;
    let error = |message: &str, index: usize| -> TokenFailure { let (line, col) = position(index); TokenFailure::Token(message.into(), line, col) };
    loop {
        if at < source.len() { read_through(at)?; }
        if line_begin && levels.is_empty() && !continued {
            // Indentation of a new logical line.
            let (mut col, mut alt) = (0usize, 0usize);
            let start = at;
            while at < source.len() {
                match source[at] { ' ' => { col += 1; alt += 1; } '\t' => { col = (col / 8 + 1) * 8; alt += 1; } '\x0c' => { col = 0; alt = 0; } _ => break }
                at += 1;
            }
            line_begin = false;
            let blank = at >= source.len() || matches!(source[at], '#' | '\n') || (source[at] == '\r' && source.get(at + 1) == Some(&'\n'));
            if !blank {
                let (top, top_alt) = *indents.last().unwrap();
                if col == top {
                    if alt != top_alt { return Err(TokenFailure::Tab(position(at).0)); }
                } else if col > top {
                    if alt <= top_alt { return Err(TokenFailure::Tab(position(at).0)); }
                    indents.push((col, alt));
                    tokens.push(Token { kind: Kind::Indent, text: source[start..at].iter().collect(), start: position(start), end: position(at) });
                } else {
                    while indents.len() > 1 && col < indents.last().unwrap().0 {
                        indents.pop();
                        tokens.push(Token { kind: Kind::Dedent, text: String::new(), start: position(at), end: position(at) });
                    }
                    let (top, top_alt) = *indents.last().unwrap();
                    if col != top { return Err(TokenFailure::Indentation("unindent does not match any outer indentation level".into(), position(at).0)); }
                    if alt != top_alt { return Err(TokenFailure::Tab(position(at).0)); }
                }
            }
        }
        continued = false;
        while at < source.len() && matches!(source[at], ' ' | '\t' | '\x0c') { at += 1; }
        if at >= source.len() {
            if !levels.is_empty() { return Err(TokenFailure::Token("unexpected EOF in multi-line statement".into(), line_count, 0)); }
            let last_line = tokens.last().map(|token: &Token| token.end.0).unwrap_or(1);
            let needs_newline = tokens.last().is_some_and(|token| !matches!(token.kind, Kind::Newline | Kind::Nl | Kind::Dedent | Kind::Indent));
            if needs_newline {
                let (line, col) = position(source.len());
                let kind = if tokens.iter().rev().take_while(|t| t.start.0 == line).all(|t| t.kind == Kind::Comment) { Kind::Nl } else { Kind::Newline };
                tokens.push(Token { kind, text: String::new(), start: (line, col), end: (line, col + 1) });
            }
            let end_line = line_count + 1;
            let _ = last_line;
            for _ in 1..indents.len() { tokens.push(Token { kind: Kind::Dedent, text: String::new(), start: (end_line, 0), end: (end_line, 0) }); }
            tokens.push(Token { kind: Kind::End, text: String::new(), start: (end_line, 0), end: (end_line, 0) });
            return Ok(tokens);
        }
        let start = at;
        let c = source[at];
        if c == '#' {
            while at < source.len() && source[at] != '\n' && !(source[at] == '\r' && source.get(at + 1) == Some(&'\n')) { at += 1; }
            tokens.push(Token { kind: Kind::Comment, text: source[start..at].iter().collect(), start: position(start), end: position(at) });
            continue;
        }
        if c == '\n' || (c == '\r' && source.get(at + 1) == Some(&'\n')) {
            at += if c == '\r' { 2 } else { 1 };
            let blank_line = tokens.last().is_none_or(|t| matches!(t.kind, Kind::Newline | Kind::Nl | Kind::Indent | Kind::Dedent)
                || (t.kind == Kind::Comment && tokens.iter().rev().take_while(|x| x.start.0 == t.start.0).all(|x| x.kind == Kind::Comment)));
            let kind = if !levels.is_empty() || blank_line { Kind::Nl } else { Kind::Newline };
            let (line, col) = position(start);
            tokens.push(Token { kind, text: source[start..at].iter().collect(), start: (line, col), end: (line, col + (at - start)) });
            line_begin = true;
            continue;
        }
        if c == '\\' {
            if source.get(at + 1) == Some(&'\n') { at += 2; continued = true; line_begin = true; continue; }
            if source.get(at + 1).is_none() { let (line, _) = position(at); return Err(TokenFailure::Token("unexpected EOF in multi-line statement".into(), line, line_length(at))); }
            let (line, _) = position(at);
            return Err(TokenFailure::Token("unexpected character after line continuation character".into(), line, line_length(at)));
        }
        // Strings, with any prefix.
        let mut prefix_end = at;
        while prefix_end < source.len() && prefix_end - at < 2 && matches!(source[prefix_end], 'r' | 'R' | 'b' | 'B' | 'u' | 'U' | 'f' | 'F') { prefix_end += 1; }
        let prefix: String = source[at..prefix_end].iter().collect::<String>().to_lowercase();
        let valid_prefix = matches!(prefix.as_str(), "" | "r" | "b" | "u" | "f" | "br" | "rb" | "fr" | "rf");
        if valid_prefix && prefix_end < source.len() && matches!(source[prefix_end], '\'' | '"') {
            let quote = source[prefix_end];
            let triple = source.get(prefix_end + 1) == Some(&quote) && source.get(prefix_end + 2) == Some(&quote);
            let mut index = prefix_end + if triple { 3 } else { 1 };
            let mut escaped_quote = false;
            loop {
                if index < source.len() { read_through(index)?; }
                if index >= source.len() || (!triple && source[index] == '\n') {
                    let (line, col) = position(start);
                    let message = if triple { "EOF in multi-line string".to_owned() } else {
                        let detected = position(index.min(source.len())).0;
                        format!("unterminated string literal (detected at line {detected}){}", if escaped_quote { "; perhaps you escaped the end quote?" } else { "" })
                    };
                    return Err(TokenFailure::Token(message, line, col + 1));
                }
                let ch = source[index];
                if ch == '\\' {
                    if source.get(index + 1) == Some(&quote) { escaped_quote = true; }
                    index += 2;
                    continue;
                }
                if ch == quote {
                    if !triple { index += 1; break; }
                    if source.get(index + 1) == Some(&quote) && source.get(index + 2) == Some(&quote) { index += 3; break; }
                }
                index += 1;
            }
            at = index.min(source.len());
            tokens.push(Token { kind: Kind::Str, text: source[start..at].iter().collect(), start: position(start), end: position(at) });
            continue;
        }
        if identifier_start(c) {
            while at < source.len() && identifier_char(source[at]) { at += 1; }
            tokens.push(Token { kind: Kind::Name, text: source[start..at].iter().collect(), start: position(start), end: position(at) });
            continue;
        }
        if c.is_ascii_digit() || (c == '.' && source.get(at + 1).is_some_and(|d| d.is_ascii_digit())) {
            at = number_end(source, at).map_err(|(message, index)| error(&message, index))?;
            tokens.push(Token { kind: Kind::Number, text: source[start..at].iter().collect(), start: position(start), end: position(at) });
            continue;
        }
        // Operators.
        let three: String = source[at..(at + 3).min(source.len())].iter().collect();
        let two: String = source[at..(at + 2).min(source.len())].iter().collect();
        let width = if ["**=", "//=", ">>=", "<<=", "...", "!=="].contains(&three.as_str()) && three != "!==" { 3 }
            else if ["!=", "%=", "&=", "**", "*=", "+=", "-=", "->", "//", "/=", ":=", "<<", "<=", "==", ">=", ">>", "@=", "^=", "|="].contains(&two.as_str()) { 2 } else { 1 };
        match c {
            '(' | '[' | '{' => {
                if levels.len() >= MAXLEVEL { return Err(error("too many nested parentheses", at + 1)); }
                levels.push(c);
            }
            ')' | ']' | '}' => { levels.pop(); }
            _ => {}
        }
        if width == 1 && (c < ' ' || c == '\x7f') {
            return Err(error(&format!("invalid non-printable character U+{:04X}", c as u32), at + 1));
        }
        at += width;
        tokens.push(Token { kind: Kind::Op, text: source[start..at].iter().collect(), start: position(start), end: position(at) });
    }
}

/// The end of a number token starting at `at`, or the tokenizer's refusal
/// with the index its position is reported at.
fn number_end(source: &[char], mut at: usize) -> Result<usize, (String, usize)> {
    let digit = |index: usize| source.get(index).is_some_and(|c| c.is_ascii_digit());
    let decimal_tail = |mut index: usize| -> Result<usize, (String, usize)> {
        loop {
            while digit(index) { index += 1; }
            if source.get(index) != Some(&'_') { return Ok(index); }
            index += 1;
            if !digit(index) { return Err(("invalid decimal literal".into(), index)); }
        }
    };
    let exponent = |index: usize| -> Result<usize, (String, usize)> {
        // source[index] is 'e' or 'E'.
        let mut next = index + 1;
        if matches!(source.get(next), Some('+' | '-')) {
            next += 1;
            if !digit(next) { return Err(("invalid decimal literal".into(), next)); }
        } else if !digit(next) {
            return Ok(index);
        }
        decimal_tail(next)
    };
    let imaginary = |index: usize| if matches!(source.get(index), Some('j' | 'J')) { index + 1 } else { index };
    let fraction = |index: usize| -> Result<usize, (String, usize)> {
        // index is just past '.'.
        let mut end = index;
        if digit(end) { end = decimal_tail(end)?; }
        if matches!(source.get(end), Some('e' | 'E')) { end = exponent(end)?; if !matches!(source.get(end), Some('e' | 'E')) { return Ok(imaginary(end)); } return Ok(end); }
        Ok(imaginary(end))
    };
    if source[at] == '.' { return fraction(at + 1); }
    if source[at] == '0' {
        match source.get(at + 1) {
            Some('x' | 'X') | Some('o' | 'O') | Some('b' | 'B') => {
                let (radix, name) = match source[at + 1].to_ascii_lowercase() { 'x' => (16, "hexadecimal"), 'o' => (8, "octal"), _ => (2, "binary") };
                at += 2;
                loop {
                    if source.get(at) == Some(&'_') { at += 1; }
                    let c = source.get(at).copied();
                    let valid = c.is_some_and(|c| c.to_digit(radix).is_some());
                    if !valid {
                        if radix != 16 && c.is_some_and(|c| c.is_ascii_digit()) {
                            return Err((format!("invalid digit '{}' in {name} literal", c.unwrap()), at + 1));
                        }
                        return Err((format!("invalid {name} literal"), at));
                    }
                    while source.get(at).is_some_and(|c| c.to_digit(radix).is_some()) { at += 1; }
                    if radix != 16 && source.get(at).is_some_and(|c| c.is_ascii_digit()) {
                        return Err((format!("invalid digit '{}' in {name} literal", source[at]), at + 1));
                    }
                    if source.get(at) != Some(&'_') { return Ok(at); }
                }
            }
            _ => {
                at += 1;
                loop {
                    if source.get(at) == Some(&'_') {
                        at += 1;
                        if !digit(at) { return Err(("invalid decimal literal".into(), at)); }
                    }
                    if source.get(at) != Some(&'0') { break; }
                    at += 1;
                }
                if digit(at) { at = decimal_tail(at)?; }
                return match source.get(at) {
                    Some('.') => fraction(at + 1),
                    Some('e' | 'E') => { let end = exponent(at)?; Ok(if end == at { at } else { imaginary(end) }) }
                    Some('j' | 'J') => Ok(at + 1),
                    _ => Ok(at),
                };
            }
        }
    }
    at = decimal_tail(at)?;
    match source.get(at) {
        Some('.') => fraction(at + 1),
        Some('e' | 'E') => { let end = exponent(at)?; Ok(if end == at { at } else { imaginary(end) }) }
        Some('j' | 'J') => Ok(at + 1),
        _ => Ok(at),
    }
}

/// `tokenize.untokenize` of full tokens.
fn untokenize(tokens: &[Token]) -> String {
    let mut out = String::new();
    let (mut prev_row, mut prev_col) = (1usize, 0usize);
    let mut indents: Vec<String> = Vec::new();
    let mut startline = false;
    for token in tokens {
        match token.kind {
            Kind::End => break,
            Kind::Indent => { indents.push(token.text.clone()); continue; }
            Kind::Dedent => { indents.pop(); (prev_row, prev_col) = token.end; continue; }
            Kind::Newline | Kind::Nl => startline = true,
            _ if startline && !indents.is_empty() => {
                let indent = indents.last().unwrap();
                if token.start.1 >= indent.chars().count() { out.push_str(indent); prev_col = indent.chars().count(); }
                startline = false;
            }
            _ => {}
        }
        let (row, col) = token.start;
        let rows = row.saturating_sub(prev_row);
        if rows > 0 { for _ in 0..rows { out.push_str("\\\n"); } prev_col = 0; }
        if col > prev_col { out.push_str(&" ".repeat(col - prev_col)); }
        out.push_str(&token.text);
        (prev_row, prev_col) = token.end;
        if matches!(token.kind, Kind::Newline | Kind::Nl) { prev_row += 1; prev_col = 0; }
    }
    out
}

/// `_filter_header`: tokenize, drop each `L` name after a number, untokenize.
pub fn filter_header(header: &str) -> Result<String, TokenFailure> {
    let chars: Vec<char> = header.chars().collect();
    let tokens = tokenize(&chars)?;
    let mut kept = Vec::with_capacity(tokens.len());
    let mut last_number = false;
    for token in tokens {
        if last_number && token.kind == Kind::Name && token.text == "L" { continue; }
        last_number = token.kind == Kind::Number;
        kept.push(token);
    }
    Ok(untokenize(&kept))
}

// ------------------------------------------------------------- parser

/// One expression node: what `literal_eval` converts, or the class it names.
#[derive(Debug)]
enum Node {
    Constant(Obj),
    Tuple(Vec<Expr>),
    List(Vec<Expr>),
    Set(Vec<Expr>),
    Dict(Vec<(Option<Expr>, Expr)>),
    Unary(char, Box<Expr>),
    Binary(char, Box<Expr>, Box<Expr>),
    Name(String),
    /// `set()` (a call to the name `set` without arguments).
    EmptySetCall,
    Other(&'static str),
}

/// A node, the line it starts on and the depth of the tree under it.
#[derive(Debug)]
struct Expr { node: Node, line: usize, depth: usize }

impl Expr {
    fn leaf(node: Node, line: usize) -> Self { Expr { node, line, depth: 1 } }
    fn over(node: Node, line: usize, children: usize) -> Self { Expr { node, line, depth: children + 1 } }
}

/// What `ast.literal_eval(text)` does.
#[derive(Debug, PartialEq)]
pub enum EvalError { Syntax, Malformed(String), Type(String), Recursion }

const KEYWORDS: [&str; 35] = ["False", "None", "True", "and", "as", "assert", "async", "await", "break", "class", "continue", "def", "del", "elif", "else",
                              "except", "finally", "for", "from", "global", "if", "import", "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise",
                              "return", "try", "while", "with", "yield"];

/// The AST depth from which CPython's builder stops with a RecursionError.
const MAX_AST_DEPTH: usize = 2996;

struct Parser { tokens: Vec<Token>, at: usize }

/// Unary `+`, `-` and `~` (outermost first) around `expr`.
fn apply_signs(mut expr: Expr, signs: Vec<(char, usize)>) -> Expr {
    for (op, line) in signs.into_iter().rev() {
        let depth = expr.depth;
        expr = Expr::over(if op == '~' { Node::Other("UnaryOp") } else { Node::Unary(op, Box::new(expr)) }, line, depth);
    }
    expr
}

type Parse<T> = Result<T, EvalError>;

impl Parser {
    fn peek(&self) -> &Token { &self.tokens[self.at] }
    fn next_is(&self, kind: Kind, text: &str) -> bool { let token = &self.tokens[(self.at + 1).min(self.tokens.len() - 1)]; token.kind == kind && token.text == text }
    fn is_op(&self, text: &str) -> bool { let token = self.peek(); token.kind == Kind::Op && token.text == text }
    fn is_name(&self, text: &str) -> bool { let token = self.peek(); token.kind == Kind::Name && token.text == text }
    fn eat_op(&mut self, text: &str) -> bool { if self.is_op(text) { self.at += 1; true } else { false } }
    fn eat_name(&mut self, text: &str) -> bool { if self.is_name(text) { self.at += 1; true } else { false } }
    fn expect_op(&mut self, text: &str) -> Parse<()> { if self.eat_op(text) { Ok(()) } else { Err(EvalError::Syntax) } }
    fn skip_layout(&mut self) { while self.peek().kind == Kind::Nl { self.at += 1; } }
    fn line(&self) -> usize { self.peek().start.0 }
    fn comprehension_follows(&self) -> bool { self.is_name("for") || self.is_name("async") }

    fn name(&mut self) -> Parse<String> {
        let token = self.peek().clone();
        if token.kind != Kind::Name || KEYWORDS.contains(&token.text.as_str()) { return Err(EvalError::Syntax); }
        self.at += 1;
        Ok(token.text)
    }

    /// `expressions`: one expression, or a tuple of them.
    fn expressions(&mut self) -> Parse<Expr> {
        let line = self.line();
        let first = self.star_expression()?;
        if !self.is_op(",") { return Ok(first); }
        let mut items = vec![first];
        while self.eat_op(",") {
            if matches!(self.peek().kind, Kind::Newline | Kind::End) { break; }
            items.push(self.star_expression()?);
        }
        let depth = items.iter().map(|item| item.depth).max().unwrap_or(0);
        Ok(Expr::over(Node::Tuple(items), line, depth))
    }

    fn star_expression(&mut self) -> Parse<Expr> {
        let line = self.line();
        if self.eat_op("*") { let inner = self.bit_or()?; return Ok(Expr::over(Node::Other("Starred"), line, inner.depth)); }
        self.named_expression()
    }

    fn named_expression(&mut self) -> Parse<Expr> {
        let line = self.line();
        if self.peek().kind == Kind::Name && self.next_is(Kind::Op, ":=") {
            self.name()?;
            self.at += 1;
            let value = self.expression()?;
            return Ok(Expr::over(Node::Other("NamedExpr"), line, value.depth));
        }
        self.expression()
    }

    fn expression(&mut self) -> Parse<Expr> {
        let line = self.line();
        if self.eat_name("lambda") {
            let mut depth = self.lambda_parameters()?;
            self.expect_op(":")?;
            depth = depth.max(self.expression()?.depth);
            return Ok(Expr::over(Node::Other("Lambda"), line, depth));
        }
        let body = self.disjunction()?;
        if self.eat_name("if") {
            let test = self.disjunction()?;
            if !self.eat_name("else") { return Err(EvalError::Syntax); }
            let other = self.expression()?;
            return Ok(Expr::over(Node::Other("IfExp"), line, body.depth.max(test.depth).max(other.depth)));
        }
        Ok(body)
    }

    fn lambda_parameters(&mut self) -> Parse<usize> {
        let mut depth = 0;
        let mut first = true;
        while !self.is_op(":") {
            if !first { self.expect_op(",")?; if self.is_op(":") { break; } }
            first = false;
            if self.eat_op("/") { continue; }
            if self.eat_op("**") { self.name()?; continue; }
            if self.eat_op("*") { if self.peek().kind == Kind::Name { self.name()?; } continue; }
            self.name()?;
            if self.eat_op("=") { depth = depth.max(self.expression()?.depth); }
        }
        Ok(depth)
    }

    fn disjunction(&mut self) -> Parse<Expr> {
        let line = self.line();
        let first = self.conjunction()?;
        if !self.is_name("or") { return Ok(first); }
        let mut depth = first.depth;
        while self.eat_name("or") { depth = depth.max(self.conjunction()?.depth); }
        Ok(Expr::over(Node::Other("BoolOp"), line, depth))
    }

    fn conjunction(&mut self) -> Parse<Expr> {
        let line = self.line();
        let first = self.inversion()?;
        if !self.is_name("and") { return Ok(first); }
        let mut depth = first.depth;
        while self.eat_name("and") { depth = depth.max(self.inversion()?.depth); }
        Ok(Expr::over(Node::Other("BoolOp"), line, depth))
    }

    fn inversion(&mut self) -> Parse<Expr> {
        let mut lines = Vec::new();
        while self.is_name("not") { lines.push(self.line()); self.at += 1; }
        let mut expr = self.comparison()?;
        for line in lines.into_iter().rev() { let depth = expr.depth; expr = Expr::over(Node::Other("UnaryOp"), line, depth); }
        Ok(expr)
    }

    fn comparison(&mut self) -> Parse<Expr> {
        let line = self.line();
        let first = self.bit_or()?;
        let mut depth = first.depth;
        let mut compared = false;
        loop {
            let token = self.peek().clone();
            if token.kind == Kind::Op && matches!(token.text.as_str(), "==" | "!=" | "<" | "<=" | ">" | ">=") { self.at += 1; }
            else if token.kind == Kind::Name && token.text == "in" { self.at += 1; }
            else if token.kind == Kind::Name && token.text == "not" && self.next_is(Kind::Name, "in") { self.at += 2; }
            else if token.kind == Kind::Name && token.text == "is" { self.at += 1; self.eat_name("not"); }
            else { break; }
            depth = depth.max(self.bit_or()?.depth);
            compared = true;
        }
        Ok(if compared { Expr::over(Node::Other("Compare"), line, depth) } else { first })
    }

    fn binary(&mut self, operators: &[&str], next: fn(&mut Self) -> Parse<Expr>) -> Parse<Expr> {
        let line = self.line();
        let mut left = next(self)?;
        loop {
            let token = self.peek();
            if token.kind != Kind::Op || !operators.contains(&token.text.as_str()) { return Ok(left); }
            let op = token.text.clone();
            self.at += 1;
            let right = next(self)?;
            let depth = left.depth.max(right.depth);
            left = match op.as_str() {
                "+" => Expr::over(Node::Binary('+', Box::new(left), Box::new(right)), line, depth),
                "-" => Expr::over(Node::Binary('-', Box::new(left), Box::new(right)), line, depth),
                _ => Expr::over(Node::Other("BinOp"), line, depth),
            };
        }
    }

    fn bit_or(&mut self) -> Parse<Expr> { self.binary(&["|"], Self::bit_xor) }
    fn bit_xor(&mut self) -> Parse<Expr> { self.binary(&["^"], Self::bit_and) }
    fn bit_and(&mut self) -> Parse<Expr> { self.binary(&["&"], Self::shift) }
    fn shift(&mut self) -> Parse<Expr> { self.binary(&["<<", ">>"], Self::sum) }
    fn sum(&mut self) -> Parse<Expr> { self.binary(&["+", "-"], Self::term) }
    fn term(&mut self) -> Parse<Expr> { self.binary(&["*", "/", "//", "%", "@"], Self::factor) }

    fn factor(&mut self) -> Parse<Expr> {
        // Unary chains iterate: a header may hold thousands of signs.
        let mut signs: Vec<(char, usize)> = Vec::new();
        loop {
            let token = self.peek();
            if token.kind == Kind::Op && matches!(token.text.as_str(), "+" | "-" | "~") {
                signs.push((token.text.chars().next().unwrap(), token.start.0));
                self.at += 1;
            } else {
                break;
            }
        }
        let expr = self.power()?;
        Ok(apply_signs(expr, signs))
    }

    /// `power`, iterated: `a ** -b ** c` nests to the right without recursing.
    fn power(&mut self) -> Parse<Expr> {
        let mut chain: Vec<(Vec<(char, usize)>, Expr)> = Vec::new();
        let mut signs: Vec<(char, usize)> = Vec::new();
        loop {
            let line = self.line();
            let awaited = self.eat_name("await");
            let mut base = self.primary()?;
            if awaited { let depth = base.depth; base = Expr::over(Node::Other("Await"), line, depth); }
            chain.push((std::mem::take(&mut signs), base));
            if !self.eat_op("**") { break; }
            loop {
                let token = self.peek();
                if token.kind == Kind::Op && matches!(token.text.as_str(), "+" | "-" | "~") {
                    signs.push((token.text.chars().next().unwrap(), token.start.0));
                    self.at += 1;
                } else {
                    break;
                }
            }
        }
        let (signs, mut expr) = chain.pop().unwrap();
        expr = apply_signs(expr, signs);
        while let Some((signs, base)) = chain.pop() {
            let line = base.line;
            expr = apply_signs(Expr::over(Node::Other("BinOp"), line, base.depth.max(expr.depth)), signs);
        }
        Ok(expr)
    }

    fn primary(&mut self) -> Parse<Expr> {
        let line = self.line();
        let mut expr = self.atom()?;
        loop {
            if self.eat_op(".") { self.name()?; let depth = expr.depth; expr = Expr::over(Node::Other("Attribute"), line, depth); continue; }
            if self.eat_op("(") {
                let set_name = matches!(&expr.node, Node::Name(name) if name == "set");
                self.skip_layout();
                let no_arguments = self.is_op(")");
                let depth = expr.depth.max(self.arguments()?);
                self.expect_op(")")?;
                expr = if set_name && no_arguments { Expr::over(Node::EmptySetCall, line, depth) } else { Expr::over(Node::Other("Call"), line, depth) };
                continue;
            }
            if self.eat_op("[") {
                let depth = expr.depth.max(self.slices()?);
                self.expect_op("]")?;
                expr = Expr::over(Node::Other("Subscript"), line, depth);
                continue;
            }
            return Ok(expr);
        }
    }

    fn arguments(&mut self) -> Parse<usize> {
        let mut depth = 0;
        let mut first = true;
        loop {
            self.skip_layout();
            if self.is_op(")") { return Ok(depth); }
            if !first { self.expect_op(",")?; self.skip_layout(); if self.is_op(")") { return Ok(depth); } }
            first = false;
            if self.eat_op("**") || self.eat_op("*") { depth = depth.max(self.expression()?.depth); continue; }
            if self.peek().kind == Kind::Name && self.next_is(Kind::Op, "=") {
                self.name()?;
                self.at += 1;
                depth = depth.max(self.expression()?.depth);
                continue;
            }
            depth = depth.max(self.named_expression()?.depth);
            if self.comprehension_follows() { depth = depth.max(self.comprehension()?); }
        }
    }

    fn slices(&mut self) -> Parse<usize> {
        let mut depth = 0;
        let mut first = true;
        loop {
            self.skip_layout();
            if self.is_op("]") { return if first { Err(EvalError::Syntax) } else { Ok(depth) }; }
            if !first { self.expect_op(",")?; self.skip_layout(); if self.is_op("]") { return Ok(depth); } }
            first = false;
            if self.eat_op("*") { depth = depth.max(self.bit_or()?.depth); continue; }
            if !self.is_op(":") { depth = depth.max(self.named_expression()?.depth); }
            if self.eat_op(":") {
                if !self.is_op(":") && !self.is_op(",") && !self.is_op("]") { depth = depth.max(self.expression()?.depth); }
                if self.eat_op(":") && !self.is_op(",") && !self.is_op("]") { depth = depth.max(self.expression()?.depth); }
            }
        }
    }

    fn comprehension(&mut self) -> Parse<usize> {
        let mut depth = 0;
        while self.comprehension_follows() {
            self.eat_name("async");
            if !self.eat_name("for") { return Err(EvalError::Syntax); }
            loop {
                self.eat_op("*");
                depth = depth.max(self.bit_or()?.depth);
                if !self.eat_op(",") || self.is_name("in") { break; }
            }
            if !self.eat_name("in") { return Err(EvalError::Syntax); }
            depth = depth.max(self.disjunction()?.depth);
            while self.eat_name("if") { depth = depth.max(self.disjunction()?.depth); }
        }
        Ok(depth)
    }

    fn strings(&mut self) -> Parse<Expr> {
        let line = self.line();
        let mut text = String::new();
        let mut bytes: Option<Vec<u8>> = None;
        let mut formatted = false;
        let mut seen_text = false;
        while self.peek().kind == Kind::Str {
            let token = self.peek().text.clone();
            self.at += 1;
            let split = token.find(['\'', '"']).unwrap();
            let prefix = token[..split].to_lowercase();
            let body: Vec<char> = token[split..].chars().collect();
            let quote_len = if body.len() >= 6 && body[1] == body[0] && body[2] == body[0] { 3 } else { 1 };
            let inner = &body[quote_len..body.len() - quote_len];
            let raw = prefix.contains('r');
            if prefix.contains('f') {
                if bytes.is_some() { return Err(EvalError::Syntax); }
                formatted = true;
                seen_text = true;
                continue;
            }
            if prefix.contains('b') {
                if seen_text { return Err(EvalError::Syntax); }
                bytes.get_or_insert_with(Vec::new).extend(decode_bytes(inner, raw)?);
            } else {
                if bytes.is_some() { return Err(EvalError::Syntax); }
                seen_text = true;
                text.push_str(&decode_text(inner, raw)?);
            }
        }
        if formatted { return Ok(Expr::leaf(Node::Other("JoinedStr"), line)); }
        Ok(Expr::leaf(Node::Constant(match bytes { Some(bytes) => Obj::Bytes(bytes), None => Obj::Str(text) }), line))
    }

    fn atom(&mut self) -> Parse<Expr> {
        let token = self.peek().clone();
        let line = token.start.0;
        match token.kind {
            Kind::Name => match token.text.as_str() {
                "True" => { self.at += 1; Ok(Expr::leaf(Node::Constant(Obj::Bool(true)), line)) }
                "False" => { self.at += 1; Ok(Expr::leaf(Node::Constant(Obj::Bool(false)), line)) }
                "None" => { self.at += 1; Ok(Expr::leaf(Node::Constant(Obj::None), line)) }
                _ => { let name = self.name()?; Ok(Expr::leaf(Node::Name(name), line)) }
            },
            Kind::Number => { self.at += 1; Ok(Expr::leaf(Node::Constant(number(&token.text)?), line)) }
            Kind::Str => self.strings(),
            Kind::Op => match token.text.as_str() {
                "..." => { self.at += 1; Ok(Expr::leaf(Node::Constant(Obj::Ellipsis), line)) }
                "(" => self.parenthesized(line),
                "[" => self.list(line),
                "{" => self.braces(line),
                _ => Err(EvalError::Syntax),
            },
            _ => Err(EvalError::Syntax),
        }
    }

    /// Items after the first up to `close`, each a starred or named expression.
    fn rest_of_items(&mut self, mut items: Vec<Expr>, close: &str) -> Parse<Vec<Expr>> {
        while self.eat_op(",") {
            self.skip_layout();
            if self.is_op(close) { break; }
            items.push(self.star_expression()?);
            self.skip_layout();
        }
        self.expect_op(close)?;
        Ok(items)
    }

    fn parenthesized(&mut self, line: usize) -> Parse<Expr> {
        self.at += 1;
        self.skip_layout();
        if self.eat_op(")") { return Ok(Expr::leaf(Node::Tuple(Vec::new()), line)); }
        if self.eat_name("yield") {
            let mut depth = 0;
            if self.eat_name("from") { depth = self.expression()?.depth; } else if !self.is_op(")") { self.skip_layout(); depth = self.expressions()?.depth; }
            self.skip_layout();
            self.expect_op(")")?;
            return Ok(Expr::over(Node::Other("Yield"), line, depth));
        }
        let first = self.star_expression()?;
        self.skip_layout();
        if self.comprehension_follows() {
            let depth = first.depth.max(self.comprehension()?);
            self.skip_layout();
            self.expect_op(")")?;
            return Ok(Expr::over(Node::Other("GeneratorExp"), line, depth));
        }
        if self.eat_op(")") {
            if matches!(first.node, Node::Other("Starred")) { return Err(EvalError::Syntax); }
            return Ok(first);
        }
        let items = self.rest_of_items(vec![first], ")")?;
        let depth = items.iter().map(|item| item.depth).max().unwrap_or(0);
        Ok(Expr::over(Node::Tuple(items), line, depth))
    }

    fn list(&mut self, line: usize) -> Parse<Expr> {
        self.at += 1;
        self.skip_layout();
        if self.eat_op("]") { return Ok(Expr::leaf(Node::List(Vec::new()), line)); }
        let first = self.star_expression()?;
        self.skip_layout();
        if self.comprehension_follows() {
            let depth = first.depth.max(self.comprehension()?);
            self.skip_layout();
            self.expect_op("]")?;
            return Ok(Expr::over(Node::Other("ListComp"), line, depth));
        }
        let items = self.rest_of_items(vec![first], "]")?;
        let depth = items.iter().map(|item| item.depth).max().unwrap_or(0);
        Ok(Expr::over(Node::List(items), line, depth))
    }

    fn braces(&mut self, line: usize) -> Parse<Expr> {
        self.at += 1;
        self.skip_layout();
        if self.eat_op("}") { return Ok(Expr::leaf(Node::Dict(Vec::new()), line)); }
        // A dict when the first item is `**x` or `key: value`, else a set.
        let (first_key, first_value) = if self.eat_op("**") {
            (None, Some(self.bit_or()?))
        } else {
            let key = self.star_expression()?;
            self.skip_layout();
            if self.eat_op(":") {
                if matches!(key.node, Node::Other("Starred")) { return Err(EvalError::Syntax); }
                self.skip_layout();
                (Some(key), Some(self.expression()?))
            } else {
                (Some(key), None)
            }
        };
        self.skip_layout();
        let Some(value) = first_value else {
            let first = first_key.unwrap();
            if self.comprehension_follows() {
                let depth = first.depth.max(self.comprehension()?);
                self.skip_layout();
                self.expect_op("}")?;
                return Ok(Expr::over(Node::Other("SetComp"), line, depth));
            }
            let items = self.rest_of_items(vec![first], "}")?;
            let depth = items.iter().map(|item| item.depth).max().unwrap_or(0);
            return Ok(Expr::over(Node::Set(items), line, depth));
        };
        if self.comprehension_follows() {
            let Some(key) = first_key else { return Err(EvalError::Syntax) };
            let depth = key.depth.max(value.depth).max(self.comprehension()?);
            self.skip_layout();
            self.expect_op("}")?;
            return Ok(Expr::over(Node::Other("DictComp"), line, depth));
        }
        let mut pairs = vec![(first_key, value)];
        while self.eat_op(",") {
            self.skip_layout();
            if self.is_op("}") { break; }
            if self.eat_op("**") { pairs.push((None, self.bit_or()?)); self.skip_layout(); continue; }
            let key = self.expression()?;
            self.skip_layout();
            self.expect_op(":")?;
            self.skip_layout();
            let value = self.expression()?;
            pairs.push((Some(key), value));
            self.skip_layout();
        }
        self.expect_op("}")?;
        let depth = pairs.iter().map(|(key, value)| key.as_ref().map_or(0, |key| key.depth).max(value.depth)).max().unwrap_or(0);
        Ok(Expr::over(Node::Dict(pairs), line, depth))
    }
}

/// A number token's value (underscores removed).
fn number(text: &str) -> Parse<Obj> {
    let clean: String = text.chars().filter(|c| *c != '_').collect();
    let lower = clean.to_ascii_lowercase();
    if let Some(imag) = lower.strip_suffix('j') { return Ok(Obj::Complex(0.0, imag.parse().map_err(|_| EvalError::Syntax)?)); }
    for (prefix, radix) in [("0x", 16), ("0o", 8), ("0b", 2)] {
        if let Some(digits) = lower.strip_prefix(prefix) { return Ok(Obj::Int(parse_int(digits, radix))); }
    }
    if lower.contains(['.', 'e']) { return Ok(Obj::Float(lower.parse().map_err(|_| EvalError::Syntax)?)); }
    if lower.len() > 1 && lower.starts_with('0') && lower.bytes().any(|b| b != b'0') { return Err(EvalError::Syntax); }
    if lower.len() > 4300 { return Err(EvalError::Syntax); }
    Ok(Obj::Int(parse_int(&lower, 10)))
}

fn hex_value(chars: &[char], at: usize, count: usize) -> Option<u32> {
    if at + count > chars.len() { return None; }
    let text: String = chars[at..at + count].iter().collect();
    if !text.chars().all(|c| c.is_ascii_hexdigit()) { return None; }
    u32::from_str_radix(&text, 16).ok()
}

/// A str literal's body with its escapes (`\N{...}` is not decoded here).
fn decode_text(inner: &[char], raw: bool) -> Parse<String> {
    let mut out = String::new();
    let mut at = 0;
    while at < inner.len() {
        let c = inner[at];
        if c != '\\' || raw {
            if raw && c == '\\' && at + 1 < inner.len() { pystr::push(&mut out, '\\' as u32); pystr::push(&mut out, inner[at + 1] as u32); at += 2; continue; }
            pystr::push(&mut out, c as u32);
            at += 1;
            continue;
        }
        let Some(&e) = inner.get(at + 1) else { pystr::push(&mut out, '\\' as u32); break };
        at += 2;
        let code = match e {
            '\n' => continue,
            '\\' => 0x5C, '\'' => 0x27, '"' => 0x22, 'a' => 7, 'b' => 8, 'f' => 12, 'n' => 10, 'r' => 13, 't' => 9, 'v' => 11,
            '0'..='7' => {
                let mut value = e.to_digit(8).unwrap();
                for _ in 0..2 { match inner.get(at).and_then(|d| d.to_digit(8)) { Some(d) => { value = value * 8 + d; at += 1; } None => break } }
                value
            }
            'x' => { let value = hex_value(inner, at, 2).ok_or(EvalError::Syntax)?; at += 2; value }
            'u' => { let value = hex_value(inner, at, 4).ok_or(EvalError::Syntax)?; at += 4; value }
            'U' => { let value = hex_value(inner, at, 8).ok_or(EvalError::Syntax)?; at += 8; if value > 0x10FFFF { return Err(EvalError::Syntax); } value }
            'N' => return Err(EvalError::Syntax),
            other => { pystr::push(&mut out, '\\' as u32); other as u32 }
        };
        pystr::push(&mut out, code);
    }
    Ok(out)
}

/// A bytes literal's body: ASCII only, with byte escapes.
fn decode_bytes(inner: &[char], raw: bool) -> Parse<Vec<u8>> {
    if inner.iter().any(|c| !c.is_ascii()) { return Err(EvalError::Syntax); }
    let mut out = Vec::new();
    let mut at = 0;
    while at < inner.len() {
        let c = inner[at] as u8;
        if c != b'\\' || raw {
            if raw && c == b'\\' && at + 1 < inner.len() { out.push(b'\\'); out.push(inner[at + 1] as u8); at += 2; continue; }
            out.push(c);
            at += 1;
            continue;
        }
        let Some(&e) = inner.get(at + 1) else { out.push(b'\\'); break };
        at += 2;
        let value = match e {
            '\n' => continue,
            '\\' => 0x5C, '\'' => 0x27, '"' => 0x22, 'a' => 7, 'b' => 8, 'f' => 12, 'n' => 10, 'r' => 13, 't' => 9, 'v' => 11,
            '0'..='7' => {
                let mut value = e.to_digit(8).unwrap();
                for _ in 0..2 { match inner.get(at).and_then(|d| d.to_digit(8)) { Some(d) => { value = value * 8 + d; at += 1; } None => break } }
                value & 0xFF
            }
            'x' => { let value = hex_value(inner, at, 2).ok_or(EvalError::Syntax)?; at += 2; value }
            other => { out.push(b'\\'); other as u32 }
        };
        out.push(value as u8);
    }
    Ok(out)
}

/// The parse of `text` as `compile(text, "<unknown>", "eval")` builds it.
fn parse(text: &str) -> Parse<Expr> {
    let chars: Vec<char> = text.chars().collect();
    let tokens = tokenize(&chars).map_err(|_| EvalError::Syntax)?;
    // What only the compiler refuses: leading zeros, invalid identifiers,
    // brackets that do not match, characters that are no token.
    let mut stack: Vec<&str> = Vec::new();
    for token in &tokens {
        match token.kind {
            Kind::Number => {
                let clean: String = token.text.chars().filter(|c| *c != '_').collect();
                let digits = clean.trim_end_matches(['j', 'J']);
                if digits.len() > 1 && digits.starts_with('0') && digits.bytes().all(|b| b.is_ascii_digit()) && digits.bytes().any(|b| b != b'0') { return Err(EvalError::Syntax); }
            }
            Kind::Name => {
                let mut chars = token.text.chars();
                if !chars.next().is_some_and(|c| valid_identifier_char(c, true)) || !chars.all(|c| valid_identifier_char(c, false)) { return Err(EvalError::Syntax); }
            }
            Kind::Op => match token.text.as_str() {
                "(" | "[" | "{" => stack.push(&token.text),
                ")" | "]" | "}" => {
                    let open = stack.pop().ok_or(EvalError::Syntax)?;
                    if !matches!((open, token.text.as_str()), ("(", ")") | ("[", "]") | ("{", "}")) { return Err(EvalError::Syntax); }
                }
                "$" | "?" | "!" | "`" | "\r" => return Err(EvalError::Syntax),
                other if other.len() == 1 && !"+-*/%@&|^~<>=.,:;".contains(other) => return Err(EvalError::Syntax),
                _ => {}
            },
            Kind::Indent | Kind::Dedent => {}
            _ => {}
        }
    }
    let tokens: Vec<Token> = tokens.into_iter().filter(|token| !matches!(token.kind, Kind::Comment | Kind::Indent | Kind::Dedent)).collect();
    let mut parser = Parser { tokens, at: 0 };
    parser.skip_layout();
    let expr = parser.expressions()?;
    while matches!(parser.peek().kind, Kind::Newline | Kind::Nl) { parser.at += 1; }
    if parser.peek().kind != Kind::End { return Err(EvalError::Syntax); }
    if expr.depth > MAX_AST_DEPTH { return Err(EvalError::Recursion); }
    Ok(expr)
}

fn malformed(expr: Option<&Expr>) -> EvalError {
    let Some(expr) = expr else { return EvalError::Malformed("malformed node or string: None".into()) };
    let name = match &expr.node {
        Node::Constant(_) => "Constant", Node::Tuple(_) => "Tuple", Node::List(_) => "List", Node::Set(_) => "Set", Node::Dict(_) => "Dict",
        Node::Unary(..) => "UnaryOp", Node::Binary(..) => "BinOp", Node::Name(_) => "Name", Node::EmptySetCall => "Call", Node::Other(name) => name,
    };
    EvalError::Malformed(format!("malformed node or string on line {}: <ast.{name} object at 0x0000000000000000>", expr.line))
}

fn convert_number(expr: &Expr) -> Parse<Obj> {
    match &expr.node {
        Node::Constant(value @ (Obj::Int(_) | Obj::Float(_) | Obj::Complex(..))) => Ok(value.clone()),
        _ => Err(malformed(Some(expr))),
    }
}

fn convert_signed(expr: &Expr) -> Parse<Obj> {
    if let Node::Unary(op, operand) = &expr.node {
        let value = convert_number(operand)?;
        if *op == '+' { return Ok(value); }
        return Ok(match value {
            Obj::Int(value) => Obj::Int(value.negate()),
            Obj::Float(value) => Obj::Float(-value),
            Obj::Complex(re, im) => Obj::Complex(-re, -im),
            other => other,
        });
    }
    convert_number(expr)
}

fn insert(pairs: &mut Vec<(Obj, Obj)>, key: Obj, value: Obj) -> Parse<()> {
    if let Some(kind) = key.unhashable() { return Err(EvalError::Type(format!("unhashable type: '{kind}'"))); }
    match pairs.iter_mut().find(|(existing, _)| existing.equals(&key)) {
        Some(slot) => slot.1 = value,
        None => pairs.push((key, value)),
    }
    Ok(())
}

fn convert(expr: &Expr) -> Parse<Obj> {
    match &expr.node {
        Node::Constant(value) => Ok(value.clone()),
        Node::Tuple(items) => Ok(Obj::Tuple(items.iter().map(convert).collect::<Parse<_>>()?)),
        Node::List(items) => Ok(Obj::List(items.iter().map(convert).collect::<Parse<_>>()?)),
        Node::Set(items) => {
            let mut set: Vec<Obj> = Vec::new();
            for item in items {
                let value = convert(item)?;
                if let Some(kind) = value.unhashable() { return Err(EvalError::Type(format!("unhashable type: '{kind}'"))); }
                if !set.iter().any(|existing| existing.equals(&value)) { set.push(value); }
            }
            Ok(Obj::Set(set))
        }
        Node::EmptySetCall => Ok(Obj::Set(Vec::new())),
        Node::Dict(pairs) => {
            let mut dict = Vec::new();
            for (key, value) in pairs {
                let key = match key { Some(key) => convert(key)?, None => return Err(malformed(None)) };
                let value = convert(value)?;
                insert(&mut dict, key, value)?;
            }
            Ok(Obj::Dict(dict))
        }
        Node::Binary(op, left, right) => {
            let a = convert_signed(left)?;
            let b = convert_number(right)?;
            if let (Some(re), Obj::Complex(bre, bim)) = (match &a { Obj::Int(v) => Some(v.to_f64()), Obj::Float(v) => Some(*v), _ => None }, &b) {
                return Ok(if *op == '+' { Obj::Complex(re + bre, *bim) } else { Obj::Complex(re - bre, -bim) });
            }
            convert_signed(expr)
        }
        _ => convert_signed(expr),
    }
}

/// `ast.literal_eval(text)`.
pub fn literal_eval(text: &str) -> Result<Obj, EvalError> {
    let expr = parse(text.trim_start_matches([' ', '\t']))?;
    convert(&expr)
}
