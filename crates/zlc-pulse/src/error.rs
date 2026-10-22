//! One error type for the pulse stack, carrying the Python exception class the
//! v2 package raised for the same refusal, so callers and goldens can tell a
//! wrong type from a wrong value without parsing text.
use std::fmt;
use std::fmt::Write as _;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// `TypeError`: the value is of the wrong kind.
    Type,
    /// `ValueError`: the value is of the right kind and not allowed.
    Value,
    /// `KeyError`: a name that does not exist.
    Key,
    /// `RuntimeError`: the device or the session is not in a state that allows it.
    Runtime,
    /// `TimeoutError`: a deadline passed before an answer.
    Timeout,
    /// `ConnectionError` and its subclasses: the channel to a board is gone.
    Connection,
    /// `OSError`: a file or port operation failed.
    Io,
}

impl Kind {
    pub fn python_name(self) -> &'static str {
        match self {
            Kind::Type => "TypeError", Kind::Value => "ValueError", Kind::Key => "KeyError",
            Kind::Runtime => "RuntimeError", Kind::Timeout => "TimeoutError", Kind::Connection => "ConnectionError",
            Kind::Io => "OSError",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error { pub kind: Kind, pub message: String }

impl Error {
    pub fn new(kind: Kind, message: impl Into<String>) -> Self { Error { kind, message: message.into() } }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str(&self.message) }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        let kind = match error.kind() {
            std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock => Kind::Timeout,
            std::io::ErrorKind::ConnectionReset | std::io::ErrorKind::ConnectionAborted | std::io::ErrorKind::ConnectionRefused
            | std::io::ErrorKind::BrokenPipe | std::io::ErrorKind::NotConnected | std::io::ErrorKind::UnexpectedEof => Kind::Connection,
            _ => Kind::Io,
        };
        Error::new(kind, error.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;

pub fn value_error<T>(message: impl Into<String>) -> Result<T> { Err(Error::new(Kind::Value, message)) }
pub fn type_error<T>(message: impl Into<String>) -> Result<T> { Err(Error::new(Kind::Type, message)) }
pub fn runtime_error<T>(message: impl Into<String>) -> Result<T> { Err(Error::new(Kind::Runtime, message)) }

/// Python `repr(text)`: the quoting every refusal uses for a name.
pub fn repr(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') { '"' } else { '\'' };
    let mut result = String::with_capacity(text.len() + 2);
    result.push(quote);
    for character in text.chars() {
        match character {
            '\\' => result.push_str("\\\\"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            c if c == quote => { result.push('\\'); result.push(c); }
            c if printable(c) => result.push(c),
            c => {
                let code = c as u32;
                if code < 0x100 { let _ = write!(result, "\\x{code:02x}"); }
                else if code < 0x10000 { let _ = write!(result, "\\u{code:04x}"); }
                else { let _ = write!(result, "\\U{code:08x}"); }
            }
        }
    }
    result.push(quote);
    result
}

/// `repr` of an optional name: `None` or the quoted text.
pub fn repr_opt(text: Option<&str>) -> String { text.map_or_else(|| "None".to_owned(), repr) }

/// Python `str.isprintable` over the ranges a pulse name can meet.
fn printable(character: char) -> bool {
    let code = character as u32;
    if character == ' ' { return true; }
    !(code < 0x20 || (0x7f..=0xa0).contains(&code) || code == 0xad || (0x2000..=0x200f).contains(&code)
        || (0x2028..=0x202f).contains(&code) || (0x205f..=0x2064).contains(&code) || code == 0x3000
        || (0xd800..=0xf8ff).contains(&code) || code == 0xfeff || (0xfff9..=0xfffb).contains(&code))
}

/// Python `str.isspace` for one character.
pub fn python_space(character: char) -> bool { character.is_whitespace() || matches!(character, '\u{1c}'..='\u{1f}') }

/// Python `text.strip()`.
pub fn py_strip(text: &str) -> &str { text.trim_matches(python_space) }

/// `[A-Za-z_][A-Za-z0-9_]*`, the identifier rule of every stable id.
pub fn is_identifier(text: &str) -> bool {
    let mut characters = text.chars();
    characters.next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && characters.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Python `repr` of a tuple of strings: `('a', 'b')`, `('a',)`, `()`.
pub fn repr_tuple(items: &[&str]) -> String {
    match items {
        [] => "()".to_owned(),
        [one] => format!("({},)", repr(one)),
        _ => format!("({})", items.iter().map(|item| repr(item)).collect::<Vec<_>>().join(", ")),
    }
}

/// Python `repr` of a list of strings.
pub fn repr_list(items: &[&str]) -> String {
    format!("[{}]", items.iter().map(|item| repr(item)).collect::<Vec<_>>().join(", "))
}
