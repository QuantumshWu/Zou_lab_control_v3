//! Matplotlib 3.10.8's artist drawing semantics, natively: from a resolved
//! artist tree ([`ir::Figure`]) to exactly the calls matplotlib's Python
//! artist code makes into its C++ `RendererAgg` -- same order, same floats,
//! same graphics-context state -- through the [`backend::Backend`] trait, with
//! text measured through [`fonts::Fonts`] (FT2Font's operations).
//!
//! Anything outside the subset zlc's figures use fails with
//! [`Error::Unsupported`]; nothing is approximated.
pub mod agg;
pub mod axes;
pub mod axis;
pub mod backend;
pub mod collections;
pub mod colors;
pub mod draw;
pub mod fonts;
pub mod ftfonts;
pub mod image;
pub mod ir;
pub mod lines;
pub mod mathtext;
pub mod patches;
pub mod query;
pub mod record;
pub mod text;
pub mod transform;
pub mod wire;

pub use draw::draw;

use std::fmt;

#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    /// The IR asks for a matplotlib feature this port does not implement.
    Unsupported(String),
    /// The IR is inconsistent (a dangling node, a wrong-shaped array, ...).
    Invalid(String),
    /// The backend or the fonts failed.
    Backend(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Unsupported(what) => write!(f, "unsupported: {what}"),
            Error::Invalid(what) => write!(f, "invalid IR: {what}"),
            Error::Backend(what) => write!(f, "backend: {what}"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn unsupported<T>(what: impl Into<String>) -> Result<T> { Err(Error::Unsupported(what.into())) }
pub(crate) fn invalid<T>(what: impl Into<String>) -> Result<T> { Err(Error::Invalid(what.into())) }
