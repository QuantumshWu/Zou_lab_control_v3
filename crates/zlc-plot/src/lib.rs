//! The native plot scene: preset layout, tick policy and the drawing of every
//! plot kind, reproducing the scientific renderer's pictures exactly.

pub mod configuration;
pub mod document;
pub mod figure;
pub mod fit;
pub mod interaction;
pub mod kinds;
pub mod layout;
pub mod numeric;
pub mod projection;
pub mod pulse;
pub mod pystr;
pub mod render;
pub mod roi;
pub mod session;
pub mod style;
pub mod ticks;
pub mod vector;

/// Python's float formatting lives with the data plane, which writes JSON too;
/// so do units and their decimal arithmetic (v2 `zlc_data.units`), which the
/// workbench reads without the plot engine.
pub use zlc_core::{decimal, pyfmt, units};
