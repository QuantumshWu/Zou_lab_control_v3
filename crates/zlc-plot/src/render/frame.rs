//! `RenderFrame`: the complete immutable input of one presentation (the
//! session's type: payload, display state, effects, overlays, selectors,
//! facet state, view limits, series readout).
pub use crate::session::renderer::RenderFrame;

/// A view request per axis: `(x limits or None, y limits or None)`.
pub type ViewLimits = (Option<(f64, f64)>, Option<(f64, f64)>);
