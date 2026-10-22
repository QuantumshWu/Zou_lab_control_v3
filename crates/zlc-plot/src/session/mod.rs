//! The plot session (`zlc_plot.session.PlotSession` and its gesture, fit
//! and state modules): one accepted scientific scene, its interactions and
//! its transactional presentation, drawn through a [`PanelRenderer`].

pub mod control;
pub mod core;
pub mod evaluation;
pub mod export;
pub mod fit;
pub mod fit_layer;
pub mod gesture;
pub mod overlay;
pub mod preparation;
pub mod projected;
pub mod renderer;

pub use overlay::{real_values, FitEllipseGlyph, FitOverlay, FitPolyline, ImagePointOverlay, OverlayArray, PointStatus};
pub use renderer::{AxesGeometry, PanelRenderer, RenderFrame, RendererFactory, RgbaFront, SeriesAction};
pub use core::{AcceptedFit, FitAnswer, PlotSession, SessionServices};
pub use fit_layer::EngineFitLayer;
pub use gesture::{PointerAction, PointerEvent};
pub use preparation::{DataPreparation, PreparedPlot};
pub use projected::{FitProjection, PlotInput, SnapshotRef};
