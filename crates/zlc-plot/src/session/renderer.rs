//! The renderer seam: everything a session asks of its panel's renderer
//! (`MatplotlibRenderer` exactly as `PlotSession` uses it).  Axes are named
//! by [`AxesId`] (the `role[:cell]` gid) and described by plain geometry
//! instead of Matplotlib handles; the session does every pointer and unit
//! computation itself from that geometry.
//!
//! A renderer owns one fixed-layout figure: [`PanelRenderer::present`]
//! prepares a frame's artists (and composes the canvas when asked),
//! the gesture calls repaint transient candidates between presents, and
//! [`PanelRenderer::rgba`] hands out the composed front.  One renderer is
//! driven by one session at a time (`&mut`); different panels' renderers
//! run concurrently.

use std::sync::Arc;

use zlc_core::OwnedBuffer;

use crate::configuration::error::Result;
use crate::configuration::parameters::RenderEffect;
use crate::configuration::selectors::{NumericRange, SelectorKind, SelectorState};
use crate::configuration::spec::PlotSpec;
use crate::configuration::state::{DisplayState, Interaction, Presentation};
use crate::interaction::controller::SelectorSnapshot;
use crate::interaction::engine::{AxesId, CameraAngles};
use crate::interaction::transform::CanvasSize;
use crate::layout::SurfacePlan;
use crate::projection::Payload;
use crate::render::scale::AxisScale;
use crate::style::PlotStyle;

use super::overlay::{FitOverlay, ImagePointOverlay};

/// One painted axes as the session reads it.
#[derive(Clone, Debug, PartialEq)]
pub struct AxesGeometry {
    pub id: AxesId,
    /// The window extent in physical canvas pixels, bottom origin:
    /// `(x0, y0, x1, y1)` (`axes.get_window_extent()`).
    pub bbox: (f64, f64, f64, f64),
    /// `get_xlim()` / `get_ylim()` as drawn (an image's y runs top-down).
    pub x_limits: (f64, f64),
    pub y_limits: (f64, f64),
    /// `renderer.axis_scale(axes, "x" | "y")`.
    pub x_scale: AxisScale,
    pub y_scale: AxisScale,
}

impl AxesGeometry {
    /// `Bbox.contains`: inclusive on both edges, either orientation.
    pub fn contains(&self, x: f64, y: f64) -> bool {
        let (x0, y0, x1, y1) = self.bbox;
        ((x0 <= x && x <= x1) || (x0 >= x && x >= x1)) && ((y0 <= y && y <= y1) || (y0 >= y && y >= y1))
    }
}

/// Complete immutable input of one presentation transaction (`RenderFrame`).
#[derive(Clone, Debug)]
pub struct RenderFrame {
    pub payload: Arc<Payload>,
    pub state: Arc<DisplayState>,
    pub effects: RenderEffect,
    pub data_revision: u64,
    pub fit_overlays: Arc<[FitOverlay]>,
    pub fit_model_id: Option<String>,
    /// One per classified distribution, aligned with the thresholds and labels.
    pub classifier_overlays: Arc<[FitOverlay]>,
    /// Display-unit thresholds, `None` where a distribution has none.
    pub classifier_thresholds: Vec<Option<f64>>,
    pub classifier_labels: Vec<String>,
    pub image_overlay: Option<Arc<ImagePointOverlay>>,
    /// Every selector in the space the axes are drawn in (display units).
    pub selectors: SelectorSnapshot,
    pub facet_index: Option<usize>,
    pub facet_focus_index: Option<usize>,
    /// Requested `(x, y)` axes limits; an absent axis autoscales.
    pub view_limits: Option<(Option<(f64, f64)>, Option<(f64, f64)>)>,
    /// The series readout to restore with this frame, if any.
    pub presentation: Option<Presentation>,
}

/// One series hover/lock step (`series_focus(action, ...)`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SeriesAction { Press, Move, Release, Leave, Clear }

impl SeriesAction {
    pub fn value(self) -> &'static str {
        match self { SeriesAction::Press => "press", SeriesAction::Move => "move", SeriesAction::Release => "release",
                     SeriesAction::Leave => "leave", SeriesAction::Clear => "clear" }
    }
}

/// One composed canvas: `height x width x 4` RGBA8, row-major, top row first.
#[derive(Clone)]
pub struct RgbaFront {
    pub width: u32,
    pub height: u32,
    pub pixels: OwnedBuffer<u8>,
}

/// One panel's renderer.  Errors carry the Python exception class the
/// Matplotlib renderer raises at the same call.
pub trait PanelRenderer: Send {
    /// The layout plan currently built (`renderer.plan`).
    fn plan(&self) -> &SurfacePlan;
    /// The specification the figure draws (`renderer.spec`).
    fn spec(&self) -> &PlotSpec;
    /// `renderer.spec = spec`: takes effect with the next present or relayout.
    fn set_spec(&mut self, spec: PlotSpec);
    /// Rebuild the figure for `plan` (`relayout`); nothing is composed.
    fn relayout(&mut self, plan: SurfacePlan, facet_index: Option<usize>, facet_focus_index: Option<usize>) -> Result<()>;
    /// Prepare every scene layer of `frame`; compose one canvas front when
    /// `compose` (an export session never composes).
    fn present(&mut self, frame: &RenderFrame, compose: bool) -> Result<()>;
    /// Compose one complete frame from the current artists.
    fn draw(&mut self) -> Result<()>;

    /// The physical-pixel canvas pointer events are measured in.
    fn canvas_size(&self) -> CanvasSize;
    /// Every axes of one role, in creation order, visible or not
    /// (`renderer.axes.get(role, ())`).
    fn axes(&self, role: &str) -> Vec<AxesId>;
    /// The visible axes in figure order (`figure.axes` filtered by visibility).
    fn visible_axes(&self) -> Vec<AxesGeometry>;
    /// One axes' current geometry, visible or not; `None` for an unknown id.
    fn geometry(&self, axes: &AxesId) -> Option<AxesGeometry>;
    /// The one painted surface a pointer gesture and its chrome act on.
    fn primary_axes(&self) -> AxesId;
    /// The visible axes owning a physical pixel (bottom origin), the
    /// primary axes first, then the visible axes from the top of the stack.
    fn interactive_axes_at(&self, x: f64, y: f64) -> Option<AxesId>;
    /// The overview cell `axes` shows, if it is a visible facet cell.
    fn facet_index_for_axes(&self, axes: &AxesId) -> Option<usize>;

    /// The clim the active image artist paints; `None` without one (the
    /// renderer's TypeError).
    fn resolved_color_limits(&self) -> Option<(f64, f64)>;

    /// Start a selector drag of `kind`; `compose` repaints at once.
    fn begin_selector_gesture(&mut self, kind: SelectorKind, compose: bool) -> Result<()>;
    /// Start a colour-limit drag with its display-unit candidate.
    fn begin_color_limit_gesture(&mut self, candidate: NumericRange) -> Result<()>;
    /// Paint a display-unit selector candidate; `false` when no gesture of
    /// its kind is held.
    fn preview_selector(&mut self, state: &SelectorState) -> Result<bool>;
    /// Paint a display-unit colour-limit candidate; `false` when no
    /// colour-limit gesture is held.
    fn preview_color_limit_candidate(&mut self, candidate: NumericRange) -> Result<bool>;
    /// Forget the gesture candidate and its captured region.
    fn end_selector_gesture(&mut self);
    /// The readings the classifier thresholds show between frames.
    fn set_classifier_labels(&mut self, labels: &[String]);
    /// Confine composes to `axes` while a hand drags its view (and capture
    /// the background now); `None` lifts the confinement.
    fn set_view_dragging(&mut self, axes: Option<&AxesId>) -> Result<()>;
    /// Compose once now under the current gesture partition.
    fn capture_gesture_background(&mut self) -> Result<()>;
    /// `set_view_dragging` on the 3D scene's axes, or lifted.
    fn set_height_bars_dragging(&mut self, dragging: bool) -> Result<()>;

    /// Whether the current frame draws the image as the 3D height-bar scene.
    fn height_bars_scene(&self) -> bool;
    /// The camera the last height-bar render used.
    fn height_bars_camera(&self) -> Option<CameraAngles>;
    /// The data coordinates of the bar under a physical pixel, if any.
    fn height_bars_pick(&self, x: f64, y: f64) -> Option<(f64, f64)>;

    /// The committed series lock (`{"series_lock": ...}`).
    fn series_interaction(&self) -> Interaction;
    /// The presented series readout (`{"series_readout": ...}`).
    fn series_presentation(&self) -> Presentation;
    /// One hover/lock step at a physical pixel without repainting; whether it
    /// changed (for a release: whether it handled the click).
    fn series_focus(&mut self, action: SeriesAction, axes: Option<&AxesId>, x: f64, y: f64, hit_radius: f64,
                    click_radius: f64) -> Result<bool>;
    /// Step a locked series by one wheel burst without repainting; whether
    /// the wheel was taken.
    fn series_focus_scroll(&mut self, axes: Option<&AxesId>, step: f64) -> Result<bool>;

    /// Monotonic count of composed frames.
    fn raster_generation(&self) -> u64;
    /// The current front; draws first when the composed frame is stale.
    fn rgba(&mut self) -> Result<RgbaFront>;
    /// Encode the figure as a file of `format` ("png", "svg", "pdf") at `dpi`.
    fn save(&mut self, format: &str, dpi: f64, restore_display: bool) -> Result<Vec<u8>>;
}

/// Makes the renderer of a new session (`MatplotlibRenderer(spec, plan, style=...)`)
/// for the panel it paints; a saved figure's export session paints none.
pub trait RendererFactory: Send + Sync {
    fn create(&self, panel: Option<&str>, spec: &PlotSpec, plan: &SurfacePlan, style: &PlotStyle) -> Result<Box<dyn PanelRenderer>>;
}
