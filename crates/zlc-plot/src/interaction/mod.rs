//! The plot interaction model, ported behaviour-exact: `zlc_plot.selectors`
//! beyond its value types (those are [`crate::configuration::selectors`]),
//! `_gesture_engine`, `_selector_scene`, `_axis_transform`, and the pure
//! rules of `_session_gesture` / `PlotSession._raster_pointer_event`.
//! Nothing here draws or holds a widget: state machines and pure functions,
//! facts in, decisions out.
//!
//! # Pieces
//!
//! * [`transform`] — [`AxisTransform`], the axes snapshot a window sends
//!   (top-origin box fractions, drawn limits and scales, canonical limits and
//!   display-unit mappings, role, cell): pointer fraction ↔ display ↔
//!   canonical coordinates, the interaction scale and drawn span of an axis.
//! * [`controller`] — [`SelectorController`]: the committed selectors (one per
//!   kind, as [`HeldSelector`]s carrying Python's object identity), their
//!   revisions and one transient pointer candidate; `install`,
//!   `retarget_facet`, `remove` / `restore_removed` / `rollback_install`,
//!   `commit_finished`, `pointer_down` / `pointer_move` / `finish_gesture` /
//!   `cancel` / `lost_pointer_capture`, `candidate_state`, `snapshot`
//!   ([`SelectorSnapshot`]), `fork` (= clone).  The owner locks it.
//! * [`drag`] — [`DragHandle`], [`GestureState`], [`drag_numeric_range`] (the
//!   one bounded range drag: NEW / BODY slide in the axis' straight space /
//!   LOW / HIGH, clamps, minimum span), [`dragged_value`], [`clamp`],
//!   [`clamp_range`], [`stored_state`].
//! * [`engine`] — [`range_endpoint_hit`], [`area_drag_handle`],
//!   [`pan_rectangle`], [`ColorLimitDrag`], and the gesture records
//!   ([`SelectorGesture`], [`ColorGesture`], [`PanGesture`], [`OrbitGesture`],
//!   [`PickGesture`] in one [`PointerGesture`]) addressed by [`AxesId`].
//! * [`gesture`] — the session's rules: [`event_canonical`] (the drag's
//!   reading fallback), [`ClickHistory`] (double clicks), [`hit_selector`] +
//!   [`session_area_hit`], [`selector_press`] / [`selector_motion`] /
//!   [`selector_release`] / [`cancel_selector_gesture`],
//!   [`begin_color_limit_drag`] / [`color_motion`] / [`color_release`],
//!   [`update_pan`], [`wheel_viewport`] / [`wheel_camera_zoom`],
//!   [`zoom_to_selection`], [`selector_bounds`], [`is_degenerate_selector`],
//!   [`raster_pixel`], [`pointer_latch`], [`InteractionDefaults`].
//! * [`scene`] — [`SelectorSceneOwner::build`]: every selector and the colour
//!   limits as lines, markers and labels (with their formatted numbers and
//!   stacked label slots), in data coordinates of the [`SelectorTarget`] axes.
//! * [`labels`] — [`selector_precision`], [`selector_number`],
//!   [`series_identity`].
//! * [`pymath`] — the Python forms of `min`/`max`/`sorted`, `math.hypot`,
//!   float `**`, `/` and the log axis' `10 ** x` (with `OverflowError` /
//!   `ZeroDivisionError` where Python raises).
//!
//! Every fallible call returns a [`PlotError`](crate::configuration::PlotError)
//! whose kind is the Python exception class at the same step.
//!
//! # How a native session drives it
//!
//! The session owns one [`SelectorController`], an `Option<PointerGesture>`,
//! a [`ClickHistory`] and its own facts (spec kind, facet focus, display
//! projection, the renderer's axes).  Per raster pointer event:
//!
//! 1. [`RasterAction::parse`] and [`raster_pixel`]: the exact physical pixel
//!    and the whole pixel the synthetic Matplotlib event carried; the axes
//!    snapshot names the [`AxisTransform`].
//! 2. Press: [`ClickHistory::is_double_click`]; cancel the held gesture; route
//!    (facet chooser, 3-D scene, button).  Middle: [`zoom_to_selection`] on a
//!    double click, else a [`PanGesture`] from
//!    [`AxisTransform::display_point`].  Otherwise [`event_canonical`], then
//!    right button → install / remove the crosshair; distribution rail →
//!    [`begin_color_limit_drag`]; else [`hit_selector`] (Areas through
//!    [`session_area_hit`] on their displayed value) and [`selector_press`].
//! 3. Move: pan → [`update_pan`] (re-render when it moved); orbit →
//!    [`OrbitGesture::follow`] (commit the camera parameters); selector →
//!    [`selector_motion`] (`started` → the renderer begins the selector
//!    gesture; [`SelectorStep`] → preview + UPDATED, or cancel); colour →
//!    [`color_motion`] (preview the candidate).
//! 4. Release: middle on a pan → [`update_pan`], clear, commit the candidate
//!    viewport; pick → [`is_pick_click`]; colour → [`color_release`];
//!    selector → [`selector_release`] ([`SelectorRelease`]: blank click
//!    removes, `Commit` goes through [`SelectorController::commit_finished`]
//!    and is announced COMMITTED, `Degenerate` removes or repaints; on an
//!    error cancel and report).
//! 5. Cancel / lost capture: [`cancel_selector_gesture`] for a started drag,
//!    then clear the gesture.
//! 6. Wheel: [wheel_ticks], then [wheel_viewport] from the committed
//!    continuous view, or [wheel_camera_zoom]. Sampling never snaps navigation.
//! 7. Frame: [`SelectorSceneOwner::build`] over the snapshot's states (with
//!    a derived classifier threshold in its kind's place) and the colour
//!    state; [`pointer_latch`] tells the frontend whether to keep the button
//!    latched and on which axes.
//!
//! `tests/interaction.rs` (`Pipeline`) is a reference composition of steps
//! 2–5 for one 2-D panel, replayed against the real `_session_gesture`.

pub mod controller;
pub mod drag;
pub mod engine;
pub mod gesture;
pub mod labels;
pub mod pymath;
pub mod scene;
pub mod transform;

pub use controller::{HeldSelector, SelectorController, SelectorSnapshot};
pub use drag::{clamp, clamp_range, drag_numeric_range, dragged_value, stored_state, DragFrame, DragHandle, GestureState};
pub use engine::{area_drag_handle, pan_rectangle, range_endpoint_hit, AxesId, CameraAngles, ColorGesture, ColorLimitDrag,
                 OrbitGesture, PanGesture, PickGesture, PointerGesture, SelectorGesture};
pub use gesture::{begin_color_limit_drag, cancel_selector_gesture, color_motion, color_release, drawn_bounds, event_canonical, hit_selector,
                  is_degenerate_selector, is_pick_click, new_area_draft, selector_motion, selector_press, selector_release, ColorRelease,
                  SelectorMotion, SelectorRelease, SelectorStep,
                  pointer_latch, raster_pixel, require_stable_selector, selector_bounds, session_area_hit, threshold_gesture_on_histogram,
                  update_pan, wheel_camera_zoom, wheel_ticks, wheel_viewport, wheel_zoom, zoom_to_selection, ClickHistory,
                  InteractionDefaults, RasterAction, SelectionZoom};
pub use labels::{selector_number, selector_precision, series_identity, series_identity_repr};
pub use scene::{ColorLimitState, Rgba, SampleValue, SceneKind, SelectorItemContext, SelectorLine, SelectorMarkers, SelectorPrimitive,
                SelectorScene, SelectorSceneOwner, SelectorSceneStyle, SelectorTarget, SelectorText};
pub use transform::{Axis, AxisTransform, CanvasSize};
