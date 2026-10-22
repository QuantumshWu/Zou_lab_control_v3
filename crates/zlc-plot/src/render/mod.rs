//! The renderer policy: what each plot kind draws, decided exactly as
//! `zlc_plot.rendering` (`MatplotlibRenderer`) decides it, resolved into
//! zlc-figure's artist IR and painted into the live front.
//!
//! # Layers
//!
//! * [`mpl`]: the matplotlib object model the renderer keeps -- a
//!   [`mpl::figure::FigureModel`] of [`mpl::axes::AxesModel`]s (box, view
//!   limits, two [`mpl::axes::AxisModel`]s with tick parameters and tick
//!   policy, titles, spines, ordered [`mpl::axes::Child`] artists) built the
//!   way matplotlib's constructors leave them under the style ([`mpl::rc`]),
//!   and its IR (`FigureModel::to_ir`), transform DAG included ([`mpl::nodes`]).
//!   Matplotlib's own AutoLocator/ScalarFormatter is [`mpl::ticker`].
//! * [`core`]: the [`core::Renderer`] (figure from the SurfacePlan,
//!   `relayout`, `present` in `MatplotlibRenderer.present`'s order,
//!   pixel-snapped boxes, home/requested views, labels, titles, grid, the
//!   automatic-limit memo) and [`core::Kind`], the one trait a plot kind
//!   implements. [`chrome`] is `_sync_axes_chrome`/`_fit_tick_label_text`.
//! * Kinds: [`curve`] (series machinery shared by Curve, Rolling and facet
//!   curve cells: prepared scene, Line2D/error-bar materialization,
//!   envelopes, isolated glyphs, slots, series focus), [`rolling`],
//!   [`histogram`] (bar collections, count limits, linear/log counts) and
//!   [`pulse`] (the pulse timeline: rows, blocks, slots, periods, brackets).
//! * [`front`]: the live front as `_compose_frame` paints it -- background
//!   with the dynamic artists withheld, dynamic artists in full-draw
//!   stacking, the series painted by zlc-render's kernels at their depth.
//! * [`panel`]: the session's `PanelRenderer` seam over all of this.
//!
//! # Adding a kind or an overlay layer
//!
//! A kind implements [`core::Kind`] (its `render` is the Python handler's
//! `render`) and is returned by [`core::kind_for`]; its retained tables live
//! beside [`curve::CurveTables`]/[`rolling::RollingTables`] on the renderer,
//! new artist types are new [`mpl::axes::Child`] variants with their IR in
//! `AxesModel::child_ir`. Its renderer-owned artists must be listed where
//! `front::Renderer::dynamic_entries` collects `_dynamic_artists`. An
//! overlay layer (selectors, fit, classifier, point overlay) adds its
//! artists to the axes, records them in `Renderer::overlay_artists` (and
//! its fit lines in `Renderer::fit_lines`), and runs inside `present` where
//! `MatplotlibRenderer.present` updates it (after the requested view, before
//! `_settle_owned_boxes` and `_sync_axes_chrome`).
//!
//! Oracle: `tests/gen/render_capture.py` records every renderer of the corpus
//! (frames, session calls, materialized IR, front sha256);
//! `tests/render_corpus.rs` replays them (IR equality, `ZLC_RENDER_PIXELS`
//! for fronts, `ZLC_RENDER_BENCH` for timing). `tests/gen/render_fuzz.py`
//! writes random sessions in corpus form for the same capture and replay.
//! `tests/session_replay.rs` tees the renderer into the session
//! (`committed_scenarios_native_renderer`, `ZLC_SESSION_NATIVE`): every
//! query it answers and front it paints against the recorded renderer's.
pub mod blake2s;
pub mod chrome;
pub mod classifier;
pub mod colormaps;
pub mod core;
pub mod curve;
pub mod envelope;
pub mod facet;
pub mod fit;
pub mod fonts;
pub mod foreground;
pub mod frame;
pub mod front;
pub mod geometry;
pub mod hashkey;
pub mod height_bars;
pub mod histogram;
pub mod image;
pub mod labels;
pub mod limits;
pub mod mpl;
pub mod points;
pub mod panel;
pub mod profile;
pub mod pulse;
pub mod replay;
pub mod rolling;
pub mod scale;
pub mod selectors;
