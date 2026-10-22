//! The plot configuration layer: what a panel shows and how it is edited,
//! ported behaviour-exact from `zlc_plot` (kinds, specs, parameters,
//! semantics, the kind registry and default table, state, selector value
//! types, the recipe codec, `resolve_configuration`) and the desktop panel
//! description.  No samples, renderer or session are involved.
//!
//! # Public API
//!
//! * [`PlotSchema`] wraps the native data plane's `zlc_core::schema::DatasetSchema`
//!   once per schema value and memoizes the facts read from it (the
//!   indexed-history layout).
//! * [`PlotSpec`] and its kinds ([`CurvePlot`], [`ImagePlot`],
//!   [`HistogramPlot`], [`RollingPlot`], [`FacetGridPlot`],
//!   [`PulseTimelinePlot`]) are the typed specifications; construction
//!   validates like the Python dataclasses.
//! * [`fitting_spec`] / [`default_spec`] / [`chosen_spec`] read the one default
//!   table; [`composed_spec`] is the single composition authority for semantic
//!   edits (wire values in, a specification or the Python exception class
//!   out); [`describe_semantics`] gives the fate table with choices;
//!   [`rebased_spec`] carries intent onto another schema.
//! * [`parameter_schema_for`] / [`parameter_schema_for_kind`] give every
//!   kind's display-parameter contract ([`ParameterSchema`]).
//! * [`resolve_configuration`] resolves a [`ResolveRequest`] (schema, wire
//!   target or typed [`ConfigurationUpdates`], base/presented candidates) into
//!   an immutable [`ResolvedConfiguration`] with its display state,
//!   selectors, viewport, focus, thresholds, fit target, [`Signature`] and
//!   wire [`ResolvedConfiguration::target`].
//! * [`describe_panel`] / [`panel_controls`] produce the JSON controls a
//!   desktop UI renders.
//! * The fit engine is reached only through the [`FitLayer`] trait.
//!
//! Wire values are [`Value`]s with Python's equality and text forms; every
//! fallible operation returns a [`PlotError`] whose [`ErrorKind`] is the
//! Python exception class.
//!
//! # Where this differs from the Python, deliberately
//!
//! * Error messages are informative but not character-identical; the class is.
//! * A fit layer must be supplied for the first resolution (Python built a
//!   default `FitEngine`); later ones inherit it, as Python inherited `engine`.
//! * An integral float of magnitude 2^127 or more stays a float coordinate
//!   (Python makes it an arbitrarily large int); JSON integers beyond 64 bits
//!   read as floats ([`Value::from_json`]).
//! * `repr` of text uses an approximation of `str.isprintable` outside the
//!   Latin and general-punctuation ranges (it only orders threshold scopes).
//! * `float(text)` accepts ASCII digits only (Python also takes other Unicode
//!   decimal digits).
//! * Schema equality is zlc-core's: two mappings naming the same rows are equal
//!   however their codes and repeats were factored.
//! * Coordinate lookup scans a non-index axis linearly (Python sorted or hashed
//!   it once per axis); results are identical.

pub mod controls;
pub mod data_contract;
pub mod display;
pub mod error;
pub mod fit;
pub mod handlers;
pub mod kinds;
pub mod metadata;
pub mod ordered;
pub mod parameters;
pub mod recipe;
pub mod resolve;
pub mod selectors;
pub mod semantics;
pub mod spec;
pub mod state;
pub mod value;

pub use controls::{describe_panel, panel_controls, plot_kinds, parameter_controls, parameter_controls_for_kind, ControlKind, ParameterControl};
pub use data_contract::{classify_axes, indexed_history_layout, resolve_axis, schemas_continue, AxisFamilies, IndexedHistoryLayout, PlotSchema, ResolvedAxis};
pub use display::{accepts_classifier_thresholds, non_portable_display_names, parameter_schema_for, parameter_schema_for_kind, validate_authored_display, FACET_FIT_PARAMETER};
pub use error::{ErrorKind, PlotError, Result};
pub use fit::{EmptyFitLayer, FitLayer, FitModelInfo, FitQuantity, FitRequest, FitResolution, FitTargetQuery, FitUnits, UnitRelation, FIT_TARGET_FIELDS};
pub use handlers::{admits, chosen_spec, default_spec, fitting_spec, handler, label_roles, FitTargetKind, KindHandler, LabelRole, HANDLERS};
pub use kinds::{AxisDomain, AxisRef, Coordinate, PlotKind};
pub use ordered::OrderedMap;
pub use parameters::{FrozenParameters, Normalizer, ParameterSchema, ParameterSpec, RenderEffect, ValueType};
pub use recipe::{decode_plot_recipe, encode_plot_recipe, DecodedRecipe, RecipeParts};
pub use resolve::{resolve_configuration, scientific_context_signature, ConfigurationUpdates, Defaults, InteractionUpdates, OverlayHandle, PreparationKey, ResolveRequest, ResolvedConfiguration, ScientificContext, Signature};
pub use selectors::{normalize_classifier_threshold_targets, normalize_viewport, CrosshairPoint, NumericRange, RectangleRange, SelectorKind, SelectorSnapshot, SelectorState, SelectorValue, ThresholdTarget, Viewport};
pub use semantics::{composed_spec, describe_semantics, fate_field_name, rebased_spec, updated_spec, CycleChoices, SemanticDescription, SemanticField};
pub use spec::{CellPlot, CurvePlot, FacetGridPlot, HistogramPlot, ImagePlot, PlotLabels, PlotSpec, PulseTimelinePlot, Reduction, RelimMode, RollingPlot};
pub use state::{DisplayState, DisplayStateStore, Interaction, InteractionUpdate, Presentation, PresentationUpdate, SeriesReadout, SeriesTarget};
pub use value::{Value, ValueMap};
