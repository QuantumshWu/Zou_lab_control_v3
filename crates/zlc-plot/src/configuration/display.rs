//! Every plot kind's display-parameter contract (the schema half of
//! `zlc_plot.specs`): names, effects, defaults, labels, choices, bounds and
//! the portable/segmented flags, in declaration order.

use std::collections::BTreeSet;
use std::sync::Arc;

use crate::style::PlotStyle;

use super::error::{PlotError, Result};
use super::kinds::PlotKind;
use super::parameters::{
    enum_choices, pulse_time_choices, spelled_choices, words, Normalizer, ParameterSchema, ParameterSpec,
    RenderEffect as E, TransitionRules, ValueType, LIMIT_MODE_NAMES,
};
use super::spec::{PlotSpec, GRID_CELL_KINDS};
use super::value::{Value, ValueMap};

/// The facet grid's per-cell fit readout parameter.
pub const FACET_FIT_PARAMETER: &str = "facet_fit_parameter";

const TEXT_EFFECTS: E = E::TEXT;
const DISPLAY_UNIT_EFFECTS: E = E(E::VIEW_PROJECTION.0 | E::PAYLOAD_PROJECTION.0 | E::BASE_GEOMETRY.0 | E::AXIS_TRANSFORM.0
    | E::TEXT.0 | E::CHROME.0 | E::OVERLAY.0 | E::INTERACTION_REPROJECT.0);
const FACET_UNIT_EFFECTS: E = E(E::VIEW_PROJECTION.0 | E::PAYLOAD_PROJECTION.0 | E::TEXT.0);
const PULSE_TIME_UNIT_EFFECTS: E = E(E::AXIS_TRANSFORM.0 | E::TEXT.0 | E::CHROME.0 | E::OVERLAY.0 | E::INTERACTION_REPROJECT.0);
const AXIS_LIMIT_EFFECTS: E = E(E::AXIS_TRANSFORM.0 | E::CHROME.0 | E::OVERLAY.0 | E::INTERACTION_REPROJECT.0);
const HISTOGRAM_VALUE_AXIS_EFFECTS: E = E(AXIS_LIMIT_EFFECTS.0 | E::PAYLOAD_PROJECTION.0);
const HISTOGRAM_REPRESENTATION_EFFECTS: E = E(E::BASE_GEOMETRY.0 | E::AXIS_TRANSFORM.0 | E::CHROME.0 | E::OVERLAY.0
    | E::FIT_SELECTION.0 | E::INTERACTION_REPROJECT.0);
const HISTOGRAM_PROJECTION_EFFECTS: E = E(E::PAYLOAD_PROJECTION.0 | HISTOGRAM_REPRESENTATION_EFFECTS.0);
const IMAGE_COLOR_EFFECTS: E = E(E::BASE_STYLE.0 | E::CHROME.0 | E::OVERLAY.0 | E::INTERACTION_REPROJECT.0);
const ROLLING_DISTRIBUTION_BIN_EFFECTS: E = E(E::BASE_GEOMETRY.0 | E::AXIS_TRANSFORM.0 | E::CHROME.0);
const UNCERTAINTY_EFFECTS: E = E(E::PAYLOAD_PROJECTION.0 | E::BASE_GEOMETRY.0 | E::FIT_SELECTION.0);
const ROLLING_WINDOW_EFFECTS: E = E(E::VIEW_PROJECTION.0 | E::PAYLOAD_PROJECTION.0 | E::BASE_GEOMETRY.0 | E::AXIS_TRANSFORM.0
    | E::CHROME.0 | E::OVERLAY.0 | E::FIT_SELECTION.0 | E::INTERACTION_REPROJECT.0);

/// The height-bar camera's home view.
pub const HOME_CAMERA: (f64, f64, f64) = (55.0, 30.0, 1.0);

const RELIM_VALUES: [&str; 3] = ["tight", "normal", "fixed"];

fn title_parameter() -> ParameterSpec {
    ParameterSpec::declare("title", ValueType::Text, TEXT_EFFECTS, Value::None)
        .normalized(Normalizer::LabelText).optional().labelled("Title").finished()
}

fn label_parameter(name: &'static str, label: &str) -> ParameterSpec {
    ParameterSpec::declare(name, ValueType::Text, TEXT_EFFECTS, Value::None)
        .normalized(Normalizer::LabelText).optional().labelled(label).finished()
}

fn show_grid_parameter(default: bool) -> ParameterSpec {
    ParameterSpec::declare("show_grid", ValueType::Bool, E::CHROME, Value::Bool(default)).labelled("Grid").finished()
}

fn unit_parameter(name: &'static str, effects: E) -> ParameterSpec {
    let label = words(name.strip_suffix("_display_unit").unwrap_or(name)) + " unit";
    ParameterSpec::declare(name, ValueType::Text, effects, Value::None)
        .normalized(Normalizer::UnitText).optional().labelled(label).finished()
}

fn pulse_x_unit_parameter() -> ParameterSpec {
    ParameterSpec::declare("x_display_unit", ValueType::Text, PULSE_TIME_UNIT_EFFECTS, Value::None)
        .normalized(Normalizer::PulseTimeUnit).optional().labelled("Time unit")
        .with_choices(pulse_time_choices()).segmented().finished()
}

fn relim_parameter(name: &'static str, effects: E, default: &str, label: &str) -> ParameterSpec {
    ParameterSpec::declare(name, ValueType::Text, effects, Value::text(default))
        .normalized(Normalizer::RelimMode).labelled(label).not_portable()
        .with_choices(enum_choices(&RELIM_VALUES)).segmented().finished()
}

fn limit_parameter(name: &'static str, effects: E, label: &str) -> ParameterSpec {
    ParameterSpec::declare(name, ValueType::Number, effects, Value::None)
        .normalized(Normalizer::FiniteOrNone).optional().labelled(label).not_portable().finished()
}

fn flag(name: &'static str, effects: E, default: bool, label: &str) -> ParameterSpec {
    ParameterSpec::declare(name, ValueType::Bool, effects, Value::Bool(default)).labelled(label).finished()
}

fn curve_parameters() -> Vec<ParameterSpec> {
    vec![
        relim_parameter("relim_mode", AXIS_LIMIT_EFFECTS, "normal", "Limits"),
        limit_parameter("y_min", AXIS_LIMIT_EFFECTS, "Y minimum"),
        limit_parameter("y_max", AXIS_LIMIT_EFFECTS, "Y maximum"),
    ]
}

fn bin_count_parameter(effects: E) -> ParameterSpec {
    ParameterSpec::declare("bin_count", ValueType::Int, effects, Value::Int(60))
        .normalized(Normalizer::Integer).labelled("Bins").bounded(1.0, 4096.0, 1.0).finished()
}

fn window_parameter(default: i64, effects: E) -> ParameterSpec {
    ParameterSpec::declare("window", ValueType::Int, effects, Value::Int(i128::from(default)))
        .normalized(Normalizer::Integer).labelled("Window").bounded(1.0, 1_000_000.0, 1.0).finished()
}

fn histogram_parameters() -> Vec<ParameterSpec> {
    vec![
        bin_count_parameter(HISTOGRAM_PROJECTION_EFFECTS),
        flag("threshold_classifier", E::OVERLAY | E::FIT_SELECTION, false, "Threshold classifier"),
        flag("density", HISTOGRAM_REPRESENTATION_EFFECTS | E::TEXT, false, "Density"),
        flag("cumulative", HISTOGRAM_REPRESENTATION_EFFECTS, false, "Cumulative"),
        relim_parameter("relim_mode", AXIS_LIMIT_EFFECTS, "normal", "Count limits"),
        limit_parameter("y_min", AXIS_LIMIT_EFFECTS, "Count minimum"),
        limit_parameter("y_max", AXIS_LIMIT_EFFECTS, "Count maximum"),
        flag("log_y", AXIS_LIMIT_EFFECTS, false, "Log count axis"),
        relim_parameter("x_relim_mode", HISTOGRAM_VALUE_AXIS_EFFECTS, "normal", "Value limits"),
        limit_parameter("x_min", HISTOGRAM_VALUE_AXIS_EFFECTS, "Value minimum"),
        limit_parameter("x_max", HISTOGRAM_VALUE_AXIS_EFFECTS, "Value maximum"),
    ]
}

fn camera_parameter(name: &'static str, normalizer: Normalizer, default: f64, label: &str) -> ParameterSpec {
    ParameterSpec::declare(name, ValueType::Number, E::BASE_GEOMETRY, Value::Float(default))
        .normalized(normalizer).labelled(label).finished()
}

fn image_parameters(style: &PlotStyle) -> Vec<ParameterSpec> {
    let policy = &style.render;
    vec![
        flag("show_image", E::BASE_STYLE, true, "Image raster"),
        relim_parameter("relim_mode", IMAGE_COLOR_EFFECTS, "tight", "Color limits"),
        ParameterSpec::declare("colormap", ValueType::Text, IMAGE_COLOR_EFFECTS, Value::text(policy.image_default_colormap))
            .normalized(Normalizer::NonEmptyText).labelled("Colormap")
            .with_choices(spelled_choices(policy.image_colormaps.iter())).finished(),
        limit_parameter("color_min", IMAGE_COLOR_EFFECTS, "Color minimum"),
        limit_parameter("color_max", IMAGE_COLOR_EFFECTS, "Color maximum"),
        flag("show_colorbar", E::CHROME, true, "Colorbar"),
        ParameterSpec::declare("presentation", ValueType::Text, IMAGE_COLOR_EFFECTS | E::BASE_GEOMETRY | E::LAYOUT, Value::text("heatmap"))
            .normalized(Normalizer::ImagePresentation).labelled("Presentation")
            .with_choices(enum_choices(&["heatmap", "height_bars"])).segmented().finished(),
        camera_parameter("camera_azimuth", Normalizer::FiniteNumber, HOME_CAMERA.0, "View azimuth"),
        camera_parameter("camera_elevation", Normalizer::CameraElevation, HOME_CAMERA.1, "View elevation"),
        camera_parameter("camera_zoom", Normalizer::CameraZoom, HOME_CAMERA.2, "View zoom"),
    ]
}

fn side_distribution_parameter() -> ParameterSpec { flag("side_distribution", E::LAYOUT, true, "Side distribution") }

/// The one schema builder for bound sessions and unbound authoring.
fn build_parameter_schema(kind: PlotKind, semantic_kind: PlotKind, has_facet: bool, style: &PlotStyle) -> ParameterSchema {
    let mut entries = vec![
        title_parameter(),
        label_parameter("x_label", "X label"),
        label_parameter("y_label", "Y label"),
        label_parameter("value_label", "Value label"),
        show_grid_parameter(kind == PlotKind::PulseTimeline),
    ];
    if kind == PlotKind::PulseTimeline {
        entries.push(pulse_x_unit_parameter());
    } else {
        if matches!(semantic_kind, PlotKind::Curve | PlotKind::Image | PlotKind::Rolling) {
            entries.push(unit_parameter("x_display_unit", DISPLAY_UNIT_EFFECTS));
        }
        if semantic_kind == PlotKind::Image { entries.push(unit_parameter("y_display_unit", DISPLAY_UNIT_EFFECTS)); }
        if kind == PlotKind::FacetGrid && has_facet { entries.push(unit_parameter("facet_display_unit", FACET_UNIT_EFFECTS)); }
        if kind == PlotKind::FacetGrid {
            entries.push(ParameterSpec::declare(FACET_FIT_PARAMETER, ValueType::Text, E::OVERLAY, Value::text("model headline"))
                .normalized(Normalizer::NonEmptyText).labelled("Cell fit value").finished());
        }
        entries.push(unit_parameter("value_display_unit", DISPLAY_UNIT_EFFECTS));
    }
    if matches!(semantic_kind, PlotKind::Curve | PlotKind::Rolling) { entries.extend(curve_parameters()); }
    if semantic_kind == PlotKind::Curve { entries.push(window_parameter(1, ROLLING_WINDOW_EFFECTS)); }
    if matches!(semantic_kind, PlotKind::Curve | PlotKind::Rolling) {
        entries.push(flag("uncertainty", UNCERTAINTY_EFFECTS, true, "Uncertainty band"));
    }
    if semantic_kind == PlotKind::Histogram {
        entries.extend(histogram_parameters());
        entries.push(window_parameter(1, ROLLING_WINDOW_EFFECTS));
    }
    if semantic_kind == PlotKind::Image {
        entries.extend(image_parameters(style));
        entries.push(side_distribution_parameter());
    }
    if kind == PlotKind::Rolling {
        entries.push(bin_count_parameter(ROLLING_DISTRIBUTION_BIN_EFFECTS));
        entries.push(window_parameter(PlotKind::Rolling.default_window() as i64, ROLLING_WINDOW_EFFECTS));
        entries.push(side_distribution_parameter());
        entries.push(ParameterSpec::declare("trailing", ValueType::Int, UNCERTAINTY_EFFECTS, Value::Int(1))
            .normalized(Normalizer::Integer).labelled("Trailing mean (shots)").bounded(1.0, 1_000_000.0, 1.0).finished());
    }
    if semantic_kind == PlotKind::Image { entries.push(flag("show_point_labels", E::OVERLAY, true, "Point labels")); }
    if kind == PlotKind::PulseTimeline { entries.push(flag("show_scan_regions", E::BASE_STYLE | E::TEXT, true, "Scan regions")); }
    let has_limits = entries.iter().any(|entry| LIMIT_MODE_NAMES.contains(&entry.name));
    ParameterSchema::new(entries, if has_limits { TransitionRules::Limits } else { TransitionRules::Conflicts })
}

/// The complete display-parameter contract of one specification.
pub fn parameter_schema_for(spec: &PlotSpec, style: &PlotStyle) -> Arc<ParameterSchema> {
    let has_facet = !spec.is_facet_grid() || spec.facet().is_some();
    Arc::new(build_parameter_schema(spec.kind(), spec.semantic().kind(), has_facet, style))
}

/// The data-independent display contract of an authored kind; a facet grid
/// needs its cell kind.
pub fn parameter_schema_for_kind(kind: PlotKind, style: &PlotStyle, facet_cell_kind: Option<PlotKind>) -> Result<Arc<ParameterSchema>> {
    let semantic_kind = if kind == PlotKind::FacetGrid {
        let cell = facet_cell_kind.ok_or_else(|| PlotError::value("facet grid display parameters require a fixed cell kind"))?;
        if !GRID_CELL_KINDS.contains(&cell) {
            return Err(PlotError::value(format!("facet grid cell kind must be one of {}",
                GRID_CELL_KINDS.iter().map(|kind| kind.value()).collect::<Vec<_>>().join(", "))));
        }
        cell
    } else {
        if facet_cell_kind.is_some() { return Err(PlotError::value("facet_cell_kind is only valid for a facet grid")); }
        kind
    };
    Ok(Arc::new(build_parameter_schema(kind, semantic_kind, true, style)))
}

/// Refuse an authored display state no host could ever accept.
pub fn validate_authored_display(kind: PlotKind, values: &ValueMap, style: &PlotStyle, facet_cell_kind: Option<PlotKind>) -> Result<()> {
    if kind == PlotKind::FacetGrid && facet_cell_kind.is_none() { return Ok(()); }
    let schema = parameter_schema_for_kind(kind, style, facet_cell_kind)?;
    let mut state = schema.initial_values(None)?;
    state.update(&schema.prepare_updates(&schema.declared_subset(values))?);
    super::parameters::validate_authored_conflicts(&state)
}

/// The display names whose value must not follow a panel across kinds.
pub fn non_portable_display_names() -> BTreeSet<&'static str> {
    let style = PlotStyle::default();
    let mut names = BTreeSet::new();
    for kind in PlotKind::ALL {
        let cells: Vec<Option<PlotKind>> = if kind == PlotKind::FacetGrid { GRID_CELL_KINDS.iter().copied().map(Some).collect() } else { vec![None] };
        for cell in cells {
            let schema = parameter_schema_for_kind(kind, &style, cell).expect("declared kind");
            names.extend(schema.specs().iter().filter(|spec| !spec.portable).map(|spec| spec.name));
        }
    }
    names
}

/// Whether this accepted plot vocabulary owns classifier levels.
pub fn accepts_classifier_thresholds(spec: &PlotSpec, display: &ValueMap) -> bool {
    spec.semantic().kind() == PlotKind::Histogram && matches!(display.get("threshold_classifier"), Some(Value::Bool(true)))
}
