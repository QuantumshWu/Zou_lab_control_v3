//! The exact plot recipe codec (`figure_artifact.encode_plot_recipe` /
//! `decode_plot_recipe`): specification, complete display parameters, size,
//! viewport, classifier thresholds, facet focus, fit, selectors, overlay,
//! interaction and presentation, as plain JSON-shaped values.

use std::sync::Arc;

use super::display::parameter_schema_for;
use super::error::{PlotError, Result};
use super::handlers::{DEFAULT_LAYOUT, DEFAULT_STYLE};
use super::kinds::{axis_ref_from_value, canonical_scalar, coordinate_value, domain_from_value, parse_kind, reduction_from_value, AxisRef, PlotKind};
use super::selectors::{
    finite_value, normalize_viewport, thresholds_wire, CrosshairPoint, NumericRange, RectangleRange, SelectorKind,
    SelectorState, SelectorValue, ThresholdTarget, Viewport,
};
use super::spec::{CellPlot, CurvePlot, FacetGridPlot, HistogramPlot, ImagePlot, PlotLabels, PlotSpec, RollingPlot, ScopeTerm};
use super::state::{py_iter, Interaction, InteractionUpdate, Presentation, PresentationUpdate};
use super::value::{Value, ValueMap};

/// The recipe's field names, in document order.
pub const RECIPE_FIELDS: [&str; 11] = ["spec", "parameters", "size", "viewport", "classifier_thresholds", "facet_focus", "fit",
                                        "selectors", "overlay", "interaction", "presentation"];

fn sorted_names(expected: &[&str]) -> String {
    let mut names: Vec<&str> = expected.to_vec();
    names.sort_unstable();
    format!("[{}]", names.iter().map(|name| format!("'{name}'")).collect::<Vec<_>>().join(", "))
}

fn keys<'a>(value: &'a Value, expected: &[&str], name: &str) -> Result<&'a ValueMap> {
    match value.as_map() {
        Some(map) if map.len() == expected.len() && expected.iter().all(|key| map.contains_key(key)) => Ok(map),
        _ => Err(PlotError::value(format!("{name} fields differ from {}", sorted_names(expected)))),
    }
}

fn axis_document(axis: Option<&AxisRef>) -> Value {
    axis.map_or(Value::None, |axis| Value::Map(ValueMap::from_pairs([
        ("domain", Value::text(axis.domain.value())), ("axis_id", Value::text(axis.axis_id.as_str()))])))
}

fn axis(value: &Value, name: &str) -> Result<Option<AxisRef>> {
    if value.is_none() { return Ok(None); }
    let entry = keys(value, &["domain", "axis_id"], name)?;
    Ok(Some(axis_ref_from_value(domain_from_value(entry.get("domain").unwrap())?, entry.get("axis_id").unwrap())?))
}

fn labels_document(labels: &PlotLabels) -> Value {
    let text = |label: &Option<Arc<str>>| label.clone().map_or(Value::None, Value::Text);
    Value::Map(ValueMap::from_pairs([("title", text(&labels.title)), ("x", text(&labels.x)), ("y", text(&labels.y)), ("value", text(&labels.value))]))
}

fn labels(value: &Value) -> Result<PlotLabels> {
    let entry = keys(value, &["title", "x", "y", "value"], "plot labels")?;
    let slot = |name: &str| -> Result<Option<Arc<str>>> {
        match entry.get(name).unwrap() {
            Value::None => Ok(None),
            Value::Text(text) => Ok(Some(text.clone())),
            _ => Err(PlotError::value(format!("label '{name}' must be non-empty text or None"))),
        }
    };
    PlotLabels::new(slot("title")?, slot("x")?, slot("y")?, slot("value")?)
}

fn scope_document(scope: &[ScopeTerm]) -> Value {
    Value::list(scope.iter().map(|(axis, coordinate)| Value::Map(ValueMap::from_pairs([
        ("axis", axis_document(Some(axis))),
        ("coordinate", Value::Map(ValueMap::from_pairs([("kind", Value::text("value")), ("value", coordinate_value(coordinate))]))),
    ]))))
}

fn scope(value: Option<&Value>) -> Result<Vec<ScopeTerm>> {
    let Some(Value::List(items)) = value else { return Err(PlotError::type_error("plot scope must be an array")) };
    let mut terms = Vec::new();
    for item in items.iter() {
        let entry = keys(item, &["axis", "coordinate"], "plot scope term")?;
        let reference = axis(entry.get("axis").unwrap(), "plot scope axis")?.ok_or_else(|| PlotError::value("plot scope axis cannot be null"))?;
        let coordinate = entry.get("coordinate").unwrap();
        if coordinate.as_map().and_then(|map| map.get("kind")) != Some(&Value::text("value")) {
            return Err(PlotError::value("plot scope coordinate is not tagged"));
        }
        let tagged = keys(coordinate, &["kind", "value"], "plot scope coordinate")?;
        terms.push((reference, canonical_scalar(tagged.get("value").unwrap(), "plot scope coordinate")?));
    }
    Ok(terms)
}

fn axes_document(axes: &[AxisRef]) -> Value { Value::list(axes.iter().map(|axis| axis_document(Some(axis)))) }

/// One data-backed specification as its recipe document.
pub fn encode_spec(spec: &PlotSpec) -> Result<ValueMap> {
    if matches!(spec, PlotSpec::PulseTimeline(_)) { return Err(PlotError::type_error("unsupported data-backed plot specification")); }
    let mut common = ValueMap::from_pairs([
        ("kind", Value::text(spec.kind().value())), ("labels", labels_document(spec.labels())),
        ("scope", scope_document(spec.scope())), ("coordinates", axes_document(spec.coordinates())),
    ]);
    match spec {
        PlotSpec::Histogram(s) => {
            common.insert("reduction", Value::text(s.reduction.value()));
            common.insert("reduced", axes_document(&s.reduced));
            common.insert("group", axis_document(s.group.as_ref()));
        }
        PlotSpec::FacetGrid(grid) => {
            common.insert("facet", axis_document(grid.facet.as_ref()));
            common.insert("cell", Value::Map(encode_spec(&grid.cell.as_spec())?));
        }
        PlotSpec::Curve(s) => {
            common.insert("reduction", Value::text(s.reduction.value()));
            common.insert("group", axis_document(s.group.as_ref()));
            common.insert("x", axis_document(Some(&s.x)));
        }
        PlotSpec::Rolling(s) => {
            common.insert("reduction", Value::text(s.reduction.value()));
            common.insert("group", axis_document(s.group.as_ref()));
            common.insert("x", axis_document(s.x.as_ref()));
        }
        PlotSpec::Image(s) => {
            common.insert("reduction", Value::text(s.reduction.value()));
            common.insert("x", axis_document(Some(&s.x)));
            common.insert("y", axis_document(Some(&s.y)));
        }
        PlotSpec::PulseTimeline(_) => unreachable!(),
    }
    Ok(common)
}

/// Decode one recipe specification document.
pub fn decode_spec(value: &Value) -> Result<PlotSpec> {
    let map = value.as_map().filter(|map| matches!(map.get("kind"), Some(Value::Text(_))))
        .ok_or_else(|| PlotError::type_error("plot recipe spec must be an object with text kind"))?;
    let kind = parse_kind(map.get("kind").unwrap().as_str().unwrap())?;
    if kind == PlotKind::PulseTimeline { return Err(PlotError::value("Pulse timelines are not data-backed Figure artifacts")); }
    let labels = labels(map.get("labels").unwrap_or(&Value::None))?;
    let scope = scope(map.get("scope"))?;
    let Some(Value::List(coordinate_documents)) = map.get("coordinates") else { return Err(PlotError::type_error("plot coordinate choices must be an array")) };
    let mut authored = Vec::new();
    for document in coordinate_documents.iter() { authored.push(axis(document, "plot coordinate choice")?); }
    // A null choice is refused where construction validates the choices.
    let coordinates = || -> Result<Vec<AxisRef>> {
        authored.iter().cloned().collect::<Option<Vec<_>>>().ok_or_else(|| PlotError::value("coordinate choices must be distinct AxisRef values"))
    };
    let missing = Value::None;
    let get = |name: &str| map.get(name).unwrap_or(&missing);
    match kind {
        PlotKind::Histogram => {
            let mut fields = vec!["kind", "labels", "scope", "coordinates", "reduction", "reduced"];
            if map.contains_key("group") { fields.push("group"); }
            keys(value, &fields, "histogram recipe")?;
            let Value::List(reduced_documents) = get("reduced") else { return Err(PlotError::type_error("histogram reduced axes must be an array")) };
            let reduction = reduction_from_value(get("reduction"))?;
            let mut reduced = Vec::new();
            for document in reduced_documents.iter() {
                reduced.push(axis(document, "histogram reduced axis")?.ok_or_else(|| PlotError::type_error("HistogramPlot.reduced must contain AxisRef values"))?);
            }
            let group = axis(get("group"), "histogram group")?;
            Ok(PlotSpec::Histogram(HistogramPlot { reduction, reduced, labels, scope, coordinates: coordinates()?, group }.validated()?))
        }
        PlotKind::FacetGrid => {
            keys(value, &["kind", "labels", "scope", "coordinates", "facet", "cell"], "facet recipe")?;
            let facet = axis(get("facet"), "facet axis")?;
            let cell = decode_spec(get("cell"))?;
            let coordinates = coordinates()?;
            let cell = CellPlot::from_spec(cell)?;
            Ok(PlotSpec::FacetGrid(FacetGridPlot { facet, cell, labels, scope, coordinates }.validated()?))
        }
        _ => {
            let mut expected = vec!["kind", "labels", "scope", "coordinates", "reduction"];
            let reduction = reduction_from_value(get("reduction"))?;
            let group = if matches!(kind, PlotKind::Curve | PlotKind::Rolling) { expected.push("group"); axis(get("group"), "plot group")? } else { None };
            let x = { expected.push("x"); axis(get("x"), "plot x")? };
            let y = if kind == PlotKind::Image { expected.push("y"); axis(get("y"), "plot y")? } else { None };
            keys(value, &expected, &format!("{} recipe", kind.value()))?;
            let coordinates = coordinates()?;
            Ok(match kind {
                PlotKind::Curve => PlotSpec::Curve(CurvePlot {
                    x: x.ok_or_else(|| PlotError::type_error("CurvePlot.x must be AxisRef"))?, group, reduction, labels, scope, coordinates,
                }.validated()?),
                PlotKind::Image => {
                    let (Some(x), Some(y)) = (x, y) else { return Err(PlotError::type_error("ImagePlot x and y must be AxisRef")) };
                    PlotSpec::Image(ImagePlot { x, y, reduction, labels, scope, coordinates }.validated()?)
                }
                _ => PlotSpec::Rolling(RollingPlot { group, x, reduction, labels, scope, coordinates }.validated()?),
            })
        }
    }
}

fn pair(values: &[f64; 2]) -> Value { Value::list([Value::Float(values[0]), Value::Float(values[1])]) }

pub fn viewport_document(viewport: Option<&Viewport>) -> Value {
    match normalize_viewport(viewport.copied()) {
        None => Value::None,
        Some((x, y)) => {
            let bounds = |range: Option<NumericRange>| range.map_or(Value::None, |range| pair(&[range.low, range.high]));
            Value::Map(ValueMap::from_pairs([("x", bounds(x)), ("y", bounds(y))]))
        }
    }
}

pub fn decode_viewport(value: &Value) -> Result<Option<Viewport>> {
    if value.is_none() { return Ok(None); }
    let entry = keys(value, &["x", "y"], "plot viewport")?;
    let mut axes = [None, None];
    for (slot, name) in axes.iter_mut().zip(["x", "y"]) {
        match entry.get(name).unwrap() {
            Value::None => {}
            Value::List(bounds) if bounds.len() == 2 => *slot = Some(NumericRange::from_values(&bounds[0], &bounds[1])?),
            _ => return Err(PlotError::value(format!("plot viewport {name} must contain two bounds"))),
        }
    }
    Ok(normalize_viewport(Some((axes[0], axes[1]))))
}

pub fn selectors_document(states: &[SelectorState]) -> Value {
    Value::list(states.iter().map(|state| {
        let payload = match state.value {
            SelectorValue::XRange(range) => pair(&[range.low, range.high]),
            SelectorValue::Area(area) => Value::Map(ValueMap::from_pairs([("x", pair(&[area.x.low, area.x.high])), ("y", pair(&[area.y.low, area.y.high]))])),
            SelectorValue::Crosshair(point) => pair(&[point.x, point.y]),
            SelectorValue::Threshold(level) => Value::Float(level),
        };
        Value::Map(ValueMap::from_pairs([
            ("kind", Value::text(state.kind().value())), ("value", payload),
            ("revision", Value::Int(i128::from(state.revision))),
            ("facet_index", state.facet_index.map_or(Value::None, |index| Value::Int(i128::from(index)))),
        ]))
    }))
}

/// `NumericRange(*bounds)`: exactly two positional values.
fn star_range(bounds: &Value) -> Result<NumericRange> {
    let items = py_iter(bounds)?;
    if items.len() != 2 { return Err(PlotError::type_error("NumericRange takes exactly a low and a high")); }
    NumericRange::from_values(&items[0], &items[1])
}

fn integer_field(value: &Value, field: &str, optional: bool) -> Result<Option<u64>> {
    match value {
        Value::None if optional => Ok(None),
        Value::Int(integer) if *integer < 0 => Err(PlotError::value(format!("{field} must be at least 0"))),
        Value::Int(integer) => u64::try_from(*integer).map(Some).map_err(|_| PlotError::overflow_error(field)),
        _ => Err(PlotError::type_error(format!("{field} must be an integer{}", if optional { " or None" } else { "" }))),
    }
}

pub fn decode_selectors(value: &Value) -> Result<Vec<SelectorState>> {
    let Value::List(items) = value else { return Err(PlotError::type_error("figure selectors must be an array")) };
    let mut result = Vec::new();
    for item in items.iter() {
        let entry = keys(item, &["kind", "value", "revision", "facet_index"], "figure selector")?;
        let kind = SelectorKind::parse(entry.get("kind").unwrap())?;
        let payload = entry.get("value").unwrap();
        let selected = match kind {
            SelectorKind::XRange => {
                let items = py_iter(payload)?;
                if items.len() != 2 { return Err(PlotError::value(format!("expected 2 values to unpack, got {}", items.len()))); }
                SelectorValue::XRange(NumericRange::from_values(&items[0], &items[1])?)
            }
            SelectorKind::Area => {
                let rectangle = keys(payload, &["x", "y"], "area selector")?;
                SelectorValue::Area(RectangleRange { x: star_range(rectangle.get("x").unwrap())?, y: star_range(rectangle.get("y").unwrap())? })
            }
            SelectorKind::Crosshair => {
                let items = py_iter(payload)?;
                if items.len() != 2 { return Err(PlotError::type_error("CrosshairPoint takes exactly x and y")); }
                SelectorValue::Crosshair(CrosshairPoint::new(finite_value(&items[0], "crosshair x")?, finite_value(&items[1], "crosshair y")?)?)
            }
            SelectorKind::Threshold => SelectorValue::Threshold(payload.py_float()?),
        };
        let revision = integer_field(entry.get("revision").unwrap(), "revision", false)?.unwrap();
        let facet_index = integer_field(entry.get("facet_index").unwrap(), "facet_index", true)?;
        result.push(SelectorState::new(selected, revision, facet_index)?);
    }
    Ok(result)
}

/// One decoded recipe.
#[derive(Clone, Debug)]
pub struct DecodedRecipe {
    pub spec: PlotSpec,
    pub parameters: ValueMap,
    pub size: Arc<str>,
    pub viewport: Option<Viewport>,
    /// Raw targets; the configuration normalizes them.
    pub classifier_thresholds: Value,
    pub facet_focus: Option<i128>,
    pub fit: ValueMap,
    pub selectors: Vec<SelectorState>,
    pub overlay: Value,
    pub interaction: InteractionUpdate,
    pub presentation: PresentationUpdate,
}

/// The arguments of one encoded recipe.
#[derive(Clone, Debug, Default)]
pub struct RecipeParts<'a> {
    pub viewport: Option<Viewport>,
    pub classifier_thresholds: &'a [ThresholdTarget],
    pub facet_focus: Option<i128>,
    pub fit: Option<&'a ValueMap>,
    pub selectors: &'a [SelectorState],
    pub interaction: Option<&'a Interaction>,
    pub presentation: Option<&'a Presentation>,
}

/// `encode_plot_recipe`: the overlay is always `None` here (arrays belong
/// to the archive writer).
pub fn encode_plot_recipe(spec: &PlotSpec, parameters: &ValueMap, size: &str, parts: &RecipeParts<'_>) -> Result<ValueMap> {
    let schema = parameter_schema_for(spec, &DEFAULT_STYLE);
    let complete = schema.initial_values(Some(parameters))?;
    let size = DEFAULT_LAYOUT.validate_preset(size).map_err(PlotError::value)?;
    let spec_document = encode_spec(spec)?;
    let default_interaction = Interaction::default();
    let default_presentation = Presentation::default();
    Ok(ValueMap::from_pairs([
        ("spec", Value::Map(spec_document)),
        ("parameters", Value::Map(complete)),
        ("size", Value::text(size)),
        ("viewport", viewport_document(parts.viewport.as_ref())),
        ("classifier_thresholds", thresholds_wire(parts.classifier_thresholds)),
        ("facet_focus", parts.facet_focus.map_or(Value::None, Value::Int)),
        ("fit", Value::Map(parts.fit.cloned().unwrap_or_default())),
        ("selectors", selectors_document(parts.selectors)),
        ("overlay", Value::None),
        ("interaction", parts.interaction.unwrap_or(&default_interaction).wire()),
        ("presentation", parts.presentation.unwrap_or(&default_presentation).wire()),
    ]))
}

/// `decode_plot_recipe`.
pub fn decode_plot_recipe(value: &Value) -> Result<DecodedRecipe> {
    let entry = keys(value, &RECIPE_FIELDS, "plot recipe")?;
    let get = |name: &str| entry.get(name).unwrap();
    let (Value::Map(parameters), Value::Map(fit)) = (get("parameters"), get("fit")) else {
        return Err(PlotError::type_error("plot recipe parameters and fit must be objects"));
    };
    let size = match get("size") { Value::Text(text) if !text.is_empty() => text.clone(), _ => return Err(PlotError::value("plot recipe size must be non-empty text")) };
    let thresholds = get("classifier_thresholds");
    if !matches!(thresholds, Value::List(_)) { return Err(PlotError::type_error("plot recipe classifier_thresholds must be an array")); }
    let facet_focus = match get("facet_focus") {
        Value::None => None,
        Value::Int(index) if *index >= 0 => Some(*index),
        _ => return Err(PlotError::type_error("plot recipe facet_focus must be a nonnegative integer or null")),
    };
    let spec = decode_spec(get("spec"))?;
    let viewport = decode_viewport(get("viewport"))?;
    let selectors = decode_selectors(get("selectors"))?;
    let interaction = InteractionUpdate::from_value(get("interaction"))?;
    let presentation = PresentationUpdate::from_value(get("presentation"))?;
    Ok(DecodedRecipe {
        spec, parameters: parameters.clone(), size, viewport, classifier_thresholds: thresholds.clone(), facet_focus,
        fit: fit.clone(), selectors, overlay: get("overlay").clone(), interaction, presentation,
    })
}

/// The decoded recipe one dataset of a figure archive was drawn with
/// (`figure_plot_recipe`), read from the info document alone.
pub fn figure_plot_recipe(info: &ValueMap, dataset: &str) -> Result<DecodedRecipe> {
    let sections = info.get("sections").and_then(Value::as_map);
    let recipes = sections.and_then(|sections| sections.get("plot")).and_then(Value::as_map);
    let datasets = sections.and_then(|sections| sections.get("dataset")).and_then(Value::as_map);
    let (Some(recipes), Some(datasets)) = (recipes, datasets) else {
        return Err(PlotError::value("figure must carry one exact plot recipe per dataset"));
    };
    if recipes.len() != datasets.len() || recipes.keys().any(|key| !datasets.contains_key(key)) {
        return Err(PlotError::value("figure must carry one exact plot recipe per dataset"));
    }
    let recipe = recipes.get(dataset).ok_or_else(|| PlotError::key(super::value::repr_str(dataset)))?;
    decode_plot_recipe(recipe)
}
