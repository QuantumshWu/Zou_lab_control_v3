//! `resolve_configuration` (`zlc_plot.configuration`): one complete,
//! immutable configuration candidate resolved without a plotting session,
//! with its scientific signature (`preparation.scientific_context_signature`)
//! and its canonical desktop wire target.

use std::any::Any;
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::layout::LayoutConfig;
use crate::style::PlotStyle;
use crate::units::{resolve_unit, UnitRegistry, DEFAULT_UNITS};

use super::data_contract::{schemas_continue, IndexedHistoryLayout, PlotSchema};
use super::display::{accepts_classifier_thresholds, parameter_schema_for};
use super::error::{PlotError, Result};
use super::fit::{FitLayer, FitModelInfo, FitRequest, FitTargetQuery, FIT_TARGET_FIELDS};
use super::handlers::{fitting_spec, label_roles, LabelRole};
use super::kinds::{kind_from_value, PlotKind};
use super::metadata::{
    configuration_fit_units, configuration_metadata, configuration_quantities, resolve_parameters, retained_parameters,
    unit_parameter_units, viewport_roles, ConfigurationMetadata,
};
use super::ordered::OrderedMap;
use super::parameters::{ParameterSchema, RenderEffect, LIMIT_PAIRS};
use super::recipe::{decode_plot_recipe, encode_plot_recipe, DecodedRecipe, RecipeParts, RECIPE_FIELDS};
use super::selectors::{committed_selectors, normalize_classifier_threshold_targets, normalize_viewport, NumericRange, SelectorKind, SelectorState, SelectorValue, ThresholdTarget, Viewport};
use super::semantics::{composed_spec, describe_semantics, rebased_spec, SemanticDescription};
use super::spec::{paints_image_surface, LabelSlot, PlotLabels, PlotSpec};
use super::state::{DisplayState, Interaction, InteractionUpdate, Presentation, PresentationUpdate};
use super::value::{repr_str, Value, ValueMap};

/// The plot library defaults a configuration reads (`PlotLibraryDefaults`).
#[derive(Clone, Debug, Default)]
pub struct Defaults {
    pub style: PlotStyle,
    pub layout: LayoutConfig,
}

/// An image point overlay, opaque to configuration (only its presence and
/// identity travel with a configuration).
pub type OverlayHandle = Arc<dyn Any + Send + Sync>;

/// Typed session updates (`configuration_updates`); `None` is "not given".
#[derive(Clone, Default)]
pub struct ConfigurationUpdates {
    pub semantic: Option<ValueMap>,
    pub parameters: Option<ValueMap>,
    pub parameter_updates: Option<ValueMap>,
    pub size: Option<String>,
    pub selectors: Option<Vec<SelectorState>>,
    pub selector_updates: Option<Vec<(SelectorKind, Option<SelectorState>)>>,
    pub viewport: Option<Option<Viewport>>,
    pub facet_focus: Option<Option<i128>>,
    pub focused_facet_index: Option<Option<i128>>,
    /// Raw threshold records, normalized here.
    pub classifier_thresholds: Option<Value>,
    pub fit: Option<ValueMap>,
    pub fit_live: Option<bool>,
    pub interaction: Option<InteractionUpdate>,
    pub presentation: Option<PresentationUpdate>,
    pub image_overlay: Option<Option<OverlayHandle>>,
    /// The overlay is a live image's: a resolved kind that paints no image drops it.
    pub overlay_follows_kind: bool,
}

/// A captured interaction (`interaction_updates`): recipe fields as wire
/// values, plus the two typed fields a gesture owner may also pass.
#[derive(Clone, Default)]
pub struct InteractionUpdates {
    pub recipe: ValueMap,
    pub selector_updates: Option<Vec<(SelectorKind, Option<SelectorState>)>>,
    pub focused_facet_index: Option<Option<i128>>,
}

const INTERACTION_RECIPE_FIELDS: [&str; 7] = ["parameters", "selectors", "viewport", "facet_focus", "classifier_thresholds", "interaction", "presentation"];
const WIRE_FIELDS: [&str; 12] = ["parameters", "size", "viewport", "classifier_thresholds", "facet_focus", "fit", "selectors", "overlay",
                                 "interaction", "presentation", "spec", "semantic"];

impl InteractionUpdates {
    fn is_empty(&self) -> bool { self.recipe.is_empty() && self.selector_updates.is_none() && self.focused_facet_index.is_none() }
    fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.recipe.keys().map(str::to_owned).collect();
        if self.selector_updates.is_some() { names.push("selector_updates".into()); }
        if self.focused_facet_index.is_some() { names.push("focused_facet_index".into()); }
        names.sort();
        names
    }
}

/// Everything one resolution is asked.
#[derive(Clone, Default)]
pub struct ResolveRequest<'a> {
    pub schema: Option<Arc<PlotSchema>>,
    pub kind: Option<PlotKind>,
    /// A desktop wire target (decoded by the recipe codec).
    pub target: Option<&'a ValueMap>,
    /// The latest pending authoring state.
    pub base: Option<&'a Arc<ResolvedConfiguration>>,
    /// What is on screen; only it decides `science_changed`.
    pub presented: Option<&'a Arc<ResolvedConfiguration>>,
    pub spec: Option<PlotSpec>,
    pub semantic_updates: Option<ValueMap>,
    pub parameter_updates: Option<ValueMap>,
    /// `same_source=True` unless the caller says otherwise.
    pub different_source: bool,
    pub window: Option<i128>,
    pub defaults: Option<Arc<Defaults>>,
    pub fit: Option<Arc<dyn FitLayer>>,
    pub unit_registry: Option<Arc<UnitRegistry>>,
    /// Captured limits keyed by low-bound name, in the previous display units.
    pub fixed_limits: OrderedMap<String, (f64, f64)>,
    pub interaction_updates: Option<InteractionUpdates>,
    pub configuration_updates: Option<ConfigurationUpdates>,
}

/// What a preparation's continuity is keyed by (`schema_preparation_key`).
#[derive(Clone, PartialEq)]
pub enum PreparationKey {
    Authored,
    Indexed(Option<usize>, Box<super::data_contract::HistoryEvent>),
    Dataset(Arc<PlotSchema>),
}

/// `preparation.scientific_context_signature`: what a numerical preparation
/// depends on.  Equal contexts share one preparation.
#[derive(Clone, PartialEq)]
pub struct ScientificContext {
    /// The specification without its labels.
    pub spec: PlotSpec,
    pub schema: PreparationKey,
    /// Fit-selecting and payload-only parameters, limits in canonical units.
    pub numerical_parameters: Vec<(&'static str, Value)>,
    pub selectors: Vec<(SelectorKind, SelectorValue, Option<u64>)>,
    /// The viewport in canonical units.
    pub viewport: Option<[Option<(f64, f64)>; 2]>,
    pub focused_facet_index: Option<i128>,
}

/// The configuration's scientific signature: its context plus the fit and
/// classifier intent.  Equal signatures need no new numerical work.
#[derive(Clone, PartialEq)]
pub struct Signature {
    pub context: ScientificContext,
    pub fit_target: ValueMap,
    pub fit_live: bool,
    pub thresholds: Vec<ThresholdTarget>,
}

/// One complete immutable configuration candidate.
pub struct ResolvedConfiguration {
    pub schema: Option<Arc<PlotSchema>>,
    pub spec: PlotSpec,
    pub parameter_schema: Arc<ParameterSchema>,
    pub display_state: DisplayState,
    pub size: &'static str,
    pub selectors: Vec<SelectorState>,
    pub viewport: Option<Viewport>,
    pub facet_focus: Option<i128>,
    pub focused_facet_index: Option<i128>,
    pub classifier_thresholds: Vec<ThresholdTarget>,
    pub fit_target: ValueMap,
    pub fit_live: bool,
    pub interaction: Interaction,
    pub presentation: Presentation,
    pub defaults: Arc<Defaults>,
    pub fit: Arc<dyn FitLayer>,
    pub unit_registry: Option<Arc<UnitRegistry>>,
    pub ignored_interaction: Vec<String>,
    pub semantics: SemanticDescription,
    /// User choices, distinct from the defaults resolved for one dataset.
    pub authored_semantics: ValueMap,
    pub metadata: ConfigurationMetadata,
    pub signature: Signature,
    pub science_changed: bool,
    pub cancellation: Arc<AtomicBool>,
    pub same_source: bool,
    pub image_overlay: Option<OverlayHandle>,
    pub discard_unmatched_classifier: bool,
    pub fit_request: Option<FitRequest>,
    pub expression_failure: Option<(String, String)>,
}

impl ResolvedConfiguration {
    pub fn kind(&self) -> PlotKind { self.spec.kind() }
    pub fn fit_models(&self) -> &[Arc<FitModelInfo>] { &self.metadata.fit_models }
    pub fn size_choices(&self) -> Vec<&'static str> { crate::layout::PANEL_PRESETS.iter().map(|preset| preset.0).collect() }
    pub fn cancelled(&self) -> bool { self.cancellation.load(Ordering::SeqCst) }
    pub fn cancel(&self) { self.cancellation.store(true, Ordering::SeqCst) }
    /// Whether a fit model or a threshold classifier needs numerical work.
    pub fn has_numerical_work(&self) -> bool {
        let model = self.fit_target.get("model").filter(|model| !model.is_none()).is_some_and(|model| !model.py_str().trim().is_empty());
        model || accepts_classifier_thresholds(&self.spec, &self.display_state.values)
    }
    pub fn fit_expression(&self) -> Result<String> {
        if let Some((expression, _)) = &self.expression_failure { return Ok(expression.clone()); }
        let Some(request) = &self.fit_request else { return Ok(String::new()) };
        let units = configuration_fit_units(self.schema.as_deref(), &self.spec, &self.display_state, self.unit_registry.as_ref())?;
        self.fit.expression_text(&units, &request.model, &self.fit_target)
    }
    pub fn fit_expression_error(&self) -> &str { self.expression_failure.as_ref().map_or("", |failure| failure.1.as_str()) }

    /// The canonical desktop wire target, encoded by the recipe owner.
    pub fn target(&self) -> Result<ValueMap> {
        let parts = RecipeParts {
            viewport: self.viewport, classifier_thresholds: &self.classifier_thresholds, facet_focus: self.facet_focus,
            fit: Some(&self.fit_target), selectors: &self.selectors, interaction: Some(&self.interaction),
            presentation: Some(&self.presentation),
        };
        let mut recipe = encode_plot_recipe(&self.spec, &self.display_state.values, self.size, &parts)?;
        recipe.remove("spec");
        recipe.remove("overlay");
        // Saving a resolved Auto axis as authored would pin it after reload.
        let automatic = self.semantics.values().get("cell_kind").and_then(Value::as_str) == Some("");
        let mut semantic = if automatic { self.authored_semantics.clone() } else { self.semantics.values() };
        semantic.insert("kind", Value::text(self.spec.kind().value()));
        let mut target = ValueMap::from_pairs([("semantic", Value::Map(semantic))]);
        target.update(&recipe);
        Ok(target)
    }
}

fn truthy(value: &Value) -> bool {
    match value {
        Value::None => false, Value::Bool(flag) => *flag, Value::Int(integer) => *integer != 0, Value::Float(real) => *real != 0.0,
        Value::Text(text) => !text.is_empty(), Value::List(items) => !items.is_empty(), Value::Map(map) => !map.is_empty(),
    }
}

fn recipe_defaults(size: &str) -> ValueMap {
    ValueMap::from_pairs([
        ("parameters", Value::Map(ValueMap::new())), ("size", Value::text(size)), ("viewport", Value::None),
        ("classifier_thresholds", Value::list([])), ("facet_focus", Value::None), ("fit", Value::Map(ValueMap::new())),
        ("selectors", Value::list([])), ("overlay", Value::None),
        ("interaction", Value::Map(ValueMap::from_pairs([("series_lock", Value::None)]))),
        ("presentation", Value::Map(ValueMap::from_pairs([("series_readout", Value::None)]))),
    ])
}

/// The decoded wire fields a target actually supplied, as typed updates.
fn wire_changes(decoded: &DecodedRecipe, supplied: impl Iterator<Item = String>, changes: &mut ConfigurationUpdates) {
    for key in supplied {
        match key.as_str() {
            "parameters" => changes.parameters = Some(decoded.parameters.clone()),
            "size" => changes.size = Some(decoded.size.to_string()),
            "viewport" => changes.viewport = Some(decoded.viewport),
            "classifier_thresholds" => changes.classifier_thresholds = Some(decoded.classifier_thresholds.clone()),
            "facet_focus" => changes.facet_focus = Some(decoded.facet_focus),
            "fit" => changes.fit = Some(decoded.fit.clone()),
            "selectors" => changes.selectors = Some(decoded.selectors.clone()),
            "interaction" => changes.interaction = Some(decoded.interaction.clone()),
            "presentation" => changes.presentation = Some(decoded.presentation.clone()),
            _ => {}
        }
    }
}

/// `changes.update(typed)`: a typed field replaces the wire one.
fn overlay_typed(changes: &mut ConfigurationUpdates, typed: &ConfigurationUpdates) {
    macro_rules! take { ($($field:ident),*) => { $( if typed.$field.is_some() { changes.$field = typed.$field.clone(); } )* }; }
    take!(semantic, parameters, parameter_updates, size, selectors, selector_updates, viewport, facet_focus, focused_facet_index,
          classifier_thresholds, fit, fit_live, interaction, presentation, image_overlay);
    changes.overlay_follows_kind |= typed.overlay_follows_kind;
}

fn rebase_viewport(base: &ResolvedConfiguration, spec: &PlotSpec, schema: Option<&PlotSchema>, values: &ValueMap,
                   viewport: Option<Viewport>, registry: &UnitRegistry) -> Result<Option<Viewport>> {
    let (Some(viewport), Some(schema), Some(base_schema)) = (viewport, schema, base.schema.as_deref()) else { return Ok(viewport) };
    let history = schema.history()?;
    let roles = label_roles(&spec.semantic_spec(), history.as_deref())?;
    let old_units = unit_parameter_units(Some(base_schema), &base.spec, Some(registry))?;
    let new_units = unit_parameter_units(Some(schema), spec, Some(registry))?;
    let mut result = [None, None];
    for (index, (slot, bounds)) in [(LabelSlot::X, viewport.0), (LabelSlot::Y, viewport.1)].into_iter().enumerate() {
        let Some(bounds) = bounds else { continue };
        let role = roles.iter().find(|(held, _)| *held == slot).map(|(_, role)| role);
        let name = match role {
            None | Some(LabelRole::Count) => { result[index] = Some(bounds); continue }
            Some(LabelRole::Value) => "value_display_unit",
            Some(_) => if slot == LabelSlot::X { "x_display_unit" } else { "y_display_unit" },
        };
        let (Some(canonical_old), Some(canonical_new)) = (old_units.get(&name), new_units.get(&name)) else { result[index] = Some(bounds); continue };
        let before = base.display_state.values.get(name).cloned().unwrap_or(Value::None);
        let after = values.get(name).cloned().unwrap_or(Value::None);
        if before == after && canonical_old == canonical_new { result[index] = Some(bounds); continue; }
        let converted = (|| -> Result<Option<NumericRange>> {
            let source = if before.is_none() { canonical_old.clone() } else { resolve_unit(&before.py_str(), Some(registry))? };
            let target = if after.is_none() { canonical_new.clone() } else { resolve_unit(&after.py_str(), Some(registry))? };
            let mut ends = [source.convert_value_to(bounds.low, &target)?, source.convert_value_to(bounds.high, &target)?];
            if !ends.iter().all(|end| end.is_finite()) { return Ok(None); }
            if ends[1] < ends[0] { ends.swap(0, 1); }
            Ok(NumericRange::new(ends[0], ends[1]).ok())
        })();
        result[index] = match converted { Ok(range) => range, Err(error) if error.is_type_value_or_key() || error.kind == super::error::ErrorKind::Overflow => None, Err(error) => return Err(error) };
    }
    Ok(normalize_viewport(Some((result[0], result[1]))))
}

/// `schema_preparation_key(schema)`.
pub fn preparation_key(schema: Option<&Arc<PlotSchema>>) -> Result<PreparationKey> {
    let Some(schema) = schema else { return Ok(PreparationKey::Authored) };
    Ok(match schema.history()? {
        Some(layout) => { let layout: &IndexedHistoryLayout = &layout; PreparationKey::Indexed(layout.inner_count, Box::new(layout.event.clone())) }
        None => PreparationKey::Dataset(schema.clone()),
    })
}

/// `scientific_context_signature(spec, schema, context, parameter_schema, registry)`.
#[allow(clippy::too_many_arguments)]
pub fn scientific_context_signature(spec: &PlotSpec, schema: Option<&Arc<PlotSchema>>, state: &DisplayState, selectors: &[SelectorState],
             viewport: Option<Viewport>, focused: Option<i128>, parameters: &ParameterSchema, registry: Option<&UnitRegistry>) -> Result<ScientificContext> {
    let mut scientific = spec.clone();
    *scientific.labels_mut() = PlotLabels::default();
    if let PlotSpec::FacetGrid(grid) = &mut scientific { *grid.cell.labels_mut() = PlotLabels::default(); }
    let quantities = configuration_quantities(schema.map(|s| &**s), spec, state, registry)?;
    let canonical = |value: &Value, slot: &'static str| -> Result<Value> {
        match (value.real(), quantities.get(&slot)) {
            (Some(number), Some(quantity)) if !value.is_none() => Ok(Value::Float(quantity.display.convert_value_to(number, &quantity.canonical)?)),
            _ => Ok(value.clone()),
        }
    };
    let mut numerical = Vec::new();
    for spec in parameters.specs() {
        let effects = spec.effects;
        if !(effects.intersects(RenderEffect::FIT_SELECTION)
             || (effects.intersects(RenderEffect::PAYLOAD_PROJECTION) && !effects.intersects(RenderEffect::VIEW_PROJECTION))) { continue; }
        let mut value = state.values.get(spec.name).cloned().unwrap_or(Value::None);
        if matches!(spec.name, "x_min" | "x_max" | "y_min" | "y_max") {
            value = canonical(&value, if spec.name.starts_with('x') { "x" } else { "y" })?;
        }
        numerical.push((spec.name, value));
    }
    let viewport = match viewport {
        None => None,
        Some((x, y)) => {
            let mut slots = [None, None];
            for (index, (bounds, slot)) in [(x, "x"), (y, "y")].into_iter().enumerate() {
                if let Some(bounds) = bounds {
                    let convert = |value: f64| -> Result<f64> {
                        match quantities.get(&slot) { Some(quantity) => Ok(quantity.display.convert_value_to(value, &quantity.canonical)?), None => Ok(value) }
                    };
                    slots[index] = Some((convert(bounds.low)?, convert(bounds.high)?));
                }
            }
            Some(slots)
        }
    };
    Ok(ScientificContext {
        spec: scientific, schema: preparation_key(schema)?, numerical_parameters: numerical,
        selectors: selectors.iter().map(|state| (state.kind(), state.value, state.facet_index)).collect(),
        viewport, focused_facet_index: focused,
    })
}

/// `KeyError(f"{label}: {sorted(names)}")` as `str()` shows it.
fn unknown_names(label: &str, mut names: Vec<&str>) -> PlotError {
    names.sort_unstable();
    let listed: Vec<String> = names.iter().map(|name| repr_str(name)).collect();
    PlotError::key(repr_str(&format!("{label}: [{}]", listed.join(", "))))
}

/// Resolve one complete immutable candidate, without touching live state.
pub fn resolve_configuration(request: ResolveRequest<'_>) -> Result<Arc<ResolvedConfiguration>> {
    let ResolveRequest { schema, kind, target, base, presented, spec, semantic_updates, parameter_updates, different_source, window,
                         defaults, fit, unit_registry, fixed_limits, interaction_updates, configuration_updates } = request;
    let same_source = !different_source;
    let defaults = defaults.unwrap_or_default();
    let wire = target.cloned().unwrap_or_default();
    let typed = configuration_updates.unwrap_or_default();
    let unknown: Vec<&str> = wire.keys().filter(|key| !WIRE_FIELDS.contains(key)).collect();
    if !unknown.is_empty() { return Err(unknown_names("unknown configuration target fields", unknown)); }
    let engine = fit.or_else(|| base.map(|base| base.fit.clone()))
        .ok_or_else(|| PlotError::type_error("a fit layer is required for the first configuration"))?;
    let registry = unit_registry.or_else(|| base.and_then(|base| base.unit_registry.clone()));
    let registry_ref = registry.as_deref();
    let size_source: String = if let Some(size) = typed.size.as_ref().filter(|size| !size.is_empty()) {
        size.clone()
    } else if let Some(size) = wire.get("size").filter(|size| truthy(size)) {
        size.as_str().ok_or_else(|| PlotError::type_error("panel preset must be text"))?.to_owned()
    } else if let Some(base) = base {
        base.size.to_owned()
    } else {
        defaults.layout.default_preset.to_owned()
    };
    let selected_size = defaults.layout.validate_preset(&size_source).map_err(PlotError::value)?;

    let wire_without_semantic = || -> ValueMap { wire.iter().filter(|(key, _)| *key != "semantic").map(|(k, v)| (k, v.clone())).collect() };
    let mut decoded: Option<DecodedRecipe> = None;
    if wire.contains_key("spec") {
        let mut document = recipe_defaults(selected_size);
        document.update(&wire_without_semantic());
        decoded = Some(decode_plot_recipe(&Value::Map(document))?);
    }
    let explicit_spec = spec.is_some() || decoded.is_some();
    let mut seed = spec.or_else(|| decoded.as_ref().map(|decoded| decoded.spec.clone())).or_else(|| base.map(|base| base.spec.clone()));
    let mut semantic_values = match (&semantic_updates, &typed.semantic, wire.get("semantic")) {
        (Some(values), _, _) => values.clone(),
        (None, Some(values), _) => values.clone(),
        (None, None, Some(Value::Map(values))) => values.clone(),
        (None, None, Some(Value::None) | None) => ValueMap::new(),
        (None, None, Some(other)) => return Err(PlotError::type_error(format!("'{}' object is not a mapping", other.type_name()))),
    };
    let role_edit = semantic_values.keys().any(|name| name.starts_with("fate:") || name.starts_with("coordinate:"));
    let mut authored_semantics = base.map(|base| base.authored_semantics.clone()).unwrap_or_default();
    authored_semantics.update(&semantic_values);
    // v2's panel cell kind "" ("automatic (from data)"): a grid whose cell the
    // data decides, again on every resolution, until a cell kind is named.
    // A resolution that names none keeps what its base was.
    let automatic_cell = match semantic_values.get("cell_kind") {
        Some(value) => value.as_str() == Some(""),
        None => base.is_some_and(|base| base.semantics.fields.iter().any(|field| field.name == "cell_kind" && field.value.as_str() == Some(""))),
    };
    if automatic_cell { semantic_values.remove("cell_kind"); }
    let schema_ref = schema.as_deref();
    if seed.is_none() {
        let selected_kind = match semantic_values.get("kind") { Some(value) => Some(kind_from_value(value)?), None => kind };
        let cell = semantic_values.get("cell_kind").map(kind_from_value).transpose()?;
        seed = Some(fitting_spec(schema_ref, selected_kind, cell)?
            .ok_or_else(|| PlotError::value("this schema has no unambiguous plot specification"))?);
    }
    let mut seed = seed.unwrap();
    let continuing = match base {
        None => true,
        Some(base) => match (base.schema.as_deref(), schema_ref) {
            (None, None) => true,
            (Some(previous), Some(following)) => schemas_continue(previous, following)?,
            _ => false,
        },
    };
    if let (Some(base), false, Some(schema)) = (base, continuing, schema_ref) {
        // Rebase the previous intent first. A new explicit edit still passes
        // through strict composition on the new schema; never silently drop it.
        let mut carried = if automatic_cell { base.authored_semantics.clone() }
            else { describe_semantics(base.schema.as_deref(), &seed, &defaults.layout)?.values() };
        if automatic_cell { carried.remove("cell_kind"); }
        seed = rebased_spec(schema, seed.kind(), &carried)?;
        let resolved = describe_semantics(Some(schema), &seed, &defaults.layout)?.values();
        if semantic_updates.is_none() && typed.semantic.is_none() {
            let previous = base.semantics.values();
            semantic_values.retain(|name, value| !(previous.get(name) == Some(value) && !resolved.contains_key(name)));
        }
        authored_semantics.retain(|name, _| name == "cell_kind" || resolved.contains_key(name));
    }
    if automatic_cell && seed.is_facet_grid() {
        // The densest cell this data admits, reached as a cell-kind edit (its scope and roles carried).
        if let Some(fitted) = fitting_spec(schema_ref, Some(PlotKind::FacetGrid), None)? {
            let densest = fitted.semantic().kind();
            if densest != seed.semantic().kind() {
                seed = composed_spec(schema_ref, &seed, &ValueMap::from_pairs([("cell_kind", Value::text(densest.value()))]))?;
            }
        }
    }
    let selected_spec = composed_spec(schema_ref, &seed, &semantic_values)?;
    let mut semantics = describe_semantics(schema_ref, &selected_spec, &defaults.layout)?;
    if role_edit || explicit_spec {
        authored_semantics = semantics.values();
        if automatic_cell { authored_semantics.insert("cell_kind", Value::text("")); }
    }
    if automatic_cell {
        if let Some(field) = semantics.fields.iter_mut().find(|field| field.name == "cell_kind") {
            field.value = Value::text("");
            field.choices.insert(0, (Value::text(""), "automatic (from data)".into()));
        }
    }
    let parameters = parameter_schema_for(&selected_spec, &defaults.style);

    if !wire.is_empty() && decoded.is_none() {
        if let PlotSpec::PulseTimeline(_) = selected_spec {
            return Err(PlotError::type_error("Pulse timelines require typed configuration, not data-backed wire recipes"));
        }
        let mut document = encode_plot_recipe(&selected_spec, &ValueMap::new(), selected_size, &RecipeParts::default())?;
        document.update(&wire_without_semantic());
        decoded = Some(decode_plot_recipe(&Value::Map(document))?);
    }
    let mut changes = ConfigurationUpdates::default();
    if let Some(decoded) = &decoded {
        wire_changes(decoded, wire.keys().filter(|key| !matches!(*key, "semantic" | "spec" | "overlay")).map(str::to_owned), &mut changes);
    }
    overlay_typed(&mut changes, &typed);
    let roles_match = match base {
        Some(base) if same_source => viewport_roles(&base.spec, base.schema.as_deref())? == viewport_roles(&selected_spec, schema_ref)?,
        _ => false,
    };
    let mut ignored = Vec::new();
    if let Some(updates) = interaction_updates.as_ref().filter(|updates| !updates.is_empty()) {
        let unknown: Vec<&str> = updates.recipe.keys().filter(|key| !INTERACTION_RECIPE_FIELDS.contains(key)).collect();
        if !unknown.is_empty() { return Err(unknown_names("unknown interaction configuration fields", unknown)); }
        let compatible = match presented {
            Some(presented) if same_source => viewport_roles(&presented.spec, presented.schema.as_deref())? == viewport_roles(&selected_spec, schema_ref)?,
            _ => false,
        };
        if compatible {
            if !updates.recipe.is_empty() {
                let mut document = encode_plot_recipe(&selected_spec, &ValueMap::new(), selected_size, &RecipeParts::default())?;
                document.update(&updates.recipe);
                let captured = decode_plot_recipe(&Value::Map(document))?;
                wire_changes(&captured, updates.recipe.keys().filter(|key| RECIPE_FIELDS.contains(key)).map(str::to_owned), &mut changes);
            }
            if updates.selector_updates.is_some() { changes.selector_updates = updates.selector_updates.clone(); }
            if updates.focused_facet_index.is_some() { changes.focused_facet_index = updates.focused_facet_index; }
        } else {
            ignored = updates.names();
        }
    }

    let old_names: Vec<&'static str> = base.map(|base| base.parameter_schema.names().collect()).unwrap_or_default();
    let declared = |values: &ValueMap| -> Result<ValueMap> {
        let unknown: Vec<&str> = values.keys().filter(|name| !parameters.contains(name) && !old_names.contains(name)).collect();
        if !unknown.is_empty() { return Err(unknown_names("unknown display parameter(s)", unknown)); }
        Ok(parameters.declared_subset(values))
    };
    let previous = match base {
        Some(base) if parameters.same_names(&base.parameter_schema) && selected_spec == base.spec => base.display_state.clone(),
        _ => {
            let retained = match base { Some(base) => retained_parameters(&base.display_state.values, &parameters)?, None => ValueMap::new() };
            DisplayState {
                revision: base.map_or(0, |base| base.display_state.revision), values: parameters.initial_values(Some(&retained))?,
                changed_names: BTreeSet::new(), effects: RenderEffect::NONE,
                interaction: base.map(|base| base.interaction.clone()).unwrap_or_default(),
            }
        }
    };
    let mut values = declared(&changes.parameters.clone().unwrap_or_default())?;
    let mut authored = match parameter_updates.or_else(|| changes.parameter_updates.clone()) {
        None => values.iter().filter(|(name, value)| previous.values.get(name) != Some(value)).map(|(name, value)| (name, value.clone())).collect(),
        Some(authored) => { let authored = declared(&authored)?; values.update(&authored); authored }
    };
    if let Some(window) = window {
        if parameters.contains("window") {
            values.insert("window", Value::Int(window));
            authored.insert("window", Value::Int(window));
        }
    }
    let resolution = resolve_parameters(&parameters, &previous, &values, Some(&authored), &fixed_limits, &selected_spec, schema_ref, registry_ref)?;
    let candidate_values = resolution.values;
    let is_grid = selected_spec.is_facet_grid();

    // Selectors, viewport and facet focus are coordinates on the base's axes
    // and source, carried only while both match; authored ones always apply.
    let mut selectors = if roles_match { base.unwrap().selectors.clone() } else { Vec::new() };
    let mut viewport = if roles_match { base.unwrap().viewport } else { None };
    let mut facet_focus = if roles_match { base.unwrap().facet_focus } else { None };
    let mut focused = if roles_match { base.unwrap().focused_facet_index } else if is_grid { Some(0) } else { None };
    let mut interaction = base.map(|base| base.interaction.clone()).unwrap_or_default();
    let mut presentation = base.map(|base| base.presentation.clone()).unwrap_or_default();
    let mut thresholds = base.map(|base| base.classifier_thresholds.clone()).unwrap_or_default();
    let mut fit_target = base.map(|base| base.fit_target.clone()).unwrap_or_default();
    let mut fit_live = base.is_some_and(|base| base.fit_live);
    if let Some(states) = &changes.selectors { selectors = committed_selectors(states)?; }
    if let Some(updates) = &changes.selector_updates {
        let mut by_kind: OrderedMap<SelectorKind, SelectorState> = OrderedMap::from_pairs(selectors.iter().map(|state| (state.kind(), *state)));
        for (selector_kind, state) in updates {
            match state {
                None => { by_kind.remove(selector_kind); }
                Some(state) if state.kind() != *selector_kind => return Err(PlotError::type_error("selector_updates values must match their SelectorKind")),
                Some(state) => by_kind.insert(*selector_kind, *state),
            }
        }
        selectors = committed_selectors(&by_kind.values().copied().collect::<Vec<_>>())?;
    }
    if let Some(update) = changes.viewport { viewport = normalize_viewport(update); }
    if let Some(update) = changes.facet_focus {
        facet_focus = update;
        if update.is_some() { focused = update; }
    }
    if let Some(update) = changes.focused_facet_index { focused = update; }
    for (name, index) in [("facet_focus", facet_focus), ("focused_facet_index", focused)] {
        if let Some(index) = index {
            if index < 0 { return Err(PlotError::type_error(format!("{name} must be a nonnegative integer or None"))); }
            if !is_grid { return Err(PlotError::type_error(format!("{name} requires a FacetGridPlot"))); }
        }
    }
    if roles_match && viewport == base.unwrap().viewport {
        viewport = rebase_viewport(base.unwrap(), &selected_spec, schema_ref, &candidate_values, viewport, registry_ref.unwrap_or(&DEFAULT_UNITS))?;
    }
    if let Some((x, y)) = viewport {
        let mut ranges = [x, y];
        for (mode, low, high) in LIMIT_PAIRS {
            if parameters.contains(low) && [mode, low, high].iter().any(|name| authored.contains_key(name)) {
                for (index, coordinate) in ["x", "y"].into_iter().enumerate() {
                    if low == format!("{coordinate}_min") { ranges[index] = None; }
                }
            }
        }
        viewport = normalize_viewport(Some((ranges[0], ranges[1])));
    }
    if let Some(update) = &changes.interaction { interaction = update.applied_to(&interaction); }
    if let Some(update) = &changes.presentation { presentation = update.applied_to(&presentation); }
    if let Some(raw) = &changes.classifier_thresholds { thresholds = normalize_classifier_threshold_targets(raw)?; }
    if !accepts_classifier_thresholds(&selected_spec, &candidate_values) { thresholds.clear(); }
    if let Some(update) = &changes.fit { fit_target = update.clone(); }
    let mut unknown_fit: Vec<&str> = fit_target.keys().filter(|key| !FIT_TARGET_FIELDS.contains(key)).collect();
    if !unknown_fit.is_empty() {
        unknown_fit.sort_unstable();
        return Err(PlotError::type_error(format!("unknown fit target fields: {unknown_fit:?}")));
    }
    if let Some(live) = changes.fit_live { fit_live = live; }
    let state = DisplayState {
        revision: previous.revision + u64::from(!resolution.changed.is_empty() || interaction != previous.interaction),
        values: candidate_values, changed_names: resolution.changed, effects: resolution.effects, interaction: interaction.clone(),
    };
    let units = configuration_fit_units(schema_ref, &selected_spec, &state, registry.as_ref())?;
    let resolution = engine.resolve_target(&FitTargetQuery {
        target: &fit_target, facet_grid: is_grid, units: &units, live: fit_live,
        previous_request: base.and_then(|base| base.fit_request.as_ref()),
        previous_expression_failure: base.and_then(|base| base.expression_failure.as_ref()),
    })?;
    let fit_target = resolution.request.as_ref().map(|request| request.target.clone()).unwrap_or_default();
    // Live is a property of a fit request: without one it names no science,
    // so the same scene has the same signature whoever asks.
    let fit_live = fit_live && resolution.request.is_some();
    let metadata = configuration_metadata(schema_ref, &selected_spec, &defaults.style, &*engine, registry_ref,
                                          resolution.request.as_ref().map(|request| &*request.model))?;
    let signature = Signature {
        context: scientific_context_signature(&selected_spec, schema.as_ref(), &state, &selectors, viewport, focused, &parameters, registry_ref)?,
        fit_target: fit_target.clone(), fit_live, thresholds: thresholds.clone(),
    };
    let cancellation = match base {
        Some(base) if same_source && !base.cancelled() && signature == base.signature => base.cancellation.clone(),
        _ => Arc::new(AtomicBool::new(false)),
    };
    let mut image_overlay = match &changes.image_overlay { Some(update) => update.clone(), None => base.and_then(|base| base.image_overlay.clone()) };
    if image_overlay.is_some() && !paints_image_surface(&selected_spec) {
        if changes.image_overlay.is_some() && !changes.overlay_follows_kind { return Err(PlotError::type_error("image overlay requires ImagePlot")); }
        image_overlay = None;
    }
    let discard_unmatched_classifier = base.is_some_and(|base| !continuing || !same_source || selected_spec != base.spec);
    let science_changed = presented.is_none_or(|presented| signature != presented.signature);
    Ok(Arc::new(ResolvedConfiguration {
        schema, spec: selected_spec, parameter_schema: parameters, display_state: state, size: selected_size, selectors, viewport,
        facet_focus, focused_facet_index: focused, classifier_thresholds: thresholds, fit_target, fit_live, interaction, presentation,
        defaults, fit: engine, unit_registry: registry, ignored_interaction: ignored, semantics, authored_semantics, metadata, signature, science_changed,
        cancellation, same_source, image_overlay, discard_unmatched_classifier, fit_request: resolution.request,
        expression_failure: resolution.expression_failure,
    }))
}

// A resolved configuration is shared by the authoring and rendering threads.
const _: () = {
    const fn shared<T: Send + Sync>() {}
    shared::<ResolvedConfiguration>();
    shared::<PlotSchema>();
};
