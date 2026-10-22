//! One frontend-neutral description of editable plot semantics
//! (`zlc_plot.semantics`): fate rows, coordinate alternatives, scope pins,
//! the single composition authority `composed_spec`, `rebased_spec`, label
//! carry-over by role, and `describe_semantics`.

use std::collections::HashSet;
use std::sync::Arc;

use zlc_core::schema::Axis;

use crate::layout::LayoutConfig;
use crate::pyfmt::format_g;

use super::data_contract::{axis_choices_for_schema, coordinate_axis_ref, coordinate_position, resolve_axis, PlotSchema, PRIMARY_INDEX};
use super::error::{ErrorKind, PlotError, Result};
use super::handlers::{admits, chosen_spec, default_spec, fitting_spec, handler, kind_label, label_roles, DEFAULT_LAYOUT, HANDLERS};
use super::kinds::{axis_ref, axis_repr, canonical_scalar, coordinate_value, kind_from_value, reduction_from_value, AxisDomain, AxisRef, Coordinate, PlotKind};
use super::ordered::OrderedMap;
use super::spec::{
    set_coordinates, set_reduced, set_reduction, set_role, CellPlot, FacetGridPlot, LabelSlot, PlotLabels, PlotSpec,
    Reduction, Role, ScopeTerm, GRID_CELL_KINDS,
};
use super::value::{Value, ValueMap};

pub const FATE_PREFIX: &str = "fate:";
pub const COORDINATE_PREFIX: &str = "coordinate:";
pub const FATE_REDUCE: &str = "reduce";
pub const FATE_POOL: &str = "pool";
const SCOPE_VALUE_TAG: &str = "scope-value";
const ROLE_FATES: [Role; 4] = [Role::X, Role::Y, Role::Group, Role::Facet];

fn role_label(role: Role) -> &'static str {
    match role { Role::X => "X axis", Role::Y => "Y axis", Role::Group => "Group", Role::Facet => "Facet" }
}

/// One `(value, label)` choice.
pub type SemanticChoice = (Value, String);

/// The lazily materialized coordinate domain behind one axis's Scope fate.
#[derive(Clone)]
pub struct CycleChoices { axis: Arc<Axis> }

impl CycleChoices {
    pub fn new(axis: Arc<Axis>) -> Self { Self { axis } }
    pub fn len(&self) -> usize { self.axis.len() }
    pub fn is_empty(&self) -> bool { self.axis.len() == 0 }
    pub fn axis(&self) -> &Arc<Axis> { &self.axis }
    /// The tagged scope value and its label at one position.
    pub fn get(&self, index: usize) -> Option<SemanticChoice> {
        let coordinate = super::data_contract::coordinate_at(&self.axis, index)?;
        let label = match self.axis.labels() {
            Some(labels) => labels[index].to_string(),
            None => match &coordinate {
                Coordinate::Missing => "(null)".to_owned(),
                Coordinate::Integer(value) => format_g(*value as f64, 6),
                Coordinate::Real(value) => format_g(*value, 6),
                Coordinate::Text(text) => text.to_string(),
            },
        };
        Some((scope_fate(&coordinate), label))
    }
    /// Where one tagged scope value sits, through the axis's own locator.
    pub fn position_of(&self, value: &Value) -> Option<usize> {
        if !is_scope_fate(value) { return None; }
        let coordinate = scope_coordinate_from_fate(value).ok()?;
        coordinate_position(&self.axis, &coordinate)
    }
    pub fn contains_value(&self, value: &Value) -> bool { self.position_of(value).is_some() }
}

impl std::fmt::Debug for CycleChoices {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "CycleChoices({})", self.axis.id()) }
}

/// One semantic editor entry.
#[derive(Clone, Debug)]
pub struct SemanticField {
    pub name: String,
    pub label: String,
    pub value: Value,
    pub choices: Vec<SemanticChoice>,
    pub required: bool,
    pub cycle_choices: Option<CycleChoices>,
}

impl SemanticField {
    pub fn new(name: String, label: String, value: Value, offered: Vec<SemanticChoice>, required: bool, cycle_choices: Option<CycleChoices>) -> Result<Self> {
        let mut choices: Vec<SemanticChoice> = Vec::new();
        for (choice, text) in offered {
            if text.trim().is_empty() { return Err(PlotError::type_error("semantic choices must be (value, non-empty label) pairs")); }
            if choices.iter().any(|(previous, _)| *previous == choice) { continue; }
            choices.push((choice, text));
        }
        if required && value.is_none() { return Err(PlotError::value(format!("required semantic field '{name}' cannot be None"))); }
        let current_is_cycle = cycle_choices.as_ref().is_some_and(|cycle| cycle.contains_value(&value));
        if !choices.is_empty() && !current_is_cycle && !choices.iter().any(|(choice, _)| *choice == value) {
            return Err(PlotError::value(format!("current value for semantic field '{name}' is not in choices")));
        }
        Ok(Self { name, label, value, choices, required, cycle_choices })
    }
    pub fn choice_values(&self) -> impl Iterator<Item = &Value> { self.choices.iter().map(|(value, _)| value) }
}

/// The complete semantic edit domain for one `(schema, spec)` pair.
#[derive(Clone, Debug)]
pub struct SemanticDescription {
    pub kind: PlotKind,
    pub kind_choices: Vec<PlotKind>,
    pub fields: Vec<SemanticField>,
    pub axis_choices: Vec<AxisRef>,
    pub x: Option<AxisRef>,
    pub y: Option<AxisRef>,
    pub group: Option<AxisRef>,
    pub reduction: Option<Reduction>,
    pub facet: Option<AxisRef>,
    pub facet_max_cells: u32,
    /// Which row belongs to which axis, in offered order.
    pub fate_rows: Vec<(AxisRef, String)>,
}

impl SemanticDescription {
    pub fn field(&self, name: &str) -> Option<&SemanticField> { self.fields.iter().find(|field| field.name == name) }
    pub fn declares(&self, name: &str) -> bool { self.field(name).is_some() }
    /// Current semantic values keyed by field name.
    pub fn values(&self) -> ValueMap { self.fields.iter().map(|field| (field.name.as_str(), field.value.clone())).collect() }
}

/// Tagged fate value for one actual pinned coordinate.
pub fn scope_fate(coordinate: &Coordinate) -> Value {
    Value::list([Value::text(SCOPE_VALUE_TAG), coordinate_value(coordinate)])
}

pub fn is_scope_fate(value: &Value) -> bool {
    matches!(value, Value::List(items) if items.len() == 2 && items[0] == Value::text(SCOPE_VALUE_TAG))
}

/// Decode one tagged pin without confusing text coordinates with verbs.
pub fn scope_coordinate_from_fate(value: &Value) -> Result<Coordinate> {
    match value {
        Value::List(items) if items.len() == 2 && items[0] == Value::text(SCOPE_VALUE_TAG) => canonical_scalar(&items[1], "scope coordinate"),
        Value::List(_) => Err(PlotError::value("scope fate tag is invalid")),
        _ => Err(PlotError::type_error("scope fate must be a tagged sequence")),
    }
}

/// Stable machine key for one exact AxisRef.
pub fn fate_field_name(reference: &AxisRef) -> String { format!("{FATE_PREFIX}{}:{}", reference.domain.value(), reference.axis_id) }

fn coordinate_field_name(schema: &PlotSchema, reference: &AxisRef) -> Result<String> {
    let primary = coordinate_axis_ref(schema, reference)?;
    Ok(format!("{COORDINATE_PREFIX}{}:{}", primary.domain.value(), primary.axis_id))
}

fn axis_label(schema: &PlotSchema, reference: &AxisRef) -> Result<String> { Ok(resolve_axis(schema, reference)?.label().to_owned()) }

/// The semantic field names of a specification's kind, in order.
pub fn field_names(spec: &PlotSpec) -> Vec<&'static str> {
    let names: Vec<&'static str> = if let PlotSpec::FacetGrid(grid) = spec {
        let mut names = vec!["kind", "cell_kind"];
        names.extend(handler(grid.cell.kind()).semantic_fields.iter().copied().filter(|name| *name != "kind"));
        names.push("facet");
        names
    } else {
        handler(spec.kind()).semantic_fields.to_vec()
    };
    let mut unique = Vec::new();
    for name in names { if !unique.contains(&name) { unique.push(name); } }
    unique
}

fn optional_roles(spec: &PlotSpec) -> HashSet<Role> {
    let semantic = spec.semantic();
    let mut optional: HashSet<Role> = ROLE_FATES.into_iter().filter(|role| semantic.optional_role(*role)).collect();
    if spec.is_facet_grid() { optional.insert(Role::Facet); }
    optional
}

fn declares_reduced(spec: &PlotSpec) -> bool { spec.semantic().reduced().is_some() }

fn default_fate(spec: &PlotSpec) -> &'static str { if spec.semantic().kind() == PlotKind::Histogram { FATE_POOL } else { FATE_REDUCE } }

fn role_holder(spec: &PlotSpec, role: Role) -> Option<AxisRef> {
    match role { Role::Facet => spec.facet().cloned(), other => spec.semantic().role(other).cloned() }
}

fn scope_terms(spec: &PlotSpec) -> OrderedMap<AxisRef, Coordinate> { OrderedMap::from_pairs(spec.scope().iter().cloned()) }

/// Every axis this specification names, in any role.
fn axes_used_by(spec: &PlotSpec) -> Vec<AxisRef> {
    let semantic = spec.semantic();
    let mut used: Vec<AxisRef> = [semantic.x(), semantic.y(), semantic.group(), spec.facet()].into_iter().flatten().cloned().collect();
    used.extend(spec.scope().iter().map(|(axis, _)| axis.clone()));
    used
}

/// What this specification has already made of one axis.
pub fn fate_of(spec: &PlotSpec, reference: &AxisRef) -> Value {
    let semantic = spec.semantic();
    for role in [Role::X, Role::Y, Role::Group] {
        if semantic.role(role) == Some(reference) { return Value::text(role.name()); }
    }
    if spec.facet() == Some(reference) { return Value::text("facet"); }
    if let Some((_, coordinate)) = spec.scope().iter().find(|(axis, _)| axis == reference) { return scope_fate(coordinate); }
    if semantic.reduced().is_some_and(|reduced| reduced.contains(reference)) { return Value::text(FATE_REDUCE); }
    Value::text(default_fate(spec))
}

/// The fixed record carrier of a Rolling plot.
pub fn is_rolling_record_axis(schema: &PlotSchema, reference: &AxisRef) -> Result<bool> {
    if schema.history()?.is_none() { return Ok(reference.domain == AxisDomain::Repeat); }
    Ok(resolve_axis(schema, reference)?.primary().role() == PRIMARY_INDEX)
}

/// The one exact editable row for each physical source axis.
pub fn fate_row_axes(schema: &PlotSchema, spec: &PlotSpec, offered: Option<&[AxisRef]>, domains: Option<&[AxisDomain]>) -> Result<Vec<AxisRef>> {
    let in_domains = |reference: &AxisRef| domains.is_none_or(|domains| domains.contains(&reference.domain));
    let all_axes;
    let axes = match offered { Some(axes) => axes, None => { all_axes = axis_choices_for_schema(schema); &all_axes } };
    let mut preferred: OrderedMap<AxisRef, AxisRef> = OrderedMap::new();
    for axis in spec.coordinates() {
        if in_domains(axis) { preferred.insert(coordinate_axis_ref(schema, axis)?, axis.clone()); }
    }
    let mut used = axes_used_by(spec);
    used.extend(spec.semantic().reduced().unwrap_or(&[]).iter().cloned());
    let mut active: OrderedMap<AxisRef, AxisRef> = OrderedMap::new();
    for reference in used {
        if !in_domains(&reference) { continue; }
        let primary = coordinate_axis_ref(schema, &reference)?;
        if active.get(&primary).is_some_and(|held| *held != reference) {
            return Err(PlotError::value("alternative coordinates of one axis cannot have different fates"));
        }
        active.insert(primary, reference);
    }
    preferred.update(&active);
    let mut listed: OrderedMap<AxisRef, AxisRef> = OrderedMap::new();
    for axis in axes {
        if !in_domains(axis) { continue; }
        let primary = coordinate_axis_ref(schema, axis)?;
        let chosen = preferred.get(&primary).cloned().unwrap_or_else(|| primary.clone());
        listed.set_default(primary, chosen);
    }
    for (primary, chosen) in preferred.iter() { listed.set_default(primary.clone(), chosen.clone()); }
    Ok(listed.values().cloned().collect())
}

/// A scope coordinate as projection reads it: a value, or the latest.
#[derive(Clone, Debug, PartialEq)]
pub enum ScopeCoordinate { Value(Coordinate), Latest }

/// Authored pins plus Last's exact last-coordinate pins.
pub fn projection_scope(schema: &PlotSchema, spec: &PlotSpec, domains: Option<&[AxisDomain]>) -> Result<Vec<(AxisRef, ScopeCoordinate)>> {
    let mut scope: OrderedMap<AxisRef, ScopeCoordinate> = OrderedMap::from_pairs(spec.scope().iter()
        .filter(|(axis, _)| domains.is_none_or(|domains| domains.contains(&axis.domain)))
        .map(|(axis, coordinate)| (axis.clone(), ScopeCoordinate::Value(coordinate.clone()))));
    if spec.semantic().reduction() == Some(Reduction::Last) {
        for reference in fate_row_axes(schema, spec, None, domains)? {
            if fate_of(spec, &reference) != Value::text(FATE_REDUCE) { continue; }
            if spec.kind() == PlotKind::Rolling && is_rolling_record_axis(schema, &reference)? { continue; }
            scope.insert(reference, ScopeCoordinate::Latest);
        }
    }
    Ok(scope.into_pairs())
}

/// Whether a scope fate names a coordinate this axis actually HAS.
pub fn axis_admits_scope(schema: &PlotSchema, reference: &AxisRef, value: &Value) -> Result<bool> {
    if !is_scope_fate(value) { return Ok(false); }
    let coordinate = scope_coordinate_from_fate(value)?;
    Ok(resolve_axis(schema, reference)?.coordinate_position(&coordinate).is_some())
}

/// One semantic value with its vocabulary type applied (`typed_choice`).
#[derive(Clone, Debug)]
enum Choice { Kind(PlotKind), Reduction(Reduction), Role(Option<AxisRef>), Raw(Value) }

fn typed_choice(name: &str, value: &Value) -> Result<Choice> {
    Ok(match name {
        "kind" | "cell_kind" => Choice::Kind(kind_from_value(value)?),
        "reduction" => Choice::Reduction(reduction_from_value(value)?),
        _ => Choice::Raw(value.clone()),
    })
}

fn choice_value(choice: &Choice) -> Value {
    match choice {
        Choice::Kind(kind) => Value::text(kind.value()), Choice::Reduction(reduction) => Value::text(reduction.value()),
        Choice::Role(Some(axis)) => Value::text(&*axis.axis_id), Choice::Role(None) => Value::None, Choice::Raw(value) => value.clone(),
    }
}

/// `replace(spec, scope=...)` but only when the terms differ.
fn with_scope_if_changed(spec: PlotSpec, terms: &OrderedMap<AxisRef, Coordinate>) -> Result<PlotSpec> {
    if terms.pairs() == spec.scope() { return Ok(spec); }
    spec.with_scope(terms.pairs().to_vec())
}

/// This specification with its collapsed axes replaced.
fn with_reduced(spec: &PlotSpec, reduced: Vec<AxisRef>) -> Result<PlotSpec> {
    spec.map_semantic(|semantic| {
        let group = semantic.semantic().group().cloned();
        let keep = group.filter(|group| !reduced.contains(group));
        set_reduced(semantic, reduced)?;
        set_role(semantic, Role::Group, keep)
    })
}

/// Change a coordinate, preserving the axis fate and pinned row position.
fn coordinate_spec(schema: &PlotSchema, spec: &PlotSpec, primary: &AxisRef, selected: &AxisRef) -> Result<PlotSpec> {
    let remap = |reference: Option<&AxisRef>| -> Result<Option<AxisRef>> {
        match reference {
            Some(reference) if coordinate_axis_ref(schema, reference)? == *primary => Ok(Some(selected.clone())),
            other => Ok(other.cloned()),
        }
    };
    let semantic = spec.semantic();
    let mut changes: Vec<(Role, Option<AxisRef>)> = Vec::new();
    for role in [Role::X, Role::Y, Role::Group] {
        if semantic.has_role(role) { changes.push((role, remap(semantic.role(role))?)); }
    }
    let reduced = match semantic.reduced() {
        Some(reduced) => Some(reduced.iter().map(|reference| Ok(remap(Some(reference))?.unwrap())).collect::<Result<Vec<_>>>()?),
        None => None,
    };
    let mut scope = Vec::new();
    for (reference, value) in spec.scope() {
        let target = remap(Some(reference))?.unwrap();
        let mut value = value.clone();
        if target != *reference {
            let Some(position) = resolve_axis(schema, reference)?.coordinate_position(&value) else { continue };
            value = resolve_axis(schema, &target)?.coordinate_at(position).ok_or_else(|| PlotError::index("axis index out of range"))?;
        }
        scope.push((target, value));
    }
    let mut choices: Vec<AxisRef> = Vec::new();
    for reference in spec.coordinates() {
        if coordinate_axis_ref(schema, reference)? != *primary { choices.push(reference.clone()); }
    }
    if selected != primary { choices.push(selected.clone()); }
    let apply = |target: &mut PlotSpec| -> Result<()> {
        for (role, value) in &changes { set_role(target, *role, value.clone())?; }
        if let Some(reduced) = &reduced { set_reduced(target, reduced.clone())?; }
        Ok(())
    };
    let mut result = match spec {
        PlotSpec::FacetGrid(grid) => {
            let mut cell = grid.cell.as_spec();
            apply(&mut cell)?;
            let cell = CellPlot::from_spec(cell.validated()?)?;
            PlotSpec::FacetGrid(FacetGridPlot { cell, facet: remap(grid.facet.as_ref())?, ..grid.clone() })
        }
        other => { let mut spec = other.clone(); apply(&mut spec)?; spec }
    };
    super::spec::set_scope_raw(&mut result, scope);
    set_coordinates(&mut result, choices);
    result.validated()
}

/// Carry authored labels across a semantic edit by role, never by slot.
pub fn merge_labels(old_spec: &PlotSpec, new_spec: &PlotSpec, schema: Option<&PlotSchema>) -> Result<PlotLabels> {
    let old_labels = old_spec.semantic().labels();
    let history = match schema { Some(schema) => schema.history()?, None => None };
    let mut carriers: Vec<(super::handlers::LabelRole, Arc<str>)> = Vec::new();
    for (slot, role) in label_roles(old_spec, history.as_deref())? {
        let label = if slot == LabelSlot::Title { old_spec.labels().title.clone() } else { old_labels.get(slot).cloned() };
        if let Some(label) = label.filter(|text| !text.is_empty()) {
            match carriers.iter_mut().find(|(held, _)| *held == role) { Some(slot) => slot.1 = label, None => carriers.push((role, label)) }
        }
    }
    let mut labels = PlotLabels::default();
    for (slot, role) in label_roles(new_spec, history.as_deref())? {
        if let Some((_, label)) = carriers.iter().find(|(held, _)| *held == role) { labels.set(slot, Some(label.clone())); }
    }
    Ok(labels)
}

/// Carry existing semantic intent onto another schema of the chosen kind.
pub fn rebased_spec(schema: &PlotSchema, kind: PlotKind, values: &ValueMap) -> Result<PlotSpec> {
    let selected_kind = match values.get("kind") { Some(value) => kind_from_value(value)?, None => kind };
    let cell_kind = values.get("cell_kind").map(kind_from_value).transpose()?;
    let candidate = fitting_spec(Some(schema), Some(selected_kind), cell_kind)?
        .ok_or_else(|| PlotError::value(format!("{} has no unambiguous default for this schema", selected_kind.value())))?;
    let vocabulary: ValueMap = ["kind", "cell_kind"].into_iter().filter_map(|name| values.get(name).map(|value| (name, value.clone()))).collect();
    let candidate = if vocabulary.is_empty() { candidate } else { composed_spec(Some(schema), &candidate, &vocabulary)? };
    let names: HashSet<String> = describe_semantics(Some(schema), &candidate, &DEFAULT_LAYOUT)?.fields.into_iter().map(|field| field.name).collect();
    let carried: ValueMap = values.iter().filter(|(name, _)| names.contains(*name)).map(|(name, value)| (name, value.clone())).collect();
    composed_spec(Some(schema), &candidate, &carried)
}

/// Compose the candidate one semantic edit produces.
pub fn updated_spec(schema: Option<&PlotSchema>, spec: &PlotSpec, name: &str, value: &Value) -> Result<PlotSpec> {
    if name.trim().is_empty() { return Err(PlotError::type_error("semantic field name must be non-empty text")); }
    composed_spec(schema, spec, &ValueMap::from_pairs([(name, value.clone())]))
}

fn chosen_kind_spec(schema: Option<&PlotSchema>, current: &PlotSpec, kind: PlotKind) -> Result<Option<PlotSpec>> {
    if kind == PlotKind::FacetGrid { return chosen_spec(schema, &current.semantic_spec()); }
    default_spec(schema, kind)
}

/// The single composition authority: a whole bag of choices is ONE edit.
pub fn composed_spec(schema: Option<&PlotSchema>, spec: &PlotSpec, values: &ValueMap) -> Result<PlotSpec> {
    if values.is_empty() { return Ok(spec.clone()); }
    let mut rest: OrderedMap<String, Choice> = OrderedMap::new();
    for (name, value) in values.iter() { rest.insert(name.to_owned(), typed_choice(name, value)?); }
    let mut candidate = spec.clone();
    let mut scope = scope_terms(spec);

    if let Some(choice) = rest.remove(&"kind".to_owned()) {
        let Choice::Kind(kind) = choice else { unreachable!("kind is typed") };
        if kind != spec.kind() {
            candidate = chosen_kind_spec(schema, spec, kind)?
                .ok_or_else(|| PlotError::value(format!("{} cannot be built for this dataset", kind.value())))?;
            if let Some(schema) = schema {
                for reference in spec.coordinates() {
                    candidate = coordinate_spec(schema, &candidate, &coordinate_axis_ref(schema, reference)?, reference)?;
                }
            }
        }
    }

    if let Some(choice) = rest.remove(&"cell_kind".to_owned()) {
        let Choice::Kind(cell_kind) = choice else { unreachable!("cell_kind is typed") };
        let PlotSpec::FacetGrid(grid) = &candidate else { return Err(PlotError::value("only a facet grid has a cell kind")) };
        if !GRID_CELL_KINDS.contains(&cell_kind) { return Err(PlotError::value(format!("a grid cell cannot be a {}", cell_kind.value()))); }
        if cell_kind != grid.cell.kind() {
            let changed = fitting_spec(schema, Some(PlotKind::FacetGrid), Some(cell_kind))?
                .ok_or_else(|| PlotError::value(format!("{} cannot be built for this dataset", cell_kind.value())))?;
            let schema = schema.expect("a facet grid has a schema");
            let mut inherited: OrderedMap<String, Choice> = OrderedMap::new();
            for (reference, value) in scope.iter() {
                inherited.insert(fate_field_name(&coordinate_axis_ref(schema, reference)?), Choice::Raw(scope_fate(value)));
            }
            let cell_roles = field_names(&changed);
            for role in [Role::X, Role::Y, Role::Group] {
                if let Some(reference) = role_holder(&candidate, role) {
                    if cell_roles.contains(&role.name()) {
                        inherited.set_default(fate_field_name(&coordinate_axis_ref(schema, &reference)?), Choice::Raw(Value::text(role.name())));
                    }
                }
            }
            if cell_roles.contains(&"reduction") {
                let reduction = candidate.semantic().reduction().expect("a cell has a reduction");
                rest.set_default("reduction".to_owned(), Choice::Reduction(reduction));
            }
            let (changed_facet, candidate_facet) = (changed.facet().cloned(), candidate.facet().cloned());
            if changed_facet != candidate_facet {
                if let Some(facet) = &changed_facet {
                    inherited.set_default(fate_field_name(&coordinate_axis_ref(schema, facet)?), Choice::Raw(Value::text(FATE_REDUCE)));
                }
                if let Some(facet) = &candidate_facet {
                    inherited.set_default(fate_field_name(&coordinate_axis_ref(schema, facet)?), Choice::Raw(Value::text("facet")));
                }
            }
            let mut merged = inherited;
            merged.update(&rest);
            rest = merged;
            let coordinates = candidate.coordinates().to_vec();
            candidate = changed.with_scope(Vec::new())?;
            for reference in &coordinates {
                candidate = coordinate_spec(schema, &candidate, &coordinate_axis_ref(schema, reference)?, reference)?;
            }
            scope = OrderedMap::new();
        }
    }

    if let (Some(schema), PlotSpec::Rolling(rolling)) = (schema, &candidate) {
        let mut record = None;
        for reference in axis_choices_for_schema(schema) {
            if is_rolling_record_axis(schema, &reference)? { record = Some(reference); break; }
        }
        let mut along = None;
        if let Some(record) = record {
            if schema.history()?.is_some() {
                along = Some(record);
                for reference in rolling.x.iter().chain(rolling.coordinates.iter()) {
                    if is_rolling_record_axis(schema, reference)? { along = Some(reference.clone()); break; }
                }
            }
        }
        let mut group = rolling.group.clone();
        if let Some(held) = &group { if is_rolling_record_axis(schema, held)? { group = None; } }
        let mut kept = OrderedMap::new();
        for (reference, value) in scope.iter() {
            if !is_rolling_record_axis(schema, reference)? { kept.insert(reference.clone(), value.clone()); }
        }
        scope = kept;
        let mut updated = rolling.clone();
        updated.x = along;
        updated.group = group;
        updated.scope = scope.pairs().to_vec();
        candidate = PlotSpec::Rolling(updated.validated()?);
    }

    if let Some(schema) = schema {
        for old in fate_row_axes(schema, &candidate, None, None)? {
            let field_name = coordinate_field_name(schema, &old)?;
            let Some(choice) = rest.remove(&field_name) else { continue };
            let selected = axis_ref(old.domain, &choice_value(&choice).py_str())?;
            let primary = coordinate_axis_ref(schema, &old)?;
            if coordinate_axis_ref(schema, &selected)? != primary { return Err(PlotError::value("coordinate choice belongs to another axis")); }
            candidate = coordinate_spec(schema, &candidate, &primary, &selected)?;
        }
        if values.keys().any(|name| name.starts_with(COORDINATE_PREFIX)) { scope = scope_terms(&candidate); }
    }

    let fate_rows: OrderedMap<String, AxisRef> = match schema {
        None => OrderedMap::new(),
        Some(schema) => {
            let row_spec = with_scope_if_changed(candidate.clone(), &scope)?;
            let mut rows = OrderedMap::new();
            for axis in fate_row_axes(schema, &row_spec, None, None)? {
                if candidate.kind() == PlotKind::Rolling && is_rolling_record_axis(schema, &axis)? { continue; }
                rows.insert(fate_field_name(&coordinate_axis_ref(schema, &axis)?), axis);
            }
            rows
        }
    };
    let mut fate_values: OrderedMap<AxisRef, Value> = OrderedMap::new();
    let names: Vec<String> = rest.keys().cloned().collect();
    for name in names {
        if !name.starts_with(FATE_PREFIX) { continue; }
        let value = choice_value(&rest.remove(&name).unwrap());
        let axis = fate_rows.get(&name).cloned().ok_or_else(|| PlotError::key(super::value::repr_str(&name)))?;
        fate_values.insert(axis.clone(), value);
        scope.remove(&axis);
    }

    if !fate_values.is_empty() {
        let schema = schema.expect("fate rows need a schema");
        let declared_names = field_names(&candidate);
        let declared_roles: Vec<Role> = ROLE_FATES.into_iter().filter(|role| declared_names.contains(&role.name())).collect();
        let current_roles: OrderedMap<Role, Option<AxisRef>> = OrderedMap::from_pairs(declared_roles.iter().map(|role| (*role, role_holder(&candidate, *role))));
        let mut desired_roles = current_roles.clone();
        let edited_axes: HashSet<AxisRef> = fate_values.keys().cloned().collect();

        let previous_scope = scope_terms(&candidate);
        let mut terms = previous_scope.clone();
        terms.retain(|axis, _| !edited_axes.contains(axis));
        candidate = with_scope_if_changed(candidate, &terms)?;
        if let Some(previous_reduced) = candidate.semantic().reduced().map(<[AxisRef]>::to_vec) {
            let reduced: Vec<AxisRef> = previous_reduced.iter().filter(|axis| !edited_axes.contains(*axis)).cloned().collect();
            if reduced != previous_reduced { candidate = with_reduced(&candidate, reduced)?; }
        }

        let mut role_targets: OrderedMap<Role, AxisRef> = OrderedMap::new();
        for (axis, value) in fate_values.iter() {
            if is_scope_fate(value) {
                if !axis_admits_scope(schema, axis, value)? { continue; }
                scope.insert(axis.clone(), scope_coordinate_from_fate(value)?);
            } else if *value == Value::text(FATE_REDUCE) || *value == Value::text(FATE_POOL) {
                if let Some(previous) = candidate.semantic().reduced() {
                    let mut reduced: OrderedMap<AxisRef, ()> = OrderedMap::from_pairs(previous.iter().map(|entry| (entry.clone(), ())));
                    if *value == Value::text(FATE_REDUCE) { reduced.insert(axis.clone(), ()); } else { reduced.remove(axis); }
                    candidate = with_reduced(&candidate, reduced.keys().cloned().collect())?;
                }
            } else if let Some(role) = value.as_str().and_then(Role::parse) {
                if !declared_roles.contains(&role) {
                    return Err(PlotError::value(format!("{} has no '{}' fate", candidate.kind().value(), role.name())));
                }
                if role_targets.get(&role).is_some_and(|previous| previous != axis) {
                    return Err(PlotError::value(format!("fate '{}' is assigned to more than one axis", role.name())));
                }
                role_targets.insert(role, axis.clone());
            } else {
                return Err(PlotError::value(format!("unknown fate {} for axis {}", value.py_repr(), axis_repr(axis))));
            }
        }

        let holders: Vec<(Role, Option<AxisRef>)> = desired_roles.iter().map(|(role, holder)| (*role, holder.clone())).collect();
        for (role, holder) in holders {
            if holder.is_some_and(|holder| edited_axes.contains(&holder)) { desired_roles.insert(role, None); }
        }
        for (role, axis) in role_targets.iter() {
            let holders: Vec<(Role, Option<AxisRef>)> = desired_roles.iter().map(|(role, holder)| (*role, holder.clone())).collect();
            for (other, holder) in holders {
                if other != *role && holder.as_ref() == Some(axis) { desired_roles.insert(other, None); }
            }
            desired_roles.insert(*role, Some(axis.clone()));
        }
        for (role, axis) in role_targets.iter() {
            let previous_role = current_roles.iter().find(|(_, holder)| holder.as_ref() == Some(axis)).map(|(name, _)| *name);
            let displaced = current_roles.get(role).cloned().flatten();
            if let (Some(previous_role), Some(displaced)) = (previous_role, displaced) {
                if previous_role != *role && !role_targets.contains_key(&previous_role) && !edited_axes.contains(&displaced) {
                    desired_roles.insert(previous_role, Some(displaced));
                }
            }
        }
        let optional = optional_roles(&candidate);
        for role in &declared_roles {
            if desired_roles.get(role).cloned().flatten().is_none() && !optional.contains(role) {
                return Err(PlotError::vacancy(role.name(), candidate.kind().value()));
            }
        }
        for role in &declared_roles {
            rest.insert(role.name().to_owned(), Choice::Role(desired_roles.get(role).cloned().flatten()));
        }
    }

    let settled = |candidate: PlotSpec, scope: &OrderedMap<AxisRef, Coordinate>| -> Result<PlotSpec> {
        let semantic = candidate.semantic();
        let role_axes: HashSet<AxisRef> = [semantic.x(), semantic.y(), semantic.group(), candidate.facet()].into_iter().flatten().cloned().collect();
        let mut pinned = scope.clone();
        pinned.retain(|axis, _| !role_axes.contains(axis));
        let candidate = with_scope_if_changed(candidate, &pinned)?;
        let labels = merge_labels(spec, &candidate, schema)?;
        match candidate {
            PlotSpec::FacetGrid(grid) => {
                let mut cell = grid.cell.clone();
                *cell.labels_mut() = PlotLabels { title: None, ..labels.clone() };
                PlotSpec::FacetGrid(FacetGridPlot { labels: labels.title_only(), cell, ..grid }).validated()
            }
            mut other => { *other.labels_mut() = labels; other.validated() }
        }
    };

    if rest.is_empty() { return settled(candidate, &scope); }
    let declared = field_names(&candidate);
    if let Some(unknown) = rest.keys().find(|name| !declared.contains(&name.as_str())) {
        return Err(PlotError::key(super::value::repr_str(unknown)));
    }
    let assign = |target: &mut PlotSpec, name: &str, choice: &Choice| -> Result<()> {
        match name {
            "reduction" => match choice {
                Choice::Reduction(reduction) => set_reduction(target, *reduction),
                _ => Err(PlotError::type_error("reduction must be Reduction")),
            },
            role_name => {
                let role = Role::parse(role_name).ok_or_else(|| PlotError::type_error(format!("unexpected field {role_name:?}")))?;
                let value = match choice {
                    Choice::Role(value) => value.clone(),
                    Choice::Raw(Value::None) => None,
                    _ => return Err(PlotError::type_error(format!("{} must be AxisRef or None", role.name()))),
                };
                set_role(target, role, value)
            }
        }
    };
    candidate = match &candidate {
        PlotSpec::FacetGrid(grid) => {
            let mut cell = grid.cell.as_spec();
            let mut cell_changed = false;
            let mut facet = grid.facet.clone();
            for (name, choice) in rest.iter() {
                if name == "facet" {
                    let mut holder = PlotSpec::FacetGrid(grid.clone());
                    assign(&mut holder, name, choice)?;
                    facet = holder.facet().cloned();
                } else {
                    assign(&mut cell, name, choice)?;
                    cell_changed = true;
                }
            }
            let cell = if cell_changed { CellPlot::from_spec(cell.validated()?)? } else { grid.cell.clone() };
            PlotSpec::FacetGrid(FacetGridPlot { facet, cell, ..grid.clone() }).validated()?
        }
        other => {
            let mut updated = other.clone();
            for (name, choice) in rest.iter() { assign(&mut updated, name, choice)?; }
            updated.validated()?
        }
    };
    settled(candidate, &scope)
}

/// Describe the complete registry/schema semantic vocabulary.
pub fn describe_semantics(schema: Option<&PlotSchema>, spec: &PlotSpec, layout: &LayoutConfig) -> Result<SemanticDescription> {
    let Some(schema) = schema else {
        if spec.kind() != PlotKind::PulseTimeline { return Err(PlotError::type_error("schema must be DatasetSchema for dataset plot kinds")); }
        let kind = PlotKind::PulseTimeline;
        let field = SemanticField::new("kind".into(), "Plot kind".into(), Value::text(kind.value()),
            vec![(Value::text(kind.value()), kind_label(kind).to_owned())], true, None)?;
        return Ok(SemanticDescription { kind, kind_choices: vec![kind], fields: vec![field], axis_choices: Vec::new(), x: None, y: None,
            group: None, reduction: None, facet: None, facet_max_cells: layout.facet_max_cells(), fate_rows: Vec::new() });
    };
    if !admits(spec.kind(), Some(schema))? {
        return Err(PlotError::value(format!("plot specification {} is not admissible for schema", spec_type_name(spec))));
    }
    let axes = axis_choices_for_schema(schema);
    let mut kinds = Vec::new();
    for item in HANDLERS { if admits(item.kind, Some(schema))? { kinds.push(item.kind); } }
    let kind_choices: Vec<SemanticChoice> = kinds.iter().map(|kind| (Value::text(kind.value()), kind_label(*kind).to_owned())).collect();
    let semantic = spec.semantic();
    let declared = field_names(spec);
    let mut fields = vec![SemanticField::new("kind".into(), "Plot kind".into(), Value::text(spec.kind().value()), kind_choices, true, None)?];
    if spec.is_facet_grid() {
        fields.push(SemanticField::new("cell_kind".into(), "Cell kind".into(), Value::text(semantic.kind().value()),
            GRID_CELL_KINDS.iter().map(|kind| (Value::text(kind.value()), kind_label(*kind).to_owned())).collect(), true, None)?);
    }
    let mut fate_rows = Vec::new();
    let default = default_fate(spec);
    let default_label = if default == FATE_POOL { "pooled" } else { "reduced" };
    let roles: Vec<Role> = ROLE_FATES.into_iter().filter(|role| declared.contains(&role.name())).collect();
    for reference in fate_row_axes(schema, spec, Some(&axes), None)? {
        let label = axis_label(schema, &reference)?;
        let primary = coordinate_axis_ref(schema, &reference)?;
        let name = fate_field_name(&primary);
        let resolved = resolve_axis(schema, &reference)?;
        let family = resolved.family();
        let fixed_record = spec.kind() == PlotKind::Rolling && is_rolling_record_axis(schema, &reference)?;
        if family.len() > 1 {
            let label = if fixed_record { "X coordinate".to_owned() } else { format!("{} coordinate", axis_label(schema, &primary)?) };
            fields.push(SemanticField::new(coordinate_field_name(schema, &reference)?, label, Value::text(&*reference.axis_id),
                family.iter().map(|axis| (Value::text(axis.id()), axis.name().to_owned())).collect(), true, None)?);
        }
        if fixed_record { continue; }
        let current = fate_of(spec, &reference);
        let mut offered = vec![(Value::text(default), format!("({default_label})"))];
        if default == FATE_POOL && declares_reduced(spec) { offered.push((Value::text(FATE_REDUCE), "reduced".to_owned())); }
        for role in &roles { offered.push((Value::text(role.name()), role_label(*role).to_owned())); }
        let pins = CycleChoices::new(resolved.axis.clone());
        fields.push(SemanticField::new(name.clone(), label, current, offered, true, Some(pins))?);
        fate_rows.push((reference, name));
    }
    if declared.contains(&"reduction") {
        let current = semantic.reduction().expect("declared reduction");
        let mut offered = Vec::new();
        for reduction in Reduction::ALL {
            let value = Value::text(reduction.value());
            let available = reduction == current || match updated_spec(Some(schema), spec, "reduction", &value) {
                Ok(_) => true,
                Err(error) => error.kind == ErrorKind::Vacancy,
            };
            if available { offered.push((value, reduction.value().to_owned())); }
        }
        fields.push(SemanticField::new("reduction".into(), "Reduction".into(), Value::text(current.value()), offered, true, None)?);
    }
    let kind_choices = fields[0].choice_values().map(|value| kind_from_value(value)).collect::<Result<Vec<_>>>()?;
    Ok(SemanticDescription {
        kind: spec.kind(), kind_choices, fields, axis_choices: axes, x: semantic.x().cloned(), y: semantic.y().cloned(),
        group: semantic.group().cloned(), reduction: semantic.reduction(), facet: spec.facet().cloned(),
        facet_max_cells: layout.facet_max_cells(), fate_rows,
    })
}

pub(crate) fn spec_type_name(spec: &PlotSpec) -> &'static str {
    match spec {
        PlotSpec::Curve(_) => "CurvePlot", PlotSpec::Image(_) => "ImagePlot", PlotSpec::Histogram(_) => "HistogramPlot",
        PlotSpec::Rolling(_) => "RollingPlot", PlotSpec::FacetGrid(_) => "FacetGridPlot", PlotSpec::PulseTimeline(_) => "PulseTimelinePlot",
    }
}

/// `ScopeTerm`s as written in a recipe or a scope listing.
pub fn scope_listing(spec: &PlotSpec) -> &[ScopeTerm] { spec.scope() }
