//! The closed plot-kind registry (`zlc_plot._kinds`): display names, fit
//! targets, semantic field lists, label roles, admissibility, and the ONE
//! table every kind's unasked-for default is read from (`_kinds.defaults`).

use std::collections::HashSet;
use std::sync::LazyLock;

use crate::layout::LayoutConfig;
use crate::style::PlotStyle;

use super::data_contract::{classify_axes, resolve_axis, AxisEntry, AxisFamilies, IndexedHistoryLayout, PlotSchema, PRIMARY_INDEX};
use super::error::{PlotError, Result};
use super::kinds::{AxisDomain, AxisRef, PlotKind};
use super::spec::{CellPlot, CurvePlot, FacetGridPlot, HistogramPlot, ImagePlot, LabelSlot, PlotLabels, PlotSpec, Reduction, RollingPlot, ScopeTerm, GRID_CELL_KINDS};

/// The library defaults the registry reads (the immutable `DEFAULTS`).
pub(crate) static DEFAULT_STYLE: LazyLock<PlotStyle> = LazyLock::new(PlotStyle::default);
pub(crate) static DEFAULT_LAYOUT: LazyLock<LayoutConfig> = LazyLock::new(LayoutConfig::default);

/// What a label slot describes; carry-over matches roles, never slots.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum LabelRole { Title, Axis(AxisRef), Value, Count, Repeat, X, PulseTime }

/// The semantic projection a kind's fit models are authored for.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum FitTargetKind { Series, Histogram, Image }

impl FitTargetKind {
    pub fn value(self) -> &'static str {
        match self { FitTargetKind::Series => "series", FitTargetKind::Histogram => "histogram", FitTargetKind::Image => "image" }
    }
}

/// One registry record.
#[derive(Clone, Copy, Debug)]
pub struct KindHandler {
    pub kind: PlotKind,
    pub display_name: &'static str,
    pub fit_target: Option<FitTargetKind>,
    pub semantic_fields: &'static [&'static str],
    pub record_input: bool,
}

pub const HANDLERS: [KindHandler; 6] = [
    KindHandler { kind: PlotKind::Curve, display_name: "Curve", fit_target: Some(FitTargetKind::Series), semantic_fields: &["kind", "x", "group", "reduction"], record_input: false },
    KindHandler { kind: PlotKind::Image, display_name: "Image", fit_target: Some(FitTargetKind::Image), semantic_fields: &["kind", "x", "y", "reduction"], record_input: false },
    KindHandler { kind: PlotKind::Histogram, display_name: "Histogram", fit_target: Some(FitTargetKind::Histogram), semantic_fields: &["kind", "group", "reduction"], record_input: false },
    KindHandler { kind: PlotKind::Rolling, display_name: "Rolling", fit_target: Some(FitTargetKind::Series), semantic_fields: &["kind", "group", "reduction"], record_input: true },
    KindHandler { kind: PlotKind::FacetGrid, display_name: "Facet grid", fit_target: None, semantic_fields: &["kind", "facet"], record_input: false },
    KindHandler { kind: PlotKind::PulseTimeline, display_name: "Pulse timeline", fit_target: None, semantic_fields: &["kind"], record_input: false },
];

pub fn handler(kind: PlotKind) -> &'static KindHandler { HANDLERS.iter().find(|handler| handler.kind == kind).expect("registered kind") }

pub fn kind_label(kind: PlotKind) -> &'static str { handler(kind).display_name }

/// Whether this dataset can host the kind (a frontend greys the rest out).
pub fn admits(kind: PlotKind, schema: Option<&PlotSchema>) -> Result<bool> {
    Ok(match kind {
        PlotKind::FacetGrid => schema.is_some(),
        PlotKind::PulseTimeline => false,
        other => default_spec(schema, other)?.is_some(),
    })
}

/// Each label slot with the semantic role it describes.
pub fn label_roles(spec: &PlotSpec, history: Option<&IndexedHistoryLayout>) -> Result<Vec<(LabelSlot, LabelRole)>> {
    Ok(match spec {
        PlotSpec::Curve(s) => vec![(LabelSlot::Title, LabelRole::Title), (LabelSlot::X, LabelRole::Axis(s.x.clone())), (LabelSlot::Y, LabelRole::Value)],
        PlotSpec::Image(s) => vec![(LabelSlot::Title, LabelRole::Title), (LabelSlot::X, LabelRole::Axis(s.x.clone())),
                                   (LabelSlot::Y, LabelRole::Axis(s.y.clone())), (LabelSlot::Value, LabelRole::Value)],
        PlotSpec::Histogram(_) => vec![(LabelSlot::Title, LabelRole::Title), (LabelSlot::X, LabelRole::Value), (LabelSlot::Y, LabelRole::Count)],
        PlotSpec::Rolling(s) => {
            let role = match &s.x {
                None => Some(PRIMARY_INDEX),
                Some(x) => match history { Some(layout) if x.domain == AxisDomain::Point => layout.coordinate_role(&x.axis_id), _ => None },
            };
            let Some(role) = role else { return Err(PlotError::value("Explicit Rolling coordinates require their resolved history layout")) };
            vec![(LabelSlot::Title, LabelRole::Title), (LabelSlot::X, if role == PRIMARY_INDEX { LabelRole::Repeat } else { LabelRole::X }),
                 (LabelSlot::Y, LabelRole::Value)]
        }
        PlotSpec::FacetGrid(grid) => {
            let mut roles = vec![(LabelSlot::Title, LabelRole::Title)];
            roles.extend(label_roles(&grid.cell.as_spec(), history)?.into_iter().filter(|(slot, _)| *slot != LabelSlot::Title));
            roles
        }
        PlotSpec::PulseTimeline(_) => vec![(LabelSlot::Title, LabelRole::Title), (LabelSlot::X, LabelRole::PulseTime)],
    })
}

/// The kind-owned default specification, or `None` when ambiguous.
pub fn default_spec(schema: Option<&PlotSchema>, kind: PlotKind) -> Result<Option<PlotSpec>> {
    if kind == PlotKind::PulseTimeline { return Ok(None); }
    table_default(schema, kind, None)
}

/// The order an unasked-for plot is inferred in, most specific first.
const INFERENCE_ORDER: [PlotKind; 3] = [PlotKind::PulseTimeline, PlotKind::Image, PlotKind::Curve];

/// A specification this dataset admits for `kind` (or, with no kind, the
/// dataset's own structure decides), or `None`.
pub fn fitting_spec(schema: Option<&PlotSchema>, kind: Option<PlotKind>, cell: Option<PlotKind>) -> Result<Option<PlotSpec>> {
    if cell.is_some() && kind != Some(PlotKind::FacetGrid) { return Err(PlotError::value("only a facet grid has a cell kind")); }
    match kind {
        Some(PlotKind::FacetGrid) => table_default(schema, PlotKind::FacetGrid, cell),
        Some(kind) => default_spec(schema, kind),
        None => {
            for kind in INFERENCE_ORDER {
                if let Some(candidate) = default_spec(schema, kind)? { return Ok(Some(candidate)); }
            }
            Ok(None)
        }
    }
}

/// A grid for an operator who ASKED for one, from the plot they had.
pub fn chosen_spec(schema: Option<&PlotSchema>, current: &PlotSpec) -> Result<Option<PlotSpec>> {
    if schema.is_none() { return Ok(None); }
    let current_kind = current.kind();
    if GRID_CELL_KINDS.contains(&current_kind) {
        if let Some(asked) = table_default(schema, PlotKind::FacetGrid, Some(current_kind))? { return Ok(Some(asked)); }
    }
    table_default(schema, PlotKind::FacetGrid, None)
}

/// Can a picture show this many lines AS separate lines?
pub fn tellable_apart(count: usize) -> bool { count <= DEFAULT_STYLE.palette.line_cycle().len() }

#[derive(Clone, Debug, Default)]
struct Plan { x: Option<AxisRef>, y: Option<AxisRef>, group: Option<AxisRef> }

impl Plan {
    fn consumed(&self) -> Vec<AxisRef> { [&self.x, &self.y, &self.group].into_iter().flatten().cloned().collect() }
}

/// What `kind` shows for `schema` unasked, read off the one table.
pub fn table_default(schema: Option<&PlotSchema>, kind: PlotKind, cell_kind: Option<PlotKind>) -> Result<Option<PlotSpec>> {
    let Some(schema) = schema else { return Ok(None) };
    if cell_kind.is_some() && kind != PlotKind::FacetGrid { return Err(PlotError::value("only a facet grid has a cell kind")); }
    let families = classify_axes(schema);
    match kind {
        PlotKind::FacetGrid => grid(schema, &families, cell_kind),
        PlotKind::Curve => Ok(curve_spec(schema, &families, &curve_plan(&families, None), true)?.map(PlotSpec::Curve)),
        PlotKind::Image => match image_plan(&families) {
            Some(plan) => Ok(Some(PlotSpec::Image(image_spec(schema, &families, &plan, true)?))),
            None => Ok(None),
        },
        PlotKind::Histogram => Ok(Some(PlotSpec::Histogram(HistogramPlot::default()))),
        PlotKind::Rolling => Ok(Some(PlotSpec::Rolling(rolling_spec(schema, &families)?))),
        PlotKind::PulseTimeline => Err(PlotError::value("PlotKind.PULSE_TIMELINE has no table entry; its handler owns its default")),
    }
}

fn densest_cell_kind(families: &AxisFamilies) -> PlotKind {
    if families.picture.is_some() || families.live_scan().len() >= 2 { return PlotKind::Image; }
    if families.live_content().len() >= 2 { return PlotKind::Image; }
    if families.live_content().is_empty() && families.live_scan().is_empty() && families.live_events().is_empty()
        && families.history.is_none() && families.repeat_size > 1 && families.has_point_axes {
        return PlotKind::Histogram;
    }
    PlotKind::Curve
}

fn grid(schema: &PlotSchema, families: &AxisFamilies, cell_kind: Option<PlotKind>) -> Result<Option<PlotSpec>> {
    let kind = cell_kind.unwrap_or_else(|| densest_cell_kind(families));
    if !GRID_CELL_KINDS.contains(&kind) { return Err(PlotError::value(format!("a grid cell cannot be a {}", kind.value()))); }
    let with_facet = |facet: Option<AxisRef>, consumed: Vec<AxisRef>, cell: CellPlot| -> Result<Option<PlotSpec>> {
        let mut used = consumed;
        used.extend(facet.clone());
        let scope = event_scope(schema, families, &used)?;
        Ok(Some(PlotSpec::FacetGrid(FacetGridPlot::new(facet, cell, PlotLabels::default(), scope)?)))
    };
    match kind {
        PlotKind::Histogram => {
            let facet = facet_axis(families, &Plan::default());
            with_facet(facet, Vec::new(), CellPlot::Histogram(HistogramPlot::default()))
        }
        PlotKind::Image => {
            let Some(plan) = image_plan(families) else { return Ok(None) };
            let facet = facet_axis(families, &plan);
            let cell = image_spec(schema, families, &plan, false)?;
            with_facet(facet, plan.consumed(), CellPlot::Image(cell))
        }
        _ => {
            let live_scan = families.live_scan();
            let facet = if live_scan.len() >= 2 {
                Some(live_scan[0].0.clone())
            } else if !live_scan.is_empty() {
                first_any(&families.live_events()).or_else(|| facetable_history(families)).or_else(|| first_any(&families.events))
            } else {
                facet_axis(families, &Plan::default())
            };
            let plan = curve_plan(families, facet.as_ref());
            let Some(cell) = curve_spec(schema, families, &plan, false)? else { return Ok(None) };
            with_facet(facet, plan.consumed(), CellPlot::Curve(cell))
        }
    }
}

fn facet_axis(families: &AxisFamilies, plan: &Plan) -> Option<AxisRef> {
    let consumed: HashSet<AxisRef> = plan.consumed().into_iter().collect();
    for (reference, _) in families.live_scan() { if !consumed.contains(&reference) { return Some(reference); } }
    for (reference, _) in families.live_events() { if !consumed.contains(&reference) { return Some(reference); } }
    if let Some(history) = facetable_history(families) { if !consumed.contains(&history) { return Some(history); } }
    families.events.iter().find(|(reference, _)| !consumed.contains(reference)).map(|(reference, _)| reference.clone())
}

fn image_plan(families: &AxisFamilies) -> Option<Plan> {
    if let Some((x, y)) = &families.picture { return Some(Plan { x: Some(x.0.clone()), y: Some(y.0.clone()), group: None }); }
    let scan = families.live_scan();
    let content = families.live_content();
    if scan.len() >= 2 {
        return Some(Plan { x: Some(scan[scan.len() - 1].0.clone()), y: Some(scan[scan.len() - 2].0.clone()), group: None });
    }
    if content.len() >= 2 {
        return Some(Plan { x: Some(content[content.len() - 1].0.clone()), y: Some(content[content.len() - 2].0.clone()), group: None });
    }
    if content.len() == 1 {
        let y = Some(content[0].0.clone());
        if !scan.is_empty() { return Some(Plan { x: Some(scan[scan.len() - 1].0.clone()), y, group: None }); }
        let events = families.live_events();
        if !events.is_empty() { return Some(Plan { x: Some(events[0].0.clone()), y, group: None }); }
        if let Some(history) = live_history(families) { return Some(Plan { x: Some(history), y, group: None }); }
    }
    None
}

fn image_spec(schema: &PlotSchema, families: &AxisFamilies, plan: &Plan, scoped: bool) -> Result<ImagePlot> {
    let scope = if scoped { event_scope(schema, families, &plan.consumed())? } else { Vec::new() };
    ImagePlot::new(plan.x.clone().expect("planned x"), plan.y.clone().expect("planned y"), Reduction::Mean, PlotLabels::default(), scope, Vec::new())
}

fn curve_plan(families: &AxisFamilies, facet: Option<&AxisRef>) -> Plan {
    let taken: HashSet<AxisRef> = facet.into_iter().cloned().collect();
    let walk = |x: &AxisRef| Plan { x: Some(x.clone()), y: None, group: group_axis(families, &[Some(x), facet]) };
    if let Some((x, _)) = &families.sample_time { if !taken.contains(x) { return walk(x); } }
    if let Some((x, _)) = &families.history_time {
        if !taken.contains(x) && families.history.as_ref().is_none_or(|(history, _)| !taken.contains(history)) { return walk(x); }
    }
    let scan: Vec<AxisEntry> = families.live_scan().into_iter().filter(|(reference, _)| !taken.contains(reference)).collect();
    if let Some((x, _)) = scan.last() { return walk(x); }
    let events: Vec<AxisEntry> = families.live_events().into_iter().filter(|(reference, _)| !taken.contains(reference)).collect();
    if let Some((x, _)) = events.first() { return walk(x); }
    let data = families.live_data();
    if let Some((x, _)) = data.last() { return walk(x); }
    if let Some(history) = live_history(families) {
        if !taken.contains(&history) { return Plan { x: Some(history), ..Plan::default() }; }
    }
    for (reference, size) in &families.repeat {
        if *size > 1 && !taken.contains(reference) { return Plan { x: Some(reference.clone()), ..Plan::default() }; }
    }
    Plan {
        x: first_free(&families.scan, &taken).or_else(|| first_free(&families.events, &taken))
            .or_else(|| first_free(&families.data, &taken)).or_else(|| first_free(&families.repeat, &taken)),
        ..Plan::default()
    }
}

fn curve_spec(schema: &PlotSchema, families: &AxisFamilies, plan: &Plan, scoped: bool) -> Result<Option<CurvePlot>> {
    let Some(x) = plan.x.clone() else { return Ok(None) };
    let scope = if scoped { event_scope(schema, families, &plan.consumed())? } else { Vec::new() };
    Ok(Some(CurvePlot::new(x, plan.group.clone(), Reduction::Mean, PlotLabels::default(), scope, Vec::new())?))
}

fn rolling_spec(schema: &PlotSchema, families: &AxisFamilies) -> Result<RollingPlot> {
    // x is a coordinate of the fixed record carrier: the shot time only as
    // the shot index's own coordinate. A shot time without one (a fit
    // faceted over shot time) is an ordinary axis of repeat records.
    let x = match (&families.history_time, &families.history) {
        (Some((stamp, _)), Some((index, _)))
            if resolve_axis(schema, stamp)?.axis.coordinate_of() == Some(index.axis_id.as_str()) => Some(stamp.clone()),
        _ => None,
    };
    RollingPlot::new(group_axis(families, &[]), x, Reduction::Mean, event_scope(schema, families, &[])?)
}

/// The one live data axis left over, when its members can be told apart.
fn group_axis(families: &AxisFamilies, taken: &[Option<&AxisRef>]) -> Option<AxisRef> {
    let used: HashSet<&AxisRef> = taken.iter().flatten().copied().collect();
    let free: Vec<AxisEntry> = families.live_data().into_iter().filter(|(reference, _)| !used.contains(reference)).collect();
    (free.len() == 1 && tellable_apart(free[0].1)).then(|| free[0].0.clone())
}

/// Pin an unused live event axis to its last declared coordinate.
fn event_scope(schema: &PlotSchema, families: &AxisFamilies, consumed: &[AxisRef]) -> Result<Vec<ScopeTerm>> {
    let used: HashSet<&AxisRef> = consumed.iter().collect();
    let mut scope = Vec::new();
    for (reference, _) in families.live_events() {
        if used.contains(&reference) { continue; }
        let axis = resolve_axis(schema, &reference)?;
        let last = axis.coordinate_at(axis.size() - 1).expect("nonempty axis");
        scope.push((reference, last));
    }
    Ok(scope)
}

fn first_any(entries: &[AxisEntry]) -> Option<AxisRef> { entries.first().map(|(reference, _)| reference.clone()) }

fn first_free(entries: &[AxisEntry], taken: &HashSet<AxisRef>) -> Option<AxisRef> {
    entries.iter().find(|(reference, _)| !taken.contains(reference)).map(|(reference, _)| reference.clone())
}

fn live_history(families: &AxisFamilies) -> Option<AxisRef> {
    families.history.as_ref().filter(|(_, size)| *size > 1).map(|(reference, _)| reference.clone())
}

/// The shot history, when the layout can lay that many cells out.
fn facetable_history(families: &AxisFamilies) -> Option<AxisRef> {
    let history = live_history(families)?;
    let size = families.history.as_ref().map(|(_, size)| *size).unwrap_or(0);
    (size as u64 <= u64::from(DEFAULT_LAYOUT.facet_max_cells())).then_some(history)
}
