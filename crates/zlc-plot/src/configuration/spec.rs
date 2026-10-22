//! Typed semantic plot specifications (`zlc_plot.specs`): what data
//! coordinates a plot uses.  Construction validates exactly what the Python
//! `__post_init__` validated, in the same order, so an edit that Python
//! refused is refused here with the same exception class.

use std::collections::HashSet;
use std::sync::Arc;

use super::error::{PlotError, Result};
use super::kinds::{axis_repr, canonical_coordinate, AxisRef, Coordinate, PlotKind};
use super::value::py_strip;

pub use super::kinds::{Reduction, RelimMode};

/// Authored labels; each present label is non-blank text.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PlotLabels {
    pub title: Option<Arc<str>>,
    pub x: Option<Arc<str>>,
    pub y: Option<Arc<str>>,
    pub value: Option<Arc<str>>,
}

/// A label slot of [`PlotLabels`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum LabelSlot { Title, X, Y, Value }

impl LabelSlot {
    pub fn name(self) -> &'static str {
        match self { LabelSlot::Title => "title", LabelSlot::X => "x", LabelSlot::Y => "y", LabelSlot::Value => "value" }
    }
}

impl PlotLabels {
    pub fn new(title: Option<Arc<str>>, x: Option<Arc<str>>, y: Option<Arc<str>>, value: Option<Arc<str>>) -> Result<Self> {
        let labels = Self { title, x, y, value };
        for slot in [LabelSlot::Title, LabelSlot::X, LabelSlot::Y, LabelSlot::Value] {
            if labels.get(slot).is_some_and(|text| py_strip(text).is_empty()) {
                return Err(PlotError::value(format!("label '{}' must be non-empty text or None", slot.name())));
            }
        }
        Ok(labels)
    }
    pub fn get(&self, slot: LabelSlot) -> Option<&Arc<str>> {
        match slot { LabelSlot::Title => self.title.as_ref(), LabelSlot::X => self.x.as_ref(),
                     LabelSlot::Y => self.y.as_ref(), LabelSlot::Value => self.value.as_ref() }
    }
    pub fn set(&mut self, slot: LabelSlot, value: Option<Arc<str>>) {
        match slot { LabelSlot::Title => self.title = value, LabelSlot::X => self.x = value,
                     LabelSlot::Y => self.y = value, LabelSlot::Value => self.value = value }
    }
    pub fn title_only(&self) -> Self { Self { title: self.title.clone(), ..Self::default() } }
}

/// One axis pinned to one of its coordinates: `(AxisRef, CoordinateScalar)`.
pub type ScopeTerm = (AxisRef, Coordinate);

#[derive(Clone, Debug, PartialEq)]
pub struct CurvePlot {
    pub x: AxisRef,
    pub group: Option<AxisRef>,
    pub reduction: Reduction,
    pub labels: PlotLabels,
    pub scope: Vec<ScopeTerm>,
    pub coordinates: Vec<AxisRef>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImagePlot {
    pub x: AxisRef,
    pub y: AxisRef,
    pub reduction: Reduction,
    pub labels: PlotLabels,
    pub scope: Vec<ScopeTerm>,
    pub coordinates: Vec<AxisRef>,
}

/// Distribution of the acquired values; `reduced` axes collapse under the
/// reduction before binning, every other axis pools.
#[derive(Clone, Debug, PartialEq)]
pub struct HistogramPlot {
    pub reduction: Reduction,
    pub reduced: Vec<AxisRef>,
    pub labels: PlotLabels,
    pub scope: Vec<ScopeTerm>,
    pub coordinates: Vec<AxisRef>,
    pub group: Option<AxisRef>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RollingPlot {
    pub group: Option<AxisRef>,
    /// The record carrier's coordinate; `None` is the relative source index.
    pub x: Option<AxisRef>,
    pub reduction: Reduction,
    pub labels: PlotLabels,
    pub scope: Vec<ScopeTerm>,
    pub coordinates: Vec<AxisRef>,
}

/// The kinds one facet-grid cell can be.
#[derive(Clone, Debug, PartialEq)]
pub enum CellPlot { Curve(CurvePlot), Image(ImagePlot), Histogram(HistogramPlot) }

/// One cell per coordinate of an optional facet axis (`None`: one cell).
#[derive(Clone, Debug, PartialEq)]
pub struct FacetGridPlot {
    pub facet: Option<AxisRef>,
    pub cell: CellPlot,
    pub labels: PlotLabels,
    pub scope: Vec<ScopeTerm>,
    pub coordinates: Vec<AxisRef>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PulseTimelinePlot { pub labels: PlotLabels }

/// Every plot specification.
#[derive(Clone, Debug, PartialEq)]
pub enum PlotSpec {
    Curve(CurvePlot),
    Image(ImagePlot),
    Histogram(HistogramPlot),
    Rolling(RollingPlot),
    FacetGrid(FacetGridPlot),
    PulseTimeline(PulseTimelinePlot),
}

/// The kinds a grid cell can be (`GRID_CELL_KINDS`).
pub const GRID_CELL_KINDS: [PlotKind; 3] = [PlotKind::Curve, PlotKind::Image, PlotKind::Histogram];

/// The roles a spec can hold an axis in.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Role { X, Y, Group, Facet }

impl Role {
    pub const ALL: [Role; 4] = [Role::X, Role::Y, Role::Group, Role::Facet];
    pub fn name(self) -> &'static str {
        match self { Role::X => "x", Role::Y => "y", Role::Group => "group", Role::Facet => "facet" }
    }
    pub fn parse(name: &str) -> Option<Role> { Self::ALL.into_iter().find(|role| role.name() == name) }
}

fn validated_coordinates(values: &[AxisRef]) -> Result<()> {
    let distinct: HashSet<&AxisRef> = values.iter().collect();
    if distinct.len() != values.len() {
        return Err(PlotError::value("coordinate choices must be distinct AxisRef values"));
    }
    Ok(())
}

/// `_validated_scope`: canonical coordinates, each axis once.
fn validated_scope(scope: &mut [ScopeTerm]) -> Result<()> {
    let mut seen = HashSet::new();
    for (axis, coordinate) in scope.iter_mut() {
        if let Coordinate::Real(value) = coordinate {
            if !value.is_finite() { return Err(PlotError::value("scope coordinate must be finite")); }
        }
        *coordinate = canonical_coordinate(coordinate);
        if !seen.insert(axis.clone()) {
            return Err(PlotError::value(format!("axis {} is scoped twice", axis_repr(axis))));
        }
    }
    Ok(())
}

fn require_distinct_axes(roles: &[(&str, Option<&AxisRef>)], scope: &[ScopeTerm]) -> Result<()> {
    let mut used: Vec<(&AxisRef, &str)> = Vec::new();
    for (name, axis) in roles {
        let Some(axis) = axis else { continue };
        if let Some((_, previous)) = used.iter().find(|(held, _)| held == axis) {
            return Err(PlotError::value(format!("one physical axis cannot be both {previous} and {name}")));
        }
        used.push((axis, name));
    }
    for (axis, _) in scope {
        if let Some((_, previous)) = used.iter().find(|(held, _)| *held == axis) {
            return Err(PlotError::value(format!("one physical axis cannot be both {previous} and scope")));
        }
        used.push((axis, "scope"));
    }
    Ok(())
}

impl CurvePlot {
    pub fn new(x: AxisRef, group: Option<AxisRef>, reduction: Reduction, labels: PlotLabels,
               scope: Vec<ScopeTerm>, coordinates: Vec<AxisRef>) -> Result<Self> {
        Self { x, group, reduction, labels, scope, coordinates }.validated()
    }
    pub fn validated(mut self) -> Result<Self> {
        validated_coordinates(&self.coordinates)?;
        validated_scope(&mut self.scope)?;
        require_distinct_axes(&[("x", Some(&self.x)), ("group", self.group.as_ref())], &self.scope)?;
        Ok(self)
    }
}

impl ImagePlot {
    pub fn new(x: AxisRef, y: AxisRef, reduction: Reduction, labels: PlotLabels,
               scope: Vec<ScopeTerm>, coordinates: Vec<AxisRef>) -> Result<Self> {
        Self { x, y, reduction, labels, scope, coordinates }.validated()
    }
    pub fn validated(mut self) -> Result<Self> {
        validated_coordinates(&self.coordinates)?;
        validated_scope(&mut self.scope)?;
        require_distinct_axes(&[("x", Some(&self.x)), ("y", Some(&self.y))], &self.scope)?;
        Ok(self)
    }
}

impl Default for HistogramPlot {
    fn default() -> Self {
        Self { reduction: Reduction::Mean, reduced: Vec::new(), labels: PlotLabels::default(),
               scope: Vec::new(), coordinates: Vec::new(), group: None }
    }
}

impl HistogramPlot {
    pub fn validated(mut self) -> Result<Self> {
        validated_coordinates(&self.coordinates)?;
        let reduced: HashSet<&AxisRef> = self.reduced.iter().collect();
        if reduced.len() != self.reduced.len() {
            return Err(PlotError::value("HistogramPlot.reduced must name distinct axes"));
        }
        validated_scope(&mut self.scope)?;
        let pinned: HashSet<&AxisRef> = self.scope.iter().map(|(axis, _)| axis).collect();
        if self.reduced.iter().any(|axis| pinned.contains(axis)) {
            return Err(PlotError::value("an axis cannot be both reduced and pinned"));
        }
        if let Some(group) = &self.group {
            if pinned.contains(group) || self.reduced.contains(group) {
                return Err(PlotError::value("a group axis cannot also be reduced or pinned"));
            }
        }
        Ok(self)
    }
}

impl RollingPlot {
    pub fn new(group: Option<AxisRef>, x: Option<AxisRef>, reduction: Reduction, scope: Vec<ScopeTerm>) -> Result<Self> {
        Self { group, x, reduction, labels: PlotLabels::default(), scope, coordinates: Vec::new() }.validated()
    }
    pub fn validated(mut self) -> Result<Self> {
        validated_coordinates(&self.coordinates)?;
        validated_scope(&mut self.scope)?;
        require_distinct_axes(&[("x", self.x.as_ref()), ("group", self.group.as_ref())], &self.scope)?;
        Ok(self)
    }
}

impl CellPlot {
    pub fn kind(&self) -> PlotKind {
        match self { CellPlot::Curve(_) => PlotKind::Curve, CellPlot::Image(_) => PlotKind::Image, CellPlot::Histogram(_) => PlotKind::Histogram }
    }
    /// The cell as the standalone specification it draws.
    pub fn as_spec(&self) -> PlotSpec {
        match self {
            CellPlot::Curve(cell) => PlotSpec::Curve(cell.clone()),
            CellPlot::Image(cell) => PlotSpec::Image(cell.clone()),
            CellPlot::Histogram(cell) => PlotSpec::Histogram(cell.clone()),
        }
    }
    /// A grid cell from a standalone specification of a cell kind.
    pub fn from_spec(spec: PlotSpec) -> Result<Self> {
        match spec {
            PlotSpec::Curve(cell) => Ok(CellPlot::Curve(cell)),
            PlotSpec::Image(cell) => Ok(CellPlot::Image(cell)),
            PlotSpec::Histogram(cell) => Ok(CellPlot::Histogram(cell)),
            _ => Err(PlotError::type_error("FacetGrid cells must be one of CurvePlot, ImagePlot, HistogramPlot")),
        }
    }
    fn view(&self) -> Surface<'_> {
        match self { CellPlot::Curve(c) => Surface::Curve(c), CellPlot::Image(c) => Surface::Image(c), CellPlot::Histogram(c) => Surface::Histogram(c) }
    }
    fn scope(&self) -> &[ScopeTerm] { self.view().scope() }
    fn coordinates(&self) -> &[AxisRef] { self.view().coordinates() }
    pub fn labels_mut(&mut self) -> &mut PlotLabels {
        match self { CellPlot::Curve(c) => &mut c.labels, CellPlot::Image(c) => &mut c.labels, CellPlot::Histogram(c) => &mut c.labels }
    }
    fn validated(self) -> Result<Self> {
        Ok(match self {
            CellPlot::Curve(cell) => CellPlot::Curve(cell.validated()?),
            CellPlot::Image(cell) => CellPlot::Image(cell.validated()?),
            CellPlot::Histogram(cell) => CellPlot::Histogram(cell.validated()?),
        })
    }
}

impl FacetGridPlot {
    pub fn new(facet: Option<AxisRef>, cell: CellPlot, labels: PlotLabels, scope: Vec<ScopeTerm>) -> Result<Self> {
        Self { facet, cell, labels, scope, coordinates: Vec::new() }.validated()
    }
    /// The grid owns its title; explicit outer axis labels move to the cell.
    pub fn validated(mut self) -> Result<Self> {
        validated_coordinates(&self.coordinates)?;
        let axis_labels = [super::spec::LabelSlot::X, LabelSlot::Y, LabelSlot::Value];
        if axis_labels.iter().any(|slot| self.labels.get(*slot).is_some()) {
            let labels = self.cell.labels_mut();
            for slot in axis_labels {
                if let Some(label) = self.labels.get(slot) { labels.set(slot, Some(label.clone())); }
            }
            self.labels = self.labels.title_only();
        }
        self.cell = self.cell.validated()?;
        if !self.cell.scope().is_empty() {
            return Err(PlotError::value("FacetGrid cell.scope is invalid; scope belongs to FacetGridPlot"));
        }
        if !self.cell.coordinates().is_empty() {
            return Err(PlotError::value("FacetGrid cell.coordinates is invalid; coordinates belong to FacetGridPlot"));
        }
        validated_scope(&mut self.scope)?;
        let view = self.cell.view();
        require_distinct_axes(&[("facet", self.facet.as_ref()), ("x", view.x()), ("y", view.y()), ("group", view.group())], &self.scope)?;
        Ok(self)
    }
}

/// A borrowed view of the specification that decides what a drawn surface IS
/// (`semantic_spec`): a grid's cell, or the specification itself.
#[derive(Clone, Copy, Debug)]
pub enum Surface<'a> {
    Curve(&'a CurvePlot),
    Image(&'a ImagePlot),
    Histogram(&'a HistogramPlot),
    Rolling(&'a RollingPlot),
    PulseTimeline(&'a PulseTimelinePlot),
}

impl<'a> Surface<'a> {
    pub fn kind(self) -> PlotKind {
        match self {
            Surface::Curve(_) => PlotKind::Curve, Surface::Image(_) => PlotKind::Image, Surface::Histogram(_) => PlotKind::Histogram,
            Surface::Rolling(_) => PlotKind::Rolling, Surface::PulseTimeline(_) => PlotKind::PulseTimeline,
        }
    }
    /// `getattr(semantic, "x", None)`.
    pub fn x(self) -> Option<&'a AxisRef> {
        match self { Surface::Curve(s) => Some(&s.x), Surface::Image(s) => Some(&s.x), Surface::Rolling(s) => s.x.as_ref(), _ => None }
    }
    pub fn y(self) -> Option<&'a AxisRef> { if let Surface::Image(s) = self { Some(&s.y) } else { None } }
    pub fn group(self) -> Option<&'a AxisRef> {
        match self { Surface::Curve(s) => s.group.as_ref(), Surface::Histogram(s) => s.group.as_ref(), Surface::Rolling(s) => s.group.as_ref(), _ => None }
    }
    pub fn role(self, role: Role) -> Option<&'a AxisRef> {
        match role { Role::X => self.x(), Role::Y => self.y(), Role::Group => self.group(), Role::Facet => None }
    }
    /// Whether this kind declares the field at all (`hasattr`).
    pub fn has_role(self, role: Role) -> bool {
        match role {
            Role::X => matches!(self, Surface::Curve(_) | Surface::Image(_) | Surface::Rolling(_)),
            Role::Y => matches!(self, Surface::Image(_)),
            Role::Group => matches!(self, Surface::Curve(_) | Surface::Histogram(_) | Surface::Rolling(_)),
            Role::Facet => false,
        }
    }
    /// Roles whose declaration defaults to `None` (optional roles).
    pub fn optional_role(self, role: Role) -> bool {
        matches!((self, role), (Surface::Curve(_) | Surface::Histogram(_) | Surface::Rolling(_), Role::Group) | (Surface::Rolling(_), Role::X))
    }
    pub fn reduction(self) -> Option<Reduction> {
        match self {
            Surface::Curve(s) => Some(s.reduction), Surface::Image(s) => Some(s.reduction), Surface::Histogram(s) => Some(s.reduction),
            Surface::Rolling(s) => Some(s.reduction), Surface::PulseTimeline(_) => None,
        }
    }
    /// `getattr(semantic, "reduced", ())`; `None` when the kind has no list.
    pub fn reduced(self) -> Option<&'a [AxisRef]> { if let Surface::Histogram(s) = self { Some(&s.reduced) } else { None } }
    pub fn labels(self) -> &'a PlotLabels {
        match self {
            Surface::Curve(s) => &s.labels, Surface::Image(s) => &s.labels, Surface::Histogram(s) => &s.labels,
            Surface::Rolling(s) => &s.labels, Surface::PulseTimeline(s) => &s.labels,
        }
    }
    pub fn scope(self) -> &'a [ScopeTerm] {
        match self {
            Surface::Curve(s) => &s.scope, Surface::Image(s) => &s.scope, Surface::Histogram(s) => &s.scope,
            Surface::Rolling(s) => &s.scope, Surface::PulseTimeline(_) => &[],
        }
    }
    pub fn coordinates(self) -> &'a [AxisRef] {
        match self {
            Surface::Curve(s) => &s.coordinates, Surface::Image(s) => &s.coordinates, Surface::Histogram(s) => &s.coordinates,
            Surface::Rolling(s) => &s.coordinates, Surface::PulseTimeline(_) => &[],
        }
    }
}

impl PlotSpec {
    pub fn kind(&self) -> PlotKind {
        match self {
            PlotSpec::Curve(_) => PlotKind::Curve, PlotSpec::Image(_) => PlotKind::Image, PlotSpec::Histogram(_) => PlotKind::Histogram,
            PlotSpec::Rolling(_) => PlotKind::Rolling, PlotSpec::FacetGrid(_) => PlotKind::FacetGrid,
            PlotSpec::PulseTimeline(_) => PlotKind::PulseTimeline,
        }
    }
    /// `semantic_spec(spec)`: the grid's cell, or the spec itself.
    pub fn semantic(&self) -> Surface<'_> {
        match self {
            PlotSpec::Curve(s) => Surface::Curve(s), PlotSpec::Image(s) => Surface::Image(s), PlotSpec::Histogram(s) => Surface::Histogram(s),
            PlotSpec::Rolling(s) => Surface::Rolling(s), PlotSpec::PulseTimeline(s) => Surface::PulseTimeline(s),
            PlotSpec::FacetGrid(grid) => grid.cell.view(),
        }
    }
    /// `semantic_spec(spec)` as an owned standalone specification.
    pub fn semantic_spec(&self) -> PlotSpec {
        match self { PlotSpec::FacetGrid(grid) => grid.cell.as_spec(), other => other.clone() }
    }
    pub fn labels(&self) -> &PlotLabels {
        match self {
            PlotSpec::Curve(s) => &s.labels, PlotSpec::Image(s) => &s.labels, PlotSpec::Histogram(s) => &s.labels,
            PlotSpec::Rolling(s) => &s.labels, PlotSpec::FacetGrid(s) => &s.labels, PlotSpec::PulseTimeline(s) => &s.labels,
        }
    }
    pub fn labels_mut(&mut self) -> &mut PlotLabels {
        match self {
            PlotSpec::Curve(s) => &mut s.labels, PlotSpec::Image(s) => &mut s.labels, PlotSpec::Histogram(s) => &mut s.labels,
            PlotSpec::Rolling(s) => &mut s.labels, PlotSpec::FacetGrid(s) => &mut s.labels, PlotSpec::PulseTimeline(s) => &mut s.labels,
        }
    }
    /// `getattr(spec, "scope", ())`: the outer scope (a grid's own).
    pub fn scope(&self) -> &[ScopeTerm] {
        match self {
            PlotSpec::Curve(s) => &s.scope, PlotSpec::Image(s) => &s.scope, PlotSpec::Histogram(s) => &s.scope,
            PlotSpec::Rolling(s) => &s.scope, PlotSpec::FacetGrid(s) => &s.scope, PlotSpec::PulseTimeline(_) => &[],
        }
    }
    /// `getattr(spec, "coordinates", ())`.
    pub fn coordinates(&self) -> &[AxisRef] {
        match self {
            PlotSpec::Curve(s) => &s.coordinates, PlotSpec::Image(s) => &s.coordinates, PlotSpec::Histogram(s) => &s.coordinates,
            PlotSpec::Rolling(s) => &s.coordinates, PlotSpec::FacetGrid(s) => &s.coordinates, PlotSpec::PulseTimeline(_) => &[],
        }
    }
    pub fn facet(&self) -> Option<&AxisRef> { if let PlotSpec::FacetGrid(grid) = self { grid.facet.as_ref() } else { None } }
    pub fn is_facet_grid(&self) -> bool { matches!(self, PlotSpec::FacetGrid(_)) }

    /// `replace(spec, scope=...)`, validated.
    pub fn with_scope(&self, scope: Vec<ScopeTerm>) -> Result<PlotSpec> {
        let mut spec = self.clone();
        match &mut spec {
            PlotSpec::Curve(s) => s.scope = scope, PlotSpec::Image(s) => s.scope = scope, PlotSpec::Histogram(s) => s.scope = scope,
            PlotSpec::Rolling(s) => s.scope = scope, PlotSpec::FacetGrid(s) => s.scope = scope,
            PlotSpec::PulseTimeline(_) => return Err(PlotError::type_error("PulseTimelinePlot has no scope")),
        }
        spec.validated()
    }

    /// Re-run construction validation after a field edit (Python `replace`).
    pub fn validated(self) -> Result<PlotSpec> {
        Ok(match self {
            PlotSpec::Curve(s) => PlotSpec::Curve(s.validated()?),
            PlotSpec::Image(s) => PlotSpec::Image(s.validated()?),
            PlotSpec::Histogram(s) => PlotSpec::Histogram(s.validated()?),
            PlotSpec::Rolling(s) => PlotSpec::Rolling(s.validated()?),
            PlotSpec::FacetGrid(s) => PlotSpec::FacetGrid(s.validated()?),
            PlotSpec::PulseTimeline(s) => PlotSpec::PulseTimeline(s),
        })
    }

    /// Edit the semantic (cell-or-self) specification, then validate the whole.
    pub fn map_semantic(&self, edit: impl FnOnce(&mut PlotSpec) -> Result<()>) -> Result<PlotSpec> {
        match self {
            PlotSpec::FacetGrid(grid) => {
                let mut cell = grid.cell.as_spec();
                edit(&mut cell)?;
                let cell = CellPlot::from_spec(cell.validated()?)?;
                PlotSpec::FacetGrid(FacetGridPlot { cell, ..grid.clone() }).validated()
            }
            other => {
                let mut spec = other.clone();
                edit(&mut spec)?;
                spec.validated()
            }
        }
    }
}

/// Mutable access to a non-grid specification's role fields.
pub(crate) fn set_role(spec: &mut PlotSpec, role: Role, value: Option<AxisRef>) -> Result<()> {
    let missing = |kind: &str, role: Role| PlotError::type_error(format!("{kind}.{} must be AxisRef", role.name()));
    match (spec, role) {
        (PlotSpec::Curve(s), Role::X) => s.x = value.ok_or_else(|| missing("CurvePlot", Role::X))?,
        (PlotSpec::Curve(s), Role::Group) => s.group = value,
        (PlotSpec::Image(s), Role::X) => s.x = value.ok_or_else(|| PlotError::type_error("ImagePlot x and y must be AxisRef"))?,
        (PlotSpec::Image(s), Role::Y) => s.y = value.ok_or_else(|| PlotError::type_error("ImagePlot x and y must be AxisRef"))?,
        (PlotSpec::Histogram(s), Role::Group) => s.group = value,
        (PlotSpec::Rolling(s), Role::X) => s.x = value,
        (PlotSpec::Rolling(s), Role::Group) => s.group = value,
        (PlotSpec::FacetGrid(s), Role::Facet) => s.facet = value,
        (spec, role) => return Err(PlotError::type_error(format!(
            "{} has no field {:?}", spec.kind().value(), role.name()))),
    }
    Ok(())
}

pub(crate) fn set_reduction(spec: &mut PlotSpec, reduction: Reduction) -> Result<()> {
    match spec {
        PlotSpec::Curve(s) => s.reduction = reduction, PlotSpec::Image(s) => s.reduction = reduction,
        PlotSpec::Histogram(s) => s.reduction = reduction, PlotSpec::Rolling(s) => s.reduction = reduction,
        other => return Err(PlotError::type_error(format!("{} has no reduction", other.kind().value()))),
    }
    Ok(())
}

pub(crate) fn set_reduced(spec: &mut PlotSpec, reduced: Vec<AxisRef>) -> Result<()> {
    match spec {
        PlotSpec::Histogram(s) => { s.reduced = reduced; Ok(()) }
        other => Err(PlotError::type_error(format!("{} has no reduced axes", other.kind().value()))),
    }
}

pub(crate) fn set_coordinates(spec: &mut PlotSpec, coordinates: Vec<AxisRef>) {
    match spec {
        PlotSpec::Curve(s) => s.coordinates = coordinates, PlotSpec::Image(s) => s.coordinates = coordinates,
        PlotSpec::Histogram(s) => s.coordinates = coordinates, PlotSpec::Rolling(s) => s.coordinates = coordinates,
        PlotSpec::FacetGrid(s) => s.coordinates = coordinates, PlotSpec::PulseTimeline(_) => {}
    }
}

pub(crate) fn set_scope_raw(spec: &mut PlotSpec, scope: Vec<ScopeTerm>) {
    match spec {
        PlotSpec::Curve(s) => s.scope = scope, PlotSpec::Image(s) => s.scope = scope, PlotSpec::Histogram(s) => s.scope = scope,
        PlotSpec::Rolling(s) => s.scope = scope, PlotSpec::FacetGrid(s) => s.scope = scope, PlotSpec::PulseTimeline(_) => {}
    }
}

/// Whether the resolved plot (including a FacetGrid cell) paints images.
pub fn paints_image_surface(spec: &PlotSpec) -> bool { spec.semantic().kind() == PlotKind::Image }
