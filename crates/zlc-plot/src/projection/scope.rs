//! Narrowing a panel's source: `FitProjection._scoped_data`, `zlc_data`'s
//! `value_selection` / `selection_indices` / `restrict_snapshot` (the single
//! cutter) and the selection subject of an accepted projection.

use std::sync::Arc;

use super::history;
use super::model::{AxisSpec, Base, Block, Coords, DomainSpec, Mapping, Placed, Schema, Segment, Snapshot, ValueInfo};
use super::nd::{Array, Mask, Scalar};
use super::output::{Payload, SelectionSubject};
use super::plan::{DisplayFacts, ProjectionPlan, ScopeValue, SpecFacts};
use super::view::{DataView, acquired_repeat_count};
use super::{ErrorKind, ProjectionError, Result};
use crate::kinds::{AxisDomain, AxisRef, PlotKind};

/// Contiguous or explicit logical/physical indices (`range | tuple`).
#[derive(Clone, Debug, PartialEq)]
pub enum Indices { Range(usize, usize), List(Vec<usize>) }

impl Indices {
    pub fn len(&self) -> usize { match self { Indices::Range(a, b) => b - a, Indices::List(v) => v.len() } }
    pub fn is_empty(&self) -> bool { self.len() == 0 }
    pub fn at(&self, i: usize) -> usize { match self { Indices::Range(a, _) => a + i, Indices::List(v) => v[i] } }
    pub fn to_vec(&self) -> Vec<usize> { (0..self.len()).map(|i| self.at(i)).collect() }
    fn keeps_everything(&self, size: usize) -> bool {
        match self { Indices::Range(a, b) => *a == 0 && *b == size, Indices::List(v) => v.len() == size && v.iter().enumerate().all(|(i, x)| i == *x) }
    }
    fn from_sorted(rows: Vec<usize>) -> Indices {
        if rows.is_empty() { return Indices::List(rows); }
        if rows[rows.len() - 1] - rows[0] + 1 == rows.len() { Indices::Range(rows[0], rows[rows.len() - 1] + 1) } else { Indices::List(rows) }
    }
}

/// One selection term (`IndexSelection`, `IndexRangeSelection`, one exact coordinate).
#[derive(Clone, Debug)]
pub enum Term { Index(usize), IndexRange(usize, usize), Coordinate(Scalar, Option<Arc<str>>) }

fn empty(message: &str) -> ProjectionError { ProjectionError::new(ErrorKind::EmptySelection, message) }

/// `resolve_selection_indices(axis, term)` for the terms a scope produces.
fn resolve_term(axis: &AxisSpec, term: &Term) -> Result<Indices> {
    match term {
        Term::Index(index) => {
            if *index >= axis.size { return Err(ProjectionError::new(ErrorKind::Index, format!("selection index {index} is outside axis {}", axis.id))); }
            Ok(Indices::Range(*index, index + 1))
        }
        Term::IndexRange(start, stop) => {
            if *stop > axis.size { return Err(ProjectionError::new(ErrorKind::Index, format!("selection range stop {stop} is outside axis {}", axis.id))); }
            Ok(Indices::Range(*start, *stop))
        }
        Term::Coordinate(value, frame) => {
            if axis.frame != *frame { return Err(ProjectionError::value(format!("coordinate frame mismatch for axis {}", axis.id))); }
            if let Coords::Implicit = axis.coords {
                let Some(v) = value.as_f64().filter(|_| value.is_numeric()) else {
                    return Err(ProjectionError::type_error(format!("implicit axis {} accepts only numeric coordinates", axis.id)));
                };
                let lower = ((v - axis.index_origin as f64).ceil() as i128).max(0);
                let upper = ((v - axis.index_origin as f64).floor() as i128).min(axis.size as i128 - 1);
                if upper < lower { return Err(empty(&format!("coordinate selection is empty on axis {}", axis.id))); }
                return Ok(Indices::Range(lower as usize, upper as usize + 1));
            }
            match axis.coordinate_position(value) {
                Some(position) => Ok(Indices::Range(position, position + 1)),
                None => Err(empty(&format!("coordinate selection is empty on axis {}", axis.id))),
            }
        }
    }
}

/// `value_selection(schema, terms)`: one named coordinate on each named axis.
fn value_selection(schema: &Schema, terms: &[(Arc<str>, ScopeValue)]) -> Result<Vec<(Arc<str>, Term)>> {
    let mut out = Vec::new();
    for (axis_id, value) in terms {
        let axis = [&schema.repeat, &schema.point, &schema.cell].iter().find_map(|d| d.axis(axis_id))
            .ok_or_else(|| ProjectionError::value(format!("selection axis AxisId(value='{axis_id}') is absent from source schema")))?;
        let coordinate = match value { ScopeValue::Latest => axis.coordinate_at(axis.size - 1), ScopeValue::Coordinate(c) => super::model::canonical_scalar(c.clone())? };
        if let Coords::Implicit = axis.coords {
            let Scalar::Int(v) = coordinate else { return Err(ProjectionError::type_error(format!("implicit axis {axis_id} requires an integer coordinate"))) };
            let index = v - axis.index_origin;
            if !(0..axis.size as i128).contains(&index) {
                return Err(ProjectionError::value(format!("axis {axis_id} has no index {v}: it runs {}..{}", axis.index_origin, axis.index_origin + axis.size as i128 - 1)));
            }
            out.push((axis_id.clone(), Term::Index(index as usize)));
        } else {
            out.push((axis_id.clone(), Term::Coordinate(coordinate, axis.frame.clone())));
        }
    }
    Ok(out)
}

/// `selection_indices(schema, selection)`.
fn selection_indices(schema: &Schema, terms: &[(Arc<str>, Term)]) -> Result<(Indices, Indices, Vec<Indices>)> {
    let logical = |domain: &DomainSpec, index: usize| -> Result<Indices> {
        let axis = &domain.axes[index];
        match terms.iter().find(|(id, _)| *id == axis.id) { None => Ok(Indices::Range(0, axis.size)), Some((_, term)) => resolve_term(axis, term) }
    };
    let physical = |domain: &DomainSpec| -> Result<Indices> {
        let selected: Vec<usize> = (0..domain.axes.len()).filter(|i| terms.iter().any(|(id, _)| *id == domain.axes[*i].id)).collect();
        if selected.is_empty() { return Ok(Indices::Range(0, domain.size())); }
        if selected.len() == 1 && domain.mapping.is_some() {
            let chosen = logical(domain, selected[0])?;
            let mapping = domain.code_mapping(domain.coordinate_axis(selected[0]));
            if let (Indices::Range(start, stop), Base::Range { start: base_start, step: 1, len }) = (&chosen, &mapping.base) {
                if mapping.outer == 1 {
                    let first = (*start as i64 - base_start).max(0) as usize;
                    let last = ((*stop as i64 - base_start).min(*len as i64)).max(0) as usize;
                    if last <= first { return Err(empty("selection names coordinates no source domain row holds together")); }
                    return Ok(Indices::Range(first * mapping.inner, last * mapping.inner));
                }
            }
        }
        let mut keep = vec![true; domain.size()];
        for index in selected {
            let chosen = logical(domain, index)?.to_vec();
            let codes = domain.codes(index);
            for (row, code) in codes.iter().enumerate() { keep[row] &= chosen.contains(&(*code as usize)); }
        }
        let rows: Vec<usize> = (0..keep.len()).filter(|r| keep[*r]).collect();
        if rows.is_empty() { return Err(empty("selection names coordinates no source domain row holds together")); }
        Ok(Indices::from_sorted(rows))
    };
    let repeat = physical(&schema.repeat)?;
    let point = physical(&schema.point)?;
    let data = (0..schema.cell.axes.len()).map(|i| logical(&schema.cell, i)).collect::<Result<Vec<_>>>()?;
    Ok((repeat, point, data))
}

/// `_subset_axis(axis, indices)`.
fn subset_axis(axis: &Arc<AxisSpec>, indices: &Indices) -> Arc<AxisSpec> {
    if indices.keeps_everything(axis.size) { return axis.clone(); }
    let picks = indices.to_vec();
    let labels = axis.labels.as_ref().map(|labels| picks.iter().map(|i| labels[*i].clone()).collect::<Arc<[Arc<str>]>>());
    let make = |coords: Coords, origin: i128| Arc::new(AxisSpec::new(axis.id.clone(), axis.name.clone(), axis.role.clone(), picks.len(), coords,
        axis.unit.clone(), axis.frame.clone(), origin, labels.clone(), axis.coordinate_of.clone()));
    match (&axis.coords, indices) {
        (Coords::Implicit, Indices::Range(start, _)) => return make(Coords::Implicit, axis.index_origin + *start as i128),
        (Coords::Factored { origins, offsets }, Indices::Range(start, stop)) if start % offsets.len() == 0 && stop % offsets.len() == 0 => {
            let width = offsets.len();
            return make(Coords::Factored { origins: origins[start / width..stop / width].into(), offsets: offsets.clone() }, 0);
        }
        _ => {}
    }
    let values = axis.values();
    let coords = match &axis.coords {
        Coords::Values(v) => Coords::Values(picks.iter().map(|i| v[*i].clone()).collect()),
        _ => match values.dtype() {
            super::nd::DType::I64 => Coords::I64(picks.iter().map(|i| values.slice::<i64>()[*i]).collect()),
            super::nd::DType::U64 => Coords::U64(picks.iter().map(|i| values.slice::<u64>()[*i]).collect()),
            super::nd::DType::F64 => {
                let floats: Vec<f64> = picks.iter().map(|i| values.slice::<f64>()[*i]).collect();
                if floats.iter().all(|v| v.fract() == 0.0 && v.abs() < 9.2e18) { Coords::I64(floats.iter().map(|v| *v as i64).collect()) } else { Coords::F64(floats.into()) }
            }
            _ => Coords::Values(picks.iter().map(|i| values.scalar(*i)).collect()),
        },
    };
    make(coords, 0)
}

/// `_subset_mapped_domain(domain, indices)`.
fn subset_mapped_domain(domain: &DomainSpec, indices: &Indices) -> DomainSpec {
    let rows: Vec<i64> = indices.to_vec().iter().map(|r| *r as i64).collect();
    let mut axes = Vec::new();
    let mut mappings = Vec::new();
    for index in 0..domain.axes.len() {
        let axis = &domain.axes[index];
        let Mapping { base, inner, outer } = domain.code_mapping(index);
        let period = base.len() * inner;
        let (kept, inner, tile): (Base, usize, usize) = match indices {
            Indices::Range(start, stop) if outer == 1 && start % inner == 0 && stop % inner == 0 => {
                let (a, b) = (start / inner, stop / inner);
                (match &base { Base::Range { start: s, step, .. } => Base::Range { start: s + a as i64 * step, step: *step, len: b - a },
                               Base::Explicit(v) => Base::Explicit(v[a..b].into()) }, inner, 1)
            }
            Indices::Range(start, stop) if start % period == 0 && stop % period == 0 => (base.clone(), inner, (stop - start) / period),
            _ => (Base::Explicit(domain.codes_at(index, &rows).into()), 1, 1),
        };
        let (used, inverse): (Vec<i64>, Base) = match &kept {
            Base::Range { start, step, len } if *step > 0 => ((0..*len as i64).map(|i| start + i * step).collect(), Base::Range { start: 0, step: 1, len: *len }),
            _ => {
                let values: Vec<i64> = (0..kept.len()).map(|i| kept.at(i)).collect();
                let mut used = values.clone();
                used.sort_unstable();
                used.dedup();
                let inverse: Vec<i64> = values.iter().map(|v| used.binary_search(v).unwrap() as i64).collect();
                let natural = inverse.iter().enumerate().all(|(i, v)| *v == i as i64);
                (used, if natural { Base::Range { start: 0, step: 1, len: inverse.len() } } else { Base::Explicit(inverse.into()) })
            }
        };
        if used.len() == axis.size { axes.push(axis.clone()); } else {
            let picks: Vec<usize> = used.iter().map(|u| *u as usize).collect();
            axes.push(subset_axis(axis, &Indices::from_sorted(picks)));
        }
        let inverse = match inverse { Base::Range { len, .. } if len == 1 => Base::Range { start: 0, step: 1, len }, other => other };
        mappings.push(Mapping { base: inverse, inner, outer: tile });
    }
    DomainSpec::new(vec![indices.len()], axes, Some(mappings))
}

fn take_axis(array: &Array, indices: &Indices, axis: usize) -> Array {
    if indices.keeps_everything(array.shape()[axis]) { return array.clone(); }
    array.take_axis(&indices.to_vec(), axis)
}

/// `restricted_values`: values cut on every axis.
fn restricted_values(values: &Array, repeat: &Indices, point: &Indices, data: &[Indices]) -> Array {
    let mut result = take_axis(values, repeat, 0);
    result = take_axis(&result, point, 1);
    for (position, indices) in data.iter().enumerate() { result = take_axis(&result, indices, 2 + position); }
    result
}

fn restricted_mask(schema: &Schema, mask: &Mask, shape: &[usize], repeat: &Indices, point: &Indices, data: &[Indices]) -> Mask {
    let Mask::Plane(plane) = mask else { return mask.clone() };
    let mut array = Array::from_vec(plane.to_vec(), shape.to_vec());
    array = take_axis(&array, repeat, 0);
    array = take_axis(&array, point, 1);
    if let Some(components) = &schema.value.components {
        for (position, id) in components.iter().enumerate() {
            let index = schema.cell.index_of(id).unwrap();
            array = take_axis(&array, &data[index], 2 + position);
        }
    }
    Mask::Plane(array.bools().into())
}

/// `selection_indices(schema, value_selection(schema, terms))`: the repeat and point rows.
pub fn selected_cells(schema: &Schema, terms: &[(Arc<str>, ScopeValue)]) -> Result<(Indices, Indices)> {
    let (repeat, point, _) = selection_indices(schema, &value_selection(schema, terms)?)?;
    Ok((repeat, point))
}

/// `restrict_snapshot(snapshot, selection, repeat_rows=...)`.
pub fn restrict_snapshot(snapshot: &Snapshot, terms: &[(Arc<str>, Term)], repeat_rows: Option<(usize, usize)>, block_id: String) -> Result<Snapshot> {
    let schema = &snapshot.schema;
    let (mut repeat, point, data) = selection_indices(schema, terms)?;
    if let Some((low, high)) = repeat_rows {
        repeat = match repeat {
            Indices::Range(a, b) => Indices::Range(a.max(low), b.min(high).max(a.max(low))),
            Indices::List(v) => Indices::List(v.into_iter().filter(|r| (low..high).contains(r)).collect()),
        };
        if repeat.is_empty() { return Err(empty("selected Repeat rows do not intersect repeat_rows")); }
    }
    let repeat_domain = if repeat.keeps_everything(schema.repeat.size()) { None } else { Some(subset_mapped_domain(&schema.repeat, &repeat)) };
    let point_domain = if point.keeps_everything(schema.point.size()) { None } else { Some(subset_mapped_domain(&schema.point, &point)) };
    let cell_axes: Vec<Arc<AxisSpec>> = schema.cell.axes.iter().zip(&data).map(|(a, i)| subset_axis(a, i)).collect();
    let cell_same = cell_axes.iter().zip(&schema.cell.axes).all(|(a, b)| Arc::ptr_eq(a, b));
    let derived = if repeat_domain.is_none() && point_domain.is_none() && cell_same { schema.clone() } else {
        let clone_domain = |d: &DomainSpec| DomainSpec::new(d.shape.clone(), d.axes.clone(), d.mapping.clone());
        Arc::new(Schema::new(
            repeat_domain.unwrap_or_else(|| clone_domain(&schema.repeat)),
            point_domain.unwrap_or_else(|| clone_domain(&schema.point)),
            if cell_same { clone_domain(&schema.cell) } else { DomainSpec::new(cell_axes.iter().map(|a| a.size).collect(), cell_axes, None) },
            ValueInfo { dtype: schema.value.dtype, unit: schema.value.unit.clone(), name: schema.value.name.clone(), components: schema.value.components.clone() },
        ))
    };
    let block = match &snapshot.block {
        Block::Values { values, mask, sigma } => {
            let values = restricted_values(values, &repeat, &point, &data);
            let mask = restricted_mask(schema, mask, &schema.mask_shape(), &repeat, &point, &data);
            let mask = match mask { Mask::Plane(p) if p.iter().all(|v| *v) => Mask::Uniform(true), Mask::Plane(p) if !p.iter().any(|v| *v) => Mask::Uniform(false), m => m };
            Block::Values { values, mask, sigma: sigma.as_ref().map(|s| restricted_values(s, &repeat, &point, &data)) }
        }
        Block::Segments(segments) => {
            let overlap = |selected: &Indices, start: usize, size: usize| -> (Indices, usize) {
                match selected {
                    Indices::Range(a, b) => {
                        let (low, high) = ((*a).max(start), (*b).min(start + size));
                        (Indices::Range(low - start, low.max(high) - start), low.saturating_sub(*a))
                    }
                    Indices::List(v) => {
                        let left = v.partition_point(|x| *x < start);
                        let right = v.partition_point(|x| *x < start + size);
                        (Indices::from_sorted(v[left..right].iter().map(|x| x - start).collect()), left)
                    }
                }
            };
            let mut kept = Vec::new();
            for segment in segments.iter() {
                let (repeats, repeat_origin) = overlap(&repeat, segment.origin[0], segment.extent[0]);
                let (points, point_origin) = overlap(&point, segment.origin[1], segment.extent[1]);
                if repeats.is_empty() || points.is_empty() { continue; }
                if repeats.keeps_everything(segment.extent[0]) && points.keeps_everything(segment.extent[1])
                    && data.iter().zip(&schema.cell.axes).all(|(i, a)| i.keeps_everything(a.size)) {
                    kept.push(Placed::new([repeat_origin, point_origin], segment.segment.clone()));
                    continue;
                }
                let mut mask_shape = segment.extent.to_vec();
                mask_shape.extend(schema.mask_shape()[2..].iter().copied());
                let values = restricted_values(&segment.values, &repeats, &points, &data);
                let mask = restricted_mask(schema, &segment.mask, &mask_shape, &repeats, &points, &data);
                let sigma = segment.sigma.as_ref().map(|s| restricted_values(s, &repeats, &points, &data));
                kept.push(Placed::new([repeat_origin, point_origin], Arc::new(Segment { extent: [repeats.len(), points.len()], values, mask, sigma })));
            }
            Block::Segments(super::model::Segments::new(kept))
        }
    };
    Ok(Snapshot { schema: derived, block, window: snapshot.window, block_id: block_id.into(), generation: snapshot.generation.clone(), revision: snapshot.revision })
}

/// `FitProjection._scoped_data`: the source cut to the panel's scope and window.
pub fn scoped_data(snapshot: &Snapshot, plan: &ProjectionPlan, display: &DisplayFacts) -> Result<Snapshot> {
    let semantic = plan.spec.semantic();
    let rolling = matches!(semantic, SpecFacts::Rolling { .. });
    let layout = if matches!(semantic, SpecFacts::Curve { .. } | SpecFacts::Histogram { .. } | SpecFacts::Rolling { .. }) {
        history::layout(&snapshot.schema)?
    } else { None };
    let window = if layout.is_some() || rolling {
        Some(display.window.ok_or_else(|| ProjectionError::new(ErrorKind::Key, "'window'"))?)
    } else { None };
    let count = match &layout { Some(layout) => layout.shot_count(), None => acquired_repeat_count(snapshot) };
    let narrowed = window.is_some_and(|w| w < count);
    if plan.scope.is_empty() && !narrowed { return Ok(snapshot.clone()); }
    let mut scoped = snapshot.clone();
    if narrowed {
        let window = window.unwrap();
        let name = format!("{}|window:{window}", snapshot.block_id);
        scoped = match &layout {
            Some(layout) => restrict_snapshot(&scoped, &[(layout.primary_axis_id.clone(), Term::IndexRange(count - window, count))], None, name)?,
            None => restrict_snapshot(&scoped, &[], Some((count - window, count)), name)?,
        };
    }
    if !plan.scope.is_empty() {
        let mut terms: Vec<(Arc<str>, ScopeValue)> = Vec::new();
        for term in &plan.scope {
            let contract = snapshot.schema.resolve(&term.axis).ok_or_else(|| ProjectionError::new(ErrorKind::Key, format!("{:?}", term.axis)))?;
            let id = contract.axis.id.clone();
            match terms.iter_mut().find(|(k, _)| *k == id) { Some(slot) => slot.1 = term.value.clone(), None => terms.push((id, term.value.clone())) }
        }
        let selection = value_selection(&scoped.schema, &terms)?;
        scoped = restrict_snapshot(&scoped, &selection, None, format!("{}|scope", snapshot.block_id))?;
    }
    Ok(scoped)
}

/// `DataView.selection_subject(spec, payload, source_window=...)`.
pub fn selection_subject(view: &DataView, plan: &ProjectionPlan, payload: &Payload) -> Result<SelectionSubject> {
    let selected = match (&plan.spec, payload) {
        (SpecFacts::FacetGrid { .. }, Payload::Facet(facet)) => &facet.cells.first()
            .ok_or_else(|| ProjectionError::value("selection subject FacetGrid has no accepted cells"))?.payload,
        (SpecFacts::FacetGrid { .. }, _) => return Err(ProjectionError::value("selection subject payload differs from FacetGrid spec")),
        (_, Payload::Facet(_)) => return Err(ProjectionError::value("standalone selection subject received FacetData")),
        (_, other) => other,
    };
    let semantic = plan.spec.semantic();
    let (x, y): (Option<AxisRef>, Option<AxisRef>) = match (semantic, selected) {
        (SpecFacts::Curve { .. }, Payload::Curve(curve)) => (curve.x_ref.clone(), None),
        (SpecFacts::Image { .. }, Payload::Image(image)) => (Some(image.x_ref.clone()), Some(image.y_ref.clone())),
        (SpecFacts::Rolling { x, .. }, _) if view.has_primary_index() =>
            (Some(x.clone().unwrap_or_else(|| AxisRef { domain: AxisDomain::Point, axis_id: view.history_layout().unwrap().primary_axis_id.to_string() })), None),
        _ => (None, None),
    };
    let frame = |axis: &Option<AxisRef>| -> Result<Option<Arc<str>>> {
        Ok(match axis { None => None, Some(axis) => view.resolve(axis)?.coordinate_frame().cloned() })
    };
    let mut scope = Vec::new();
    for term in &plan.scope {
        let value = match &term.value {
            ScopeValue::Latest => { let resolved = view.resolve(&term.axis)?; resolved.contract.axis.coordinate_at(resolved.contract.axis.size - 1) }
            ScopeValue::Coordinate(c) => super::model::canonical_scalar(c.clone())?,
        };
        scope.push((term.axis.clone(), value));
    }
    let kind = match semantic { SpecFacts::Curve { .. } => PlotKind::Curve, SpecFacts::Image { .. } => PlotKind::Image,
        SpecFacts::Histogram { .. } => PlotKind::Histogram, _ => PlotKind::Rolling };
    Ok(SelectionSubject { plot_kind: kind, x_coordinate_frame: frame(&x)?, y_coordinate_frame: frame(&y)?, x, y, scope,
        source_window: if view.has_primary_index() { plan.display.window } else { None } })
}
