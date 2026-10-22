//! `DataView`: resolve and aggregate one immutable dataset revision.
//! Axis resolution, used coordinate domains, the packed calculation scratch
//! and the validation authority shared by every projection kind.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use super::history::{self, IndexedHistoryLayout};
use super::model::{AxisContract, Base, Block, Schema, Snapshot, PRIMARY_INDEX};
use super::nd::{Array, DType, Mask, Scalar, product};
use super::output::{AxisValue, convert_array};
use super::{ErrorKind, ProjectionError, Result};
use crate::kinds::{AxisDomain, AxisRef};
use crate::units::{Unit, UnitRegistry, resolve_unit};

/// `_ProjectedAxis`: one exact axis contract with its unit-aware domain.
pub struct ProjectedAxis {
    pub axis_ref: AxisRef,
    pub contract: AxisContract,
    /// `np.asarray(axis)`: the declared coordinates.
    pub domain_canonical: Array,
    /// The same coordinates in the display unit (the same array when equal).
    pub domain_display: Array,
    pub canonical_unit: Arc<Unit>,
    pub display_unit: Arc<Unit>,
    pub label: Arc<str>,
    /// Each carrier row's coordinate code (`source_indices`).
    pub source_codes: Arc<[i64]>,
    used: OnceLock<Arc<UsedDomain>>,
}

impl ProjectedAxis {
    pub fn dimension(&self) -> usize { self.contract.dimension }
    pub fn coordinate_frame(&self) -> Option<&Arc<str>> { self.contract.axis.frame.as_ref() }
    pub fn coordinate_labels(&self) -> Option<&Arc<[Arc<str>]>> { self.contract.axis.labels.as_ref() }
    /// The canonical coordinate of carrier row `row` along this axis's dimension.
    pub fn row_canonical(&self, row: usize) -> f64 { self.domain_canonical.get_f64(self.source_codes[row] as usize) }
}

/// `_Domain`: the used coordinate domain of one axis and each row's code into it.
pub struct UsedDomain {
    pub axis_ref: AxisRef,
    pub canonical: Array,
    pub display: Array,
    pub codes: Arc<[i64]>,
    pub used_indices: Vec<usize>,
    labels: Option<Arc<[Arc<str>]>>,
    axis_label: Arc<str>,
    display_unit: Arc<Unit>,
    values: OnceLock<Vec<AxisValue>>,
}

impl UsedDomain {
    pub fn size(&self) -> usize { self.canonical.len() }
    /// The labelled `AxisValue`s, built on first request.
    pub fn values(&self) -> &[AxisValue] {
        self.values.get_or_init(|| (0..self.size()).map(|position| {
            let index = self.used_indices[position];
            let display = self.display.scalar(position);
            let label = axis_value_label(&self.axis_label, &display, &self.display_unit,
                self.labels.as_ref().map(|labels| labels[index].as_ref()));
            AxisValue { axis: self.axis_ref.clone(), index: Some(index), canonical: self.canonical.scalar(position), display, label }
        }).collect())
    }
}

/// `_axis_value_label`.
pub fn axis_value_label(label: &str, value: &Scalar, unit: &Unit, coordinate_label: Option<&str>) -> String {
    if let Some(coordinate) = coordinate_label { return format!("{label}={coordinate}"); }
    let suffix = if unit.symbol == "1" { String::new() } else { format!(" {}", unit.symbol) };
    format!("{label}={}{suffix}", value.python_str())
}

/// `_scalar_kind_array`: the dtype Python-scalar materialization produced.
pub fn scalar_kind_array(values: Array) -> Array {
    match values.dtype() {
        DType::F16 | DType::F32 => values.as_f64_array(),
        DType::I8 | DType::I16 | DType::I32 => Array::from_vec((0..values.len()).map(|i| values.get_f64(i) as i64).collect::<Vec<i64>>(), values.shape().to_vec()),
        DType::U8 | DType::U16 | DType::U32 => Array::from_vec((0..values.len()).map(|i| values.get_f64(i) as i64).collect::<Vec<i64>>(), values.shape().to_vec()),
        DType::U64 => {
            let slice = values.slice::<u64>();
            if slice.iter().all(|v| *v <= i64::MAX as u64) { Array::from_vec(slice.iter().map(|v| *v as i64).collect::<Vec<i64>>(), values.shape().to_vec()) } else { values }
        }
        _ => values,
    }
}

/// The packed numeric scratch every projection reads (`_segment_arrays`).
pub struct Packed {
    /// Full physical shape, or `(rows, *cell)` with `rows` naming each.
    pub values: Array,
    /// Validity merged with finiteness, over `values`.
    pub valid: Mask,
    pub sigma: Option<Array>,
    pub rows: Option<(Arc<[i64]>, Arc<[i64]>)>,
    pub sigma_read: bool,
    /// Whether numpy holds `valid` as a C-contiguous plane (a broadcast
    /// mask is not): `_centred_moment_sums` declines any other layout.
    pub valid_contiguous: bool,
}

pub struct DataView {
    pub(crate) snapshot: Snapshot,
    pub(crate) schema: Arc<Schema>,
    pub(crate) registry: Option<Arc<UnitRegistry>>,
    pub(crate) axis_display_units: Vec<(AxisRef, Arc<str>)>,
    pub(crate) value_unit: Arc<Unit>,
    pub(crate) value_display_unit: Arc<Unit>,
    pub(crate) layout: Option<Arc<IndexedHistoryLayout>>,
    axis_cache: RefCell<HashMap<AxisRef, Arc<ProjectedAxis>>>,
    packed: RefCell<Option<Arc<Packed>>>,
    pub(crate) caches: RefCell<ViewCarries>,
    pub(crate) pools: Arc<super::nd::Pools>,
}

fn axis_repr(axis: &AxisRef) -> String {
    let domain = match axis.domain { AxisDomain::Repeat => "REPEAT", AxisDomain::Point => "POINT", AxisDomain::CellData => "CELL_DATA" };
    format!("AxisRef(domain=<AxisDomain.{domain}: '{}'>, axis_id='{}')", axis.domain.value(), axis.axis_id)
}

impl DataView {
    pub fn new(snapshot: Snapshot, axis_display_units: Vec<(AxisRef, Arc<str>)>, value_display_unit: Option<&str>,
               registry: Option<Arc<UnitRegistry>>) -> Result<Self> {
        let schema = snapshot.schema.clone();
        let value_unit = resolve_unit(schema.value.unit.as_deref().unwrap_or("1"), registry.as_deref())?;
        let value_display_unit = match value_display_unit { None => value_unit.clone(), Some(unit) => resolve_unit(unit, registry.as_deref())? };
        if !value_unit.compatible_with(&value_display_unit) {
            return Err(ProjectionError::view("value display unit is incompatible with dataset values"));
        }
        let layout = history::layout(&schema)?;
        let view = DataView { snapshot, schema, registry, axis_display_units, value_unit, value_display_unit, layout,
            axis_cache: RefCell::new(HashMap::new()), packed: RefCell::new(None), caches: RefCell::new(Default::default()),
            pools: Arc::new(Default::default()) };
        for (axis, _) in view.axis_display_units.clone() { view.resolve(&axis)?; }
        Ok(view)
    }

    pub fn snapshot(&self) -> &Snapshot { &self.snapshot }
    pub fn schema(&self) -> &Arc<Schema> { &self.schema }
    pub fn revision(&self) -> u64 { self.snapshot.revision }
    pub fn generation(&self) -> &Arc<str> { &self.snapshot.generation }
    pub fn has_primary_index(&self) -> bool { self.layout.is_some() }
    pub fn history_layout(&self) -> Option<&Arc<IndexedHistoryLayout>> { self.layout.as_ref() }
    pub fn value_label(&self) -> String { self.schema.value.name.as_deref().unwrap_or("value").to_owned() }
    pub fn registry(&self) -> Option<&UnitRegistry> { self.registry.as_deref() }

    /// `_resolve(ref)`.
    pub fn resolve(&self, axis: &AxisRef) -> Result<Arc<ProjectedAxis>> {
        if let Some(found) = self.axis_cache.borrow().get(axis) { return Ok(found.clone()); }
        let contract = self.schema.resolve(axis).ok_or_else(|| ProjectionError::new(ErrorKind::AxisResolution,
            format!("dataset has no exact {} axis '{}'", axis.domain.value(), axis.axis_id)))?;
        let spec = contract.axis.clone();
        let domain_canonical = spec.values().clone();
        let source_codes = self.schema.domain_of(&contract).codes(contract.index);
        let canonical_unit = resolve_unit(spec.unit.as_deref().unwrap_or("1"), self.registry.as_deref())?;
        let display_unit = match self.axis_display_units.iter().find(|(key, _)| key == axis) {
            None => canonical_unit.clone(),
            Some((_, unit)) => resolve_unit(unit, self.registry.as_deref())?,
        };
        if !canonical_unit.compatible_with(&display_unit) {
            return Err(ProjectionError::view(format!("display unit for {} is incompatible with its axis", axis_repr(axis))));
        }
        let domain_display = convert_array(&domain_canonical, &canonical_unit, &display_unit);
        let resolved = Arc::new(ProjectedAxis { axis_ref: axis.clone(), contract, domain_canonical, domain_display, canonical_unit,
            display_unit, label: spec.name.clone(), source_codes, used: OnceLock::new() });
        self.axis_cache.borrow_mut().insert(axis.clone(), resolved.clone());
        Ok(resolved)
    }

    /// `_domain(ref)`: the used coordinate domain of one declared axis mapping.
    pub fn domain(&self, axis: &AxisRef) -> Result<Arc<UsedDomain>> {
        let resolved = self.resolve(axis)?;
        if let Some(found) = resolved.used.get() { return Ok(found.clone()); }
        let mapping = self.schema.domain_of(&resolved.contract).code_mapping(resolved.contract.index);
        let size = resolved.domain_canonical.len();
        let source = &resolved.source_codes;
        let (used_indices, codes): (Vec<usize>, Arc<[i64]>) = match &mapping.base {
            Base::Range { start, step, len } => {
                let last = start + (*len as i64 - 1) * step;
                let (low, high, stride) = ((*start).min(last), (*start).max(last), step.unsigned_abs() as usize);
                let used: Vec<usize> = (low as usize..=high as usize).step_by(stride.max(1)).collect();
                if used.len() == size && low == 0 && stride == 1 { (used, source.clone()) }
                else { (used, source.iter().map(|code| (code - low) / stride as i64).collect()) }
            }
            Base::Explicit(base) => {
                let mut present = vec![false; size];
                for code in base.iter() { present[*code as usize] = true; }
                let used: Vec<usize> = (0..size).filter(|i| present[*i]).collect();
                if used.len() == size { (used, source.clone()) } else {
                    let mut remap = vec![-1i64; size];
                    for (position, index) in used.iter().enumerate() { remap[*index] = position as i64; }
                    (used, source.iter().map(|code| remap[*code as usize]).collect())
                }
            }
        };
        let n = used_indices.len();
        let gather = |values: &Array| -> Array {
            if n == size && used_indices.iter().enumerate().all(|(i, u)| i == *u) { values.clone() } else { values.take_flat(&used_indices, vec![n]) }
        };
        let canonical = scalar_kind_array(gather(&resolved.domain_canonical));
        let display = if resolved.domain_display.same_storage(&resolved.domain_canonical) { canonical.clone() }
                      else { scalar_kind_array(gather(&resolved.domain_display)) };
        let domain = Arc::new(UsedDomain { axis_ref: axis.clone(), canonical, display, codes, used_indices,
            labels: resolved.coordinate_labels().cloned(), axis_label: resolved.label.clone(),
            display_unit: resolved.display_unit.clone(), values: OnceLock::new() });
        let _ = resolved.used.set(domain.clone());
        Ok(domain)
    }

    /// `_axis_projection(refs)`: each kept axis's codes and tensor dimension.
    pub fn axis_projection(&self, refs: &[AxisRef]) -> Result<(Vec<Arc<UsedDomain>>, Vec<Arc<[i64]>>, Vec<usize>)> {
        let mut domains = Vec::with_capacity(refs.len());
        let mut codes = Vec::with_capacity(refs.len());
        let mut dimensions = Vec::with_capacity(refs.len());
        for axis in refs {
            let resolved = self.resolve(axis)?;
            let domain = self.domain(axis)?;
            codes.push(domain.codes.clone());
            dimensions.push(resolved.dimension());
            domains.push(domain);
        }
        Ok((domains, codes, dimensions))
    }

    /// `_require_real_numeric` on an axis's coordinates.
    pub fn require_real_numeric(&self, axis: &AxisRef) -> Result<()> {
        if !"biuf".contains(self.resolve(axis)?.domain_canonical.kind()) {
            return Err(ProjectionError::view(format!("{} must be real numeric for this projection", axis_repr(axis))));
        }
        Ok(())
    }

    /// `validate_curve`.
    pub fn validate_curve(&self, x: &AxisRef, groups: &[AxisRef]) -> Result<()> {
        for (index, group) in groups.iter().enumerate() {
            if groups[..index].contains(group) { return Err(ProjectionError::view("group_by axes must be unique")); }
        }
        if groups.contains(x) { return Err(ProjectionError::view("curve x axis cannot also be a group axis")); }
        self.require_real_numeric(x)?;
        for group in groups { self.resolve(group)?; }
        Ok(())
    }

    /// `validate_image`.
    pub fn validate_image(&self, x: &AxisRef, y: &AxisRef) -> Result<()> {
        if x == y { return Err(ProjectionError::view("image x and y axes must differ")); }
        self.require_real_numeric(x)?;
        self.require_real_numeric(y)
    }

    /// `rolling_coordinate_role(x)`.
    pub fn rolling_coordinate_role(&self, x: Option<&AxisRef>) -> Result<&'static str> {
        let Some(x) = x else { return Ok(PRIMARY_INDEX) };
        let role = match &self.layout {
            Some(layout) if x.domain == AxisDomain::Point => layout.coordinate_role(&x.axis_id),
            _ => None,
        };
        role.ok_or_else(|| ProjectionError::view("Rolling has a fixed record axis; choose its source index or shot time coordinate"))
    }

    /// `semantics._is_rolling_record_axis`.
    pub fn is_rolling_record_axis(&self, axis: &AxisRef) -> Result<bool> {
        if self.layout.is_none() { return Ok(axis.domain == AxisDomain::Repeat); }
        let contract = self.schema.resolve(axis).ok_or_else(|| ProjectionError::new(ErrorKind::Key, format!("{axis:?}")))?;
        let domain = self.schema.domain_of(&contract);
        Ok(&*domain.axes[domain.coordinate_axis(contract.index)].role == PRIMARY_INDEX)
    }

    /// `validate_rolling`.
    pub fn validate_rolling(&self, group: Option<&AxisRef>, x: Option<&AxisRef>) -> Result<()> {
        self.rolling_coordinate_role(x)?;
        if let Some(group) = group {
            if self.is_rolling_record_axis(group)? {
                return Err(ProjectionError::view("Rolling's fixed record axis cannot also be Group"));
            }
            self.resolve(group)?;
        }
        Ok(())
    }

    /// `acquired_repeat_count(snapshot)`.
    pub fn acquired_repeat_count(&self) -> usize { acquired_repeat_count(&self.snapshot) }

    // ------------------------------------------------------------- scratch
    /// `_segment_arrays(sigma=...)`: one packed numeric scratch for this view.
    pub fn packed(&self, sigma: bool) -> Result<Arc<Packed>> {
        if let Some(found) = self.packed.borrow().as_ref() {
            if !sigma || found.sigma_read { return Ok(found.clone()); }
        }
        let packed = Arc::new(pack(&self.snapshot, None, sigma, Some(&self.pools))?);
        *self.packed.borrow_mut() = Some(packed.clone());
        Ok(packed)
    }

    /// Whether the scratch is the whole physical tensor, not rows named one
    /// by one (`packed().rows` is None), known without packing it.
    pub fn packs_whole(&self) -> bool {
        match &self.snapshot.block {
            Block::Values { .. } => true,
            Block::Segments(all) => fills_in_order(&self.schema.physical_shape, &all.iter().collect::<Vec<_>>()),
        }
    }
}

/// `acquired_repeat_count`: leading Repeat rows a snapshot has begun.
pub fn acquired_repeat_count(snapshot: &Snapshot) -> usize {
    let declared = snapshot.schema.repeat.size();
    match &snapshot.block {
        Block::Values { .. } => declared,
        Block::Segments(segments) if segments.is_empty() => declared.min(1),
        Block::Segments(segments) => declared.min(segments.iter().map(|s| s.origin[0] + s.extent[0]).max().unwrap_or(0).max(1)),
    }
}

/// Expand a compact `(leading, *components)` mask over `(leading, *cell)`.
pub fn expand_mask(schema: &Schema, compact: &[bool], leading: usize) -> Vec<bool> {
    let cell: Vec<usize> = schema.cell.shape.clone();
    let cell_size = product(&cell);
    let named: Vec<bool> = schema.cell.axes.iter().map(|axis| schema.value.components.as_ref()
        .is_some_and(|ids| ids.iter().any(|id| *id == axis.id))).collect();
    let component_size: usize = cell.iter().zip(&named).filter(|(_, n)| **n).map(|(s, _)| *s).product();
    let mut map = vec![0usize; cell_size];
    for (flat, slot) in map.iter_mut().enumerate() {
        let (mut rest, mut compact_index, mut stride) = (flat, 0usize, 1usize);
        for axis in (0..cell.len()).rev() {
            let index = rest % cell[axis];
            rest /= cell[axis];
            if named[axis] { compact_index += index * stride; stride *= cell[axis]; }
        }
        *slot = compact_index;
    }
    let mut full = Vec::with_capacity(leading * cell_size);
    for row in 0..leading {
        let base = row * component_size;
        full.extend(map.iter().map(|m| compact[base + m]));
    }
    full
}

/// The valid/finite merge, and whether numpy's plane is C-contiguous after
/// it: `np.broadcast_to` of a whole-block bool has stride 0 everywhere, of
/// a compact mask stride 0 along every cell axis outside the validity
/// components; merging finiteness writes a new plane.
fn merge_finite(schema: &Schema, values: &Array, valid: Mask) -> (Mask, bool) {
    let contiguous = values.len() <= 1 || match &valid {
        Mask::Uniform(_) => false,
        Mask::Plane(_) => schema.cell.axes.iter().zip(&schema.cell.shape).all(|(axis, size)| *size <= 1
            || schema.value.components.as_ref().is_some_and(|ids| ids.iter().any(|id| *id == axis.id))),
    };
    if "biu".contains(values.kind()) || values.all_finite() { return (valid, contiguous); }
    let finite = values.finite_mask();
    (match valid {
        Mask::Uniform(true) => Mask::Plane(finite.into()),
        Mask::Uniform(false) => Mask::Uniform(false),
        Mask::Plane(plane) => Mask::Plane(plane.iter().zip(finite).map(|(a, b)| *a && b).collect()),
    }, true)
}

/// Whether `segments` fill every Repeat x Point cell of `shape` in row
/// order: they are then packed whole, with no rows named one by one.
fn fills_in_order(shape: &[usize], segments: &[super::model::PlacedRef<'_>]) -> bool {
    let counts: Vec<usize> = segments.iter().map(|s| s.extent[0] * s.extent[1]).collect();
    let starts: Vec<usize> = counts.iter().scan(0usize, |total, count| { let start = *total; *total += count; Some(start) }).collect();
    counts.iter().sum::<usize>() == shape[0] * shape[1]
        && segments.iter().zip(&starts).all(|(s, start)| s.origin[0] * shape[1] + s.origin[1] == *start)
        && segments.iter().zip(&counts).all(|(s, count)| (s.extent[0] - 1) * shape[1] + s.extent[1] == *count)
}

/// `DataBlock.packed_planes` + the valid/finite merge of `_segment_arrays`.
pub fn pack(snapshot: &Snapshot, selection: Option<&[usize]>, sigma: bool, pools: Option<&super::nd::Pools>) -> Result<Packed> {
    let schema = &snapshot.schema;
    let shape = &schema.physical_shape;
    let cell: Vec<usize> = shape[2..].to_vec();
    let cell_size = product(&cell);
    match &snapshot.block {
        Block::Values { values, mask, sigma: errors } => {
            let valid = match mask {
                Mask::Uniform(v) => Mask::Uniform(*v),
                Mask::Plane(compact) => Mask::from_vec(expand_mask(schema, compact, shape[0] * shape[1])),
            };
            let (valid, valid_contiguous) = merge_finite(schema, values, valid);
            Ok(Packed { values: values.clone(), valid, sigma: if sigma { errors.clone() } else { None }, rows: None, sigma_read: sigma, valid_contiguous })
        }
        Block::Segments(all) => {
            let segments: Vec<super::model::PlacedRef<'_>> = match selection {
                None => all.iter().collect(),
                Some(chosen) => chosen.iter().map(|i| all.get(*i)).collect(),
            };
            let counts: Vec<usize> = segments.iter().map(|s| s.extent[0] * s.extent[1]).collect();
            let total: usize = counts.iter().sum();
            let complete = fills_in_order(shape, &segments);
            let dtype = schema.value.dtype;
            let parts: Vec<&Array> = segments.iter().map(|s| &s.segment.values).collect();
            let mut values = if parts.is_empty() { crate::with_dtype!(dtype, T => Array::from_vec(Vec::<T>::new(), vec![0])) }
                             else if let Some(pools) = pools { pools.concat(&parts, vec![total * cell_size]) } else { Array::concat(&parts, vec![total * cell_size]) };
            let mut rows = None;
            let mut order: Option<Vec<usize>> = None;
            if complete {
                values = values.reshape(shape.clone());
            } else {
                let mut repeat = Vec::with_capacity(total);
                let mut point = Vec::with_capacity(total);
                for segment in &segments {
                    for local in 0..segment.extent[0] * segment.extent[1] {
                        repeat.push((segment.origin[0] + local / segment.extent[1]) as i64);
                        point.push((segment.origin[1] + local % segment.extent[1]) as i64);
                    }
                }
                let linear: Vec<i64> = repeat.iter().zip(&point).map(|(r, p)| r * shape[1] as i64 + p).collect();
                values = values.reshape([vec![total], cell.clone()].concat());
                if linear.windows(2).any(|pair| pair[1] < pair[0]) {
                    let mut sorted: Vec<usize> = (0..total).collect();
                    sorted.sort_by_key(|i| linear[*i]);
                    values = values.take_rows(&sorted);
                    repeat = sorted.iter().map(|i| repeat[*i]).collect();
                    point = sorted.iter().map(|i| point[*i]).collect();
                    order = Some(sorted);
                }
                rows = Some((Arc::from(repeat), Arc::from(point)));
            }
            let all_true = segments.iter().all(|s| matches!(s.mask, Mask::Uniform(true)));
            let all_false = segments.iter().all(|s| matches!(s.mask, Mask::Uniform(false)));
            let leading_rows = total;
            let valid = if all_true { Mask::Uniform(true) } else if all_false { Mask::Uniform(false) } else {
                let component_size: usize = schema.mask_shape()[2..].iter().product();
                let mut compact = Vec::with_capacity(total * component_size);
                for (segment, count) in segments.iter().zip(&counts) {
                    match &segment.mask {
                        Mask::Uniform(v) => compact.extend(std::iter::repeat_n(*v, count * component_size)),
                        Mask::Plane(plane) => compact.extend_from_slice(plane),
                    }
                }
                if let Some(order) = &order {
                    let source = compact.clone();
                    compact = order.iter().flat_map(|row| source[row * component_size..(row + 1) * component_size].iter().copied()).collect();
                }
                Mask::from_vec(expand_mask(schema, &compact, leading_rows))
            };
            let errors = if sigma && segments.iter().any(|s| s.sigma.is_some()) {
                let mut plane: Vec<f64> = Vec::with_capacity(total * cell_size);
                for segment in &segments {
                    match &segment.sigma {
                        Some(sigma) => plane.extend_from_slice(sigma.slice::<f64>()),
                        None => plane.extend(std::iter::repeat_n(f64::NAN, segment.values.len())),
                    }
                }
                let mut array = Array::from_vec(plane, values.shape().to_vec());
                if let Some(order) = &order { array = array.reshape([vec![total], cell.clone()].concat()).take_rows(order).reshape(values.shape().to_vec()); }
                Some(array)
            } else { None };
            let (valid, valid_contiguous) = merge_finite(schema, &values, valid);
            Ok(Packed { valid, values, sigma: errors, rows, sigma_read: sigma, valid_contiguous })
        }
    }
}

/// Work carried from the previous revision's view (`inherit_domains_from`).
#[derive(Default)]
pub struct ViewCarries {
    pub histogram_plan: Option<(String, Arc<super::histogram::HistogramPlan>)>,
    pub frequency: Option<super::histogram::FrequencyCarry>,
    /// Resolved axes and their used domains: immutable schema and unit facts.
    pub axes: Option<CarriedAxes>,
    /// The previous history by segment identity (`_rolling_carry`).
    pub rolling: Option<super::rolling::RollingCarry>,
    /// The previous axis reduction's walked state by segment identity.
    pub accumulation: Option<super::reduce::AccumulationCarry>,
    /// The previous revision's histogram of a still-filling run, by segment identity.
    pub counted: Option<super::histogram::HistogramCarry>,
}

pub struct CarriedAxes {
    schema: Arc<Schema>,
    units: Vec<(AxisRef, Arc<str>)>,
    registry: Option<Arc<UnitRegistry>>,
    cache: HashMap<AxisRef, Arc<ProjectedAxis>>,
}

impl DataView {
    /// Take over the previous revision's carried work, where Python would.
    pub(crate) fn adopt(&self, carry: ViewCarries) {
        let integer_history = self.layout.is_some() && self.snapshot.window.is_some() && "iu".contains(self.schema.value.dtype.kind());
        if integer_history {
            self.caches.borrow_mut().frequency = carry.frequency;
        }
        if let Some(axes) = carry.axes {
            let same_registry = match (&axes.registry, &self.registry) { (None, None) => true, (Some(a), Some(b)) => Arc::ptr_eq(a, b), _ => false };
            let same_units = axes.units == self.axis_display_units && same_registry;
            if same_units && self.snapshot.is_segmented() {
                let mut caches = self.caches.borrow_mut();
                caches.rolling = carry.rolling.filter(|rolling| rolling.indexed == self.layout.is_some());
                caches.accumulation = carry.accumulation;
                caches.counted = carry.counted;
            }
            if Arc::ptr_eq(&axes.schema, &self.schema) && same_units {
                let mut cache = self.axis_cache.borrow_mut();
                for (key, value) in axes.cache { cache.entry(key).or_insert(value); }
            }
        }
    }
    /// Hand this view's carried work to the next revision.
    pub(crate) fn release(&self) -> ViewCarries {
        let mut caches = self.caches.borrow_mut();
        let axes = CarriedAxes { schema: self.schema.clone(), units: self.axis_display_units.clone(), registry: self.registry.clone(),
            cache: self.axis_cache.borrow().clone() };
        ViewCarries { histogram_plan: None, frequency: caches.frequency.take(), axes: Some(axes), rolling: caches.rolling.take(),
                      accumulation: caches.accumulation.take(), counted: caches.counted.take() }
    }
}
