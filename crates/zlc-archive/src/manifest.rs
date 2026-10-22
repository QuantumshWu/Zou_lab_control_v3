//! One dataset in a file (`zlc_data.io` manifests): its revision identity,
//! schema tree, materialized planes and, for a partial acquisition, where its
//! acquired rectangles lie.

use std::sync::Arc;

use zlc_core::dataset::{DataBlock, Validity};
use zlc_core::restrict::{gather, gather_buffer, restricted_schema, Indices, SelectedIndices, Source};
use zlc_core::runtime::{Read, ReadKind};
use zlc_core::schema::{DatasetSchema, DomainRole, ValidityContract};
use zlc_core::store::{DatasetStore, Placement, Snapshot};
use zlc_core::{AnyBuffer, Complex128, Complex64, DType, F16, OwnedBuffer};

use crate::error::{ArchiveError, Result};
use crate::json::{py_repr, py_repr_names, py_repr_str, Json};
use crate::figure::Members;
use crate::npy::{NpyArray, NpyData, NpyType};
use crate::tree::{canonical_text, exact_mapping, integer, schema_from_tree, schema_tree};

pub const FORMAT: &str = "zlc.dataset";
pub const REF_SCHEMA: &str = "zlc_data.DatasetRevisionRef";

/// `DatasetRevisionRef` without its schema fingerprint (derived from the schema).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DatasetRef { pub block_id: String, pub stream_generation: String, pub revision: u64 }

impl DatasetRef {
    /// The identity `zlc_python.NativeRead.to_science` gives one read.
    pub fn of_read(read: &Read) -> Self {
        let key = read.key();
        let block_id = match read.kind() {
            ReadKind::Event(_) => format!("event:{}", key.signal),
            ReadKind::Canonical(_) => format!("canonical:{}", key.signal),
            ReadKind::Window(window) => format!("window:{}:{}:{}:{}", key.signal.len(), key.signal, window.width, window.view_epoch),
        };
        Self { block_id, stream_generation: key.generation.as_uuid().to_string(), revision: key.revision }
    }
}

/// One dataset to archive: its identity and its planes as the scientific
/// value holds them (a dense event or acquired placements).
#[derive(Clone)]
pub struct ArchiveDataset { pub reference: DatasetRef, pub source: Source }

impl ArchiveDataset {
    pub fn of_read(read: &Read) -> std::result::Result<Self, String> {
        Ok(Self { reference: DatasetRef::of_read(read), source: Source::of_read(read)? })
    }
}

/// Materialized planes and validity as `DataBlock.materialize` gives them.
pub(crate) enum Mark { Valid, Invalid, Plane(OwnedBuffer<bool>) }

pub(crate) struct Planes {
    schema: Arc<DatasetSchema>,
    values: AnyBuffer,
    mark: Mark,
    sigma: Option<OwnedBuffer<f64>>,
    /// Acquired rectangles when they do not cover the dataset.
    placements: Option<(Vec<[usize; 2]>, Vec<[usize; 2]>)>,
}

fn storage_shape(schema: &DatasetSchema) -> Vec<usize> {
    let cell = schema.domain(DomainRole::Cell);
    let mut shape = vec![schema.domain(DomainRole::Repeat).size(), schema.domain(DomainRole::Point).size()];
    if let ValidityContract::Components(ids) = schema.value().validity() {
        shape.extend(ids.iter().map(|id| cell.axis(id).expect("validated component").len()));
    }
    shape
}

/// Copy one acquired block into its rectangle of complete C-order planes.
fn scatter<T: Copy>(target: &mut [T], target_points: usize, unit: usize, origin: [usize; 2], extent: [usize; 2], source: &[T]) {
    for row in 0..extent[0] {
        let from = row * extent[1] * unit;
        let to = ((origin[0] + row) * target_points + origin[1]) * unit;
        target[to..to + extent[1] * unit].copy_from_slice(&source[from..from + extent[1] * unit]);
    }
}

/// A mutable typed vector of any dtype, for filling complete planes.
enum Filling { Bool(Vec<bool>), U8(Vec<u8>), I8(Vec<i8>), U16(Vec<u16>), I16(Vec<i16>), U32(Vec<u32>), I32(Vec<i32>), U64(Vec<u64>), I64(Vec<i64>),
               F16(Vec<F16>), F32(Vec<f32>), F64(Vec<f64>), C64(Vec<Complex64>), C128(Vec<Complex128>) }

macro_rules! filling_dispatch {
    ($filling:expr, $source:expr, |$t:ident, $s:ident| $body:expr) => {
        match ($filling, $source) {
            (Filling::Bool($t), AnyBuffer::Bool($s)) => $body, (Filling::U8($t), AnyBuffer::U8($s)) => $body,
            (Filling::I8($t), AnyBuffer::I8($s)) => $body, (Filling::U16($t), AnyBuffer::U16($s)) => $body,
            (Filling::I16($t), AnyBuffer::I16($s)) => $body, (Filling::U32($t), AnyBuffer::U32($s)) => $body,
            (Filling::I32($t), AnyBuffer::I32($s)) => $body, (Filling::U64($t), AnyBuffer::U64($s)) => $body,
            (Filling::I64($t), AnyBuffer::I64($s)) => $body, (Filling::F16($t), AnyBuffer::F16($s)) => $body,
            (Filling::F32($t), AnyBuffer::F32($s)) => $body, (Filling::F64($t), AnyBuffer::F64($s)) => $body,
            (Filling::C64($t), AnyBuffer::C64($s)) => $body, (Filling::C128($t), AnyBuffer::C128($s)) => $body,
            _ => unreachable!("segment dtype differs from its dataset"),
        }
    };
}

impl Filling {
    fn zeros(dtype: DType, len: usize) -> Self {
        match dtype {
            DType::Bool => Filling::Bool(vec![false; len]), DType::U8 => Filling::U8(vec![0; len]), DType::I8 => Filling::I8(vec![0; len]),
            DType::U16 => Filling::U16(vec![0; len]), DType::I16 => Filling::I16(vec![0; len]), DType::U32 => Filling::U32(vec![0; len]),
            DType::I32 => Filling::I32(vec![0; len]), DType::U64 => Filling::U64(vec![0; len]), DType::I64 => Filling::I64(vec![0; len]),
            DType::F16 => Filling::F16(vec![F16::default(); len]), DType::F32 => Filling::F32(vec![0.0; len]), DType::F64 => Filling::F64(vec![0.0; len]),
            DType::C64 => Filling::C64(vec![Complex64::default(); len]), DType::C128 => Filling::C128(vec![Complex128::default(); len]),
        }
    }
    fn place(&mut self, source: &AnyBuffer, points: usize, unit: usize, origin: [usize; 2], extent: [usize; 2]) {
        filling_dispatch!(self, source, |t, s| scatter(t, points, unit, origin, extent, s.as_slice()));
    }
    fn finish(self) -> AnyBuffer {
        match self {
            Filling::Bool(v) => OwnedBuffer::from_vec(v).into(), Filling::U8(v) => OwnedBuffer::from_vec(v).into(), Filling::I8(v) => OwnedBuffer::from_vec(v).into(),
            Filling::U16(v) => OwnedBuffer::from_vec(v).into(), Filling::I16(v) => OwnedBuffer::from_vec(v).into(), Filling::U32(v) => OwnedBuffer::from_vec(v).into(),
            Filling::I32(v) => OwnedBuffer::from_vec(v).into(), Filling::U64(v) => OwnedBuffer::from_vec(v).into(), Filling::I64(v) => OwnedBuffer::from_vec(v).into(),
            Filling::F16(v) => OwnedBuffer::from_vec(v).into(), Filling::F32(v) => OwnedBuffer::from_vec(v).into(), Filling::F64(v) => OwnedBuffer::from_vec(v).into(),
            Filling::C64(v) => OwnedBuffer::from_vec(v).into(), Filling::C128(v) => OwnedBuffer::from_vec(v).into(),
        }
    }
}

/// `DataBlock.materialize()` of a dataset's planes.
pub(crate) fn materialize(source: &Source) -> Planes {
    match source {
        Source::Dense(block) => Planes {
            schema: block.schema().clone(), values: block.values().clone(), sigma: block.sigma().cloned(), placements: None,
            mark: match block.validity() { Validity::Uniform(true) => Mark::Valid, Validity::Uniform(false) => Mark::Invalid, Validity::Mask(mask) => Mark::Plane(mask.clone()) },
        },
        Source::Placed { schema, segments, .. } => {
            let (repeats, points) = (schema.domain(DomainRole::Repeat).size(), schema.domain(DomainRole::Point).size());
            let cells = schema.domain(DomainRole::Cell).size();
            let marks = storage_shape(schema)[2..].iter().product::<usize>();
            let mut values = Filling::zeros(schema.dtype(), schema.value_len());
            let mut mask = vec![false; schema.component_mask_len()];
            let mut sigma = segments.iter().any(|(_, block)| block.sigma().is_some()).then(|| vec![f64::NAN; schema.value_len()]);
            let mut acquired = 0;
            for (origin, block) in segments {
                let block_schema = block.schema();
                let extent = [block_schema.domain(DomainRole::Repeat).size(), block_schema.domain(DomainRole::Point).size()];
                acquired += extent[0] * extent[1];
                values.place(block.values(), points, cells, *origin, extent);
                match block.validity() {
                    Validity::Uniform(flag) => {
                        let plane = vec![*flag; extent[0] * extent[1] * marks];
                        scatter(&mut mask, points, marks, *origin, extent, &plane);
                    }
                    Validity::Mask(plane) => scatter(&mut mask, points, marks, *origin, extent, plane.as_slice()),
                }
                if let Some(target) = &mut sigma {
                    match block.sigma() {
                        Some(errors) => scatter(target, points, cells, *origin, extent, errors.as_slice()),
                        None => scatter(target, points, cells, *origin, extent, &vec![f64::NAN; extent[0] * extent[1] * cells]),
                    }
                }
            }
            let mark = if mask.iter().all(|flag| *flag) { Mark::Valid } else if !mask.iter().any(|flag| *flag) { Mark::Invalid } else { Mark::Plane(OwnedBuffer::from_vec(mask)) };
            let placements = (acquired != repeats * points).then(|| (segments.iter().map(|(origin, _)| *origin).collect(),
                segments.iter().map(|(_, block)| [block.schema().domain(DomainRole::Repeat).size(), block.schema().domain(DomainRole::Point).size()]).collect()));
            Planes { schema: schema.clone(), values: values.finish(), mark, sigma: sigma.map(OwnedBuffer::from_vec), placements }
        }
    }
}

/// A dataset's complete planes (`DataBlock.materialize()`): values in its
/// physical shape, its validity as one flag or a mask over its component
/// layout, and its sigma.  What the Figure Viewer's data editor expands.
pub struct DensePlanes { pub values: AnyBuffer, pub valid: bool, pub mask: Option<Vec<bool>>, pub sigma: Option<Vec<f64>> }

pub fn dense_planes(source: &Source) -> DensePlanes {
    let planes = materialize(source);
    let (valid, mask) = match planes.mark { Mark::Valid => (true, None), Mark::Invalid => (false, None), Mark::Plane(mask) => (false, Some(mask.as_slice().to_vec())) };
    DensePlanes { values: planes.values, valid, mask, sigma: planes.sigma.map(|sigma| sigma.as_slice().to_vec()) }
}

/// `_snapshot_members(key, snapshot)`: which members a dataset's planes take.
pub fn snapshot_members(key: &str, source: &Source) -> Vec<String> { members_of(key, &materialize(source)) }

pub(crate) fn members_of(key: &str, planes: &Planes) -> Vec<String> {
    let mut members = vec![key.to_owned()];
    if let Mark::Plane(_) = planes.mark { members.push(format!("{key}.validity")); }
    if planes.sigma.is_some() { members.push(format!("{key}.sigma")); }
    members
}

/// `snapshot_manifest(snapshot, arrays, **_snapshot_members(key, snapshot))`.
pub(crate) fn snapshot_manifest(key: &str, reference: &DatasetRef, planes: Planes, stored: &mut Vec<(String, NpyArray)>) -> Json {
    let schema = &planes.schema;
    let physical = schema.physical_shape().to_vec();
    stored.push((key.to_owned(), NpyArray::numeric(planes.values, physical.clone())));
    let validity = match planes.mark {
        Mark::Valid => Json::object([("kind", Json::str("valid"))]),
        Mark::Invalid => Json::object([("kind", Json::str("invalid"))]),
        Mark::Plane(mask) => {
            let mask_key = format!("{key}.validity");
            stored.push((mask_key.clone(), NpyArray::numeric(mask.into(), storage_shape(schema))));
            match schema.value().validity() {
                ValidityContract::Value => Json::object([("kind", Json::str("cell")), ("mask_key", Json::Str(mask_key))]),
                ValidityContract::Components(ids) => Json::object([("kind", Json::str("dataset_component")),
                    ("axis_ids", Json::Array(ids.iter().map(|id| Json::str(id.as_ref())).collect())), ("mask_key", Json::Str(mask_key))]),
            }
        }
    };
    let mut manifest = vec![
        ("format", Json::str(FORMAT)), ("schema", schema_tree(schema)),
        ("ref", Json::object([("schema", Json::str(REF_SCHEMA)), ("block_id", Json::str(reference.block_id.as_str())),
                              ("stream_generation", Json::str(reference.stream_generation.as_str())), ("revision", Json::Int(i128::from(reference.revision)))])),
        ("values_key", Json::str(key)), ("validity", validity),
    ];
    if let Some(sigma) = planes.sigma {
        let sigma_key = format!("{key}.sigma");
        stored.push((sigma_key.clone(), NpyArray::numeric(sigma.into(), physical)));
        manifest.push(("sigma_key", Json::Str(sigma_key)));
    }
    if let Some((origins, shapes)) = planes.placements {
        let pairs = |items: Vec<[usize; 2]>| Json::Array(items.into_iter().map(|pair| Json::Array(vec![Json::Int(pair[0] as i128), Json::Int(pair[1] as i128)])).collect());
        manifest.push(("placements", Json::object([("origins", pairs(origins)), ("shapes", pairs(shapes))])));
    }
    Json::object(manifest)
}

/// `snapshot_manifest` of one dataset written as plain members under `key`
/// (a figure overlay's measured status): its manifest and its arrays.
pub fn dataset_manifest(key: &str, dataset: &ArchiveDataset) -> (Json, Vec<(String, NpyArray)>) {
    let mut stored = Vec::new();
    let manifest = snapshot_manifest(key, &dataset.reference, materialize(&dataset.source), &mut stored);
    (manifest, stored)
}

/// `manifest_array_keys(manifest)`.
pub fn manifest_array_keys(manifest: &Json) -> Vec<String> {
    let mut keys = Vec::new();
    match manifest.get("values_key") { Some(Json::Str(key)) => keys.push(key.clone()), Some(other) => keys.push(py_repr(other)), None => {} }
    if let Some(Json::Str(key)) = manifest.get("validity").and_then(|validity| validity.get("mask_key")) { keys.push(key.clone()); }
    if let Some(Json::Str(key)) = manifest.get("sigma_key") { keys.push(key.clone()); }
    keys
}

/// `exact_keys(mapping, expected, path)`.
pub fn exact_keys(value: &Json, expected: &[&str], path: &str) -> Result<()> {
    let actual: Vec<&str> = value.as_object().map(|pairs| pairs.iter().map(|(key, _)| key.as_str()).collect()).unwrap_or_default();
    let missing: Vec<&str> = expected.iter().filter(|key| !actual.contains(key)).copied().collect();
    let extra: Vec<&str> = actual.iter().filter(|key| !expected.contains(key)).copied().collect();
    if !missing.is_empty() || !extra.is_empty() {
        return Err(ArchiveError::npz(format!("{path} keys mismatch; missing={}, extra={}", py_repr_names(&missing), py_repr_names(&extra))));
    }
    Ok(())
}

/// A decoded member by name, as `snapshot_from_manifest`'s `_array` takes it.
fn array<'a>(arrays: &'a Members, key: &Json, referenced: &mut Vec<String>, path: &str) -> Result<&'a NpyArray> {
    let Json::Str(key) = key else { return Err(ArchiveError::npz(format!("{path} must be a non-empty array key"))) };
    if key.is_empty() { return Err(ArchiveError::npz(format!("{path} must be a non-empty array key"))); }
    if referenced.contains(key) { return Err(ArchiveError::npz(format!("{path} contains duplicate array reference {}", py_repr_str(key)))); }
    referenced.push(key.clone());
    arrays.get(key).ok_or_else(|| ArchiveError::npz(format!("missing array {}", py_repr_str(key))))
}

fn shape_repr(shape: &[usize]) -> String {
    match shape.len() { 1 => format!("({},)", shape[0]), _ => format!("({})", shape.iter().map(usize::to_string).collect::<Vec<_>>().join(", ")) }
}

/// A bool member as `immutable_bool_array` admits it.
fn bool_mask(array: &NpyArray) -> Result<OwnedBuffer<bool>> {
    match &array.data {
        NpyData::Numeric(AnyBuffer::Bool(mask)) => Ok(mask.clone()),
        _ => Err(ArchiveError::type_error(format!("validity mask dtype must be bool, got {}", crate::tree::dtype_name(&array.dtype)))),
    }
}

/// What a manifest's validity says, before it meets the schema.
enum Declared { Valid, Invalid, Cell(OwnedBuffer<bool>, Vec<usize>), Component(Vec<String>, OwnedBuffer<bool>, Vec<usize>) }

/// `_validity_from_manifest(raw, arrays, referenced)`.
fn validity_from_manifest(raw: &Json, arrays: &Members, referenced: &mut Vec<String>) -> Result<Declared> {
    let kind = match raw.get("kind") { Some(Json::Str(kind)) if raw.as_object().is_some() => kind.clone(), _ => return Err(ArchiveError::npz("manifest.validity must be a tagged object")) };
    match kind.as_str() {
        "valid" | "invalid" => {
            exact_keys(raw, &["kind"], "manifest.validity")?;
            Ok(if kind == "valid" { Declared::Valid } else { Declared::Invalid })
        }
        "cell" => {
            exact_keys(raw, &["kind", "mask_key"], "manifest.validity")?;
            let mask = array(arrays, raw.get("mask_key").unwrap(), referenced, "validity.mask_key")?;
            if mask.shape.len() != 2 { return Err(ArchiveError::value("CellValidity mask must have shape (R, P)")); }
            Ok(Declared::Cell(bool_mask(mask)?, mask.shape.clone()))
        }
        "dataset_component" => {
            exact_keys(raw, &["kind", "axis_ids", "mask_key"], "manifest.validity")?;
            let ids = match raw.get("axis_ids") {
                Some(Json::Array(items)) if items.iter().all(|item| item.as_str().is_some()) => items.clone(),
                _ => return Err(ArchiveError::npz("manifest.validity.axis_ids must be a string list")),
            };
            let ids = ids.iter().map(|id| canonical_text(id, "AxisId").map(str::to_owned)).collect::<Result<Vec<_>>>()?;
            let mask = array(arrays, raw.get("mask_key").unwrap(), referenced, "validity.mask_key")?;
            if ids.is_empty() { return Err(ArchiveError::value("DatasetComponentValidity requires a named component axis")); }
            if (1..ids.len()).any(|index| ids[..index].contains(&ids[index])) { return Err(ArchiveError::value("DatasetComponentValidity axis_ids must be unique")); }
            if mask.shape.len() != 2 + ids.len() { return Err(ArchiveError::value("DatasetComponentValidity mask rank must be 2 + named-axis count")); }
            Ok(Declared::Component(ids, bool_mask(mask)?, mask.shape.clone()))
        }
        other => Err(ArchiveError::npz(format!("invalid validity kind {}", py_repr_str(other)))),
    }
}

/// `_validate_dataset_validity` then `dataset_validity_storage`: the compact
/// plane over `(R, P, *declared component axes)`.
fn storage_validity(declared: Declared, schema: &DatasetSchema) -> Result<Validity> {
    let leading = [schema.domain(DomainRole::Repeat).size(), schema.domain(DomainRole::Point).size()];
    let cell = schema.domain(DomainRole::Cell);
    let components: Vec<&str> = match schema.value().validity() { ValidityContract::Value => Vec::new(), ValidityContract::Components(ids) => ids.iter().map(|id| id.as_ref()).collect() };
    let component_sizes: Vec<usize> = components.iter().map(|id| cell.axis(id).unwrap().len()).collect();
    // Broadcast a mask over `named` axes to every declared component axis.
    let broadcast = |mask: &OwnedBuffer<bool>, named: &[usize]| -> OwnedBuffer<bool> {
        let total: usize = leading.iter().product::<usize>() * component_sizes.iter().product::<usize>();
        let named_sizes: Vec<usize> = named.iter().map(|position| component_sizes[*position]).collect();
        let values = mask.as_slice();
        let mut out = Vec::with_capacity(total);
        for flat in 0..total {
            let mut rest = flat;
            let mut index = vec![0; component_sizes.len()];
            for axis in (0..component_sizes.len()).rev() { index[axis] = rest % component_sizes[axis]; rest /= component_sizes[axis]; }
            let mut source = rest;
            for (position, size) in named.iter().zip(&named_sizes) { source = source * size + index[*position]; }
            out.push(values[source]);
        }
        OwnedBuffer::from_vec(out)
    };
    match declared {
        Declared::Valid => Ok(Validity::Uniform(true)),
        Declared::Invalid => Ok(Validity::Uniform(false)),
        Declared::Cell(mask, shape) => {
            if shape != leading { return Err(ArchiveError::value(format!("cell validity shape {} does not match dataset cells {}", shape_repr(&shape), shape_repr(&leading)))); }
            Ok(Validity::Mask(if components.is_empty() { mask } else { broadcast(&mask, &[]) }))
        }
        Declared::Component(ids, mask, shape) => {
            if components.is_empty() { return Err(ArchiveError::value("component validity is forbidden by VALUE validity contract")); }
            if ids.iter().any(|id| !components.contains(&id.as_str())) { return Err(ArchiveError::value("component validity uses an axis absent from the schema contract")); }
            let positions = ids.iter().map(|id| cell.axes().iter().position(|axis| axis.id() == id)).collect::<Option<Vec<_>>>()
                .ok_or_else(|| ArchiveError::value("validity axis is absent from the Cell domain"))?;
            if positions.windows(2).any(|pair| pair[0] > pair[1]) { return Err(ArchiveError::value("validity axes must follow Cell-domain axis order")); }
            let expected: Vec<usize> = leading.iter().copied().chain(positions.iter().map(|position| cell.axes()[*position].len())).collect();
            if shape != expected { return Err(ArchiveError::value(format!("component validity shape {} does not match named axes {}", shape_repr(&shape), shape_repr(&expected)))); }
            let named: Vec<usize> = ids.iter().map(|id| components.iter().position(|item| item == id).unwrap()).collect();
            Ok(Validity::Mask(if named.len() == components.len() { mask } else { broadcast(&mask, &named) }))
        }
    }
}

/// `immutable_array(values, dtype=schema dtype, shape=physical shape)`.
fn plane(array: &NpyArray, dtype: DType, shape: &[usize]) -> Result<AnyBuffer> {
    let NpyData::Numeric(values) = &array.data else {
        return Err(ArchiveError::type_error(format!("data values require bool or numeric dtype, got {}", crate::tree::dtype_name(&array.dtype))));
    };
    if values.dtype() != dtype {
        return Err(ArchiveError::type_error(format!("values dtype {} does not match schema dtype {}", crate::tree::dtype_name(&array.dtype),
            crate::tree::dtype_name(&NpyType::native(dtype)))));
    }
    if array.shape != shape { return Err(ArchiveError::value(format!("values shape {} does not match expected {}", shape_repr(&array.shape), shape_repr(shape)))); }
    Ok(values.clone())
}

/// Python truthiness of each value (`np.any(values, where=...)`).
fn nonzero(values: &AnyBuffer, index: usize) -> bool {
    match values {
        AnyBuffer::Bool(v) => v.as_slice()[index], AnyBuffer::U8(v) => v.as_slice()[index] != 0, AnyBuffer::I8(v) => v.as_slice()[index] != 0,
        AnyBuffer::U16(v) => v.as_slice()[index] != 0, AnyBuffer::I16(v) => v.as_slice()[index] != 0, AnyBuffer::U32(v) => v.as_slice()[index] != 0,
        AnyBuffer::I32(v) => v.as_slice()[index] != 0, AnyBuffer::U64(v) => v.as_slice()[index] != 0, AnyBuffer::I64(v) => v.as_slice()[index] != 0,
        AnyBuffer::F16(v) => v.as_slice()[index].to_bits() & 0x7fff != 0, AnyBuffer::F32(v) => v.as_slice()[index] != 0.0,
        AnyBuffer::F64(v) => v.as_slice()[index] != 0.0,
        AnyBuffer::C64(v) => { let c = v.as_slice()[index]; c.re != 0.0 || c.im != 0.0 }
        AnyBuffer::C128(v) => { let c = v.as_slice()[index]; c.re != 0.0 || c.im != 0.0 }
    }
}

/// `with_placements(origins, shapes)` pairs.
fn placement_pairs(value: &Json, name: &str) -> Result<Vec<[i128; 2]>> {
    let refuse = || ArchiveError::type_error(format!("{name} must contain integer Repeat/Point pairs"));
    let Json::Array(items) = value else { return Err(refuse()) };
    let mut pairs = Vec::with_capacity(items.len());
    let mut wide = false;
    for item in items {
        match item.as_array() {
            Some([a, b]) => match (a, b) {
                (Json::Int(a), Json::Int(b)) => pairs.push([*a, *b]),
                (Json::Int(_) | Json::BigInt(_), Json::Int(_) | Json::BigInt(_)) => wide = true,
                _ => return Err(refuse()),
            },
            _ => return Err(refuse()),
        }
    }
    // `np.asarray(pairs)`: each int is int64 or else uint64 or else an
    // object; int64 beside uint64 promotes to float64.
    let values: Vec<i128> = pairs.iter().flatten().copied().collect();
    let unsigned = values.iter().filter(|value| **value > i128::from(i64::MAX)).count();
    let object = wide || values.iter().any(|value| *value < i128::from(i64::MIN) || *value > i128::from(u64::MAX));
    if object || (unsigned > 0 && unsigned < values.len()) { return Err(ArchiveError::type_error(format!("{name} must be an N by 2 integer array"))); }
    if unsigned > 0 { return Err(ArchiveError::value(format!("{name} exceeds int64"))); }
    Ok(pairs)
}

/// One dataset read back: identity, the native snapshot
/// `NativeDataset.from_science` makes of the scientific block, and whether
/// that block is one dense array (a manifest without placements).
#[derive(Clone)]
pub struct ReadDataset { pub reference: DatasetRef, pub snapshot: Snapshot, pub dense: bool }

impl ReadDataset {
    /// The planes as the scientific block holds them: a dense block keeps
    /// its mask when uniform (`snapshot_manifest` writes it as read), placed
    /// segments are materialized and compacted.
    pub fn source(&self) -> Source {
        match self.snapshot.segments().next() {
            Some(placement) if self.dense => Source::Dense(placement.data().clone()),
            _ => Source::of_snapshot(&self.snapshot),
        }
    }
}

/// `snapshot_from_manifest(manifest, arrays)` adopted as a native snapshot.
pub fn snapshot_from_manifest(manifest: &Json, arrays: &Members) -> Result<ReadDataset> {
    if manifest.as_object().is_none() { return Err(ArchiveError::npz("manifest root must be an object")); }
    let mut expected = vec!["format", "schema", "ref", "values_key", "validity"];
    if manifest.get("sigma_key").is_some() { expected.push("sigma_key"); }
    if manifest.get("placements").is_some() { expected.push("placements"); }
    exact_keys(manifest, &expected, "manifest")?;
    let format = manifest.get("format").unwrap();
    if format.as_str() != Some(FORMAT) { return Err(ArchiveError::npz(format!("unsupported data format {}", py_repr(format)))); }
    let mut referenced = Vec::new();
    let schema = schema_from_tree(manifest.get("schema").unwrap())?;
    let reference = manifest.get("ref").unwrap();
    if reference.as_object().is_none() { return Err(ArchiveError::npz("manifest.ref must be an object")); }
    if reference.get("schema_fingerprint").is_some() { return Err(ArchiveError::npz("manifest ref must not repeat schema_fingerprint")); }
    let mut tree = reference.clone();
    tree.insert("schema_fingerprint", Json::str("0".repeat(32)));
    exact_mapping(&tree, &["schema", "block_id", "stream_generation", "schema_fingerprint", "revision"], REF_SCHEMA, Some("schema"))?;
    let block_id = canonical_text(tree.get("block_id").unwrap(), "BlockId")?.to_owned();
    let stream_generation = canonical_text(tree.get("stream_generation").unwrap(), "StreamGenerationId")?.to_owned();
    let revision = integer(tree.get("revision").unwrap(), "DatasetRevision", Some(0))?;
    let revision = u64::try_from(revision).map_err(|_| ArchiveError::new(crate::ErrorKind::Overflow, "DatasetRevision is too large"))?;
    let values = array(arrays, manifest.get("values_key").unwrap(), &mut referenced, "manifest.values_key")?;
    let declared = validity_from_manifest(manifest.get("validity").unwrap(), arrays, &mut referenced)?;
    let sigma = match manifest.get("sigma_key") {
        None => None,
        Some(Json::Str(key)) if !key.is_empty() => Some(array(arrays, manifest.get("sigma_key").unwrap(), &mut referenced, "manifest.sigma_key")?),
        Some(_) => return Err(ArchiveError::npz("manifest.sigma_key must be a non-empty string")),
    };
    // `DataBlock.__post_init__`: validity, values, then sigma.
    let validity = storage_validity(declared, &schema)?;
    let values = plane(values, schema.dtype(), schema.physical_shape())?;
    let sigma = match sigma {
        None => None,
        Some(array) => {
            let AnyBuffer::F64(errors) = plane(array, DType::F64, schema.physical_shape())? else { unreachable!() };
            if errors.as_slice().iter().any(|value| value.is_finite() && *value < 0.0) { return Err(ArchiveError::value("sample sigma must be non-negative")); }
            Some(errors)
        }
    };
    let reference = DatasetRef { block_id, stream_generation, revision };
    let extent = [schema.domain(DomainRole::Repeat).size(), schema.domain(DomainRole::Point).size()];
    let placements = match manifest.get("placements") {
        None => vec![Placement::new([0, 0], extent, DataBlock::new(schema.clone(), values, validity, sigma).map_err(ArchiveError::value)?)
            .map_err(ArchiveError::value)?],
        Some(placements) => {
            if placements.as_object().is_none() { return Err(ArchiveError::npz("manifest.placements must be an object")); }
            exact_keys(placements, &["origins", "shapes"], "manifest.placements")?;
            place(&schema, values, validity, sigma, placements).map_err(|error| ArchiveError::npz(format!("invalid manifest.placements: {}", error.message)))?
        }
    };
    let dense = manifest.get("placements").is_none();
    let snapshot = DatasetStore::new(schema).commit(placements).map_err(ArchiveError::value)?;
    Ok(ReadDataset { reference, snapshot, dense })
}

/// `DataBlock.with_placements` over complete planes, then the segments
/// `NativeDataset.from_science` adopts in their restricted schemas.
fn place(schema: &Arc<DatasetSchema>, values: AnyBuffer, validity: Validity, sigma: Option<OwnedBuffer<f64>>, placements: &Json) -> Result<Vec<Placement>> {
    let origins = placement_pairs(placements.get("origins").unwrap(), "placement origins")?;
    let shapes = placement_pairs(placements.get("shapes").unwrap(), "placement shapes")?;
    if origins.len() != shapes.len() { return Err(ArchiveError::value("placement origins and shapes must have the same length")); }
    let extent = [schema.domain(DomainRole::Repeat).size() as i128, schema.domain(DomainRole::Point).size() as i128];
    let inside = origins.iter().zip(&shapes).all(|(origin, shape)| (0..2).all(|axis| origin[axis] >= 0 && origin[axis] <= extent[axis] && shape[axis] >= 1 && shape[axis] <= extent[axis] - origin[axis]));
    if !inside { return Err(ArchiveError::value("placement must be a positive rectangle inside the dataset")); }
    let (repeats, points) = (extent[0] as usize, extent[1] as usize);
    let mut occupied = vec![false; repeats * points];
    for (origin, shape) in origins.iter().zip(&shapes) {
        for row in origin[0] as usize..(origin[0] + shape[0]) as usize {
            for column in origin[1] as usize..(origin[1] + shape[1]) as usize {
                if occupied[row * points + column] { return Err(ArchiveError::value("data placements must not overlap")); }
                occupied[row * points + column] = true;
            }
        }
    }
    let cells = schema.domain(DomainRole::Cell).size();
    let marks = schema.component_mask_len() / (repeats * points).max(1);
    if occupied.iter().any(|kept| !kept) {
        let holes = |unit: usize| (0..repeats * points).filter(|cell| !occupied[*cell]).flat_map(move |cell| cell * unit..(cell + 1) * unit);
        if holes(cells).any(|index| nonzero(&values, index)) { return Err(ArchiveError::value("unplaced values must contain materialization's zero fill")); }
        let invalid = match &validity { Validity::Uniform(flag) => !*flag, Validity::Mask(mask) => !holes(marks).any(|index| mask.as_slice()[index]) };
        if !invalid { return Err(ArchiveError::value("unplaced samples must be invalid")); }
        if let Some(errors) = &sigma {
            if holes(cells).any(|index| !errors.as_slice()[index].is_nan()) { return Err(ArchiveError::value("unplaced sigma must contain materialization's NaN fill")); }
        }
    }
    if origins.is_empty() && sigma.is_some() { return Err(ArchiveError::value("empty placements cannot carry a sigma plane")); }
    let cell_shape = schema.domain(DomainRole::Cell).physical_shape().to_vec();
    let shape: Vec<usize> = [repeats, points].into_iter().chain(cell_shape.iter().copied()).collect();
    let mask_shape = storage_shape(schema);
    let all_cells: Vec<Indices> = cell_shape.iter().map(|size| Indices::all(*size)).collect();
    let all_marks: Vec<Indices> = mask_shape[2..].iter().map(|size| Indices::all(*size)).collect();
    let mut segments = Vec::with_capacity(origins.len());
    for (origin, size) in origins.iter().zip(&shapes) {
        let origin = [origin[0] as usize, origin[1] as usize];
        let size = [size[0] as usize, size[1] as usize];
        let (rows, columns) = (Indices::Range(origin[0], origin[0] + size[0]), Indices::Range(origin[1], origin[1] + size[1]));
        let picks: Vec<&Indices> = [&rows, &columns].into_iter().chain(all_cells.iter()).collect();
        let mark_picks: Vec<&Indices> = [&rows, &columns].into_iter().chain(all_marks.iter()).collect();
        let chunk = restricted_schema(schema, &SelectedIndices::rows(schema, rows.clone(), columns.clone()))?;
        let segment_validity = match &validity {
            Validity::Uniform(flag) => Validity::Uniform(*flag),
            Validity::Mask(mask) => Validity::Mask(OwnedBuffer::from_vec(gather(mask.as_slice(), &mask_shape, &mark_picks))),
        };
        let segment_sigma = sigma.as_ref().map(|errors| OwnedBuffer::from_vec(gather(errors.as_slice(), &shape, &picks)));
        let data = DataBlock::new(chunk, gather_buffer(&values, &shape, &picks), segment_validity, segment_sigma).map_err(ArchiveError::value)?;
        segments.push(Placement::new(origin, size, data).map_err(ArchiveError::value)?);
    }
    Ok(segments)
}
