//! Data-backed figures (`zlc_plot.figure_artifact`, `zlc_science.figure`):
//! one exact plot recipe beside its typed data, archive first, image second.
//!
//! * [`read_document`] decodes a figure archive into the planes to mount, each
//!   panel's image point geometry and measured status, and the canonical
//!   wire target its recipe resolves to, without fitting or drawing.
//! * [`ArchiveSaver`] is the document's [`FigureSaver`]: the provenance and
//!   checked paths first, then the archive of the export session's recipe and
//!   the image it renders.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use zlc_archive::durable::{atomic_write_file, durable_makedirs, resolve};
use zlc_archive::json::{py_repr, py_repr_bytes, Json};
use zlc_archive::manifest::{dataset_manifest, snapshot_from_manifest, ArchiveDataset, DatasetRef, ReadDataset};
use zlc_archive::npdtype::Kind;
use zlc_archive::npy::{NpyArray, NpyData};
use zlc_archive::provenance::capture_figure_provenance;
use zlc_archive::{read_archive, write_figure_file, ArchiveError, Entry, FigureArchive};
use zlc_core::store::Snapshot;
use zlc_core::AnyBuffer;

use crate::configuration::data_contract::PlotSchema;
use crate::configuration::error::{ErrorKind, PlotError, Result};
use crate::configuration::fit::FitLayer;
use crate::configuration::recipe::{decode_plot_recipe, figure_plot_recipe, DecodedRecipe};
use crate::configuration::resolve::{resolve_configuration, ConfigurationUpdates, Defaults, OverlayHandle, ResolveRequest};
use crate::configuration::value::{parse_python_float, py_strip, repr_str, Value, ValueMap};
use crate::document::{DocumentRead, FigureSaver, FigureWrite, SaveRequest};
use crate::projection::{Source, SourceData};
use crate::session::{real_values, ImagePointOverlay, OverlayArray, PointStatus};
use crate::units::UnitRegistry;

const OVERLAY_PREFIX: &str = "data.overlay";

impl From<ArchiveError> for PlotError {
    fn from(error: ArchiveError) -> Self {
        use zlc_archive::ErrorKind as Kind;
        let kind = match error.kind {
            Kind::Value => ErrorKind::Value, Kind::Type => ErrorKind::Type, Kind::Key => ErrorKind::Key, Kind::Index => ErrorKind::Index,
            Kind::Runtime => ErrorKind::Runtime, Kind::Overflow => ErrorKind::Overflow, other => ErrorKind::Other(other.python_name()),
        };
        PlotError::new(kind, error.message)
    }
}

// ------------------------------------------------------------- values

/// A metadata value as the configuration layer holds it (a Python int past
/// 128 bits becomes its nearest float).
fn value_of(json: &Json) -> Value {
    match json {
        Json::Null => Value::None,
        Json::Bool(flag) => Value::Bool(*flag),
        Json::Int(integer) => Value::Int(*integer),
        Json::BigInt(text) => Value::Float(text.parse().unwrap_or(f64::NAN)),
        Json::Float(real) => Value::Float(*real),
        Json::Str(text) => Value::text(text.as_str()),
        Json::Array(items) => Value::list(items.iter().map(value_of)),
        Json::Object(pairs) => Value::Map(ValueMap::from_pairs(pairs.iter().map(|(key, item)| (key.as_str(), value_of(item))))),
    }
}

fn json_of(value: &Value) -> Json {
    match value {
        Value::None => Json::Null,
        Value::Bool(flag) => Json::Bool(*flag),
        Value::Int(integer) => Json::Int(*integer),
        Value::Float(real) => Json::Float(*real),
        Value::Text(text) => Json::str(&**text),
        Value::List(items) => Json::Array(items.iter().map(json_of).collect()),
        Value::Map(map) => Json::Object(map.iter().map(|(key, item)| (key.to_owned(), json_of(item))).collect()),
    }
}

/// `json.dumps(value, allow_nan=False)` parsed back: plain JSON, refusing the
/// first non-finite number in document order.
fn finite_json(value: &Json) -> Result<serde_json::Value> {
    fn check(value: &Json) -> Result<()> {
        match value {
            Json::Float(real) if !real.is_finite() => Err(PlotError::value(format!(
                "Out of range float values are not JSON compliant: {}", crate::pyfmt::repr(*real)))),
            Json::Array(items) => items.iter().try_for_each(check),
            Json::Object(pairs) => pairs.iter().try_for_each(|(_, item)| check(item)),
            _ => Ok(()),
        }
    }
    check(value)?;
    Ok(value.to_serde())
}

/// Python `str(value)`.
fn py_str(value: &Json) -> String { match value { Json::Str(text) => text.clone(), other => py_repr(other) } }

/// Python `int(value)` of a metadata value.
fn py_int(value: &Json) -> Result<i128> {
    let too_large = || PlotError::new(ErrorKind::Overflow, "revision is too large");
    match value {
        Json::Int(integer) => Ok(*integer),
        Json::Bool(flag) => Ok(i128::from(*flag)),
        Json::Float(real) => { let whole = real.trunc(); if whole.abs() < 1.7e38 { Ok(whole as i128) } else { Err(too_large()) } }
        Json::BigInt(_) => Err(too_large()),
        Json::Str(text) => {
            let invalid = || PlotError::value(format!("invalid literal for int() with base 10: {}", repr_str(text)));
            let body = py_strip(text);
            let digits = body.strip_prefix(['+', '-']).unwrap_or(body);
            let bytes = digits.as_bytes();
            let underscores_ok = bytes.iter().enumerate().all(|(index, byte)| *byte != b'_'
                || (index > 0 && bytes[index - 1].is_ascii_digit() && bytes.get(index + 1).is_some_and(u8::is_ascii_digit)));
            if bytes.is_empty() || !underscores_ok || !bytes.iter().all(|byte| byte.is_ascii_digit() || *byte == b'_') { return Err(invalid()); }
            let magnitude: i128 = digits.replace('_', "").parse().map_err(|_| too_large())?;
            Ok(if body.starts_with('-') { -magnitude } else { magnitude })
        }
        other => Err(PlotError::type_error(format!(
            "int() argument must be a string, a bytes-like object or a real number, not {}", repr_str(other.type_name())))),
    }
}

// ------------------------------------------------------------- overlay codec

/// `str(item) for item in array` of a one-dimensional member.
fn member_strings(name: &str, array: &NpyArray) -> Result<Vec<String>> {
    if array.shape.is_empty() { return Err(PlotError::type_error("iteration over a 0-d array")); }
    if array.shape.len() != 1 { return Err(PlotError::value(format!("figure overlay {name} member must be one-dimensional"))); }
    match (&array.data, &array.dtype.kind) {
        (NpyData::Text(_), _) => (0..array.size()).map(|index| array.text(index).map_err(PlotError::from)).collect(),
        (NpyData::Raw(_), Kind::Bytes) => Ok((0..array.size()).map(|index| py_repr_bytes(array.bytes_item(index))).collect()),
        (NpyData::Numeric(values), _) => {
            fn text<T: ToString + Copy>(values: &[T]) -> Vec<String> { values.iter().map(ToString::to_string).collect() }
            match values {
                AnyBuffer::Bool(values) => Ok(values.as_slice().iter().map(|flag| if *flag { "True" } else { "False" }.to_owned()).collect()),
                AnyBuffer::U8(values) => Ok(text(values.as_slice())), AnyBuffer::I8(values) => Ok(text(values.as_slice())),
                AnyBuffer::U16(values) => Ok(text(values.as_slice())), AnyBuffer::I16(values) => Ok(text(values.as_slice())),
                AnyBuffer::U32(values) => Ok(text(values.as_slice())), AnyBuffer::I32(values) => Ok(text(values.as_slice())),
                AnyBuffer::U64(values) => Ok(text(values.as_slice())), AnyBuffer::I64(values) => Ok(text(values.as_slice())),
                AnyBuffer::F64(values) => Ok(values.as_slice().iter().map(|value| crate::pyfmt::repr(*value)).collect()),
                _ => Err(PlotError::type_error(format!("figure overlay {name} member must hold text, integers or float64"))),
            }
        }
        _ => Err(PlotError::type_error(format!("figure overlay {name} member must hold text, integers or float64"))),
    }
}

/// `np.asarray(member, dtype="<f8")`.
fn member_floats(array: &NpyArray) -> Result<Vec<f64>> {
    let parse = |text: &str, shown: String| parse_python_float(text).ok_or_else(|| PlotError::value(format!("could not convert string to float: {shown}")));
    match (&array.data, &array.dtype.kind) {
        (NpyData::Numeric(AnyBuffer::C64(values)), _) => Ok(values.as_slice().iter().map(|value| f64::from(value.re)).collect()),
        (NpyData::Numeric(AnyBuffer::C128(values)), _) => Ok(values.as_slice().iter().map(|value| value.re).collect()),
        (NpyData::Numeric(values), _) => Ok(real_values(values).expect("real dtype")),
        (NpyData::Text(_), _) => (0..array.size()).map(|index| {
            let item = array.text(index).map_err(PlotError::from)?;
            parse(&item, format!("np.str_({})", repr_str(&item)))
        }).collect(),
        (NpyData::Raw(_), Kind::Bytes) => (0..array.size()).map(|index| {
            let item = array.bytes_item(index);
            parse(&String::from_utf8_lossy(item), format!("np.bytes_({})", py_repr_bytes(item)))
        }).collect(),
        _ => Err(PlotError::type_error("figure overlay coordinates must hold numbers or text")),
    }
}

/// `np.asarray(member)` as the overlay checks it.
fn member_array(array: &NpyArray) -> OverlayArray {
    let values = match &array.data { NpyData::Numeric(values) => real_values(values), _ => None };
    OverlayArray { values, shape: array.shape.clone() }
}

/// `_restore_overlay`: the image point geometry a recipe names, with the
/// measured status dataset it carries.
fn restore_overlay(archive: &FigureArchive, value: &Json) -> Result<Option<(ImagePointOverlay, Option<ReadDataset>)>> {
    if value.is_null() { return Ok(None); }
    const ALLOWED: [&str; 7] = ["revision", "coordinates", "point_ids", "labels", "static_statuses", "status", "paths_xy"];
    let fields = value.as_object().filter(|pairs| ["revision", "coordinates"].iter().all(|key| pairs.iter().any(|(name, _)| name == key))
        && pairs.iter().all(|(name, _)| ALLOWED.contains(&name.as_str())));
    if fields.is_none() { return Err(PlotError::value("figure overlay fields differ")); }
    let member = |field: &str| -> Result<&NpyArray> {
        let key = py_str(value.get(field).expect("present field"));
        archive.array(&key).ok_or_else(|| PlotError::key(repr_str(&key)))
    };
    let strings = |field: &str| -> Result<Option<Vec<String>>> {
        if value.get(field).is_none() { return Ok(None); }
        member_strings(field, member(field)?).map(Some)
    };
    let revision = py_int(value.get("revision").expect("present field"))?;
    let coordinates = member("coordinates")?;
    let coordinates = OverlayArray { values: Some(member_floats(coordinates)?), shape: coordinates.shape.clone() };
    let point_ids = strings("point_ids")?;
    let labels = strings("labels")?.map(|labels| labels.into_iter().map(|label| (!label.is_empty()).then_some(label)).collect());
    let static_statuses = strings("static_statuses")?
        .map(|statuses| statuses.iter().map(|status| PointStatus::parse(status)).collect::<Result<Vec<_>>>()).transpose()?;
    let status = match value.get("status") {
        None => None,
        Some(manifest) => Some(snapshot_from_manifest(manifest, &archive.arrays)?),
    };
    let paths = value.get("paths_xy").map(|_| member("paths_xy").map(member_array)).transpose()?;
    let status_source = status.as_ref().map(|dataset| Arc::new(Source::of_archived(dataset)));
    let overlay = ImagePointOverlay::new(revision, coordinates, point_ids, labels, static_statuses, status_source, paths)?;
    Ok(Some((overlay, status)))
}

/// `_overlay_payload(plot_input, "data.overlay")`: the overlay's members and
/// the recipe document naming them.
fn overlay_payload(overlay: Option<&ImagePointOverlay>) -> (Vec<(String, Entry)>, Json) {
    let Some(overlay) = overlay else { return (Vec::new(), Json::Null) };
    let count = overlay.count();
    let key = |name: &str| format!("{OVERLAY_PREFIX}.{name}");
    let floats = |values: &[f64], shape: Vec<usize>| Entry::Array(NpyArray::numeric(zlc_core::OwnedBuffer::from_vec(values.to_vec()).into(), shape));
    let text = |items: Vec<String>| Entry::Array(NpyArray::unicode(items, vec![count]));
    let mut arrays = vec![(key("coordinates"), floats(&overlay.coordinates, vec![count, 2]))];
    let mut document = vec![("revision", Json::Int(i128::from(overlay.revision))), ("coordinates", Json::Str(key("coordinates")))];
    if let Some((paths, vertices)) = &overlay.paths_xy {
        arrays.push((key("paths_xy"), floats(paths, vec![count, *vertices, 2])));
        document.push(("paths_xy", Json::Str(key("paths_xy"))));
    }
    if let Some(ids) = &overlay.point_ids {
        arrays.push((key("point_ids"), text(ids.clone())));
        document.push(("point_ids", Json::Str(key("point_ids"))));
    }
    if let Some(labels) = &overlay.labels {
        arrays.push((key("labels"), text(labels.iter().map(|label| label.clone().unwrap_or_default()).collect())));
        document.push(("labels", Json::Str(key("labels"))));
    }
    if let Some(statuses) = &overlay.static_statuses {
        arrays.push((key("static_statuses"), text(statuses.iter().map(|status| status.value().to_owned()).collect())));
        document.push(("static_statuses", Json::Str(key("static_statuses"))));
    }
    if let Some(status) = &overlay.status {
        let (manifest, stored) = dataset_manifest(&key("status"), &archive_dataset(status));
        arrays.extend(stored.into_iter().map(|(name, array)| (name, Entry::Array(array))));
        document.push(("status", manifest));
    }
    (arrays, Json::object(document))
}

/// A read's planes and identity as the archive writes them.
fn archive_dataset(source: &Source) -> ArchiveDataset {
    ArchiveDataset {
        reference: DatasetRef { block_id: source.block_id.to_string(), stream_generation: source.generation.to_string(), revision: source.revision },
        source: match &source.data {
            SourceData::Dense(block) => zlc_core::restrict::Source::Dense(block.clone()),
            SourceData::Segments(segments) => zlc_core::restrict::Source::Placed {
                schema: source.schema.clone(), segments: segments.clone(),
                window: source.window.map(|(start, latest)| (start, latest as u64)),
            },
        },
    }
}

// ------------------------------------------------------------- reading

/// One dataset of a figure (`read_figure_plot`): its typed data, the image
/// point overlay drawn over it (its status the measured dataset beside it)
/// and the recipe it was drawn with.
pub struct FigurePlot { pub data: ReadDataset, pub overlay: Option<ImagePointOverlay>, pub status: Option<ReadDataset>, pub recipe: DecodedRecipe }

/// `read_figure_plot(info, arrays, datasets, key)`.
pub fn read_figure_plot(archive: &FigureArchive, key: &str) -> Result<FigurePlot> {
    let info = value_of(&archive.info);
    let recipe = figure_plot_recipe(info.as_map().expect("archive metadata is an object"), key)?;
    let overlay = archive.info.get("sections").and_then(|sections| sections.get("plot")).and_then(|plot| plot.get(key))
        .and_then(|recipe| recipe.get("overlay")).expect("a decoded recipe has an overlay field");
    let data = archive.dataset(key).ok_or_else(|| PlotError::key(repr_str(key)))?.clone();
    let (overlay, status) = match restore_overlay(archive, overlay)? { Some((overlay, status)) => (Some(overlay), status), None => (None, None) };
    Ok(FigurePlot { data, overlay, status, recipe })
}

/// One dataset of an opened figure.
pub struct OpenedItem {
    pub key: String,
    /// The typed data to mount (`NativeDataset.from_science(snapshot.block)`).
    pub plane: Snapshot,
    /// The measured status of a dynamic point layer, mounted beside the image.
    pub overlay_plane: Option<Snapshot>,
    /// The passive point geometry (its status lives in `overlay_plane`).
    pub overlay: Option<Arc<ImagePointOverlay>>,
    /// The canonical desktop wire target the recipe resolves to.
    pub target: serde_json::Value,
}

/// `read_document(path)`: an archive's resolved path, metadata and datasets.
pub struct OpenedFigure { pub path: String, pub info: serde_json::Value, pub items: Vec<OpenedItem> }

/// `zlc_science.figure.read_document`: typed data and the existing recipe,
/// without fitting or drawing.
pub fn read_document(path: &str, fit: Arc<dyn FitLayer>, defaults: Arc<Defaults>, registry: Option<Arc<UnitRegistry>>) -> Result<OpenedFigure> {
    let resolved = resolve(Path::new(path))?;
    let archive = read_archive(&resolved)?;
    let mut items = Vec::with_capacity(archive.datasets.len());
    for (key, _) in &archive.datasets {
        let FigurePlot { data, overlay, status, recipe } = read_figure_plot(&archive, key)?;
        // Geometry is passive authoring; the measured status has one data
        // owner, the companion dataset mounted with this image.
        let overlay = overlay.map(|overlay| overlay.with_status(None).map(Arc::new)).transpose()?;
        let configuration = resolve_configuration(ResolveRequest {
            schema: Some(PlotSchema::new(data.snapshot.schema().clone())), spec: Some(recipe.spec),
            defaults: Some(defaults.clone()), fit: Some(fit.clone()), unit_registry: registry.clone(),
            configuration_updates: Some(ConfigurationUpdates {
                parameters: Some(recipe.parameters), size: Some(recipe.size.to_string()), viewport: Some(recipe.viewport),
                classifier_thresholds: Some(recipe.classifier_thresholds), facet_focus: Some(recipe.facet_focus), fit: Some(recipe.fit),
                selectors: Some(recipe.selectors), interaction: Some(recipe.interaction), presentation: Some(recipe.presentation),
                fit_live: Some(true), image_overlay: Some(overlay.clone().map(|overlay| overlay as OverlayHandle)),
                ..Default::default()
            }),
            ..Default::default()
        })?;
        let target = finite_json(&json_of(&Value::Map(configuration.target()?)))?;
        items.push(OpenedItem { key: key.clone(), plane: data.snapshot, overlay_plane: status.map(|status| status.snapshot), overlay, target });
    }
    if items.is_empty() { return Err(PlotError::value("Figure archive contains no datasets")); }
    Ok(OpenedFigure { path: resolved.display().to_string(), info: finite_json(&archive.info)?, items })
}

// ------------------------------------------------------------- saving

/// The document's figure writer over the native archive.
pub struct ArchiveSaver;

/// `_prepare_figure_artifact`'s result: the checked paths, the data and the
/// provenance one archive is written with.
pub struct PreparedFigure {
    image_path: PathBuf,
    archive_path: PathBuf,
    data: ArchiveDataset,
    overlay_arrays: Vec<(String, Entry)>,
    overlay: Json,
    lineage: Json,
    source: Json,
}

/// pathlib's `suffix` of a final component.
fn suffix(name: &str) -> &str { match name.rfind('.') { Some(dot) if dot > 0 && dot + 1 < name.len() => &name[dot..], _ => "" } }

/// pathlib's `with_suffix`.
fn with_suffix(path: &Path, replacement: &str) -> Result<PathBuf> {
    let name = path.file_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_default();
    if name.is_empty() {
        let class = if cfg!(windows) { "WindowsPath" } else { "PosixPath" };
        return Err(PlotError::value(format!("{class}({}) has an empty name", repr_str(&path.to_string_lossy().replace('\\', "/")))));
    }
    let stem = &name[..name.len() - suffix(&name).len()];
    Ok(path.with_file_name(format!("{stem}{replacement}")))
}

impl<R: DocumentRead> FigureSaver<R> for ArchiveSaver {
    fn prepare(&self, request: &SaveRequest<'_, R>) -> Result<Box<dyn FigureWrite>> {
        let tree = zlc_archive::json::loads(&request.read.provenance_json())?;
        let (lineage, source) = capture_figure_provenance((!tree.is_null()).then_some(&tree))?;
        Ok(Box::new(PreparedFigure::new(Path::new(request.path), request.read.source(), request.overlay.as_deref(), lineage, source)?))
    }
}

impl PreparedFigure {
    /// The checked paths and folder, the data and the archive sections.
    fn new(path: &Path, data: &Source, overlay: Option<&ImagePointOverlay>, lineage: Json, source: Json) -> Result<Self> {
        let selected = resolve(path)?;
        let selected_name = selected.file_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_default();
        let image_path = if suffix(&selected_name).is_empty() { with_suffix(&selected, ".png")? } else { selected };
        let image_name = image_path.file_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_default();
        if ![".png", ".pdf", ".svg"].contains(&suffix(&image_name).to_lowercase().as_str()) {
            return Err(PlotError::value("figure image format must be PNG, PDF, or SVG"));
        }
        durable_makedirs(image_path.parent().unwrap_or(&image_path))?;
        let archive_path = with_suffix(&image_path, ".npz")?;
        let (overlay_arrays, overlay) = overlay_payload(overlay);
        Ok(PreparedFigure { image_path, archive_path, data: archive_dataset(data), overlay_arrays, overlay, lineage, source })
    }

    /// The archive of `recipe`, then the image `image(format)` renders (none
    /// without it): the written image and archive.
    fn finish(self, recipe: ValueMap, image: Option<&mut dyn FnMut(&str) -> Result<Vec<u8>>>) -> Result<(Option<PathBuf>, PathBuf)> {
        let PreparedFigure { image_path, archive_path, data, overlay_arrays, overlay, lineage, source } = self;
        let mut recipe = json_of(&Value::Map(recipe));
        recipe.insert("overlay", overlay);
        let mut source = source;
        if let Some(title) = recipe.get("spec").and_then(|spec| spec.get("labels")).and_then(|labels| labels.get("title")).filter(|title| !title.is_null()) {
            if source.get("title").is_none() { source.insert("title", title.clone()); }
        }
        let sections = Json::object([("plot", Json::object([("data", recipe)])), ("lineage", lineage), ("source", source)]);
        let image_name = image_path.file_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_default();
        let mut arrays = vec![("data".to_owned(), Entry::Dataset(data))];
        arrays.extend(overlay_arrays);
        write_figure_file(&archive_path, &image_name, &arrays, &sections)?;
        let Some(image) = image else { return Ok((None, archive_path)) };
        let format = suffix(&image_name).trim_start_matches('.').to_lowercase();
        image(&format).and_then(|bytes| {
            atomic_write_file(&image_path, |mut file| {
                std::io::Write::write_all(&mut file, &bytes).map_err(|error| ArchiveError::io(&error, &image_path.to_string_lossy()))?;
                Ok(file)
            })?;
            Ok(())
        }).map_err(|error| PlotError::runtime(format!(
            "figure archive {} was saved, but image rendering failed: {}", archive_path.display(), error.message)))?;
        Ok((Some(image_path), archive_path))
    }
}

impl FigureWrite for PreparedFigure {
    fn write(self: Box<Self>, recipe: ValueMap, image: &mut dyn FnMut(&str) -> Result<Vec<u8>>) -> Result<Vec<String>> {
        let (image_path, archive_path) = self.finish(recipe, Some(image))?;
        Ok(image_path.into_iter().chain([archive_path]).map(|path| path.display().to_string()).collect())
    }
}

/// What a run asks the core to save (`RunContext.save_figure`, the
/// `run.figure` request): one complete dataset and its point overlay, the
/// recipe `{spec, parameters, size}` with any further recipe parts, and the
/// archive's `source` and `lineage` sections.
pub struct RunFigure {
    pub path: PathBuf,
    pub data: Source,
    pub overlay: Option<ImagePointOverlay>,
    pub recipe: serde_json::Value,
    pub source: serde_json::Value,
    pub lineage: serde_json::Value,
    pub image: bool,
}

/// `save_figure_artifact` without a host: one fresh export session of the
/// recipe gives the archive's recipe and, when `image`, the picture, so the
/// image is what reopening the archive draws. The written image and archive.
pub fn save_run_figure(figure: RunFigure, services: &crate::session::SessionServices) -> Result<(Option<PathBuf>, PathBuf)> {
    let RunFigure { path, data, overlay, recipe, source, lineage, image } = figure;
    let lineage = match Json::from_serde(&lineage) {
        Json::Null => Json::Object(Vec::new()),
        other => other,
    };
    let lineage = if lineage.as_object().is_some_and(|pairs| pairs.is_empty()) {
        Json::object([("root", Json::Null), ("nodes", Json::Array(Vec::new())), ("device_settings", Json::Array(Vec::new()))])
    } else { lineage };
    let source = match Json::from_serde(&source) {
        Json::Null => Json::Object(Vec::new()),
        object @ Json::Object(_) => object,
        _ => return Err(PlotError::type_error("a figure's source must be an object")),
    };
    let overlay = overlay.map(Arc::new);
    let prepared = PreparedFigure::new(&path, &data, overlay.as_deref(), lineage, source)?;
    // The parts a run leaves out are the recipe's defaults (`_export`).
    let mut complete = ValueMap::from_pairs([
        ("viewport", Value::None), ("classifier_thresholds", Value::list([])), ("facet_focus", Value::None),
        ("fit", Value::Map(ValueMap::new())), ("selectors", Value::list([])), ("overlay", Value::None),
        ("interaction", Value::Map(ValueMap::new())), ("presentation", Value::Map(ValueMap::new())),
    ]);
    let Value::Map(given) = value_of(&Json::from_serde(&recipe)) else { return Err(PlotError::type_error("a figure recipe must be an object")) };
    for (name, value) in given.iter() { complete.insert(name, value.clone()); }
    let recipe = decode_plot_recipe(&Value::Map(complete))?;
    let updates = ConfigurationUpdates {
        size: Some(recipe.size.to_string()), image_overlay: Some(overlay.map(|overlay| overlay as OverlayHandle)),
        viewport: Some(recipe.viewport), classifier_thresholds: Some(recipe.classifier_thresholds), facet_focus: Some(recipe.facet_focus),
        selectors: Some(recipe.selectors), fit: Some(recipe.fit), interaction: Some(recipe.interaction), presentation: Some(recipe.presentation),
        ..Default::default()
    };
    let input = crate::session::PlotInput::Dataset(Arc::new(data));
    let mut export = crate::session::PlotSession::for_export(input, recipe.spec, recipe.parameters, updates, services)?;
    let written = export.figure_recipe()?;
    let mut render = |format: &str| export.save_image(format);
    let saved = prepared.finish(written, image.then_some(&mut render as &mut dyn FnMut(&str) -> Result<Vec<u8>>));
    export.close();
    saved
}
