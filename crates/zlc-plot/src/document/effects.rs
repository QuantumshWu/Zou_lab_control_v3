//! What a command leaves for the host besides fronts (`_derivative_effects`,
//! `fit_outputs.fit_parameter_data`): the ROI a panel's selection plans,
//! the fitted parameters as typed datasets, and removals; and the seam the
//! figure archive fills (saved figures).

use std::collections::BTreeMap;
use std::sync::Arc;

use serde_json::{json, Value as Json};
use zlc_core::dataset::{DataBlock, Validity};
use zlc_core::schema::{Axis, CodeBase, CodeMapping, Coordinate, Coordinates, DatasetSchema, Domain, ValidityContract, ValueSchema};
use zlc_core::store::{DatasetStore, Placement, Snapshot};
use zlc_core::{DType, OwnedBuffer};

use crate::configuration::error::{PlotError, Result};
use crate::configuration::selectors::{SelectorKind, SelectorState, SelectorValue};
use crate::configuration::value::ValueMap;
use crate::kinds::AxisDomain;
use crate::projection::model::{PRIMARY_INDEX, SHOT_TIME};
use crate::projection::nd::Scalar;
use crate::projection::output::SelectionSubject;
use crate::projection::Source;
use crate::roi::RoiPlan;
use crate::session::evaluation::FitEvaluation;
use crate::session::projected::{SampleAxis, SnapshotRef};

use super::{lock, DocumentRead, FigureDocument, RoiKey};

/// What a saved figure is made of (`save_figure_artifact`'s arguments).
pub struct SaveRequest<'a, R> {
    pub path: &'a str,
    pub read: &'a R,
    pub overlay: Option<Arc<crate::session::ImagePointOverlay>>,
}

/// Writes a figure archive and its image (`save_figure_artifact`).
pub trait FigureSaver<R>: Send + Sync {
    /// `_prepare_figure_artifact`: the provenance, the checked paths and
    /// folder, before any export session exists.
    fn prepare(&self, request: &SaveRequest<'_, R>) -> Result<Box<dyn FigureWrite>>;
}

/// The write `_prepare_figure_artifact` returns.
pub trait FigureWrite {
    /// The archive of the export session's `recipe`, then the image
    /// `image(format)` renders; the written paths.
    fn write(self: Box<Self>, recipe: ValueMap, image: &mut dyn FnMut(&str) -> Result<Vec<u8>>) -> Result<Vec<String>>;
}

/// One derivative effect of a command.
pub enum Effect<R> {
    Remove { panel: String, output: String },
    Roi { panel: String, source: R, plan: Arc<RoiPlan>, metadata: Json },
    Fit { panel: String, source: R, values: BTreeMap<String, Snapshot>, records: bool, metadata: Json },
    /// The panel's region changed: v2's `panel_selection_document` of it (None: no
    /// region any more), on the read it was drawn on. What it means to the
    /// producer of the panel's signal is the producer's (v2 `selection_patch`).
    Selection { panel: String, source: R, selection: Option<Json> },
}

impl<R> Effect<R> {
    pub fn panel(&self) -> &str {
        match self { Effect::Remove { panel, .. } | Effect::Roi { panel, .. } | Effect::Fit { panel, .. } | Effect::Selection { panel, .. } => panel }
    }
}

/// v2 `PlotSelectionSource._translate` + `panel_selection_document`: one region
/// in the runtime's words. A box whose y names no axis is the x range it spans
/// (the box kept as `drawn`); an x naming no axis is a band on the measured value.
/// A rolling region names no upstream rows: None.
pub fn selection_document(subject: &SelectionSubject, selector: &SelectorState) -> Option<Json> {
    use crate::kinds::PlotKind;
    if subject.plot_kind == PlotKind::Rolling { return None; }
    let range = |axis: &Option<crate::kinds::AxisRef>, bounds: &crate::configuration::selectors::NumericRange, frame: &Option<Arc<str>>| json!({
        "axis": axis.as_ref().map_or("", |axis| axis.axis_id.as_str()), "lower": bounds.low.min(bounds.high), "upper": bounds.low.max(bounds.high),
        "coordinate_frame": frame.as_deref(), "domain": axis.as_ref().map_or("value", |axis| axis.domain.value()),
    });
    let (kind, ranges, drawn) = match &selector.value {
        SelectorValue::Area(area) if subject.y.is_some() => ("area", vec![range(&subject.x, &area.x, &subject.x_coordinate_frame), range(&subject.y, &area.y, &subject.y_coordinate_frame)], Json::Null),
        SelectorValue::Area(area) => ("x_range", vec![range(&subject.x, &area.x, &subject.x_coordinate_frame)],
            json!({"kind": "area", "lower": area.y.low.min(area.y.high), "upper": area.y.low.max(area.y.high)})),
        SelectorValue::XRange(bounds) => ("x_range", vec![range(&subject.x, bounds, &subject.x_coordinate_frame)], Json::Null),
        _ => return None,
    };
    let facets: Vec<Json> = subject.scope.iter().map(|(axis, value)| json!({"axis": axis.axis_id, "value": crate::session::projected::scalar_value(value).to_json(),
        "domain": axis.domain.value()})).collect();
    Some(json!({"plot_kind": subject.plot_kind.value(), "selector_kind": kind, "source_window": subject.source_window, "drawn": drawn,
        "ranges": ranges, "facets": facets}))
}

/// `asdict(subject)` as JSON.
pub fn subject_json(subject: &SelectionSubject) -> Json {
    let axis = |axis: &Option<crate::kinds::AxisRef>| axis.as_ref().map_or(Json::Null, |axis| json!({"domain": axis.domain.value(), "axis_id": axis.axis_id}));
    json!({
        "plot_kind": subject.plot_kind.value(), "x": axis(&subject.x), "y": axis(&subject.y),
        "x_coordinate_frame": subject.x_coordinate_frame.as_deref(), "y_coordinate_frame": subject.y_coordinate_frame.as_deref(),
        "scope": subject.scope.iter().map(|(axis, value)| json!([{"domain": axis.domain.value(), "axis_id": axis.axis_id},
                                                                   crate::session::projected::scalar_value(value).to_json()])).collect::<Vec<_>>(),
        "source_window": subject.source_window,
    })
}

/// `asdict(selector)` as JSON.
pub fn selector_json(state: &SelectorState) -> Json {
    let range = |range: &crate::configuration::selectors::NumericRange| json!({"low": range.low, "high": range.high});
    let value = match &state.value {
        SelectorValue::XRange(value) => range(value),
        SelectorValue::Area(value) => json!({"x": range(&value.x), "y": range(&value.y)}),
        SelectorValue::Crosshair(point) => json!({"x": point.x, "y": point.y}),
        SelectorValue::Threshold(level) => json!(level),
    };
    json!({"kind": state.kind().value(), "value": value, "revision": state.revision, "facet_index": state.facet_index})
}

fn coordinate(value: &Scalar) -> Coordinate {
    match value {
        Scalar::Int(v) => Coordinate::Integer(*v), Scalar::Float(v) => Coordinate::Real(*v), Scalar::Text(v) => Coordinate::Text(v.clone()),
        Scalar::Bool(v) => Coordinate::Integer(i128::from(*v)), Scalar::None => Coordinate::Missing,
    }
}

fn axis(sample: &SampleAxis) -> std::result::Result<Arc<Axis>, String> {
    Axis::new(sample.id.clone(), sample.name.clone(), sample.role.clone(), Coordinates::Values(sample.coordinates.iter().map(coordinate).collect::<Vec<_>>().into()),
              sample.unit.clone(), sample.frame.clone(), sample.labels.clone(), None)
}

/// `fit_outputs._domain(axes)`: one flat explicitly mapped domain.
fn flat_domain(axes: Vec<Arc<Axis>>) -> std::result::Result<Arc<Domain>, String> {
    let size: usize = axes.iter().map(|axis| axis.len()).product();
    let (mut inner, mut outer) = (size, 1usize);
    let mut mappings = Vec::with_capacity(axes.len());
    for axis in &axes {
        inner /= axis.len();
        mappings.push(CodeMapping::new(CodeBase::Range { start: 0, step: 1, len: axis.len() }, inner, outer)?);
        outer *= axis.len();
    }
    Domain::new(vec![size], axes, mappings)
}

pub(crate) fn scalar_domain() -> std::result::Result<Arc<Domain>, String> {
    let axis = Axis::new("zlc_data.scalar".into(), "value".into(), "scalar".into(), Coordinates::Values(vec![Coordinate::Integer(0)].into()),
                         None, None, None, None)?;
    Domain::dense(vec![1], vec![axis])
}

/// `fit_parameter_data(evaluation, source)`: each fitted parameter as a
/// typed dataset over the fit's sample axes (uncertainty as its sigma),
/// and whether the outputs keep a record coordinate.
pub fn fit_parameter_data(evaluation: &FitEvaluation, source: &Source) -> Result<(BTreeMap<String, Snapshot>, bool)> {
    if evaluation.source_ref != SnapshotRef::of(source) { return Err(PlotError::value("fit evaluation does not match the exact source ref")); }
    let failure = |error: String| PlotError::value(error);
    let axes = &evaluation.sample_axes;
    let repeat: Vec<usize> = (0..axes.len()).filter(|index| axes[*index].domain == AxisDomain::Repeat).collect();
    let order: Vec<usize> = repeat.iter().copied().chain((0..axes.len()).filter(|index| axes[*index].domain != AxisDomain::Repeat)).collect();
    let built: Vec<Arc<Axis>> = axes.iter().map(axis).collect::<std::result::Result<_, _>>().map_err(failure)?;
    let repeat_domain = if repeat.is_empty() {
        let fit_repeat = Axis::new("fit.repeat".into(), "repeat".into(), "repeat".into(), Coordinates::Values(vec![Coordinate::Integer(0)].into()),
                                   None, None, None, None).map_err(failure)?;
        flat_domain(vec![fit_repeat]).map_err(failure)?
    } else {
        flat_domain(repeat.iter().map(|index| built[*index].clone()).collect()).map_err(failure)?
    };
    let point_axes: Vec<Arc<Axis>> = order[repeat.len()..].iter().map(|index| built[*index].clone()).collect();
    let record_axis = point_axes.iter().any(|axis| axis.role() == PRIMARY_INDEX || axis.role() == SHOT_TIME);
    let point_domain = flat_domain(point_axes).map_err(failure)?;
    let cell = scalar_domain().map_err(failure)?;
    let shape: Vec<usize> = axes.iter().map(SampleAxis::size).collect();
    let count = evaluation.results.len();
    // `values.reshape(shape).transpose(order)` read in C order.
    let permutation: Vec<usize> = {
        let mut strides = vec![1usize; shape.len()];
        for index in (0..shape.len().saturating_sub(1)).rev() { strides[index] = strides[index + 1] * shape[index + 1]; }
        let permuted: Vec<usize> = order.iter().map(|axis| shape[*axis]).collect();
        (0..count).map(|flat| {
            let mut rest = flat;
            let mut position = vec![0usize; permuted.len()];
            for slot in (0..permuted.len()).rev() { position[slot] = rest % permuted[slot]; rest /= permuted[slot]; }
            order.iter().zip(&position).map(|(axis, at)| at * strides[*axis]).sum()
        }).collect()
    };
    let success = evaluation.success();
    let validity: Vec<bool> = permutation.iter().map(|index| success[*index]).collect();
    let values = evaluation.parameter_values();
    let errors = evaluation.parameter_errors();
    let window = if record_axis { source.window.map(|(start, latest)| (start, latest as u64)) } else { None };
    let mut outputs = BTreeMap::new();
    for ((name, column), (_, sigma)) in values.iter().zip(&errors) {
        let unit = evaluation.parameter_units.iter().find(|(held, _)| held == name).map(|(_, unit)| unit.clone()).unwrap_or_default();
        let value = ValueSchema::new(DType::F64, (!unit.is_empty()).then(|| Arc::from(unit.as_str())), Some(Arc::from(name.as_str())), ValidityContract::Value)
            .map_err(failure)?;
        let schema = DatasetSchema::new(repeat_domain.clone(), point_domain.clone(), cell.clone(), value).map_err(failure)?;
        let block = DataBlock::new(schema.clone(), OwnedBuffer::from_vec(permutation.iter().map(|index| column[*index]).collect::<Vec<f64>>()).into(),
                                   Validity::Mask(OwnedBuffer::from_vec(validity.clone())),
                                   Some(OwnedBuffer::from_vec(permutation.iter().map(|index| sigma[*index]).collect::<Vec<f64>>()))).map_err(failure)?;
        let extent = [repeat_domain.size(), point_domain.size()];
        let mut store = DatasetStore::new(schema);
        let snapshot = store.commit(vec![Placement::new([0, 0], extent, block).map_err(failure)?]).map_err(failure)?
            .with_window(window).map_err(failure)?;
        outputs.insert(name.clone(), snapshot);
    }
    Ok((outputs, record_axis))
}

impl<R: DocumentRead> FigureDocument<R> {
    /// `_derivative_effects`: removals, ROI plans of changed selections,
    /// withdrawn fits and this command's manual fit results.
    pub(crate) fn derivative_effects(&mut self) -> Result<Vec<Effect<R>>> {
        let mut effects: Vec<Effect<R>> = std::mem::take(&mut self.removed_outputs).into_iter()
            .map(|(panel, output)| Effect::Remove { panel, output }).collect();
        let keys: Vec<String> = self.sessions.keys().cloned().collect();
        for key in keys {
            let Some(read) = self.inputs.get(&key).cloned() else { continue };
            let handle = self.sessions.get(&key).cloned().expect("session");
            let (subject, selector) = {
                let session = lock(&handle);
                let subject = session.selection_subject()?;
                // The panel's region: its box, else its x range (v2 keeps one region a panel).
                let selectors = session.configured_selectors()?;
                let selector = [SelectorKind::Area, SelectorKind::XRange].into_iter().find_map(|kind| selectors.iter().find(|item| item.kind() == kind).copied());
                (subject, selector)
            };
            let binding = self.panel_status.get(&key).and_then(|status| status.binding.clone());
            let shown = self.panel_sources.get(&key);
            let shown_pair = (shown.map(|source| source.signal.clone()), shown.map_or(1, |source| source.window));
            let binding_pair = (binding.as_ref().map(|binding| binding.signal.clone()), binding.as_ref().map_or(1, |binding| binding.window));
            if shown_pair != binding_pair { continue; }
            let schema = read.source().schema.clone();
            let identity = selector.as_ref().map(|selector| RoiKey {
                signal: binding_pair.0.clone(), window: binding_pair.1, schema: schema.clone(), subject: subject_json(&subject),
                kind: selector.kind().value().to_owned(), value: selector_json(selector)["value"].clone(),
            });
            if identity.as_ref() == self.roi_keys.get(&key) { continue; }
            self.roi_keys.remove(&key);
            self.status_mut(&key).derivation_error = None;
            effects.push(Effect::Selection { panel: key.clone(), source: read.clone(), selection: selector.as_ref().and_then(|selector| selection_document(&subject, selector)) });
            let (Some(selector), Some(identity)) = (selector, identity) else {
                effects.push(Effect::Remove { panel: key.clone(), output: "roi".into() });
                continue;
            };
            match RoiPlan::for_region(&schema, &subject, &selector) {
                Ok(Some(plan)) => {
                    let plan = Arc::new(plan);
                    let metadata = json!({"operation": "coordinate-roi", "subject": subject_json(&subject), "selector": selector_json(&selector)});
                    effects.push(Effect::Roi { panel: key.clone(), source: read.clone(), plan, metadata });
                    self.roi_keys.insert(key.clone(), identity);
                }
                // v2: a rolling trace's region is the panel's own mark and derives nothing.
                Ok(None) => {
                    effects.push(Effect::Remove { panel: key.clone(), output: "roi".into() });
                    self.roi_keys.insert(key.clone(), identity);
                }
                Err(error) => {
                    effects.push(Effect::Remove { panel: key.clone(), output: "roi".into() });
                    self.roi_keys.insert(key.clone(), identity);
                    self.status_mut(&key).derivation_error = Some(format!("ROI output: {error}"));
                }
            }
        }
        let withdrawn: Vec<String> = self.configurations.iter()
            .filter(|(key, configured)| self.fit_outputs.contains(*key) && configured.configuration.fit_target.get("model")
                .is_none_or(|model| !super::truthy_json(&model.to_json())))
            .map(|(key, _)| key.clone()).collect();
        for key in withdrawn {
            self.fit_outputs.remove(&key);
            effects.push(Effect::Remove { panel: key, output: "fit".into() });
        }
        effects.append(&mut self.manual_fits);
        Ok(effects)
    }

    /// The `save` action: the figure archive and its image.
    pub(crate) fn save(&mut self, key: &str, command: &Json) -> Result<Json> {
        let handle = self.sessions.get(&key.to_owned()).cloned().ok_or_else(|| PlotError::value("There is no completed figure to save yet"))?;
        let saver = self.saver.clone().ok_or_else(|| PlotError::runtime("this document cannot save figures"))?;
        let read = self.inputs.get(&key.to_owned()).cloned().ok_or_else(|| PlotError::key(format!("'{key}'")))?;
        let path = command.get("path").and_then(Json::as_str).ok_or_else(|| PlotError::key("'path'"))?;
        let write = saver.prepare(&SaveRequest { path, read: &read, overlay: lock(&handle).image_overlay() })?;
        let mut export = lock(&handle).export_session(&self.services)?;
        let saved = write.write(export.figure_recipe()?, &mut |format| export.save_image(format))?;
        Ok(Json::Array(saved.into_iter().map(Json::String).collect()))
    }
}

