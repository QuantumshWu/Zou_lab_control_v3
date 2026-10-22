//! Display-space values the session hands its renderer beside the payload:
//! fitted-model scenes (`_fit_scene`) and the image point layer
//! (`primitives.ImagePointOverlay`).

use std::sync::Arc;

use zlc_core::dataset::Validity;
use zlc_core::schema::DomainRole;
use zlc_core::{AnyBuffer, DType};

use crate::configuration::error::{PlotError, Result};
use crate::fit::FitParameterDisplay;
use crate::projection::nd::half_to_f64;
use crate::projection::output::AxisValue;
use crate::projection::{Source, SourceData};

/// One fitted curve in display coordinates (`FitPolyline`).
#[derive(Clone, Debug, PartialEq)]
pub struct FitPolyline {
    pub x: Arc<[f64]>,
    pub y: Arc<[f64]>,
    /// `"primary"` or a component role.
    pub role: &'static str,
    pub component_index: usize,
}

/// One indivisible centre-and-two-radii glyph in display coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FitEllipseGlyph { pub center_x: f64, pub center_y: f64, pub radius_x: f64, pub radius_y: f64 }

/// One complete display-space fit scene (`FitOverlay`).
#[derive(Clone, Debug, PartialEq)]
pub struct FitOverlay {
    pub polylines: Vec<FitPolyline>,
    pub ellipse_glyph: Option<FitEllipseGlyph>,
    pub success: bool,
    pub formula: String,
    pub parameter_display: Vec<FitParameterDisplay>,
    pub diagnostic: String,
    pub facet_index: Option<usize>,
    pub group_key: Vec<AxisValue>,
    pub headline_parameter: Option<FitParameterDisplay>,
    /// How the two-population question was decided; empty for other models.
    pub evidence: String,
}

impl Default for FitOverlay {
    fn default() -> Self {
        Self { polylines: Vec::new(), ellipse_glyph: None, success: true, formula: String::new(), parameter_display: Vec::new(),
               diagnostic: String::new(), facet_index: None, group_key: Vec::new(), headline_parameter: None, evidence: String::new() }
    }
}

/// A hand-authored point judgement (`PointStatus`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PointStatus { Unknown, Empty, Occupied, Invalid }

impl PointStatus {
    pub const ALL: [PointStatus; 4] = [PointStatus::Unknown, PointStatus::Empty, PointStatus::Occupied, PointStatus::Invalid];
    pub fn value(self) -> &'static str {
        match self { PointStatus::Unknown => "unknown", PointStatus::Empty => "empty", PointStatus::Occupied => "occupied", PointStatus::Invalid => "invalid" }
    }
    /// `PointStatus(text)`.
    pub fn parse(text: &str) -> Result<Self> {
        Self::ALL.into_iter().find(|status| status.value() == text)
            .ok_or_else(|| PlotError::value(format!("{} is not a valid PointStatus", crate::configuration::value::repr_str(text))))
    }
}

/// An array as `np.asarray` gives it to the overlay: its values as float64
/// when its dtype is real numeric (kind `b`, `i`, `u` or `f`), and its shape.
pub struct OverlayArray { pub values: Option<Vec<f64>>, pub shape: Vec<usize> }

/// One immutable revision of canonical x/y annotations for an image.
#[derive(Clone)]
pub struct ImagePointOverlay {
    pub revision: u64,
    /// `(N, 2)` canonical x then y, row-major.
    pub coordinates: Arc<[f64]>,
    pub point_ids: Option<Vec<String>>,
    pub labels: Option<Vec<Option<String>>>,
    pub static_statuses: Option<Vec<PointStatus>>,
    /// The measured EMPTY/OCCUPIED status dataset of a dynamic layer.
    pub status: Option<Arc<Source>>,
    /// `(N, K, 2)` canonical x/y polylines, row-major, with `K`.
    pub paths_xy: Option<(Arc<[f64]>, usize)>,
}

impl ImagePointOverlay {
    /// `ImagePointOverlay(...)`: `__post_init__`'s checks in its order.
    pub fn new(revision: i128, coordinates: OverlayArray, point_ids: Option<Vec<String>>, labels: Option<Vec<Option<String>>>,
               static_statuses: Option<Vec<PointStatus>>, status: Option<Arc<Source>>, paths_xy: Option<OverlayArray>) -> Result<Self> {
        if revision < 0 { return Err(PlotError::value("revision must be at least 0")); }
        let revision = u64::try_from(revision).map_err(|_| PlotError::overflow_error("revision"))?;
        if coordinates.shape.len() != 2 || coordinates.shape[1] != 2 { return Err(PlotError::value("coordinates must have shape (N, 2)")); }
        let coordinates_values = coordinates.values.ok_or_else(|| PlotError::type_error("coordinates must be numeric"))?;
        if !coordinates_values.iter().all(|value| value.is_finite()) { return Err(PlotError::value("coordinates must be finite")); }
        let count = coordinates.shape[0];
        let paths_xy = match paths_xy {
            None => None,
            Some(paths) => {
                if paths.shape.len() != 3 || paths.shape[0] != count || paths.shape[1] < 1 || paths.shape[2] != 2 {
                    return Err(PlotError::value("paths_xy must have shape (N, K, 2), K >= 1"));
                }
                let values = paths.values.ok_or_else(|| PlotError::type_error("paths_xy must be numeric"))?;
                if !values.iter().all(|value| value.is_finite()) { return Err(PlotError::value("paths_xy must be finite")); }
                Some((values.into(), paths.shape[1]))
            }
        };
        if let Some(ids) = &point_ids {
            if ids.len() != count { return Err(PlotError::value("point_ids must have one item per coordinate")); }
            if ids.iter().enumerate().any(|(index, id)| ids[..index].contains(id)) { return Err(PlotError::value("point_ids must be unique")); }
        }
        if labels.as_ref().is_some_and(|labels| labels.len() != count) { return Err(PlotError::value("labels must have one item per coordinate")); }
        if static_statuses.as_ref().is_some_and(|statuses| statuses.len() != count) {
            return Err(PlotError::value("static_statuses must have one item per coordinate"));
        }
        let overlay = ImagePointOverlay { revision, coordinates: coordinates_values.into(), point_ids, labels, static_statuses, status: None, paths_xy };
        match status { Some(status) => overlay.with_status(Some(status)), None => Ok(overlay) }
    }

    pub fn count(&self) -> usize { self.coordinates.len() / 2 }

    /// `dataclasses.replace(overlay, status=status)`: the measured status this
    /// geometry shows, checked against it.
    pub fn with_status(&self, status: Option<Arc<Source>>) -> Result<Self> {
        if let Some(status) = &status {
            if self.static_statuses.is_some() {
                return Err(PlotError::value("static statuses and a dynamic status snapshot are mutually exclusive"));
            }
            validate_dynamic_status(status, self.count())?;
        }
        Ok(ImagePointOverlay { status, ..self.clone() })
    }
}

/// `np.asarray(values, dtype=np.float64)` of a real (kind `b`, `i`, `u`
/// or `f`) array; `None` for a complex one.
pub fn real_values(values: &AnyBuffer) -> Option<Vec<f64>> {
    fn cast<T: Copy>(values: &[T], to: impl Fn(T) -> f64) -> Option<Vec<f64>> { Some(values.iter().map(|value| to(*value)).collect()) }
    match values {
        AnyBuffer::Bool(values) => cast(values.as_slice(), |value| f64::from(u8::from(value))),
        AnyBuffer::U8(values) => cast(values.as_slice(), f64::from), AnyBuffer::I8(values) => cast(values.as_slice(), f64::from),
        AnyBuffer::U16(values) => cast(values.as_slice(), f64::from), AnyBuffer::I16(values) => cast(values.as_slice(), f64::from),
        AnyBuffer::U32(values) => cast(values.as_slice(), f64::from), AnyBuffer::I32(values) => cast(values.as_slice(), f64::from),
        AnyBuffer::U64(values) => cast(values.as_slice(), |value| value as f64), AnyBuffer::I64(values) => cast(values.as_slice(), |value| value as f64),
        AnyBuffer::F16(values) => cast(values.as_slice(), |value| half_to_f64(value.to_bits())),
        AnyBuffer::F32(values) => cast(values.as_slice(), f64::from), AnyBuffer::F64(values) => Some(values.as_slice().to_vec()),
        AnyBuffer::C64(_) | AnyBuffer::C128(_) => None,
    }
}

/// `ImagePointOverlay._validate_dynamic_status`: one complete point axis of
/// bool or real values, each measured one 0 or 1.
fn validate_dynamic_status(status: &Source, count: usize) -> Result<()> {
    let axes = status.schema.domain(DomainRole::Cell).axes();
    if axes.len() != 1 || axes[0].len() != count {
        return Err(PlotError::value(format!("overlay status must declare one complete point axis of {count} items")));
    }
    if matches!(status.schema.dtype(), DType::C64 | DType::C128) { return Err(PlotError::type_error("overlay status values must be bool or real numeric")); }
    let blocks: Vec<&zlc_core::dataset::DataBlock> = match &status.data {
        SourceData::Dense(block) => vec![block],
        SourceData::Segments(entries) => entries.iter().map(|(_, block)| &**block).collect(),
    };
    for block in blocks {
        let values = real_values(block.values()).expect("real dtype");
        let validity = block.validity();
        // The mask is per value, or per (repeat, point) cell of `count` values.
        let valid = |index: usize| match validity {
            Validity::Uniform(flag) => *flag,
            Validity::Mask(mask) => mask.as_slice()[if mask.len() == values.len() { index } else { index / count }],
        };
        if values.iter().enumerate().any(|(index, value)| valid(index) && *value != 0.0 && *value != 1.0) {
            return Err(PlotError::value("numeric overlay status values must be 0 or 1"));
        }
    }
    Ok(())
}

impl std::fmt::Debug for ImagePointOverlay {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImagePointOverlay").field("revision", &self.revision).field("count", &self.count())
            .field("dynamic", &self.status.is_some()).finish()
    }
}
