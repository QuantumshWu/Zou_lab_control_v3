//! A live image's point layer (v2 `_image_point_overlay`): the site rings a
//! panel's Overlay draws, made from the judgement publication the panel read
//! with its image -- geometry from that run's record, status from its data --
//! or, without an Overlay, the passive rings a producer recorded with the image.
use std::collections::BTreeMap;
use std::sync::Arc;

use serde_json::Value;
use zlc_core::runtime::{Read, RunRecord};
use zlc_core::schema::{DatasetSchema, DomainRole};
use zlc_plot::projection::Source;
use zlc_plot::session::{ImagePointOverlay, OverlayArray, PointStatus};

/// The run/event record key the geometry document travels under.
const GEOMETRY_RECORD: &str = "image_point_overlay_geometry";
const FIELDS: [&str; 8] = ["coordinate_frame", "x_axis_id", "y_axis_id", "point_ids", "coordinates_xy", "labels", "status_axis_id", "status_coordinates"];
const OPTIONAL: [&str; 2] = ["paths_xy", "static_statuses"];

/// What a layer was made from: one judgement read, or one image run's record.
enum Made { Status(Read), Record(Arc<RunRecord>) }

#[derive(Default)]
pub struct LiveOverlays { made: BTreeMap<String, (Made, Arc<ImagePointOverlay>)>, revision: u64 }

impl LiveOverlays {
    /// The layer `panel` draws on `image`: its Overlay's judgement when one
    /// was read with the picture, else the rings the image's run recorded.
    pub fn layer(&mut self, panel: &str, image: &Read, status: Option<&Read>) -> Result<Option<Arc<ImagePointOverlay>>, String> {
        let reused = match (self.made.get(panel), status) {
            (Some((Made::Status(held), layer)), Some(status)) if held.same_read(status) => Some(layer.clone()),
            (Some((Made::Record(held), layer)), None) if Arc::ptr_eq(held, image.anchor().run()) => Some(layer.clone()),
            _ => None,
        };
        if let Some(layer) = reused { return Ok(Some(layer)); }
        self.made.remove(panel);
        let made = match status {
            Some(status) => {
                let geometry = status.anchor().run().metadata.get(GEOMETRY_RECORD).ok_or("The overlay's run records no point geometry")?;
                let layer = self.build(geometry, image.schema(), Some(status))?;
                (Made::Status(status.clone()), layer)
            }
            None => {
                let Some(geometry) = image.anchor().run().metadata.get(GEOMETRY_RECORD).filter(|geometry| matches(geometry, image.schema()))
                    else { return Ok(None) };
                (Made::Record(image.anchor().run().clone()), self.build(geometry, image.schema(), None)?)
            }
        };
        let layer = made.1.clone();
        self.made.insert(panel.to_owned(), made);
        Ok(Some(layer))
    }

    /// Keep only the layers of panels still shown.
    pub fn retain(&mut self, keep: impl Fn(&str) -> bool) { self.made.retain(|panel, _| keep(panel)); }

    fn build(&mut self, geometry: &Value, image: &DatasetSchema, status: Option<&Read>) -> Result<Arc<ImagePointOverlay>, String> {
        let document = geometry.as_object().ok_or("image overlay geometry must be a mapping")?;
        if FIELDS.iter().any(|field| !document.contains_key(*field)) || document.keys().any(|key| !FIELDS.contains(&key.as_str()) && !OPTIONAL.contains(&key.as_str())) {
            return Err("image overlay geometry fields are not canonical".into());
        }
        let texts = |field: &str| document[field].as_array().map(|items| items.iter().map(|item| match item {
            Value::String(text) => text.trim().to_owned(), other => other.to_string() }).collect::<Vec<_>>()).unwrap_or_default();
        let (ids, labels) = (texts("point_ids"), texts("labels"));
        if ids.is_empty() || ids.iter().any(String::is_empty) || labels.len() != ids.len() || labels.iter().any(String::is_empty)
            || ids.iter().enumerate().any(|(index, id)| ids[..index].contains(id)) {
            return Err("image overlay geometry identities are invalid".into());
        }
        let coordinates = numbers(&document["coordinates_xy"], &[ids.len(), 2]).ok_or("image overlay geometry coordinates are invalid")?;
        if !matches(geometry, image) { return Err("image overlay geometry differs from image axes".into()); }
        let paths = match document.get("paths_xy") {
            None | Some(Value::Null) => None,
            Some(paths) => {
                let steps = paths.get(0).and_then(Value::as_array).map_or(0, Vec::len);
                Some(OverlayArray { values: Some(numbers(paths, &[ids.len(), steps, 2]).ok_or("image overlay paths are invalid")?), shape: vec![ids.len(), steps, 2] })
            }
        };
        let source = match status {
            Some(status) => {
                let schema = status.schema();
                if schema.domain(DomainRole::Repeat) != image.domain(DomainRole::Repeat) || schema.domain(DomainRole::Point) != image.domain(DomainRole::Point) {
                    return Err("image overlay status leading geometry differs from image".into());
                }
                let axes = schema.domain(DomainRole::Cell).axes();
                if axes.len() != 1 || document["status_axis_id"].as_str() != Some(axes[0].id()) {
                    return Err("image overlay geometry differs from status axis".into());
                }
                Some(Arc::new(Source::from_read(status).map_err(|error| error.to_string())?))
            }
            None => None,
        };
        let statuses = match (&source, document.get("static_statuses")) {
            (Some(_), _) => None,
            (None, Some(Value::Array(items))) => Some(items.iter().map(|item| PointStatus::parse(item.as_str().unwrap_or("")))
                .collect::<Result<Vec<_>, _>>().map_err(|error| error.message)?),
            (None, _) => Some(vec![PointStatus::Unknown; ids.len()]),
        };
        self.revision += 1;
        let count = ids.len();
        ImagePointOverlay::new(i128::from(self.revision), OverlayArray { values: Some(coordinates), shape: vec![count, 2] }, Some(ids),
            Some(labels.into_iter().map(Some).collect()), statuses, source, paths).map(Arc::new).map_err(|error| error.message)
    }
}

/// `image_point_overlay_geometry_matches`: the geometry names this image's x/y axes.
fn matches(geometry: &Value, image: &DatasetSchema) -> bool {
    let axes = image.domain(DomainRole::Cell).axes();
    let [y, x] = axes else { return false };
    geometry.get("coordinate_frame").map(|frame| frame.as_str()) == Some(x.coordinate_frame())
        && geometry["x_axis_id"].as_str() == Some(x.id()) && geometry["y_axis_id"].as_str() == Some(y.id())
}

/// A finite numeric array of exactly `shape`, flattened row-major.
fn numbers(value: &Value, shape: &[usize]) -> Option<Vec<f64>> {
    fn walk(value: &Value, shape: &[usize], out: &mut Vec<f64>) -> Option<()> {
        match shape.split_first() {
            None => { let number = value.as_f64().filter(|number| number.is_finite())?; out.push(number); Some(()) }
            Some((count, rest)) => {
                let items = value.as_array().filter(|items| items.len() == *count)?;
                for item in items { walk(item, rest, out)?; }
                Some(())
            }
        }
    }
    let mut out = Vec::with_capacity(shape.iter().product());
    walk(value, shape, &mut out).map(|()| out)
}
