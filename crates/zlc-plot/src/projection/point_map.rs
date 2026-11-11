//! Project one measurement per SITE coordinate and join it to the geometry
//! belonging to this read. Other axes have already passed through normal Scope.
use super::nd::{Array, DType};
use std::sync::Arc;
use super::output::{ImageData, PointMapData, QuantityArray};
use super::site_geometry::{SiteGeometry, point_ring_radius};
use super::{DataView, ProjectionError, Result};
use crate::kinds::{AxisRef, Reduction};
use crate::session::{ImagePointOverlay, OverlayArray, PointStatus};
use crate::units::resolve_unit;

impl DataView {
    pub(super) fn point_map(&self, site: &AxisRef, reduction: Reduction) -> Result<PointMapData> {
        if self.resolve(site)?.contract.axis.role.as_ref() != crate::configuration::data_contract::SITE {
            return Err(ProjectionError::value("Point Map requires a site axis"));
        }
        let geometry = self.snapshot().site_geometry.resolve(&site.axis_id)?;
        let refs = std::slice::from_ref(site);
        let uncertainty = reduction.statistic() == Reduction::Mean;
        // The dense SEM kernel has no binary16 implementation. The common
        // code-bucket reduction supports those inputs at their decoded values.
        let dense = if self.schema.value.dtype == DType::F16 && uncertainty { None }
            else { self.dense_tensor_projection(refs, reduction, uncertainty)? };
        let (domain, values, mut valid, counts, sigma) = match dense {
            Some(dense) => (dense.domains[0].clone(), dense.values, dense.valid.to_vec(), dense.counts.to_vec(), dense.sem),
            None => {
                let (domains, codes, dimensions) = self.axis_projection(refs)?;
                let domain = domains[0].clone();
                let reduced = self.segmented_axes(&codes, &dimensions, &[domain.size()], reduction.statistic(), uncertainty)?;
                let valid = reduced.counts.iter().zip(&reduced.values).map(|(&n, value)| n > 0 && value.is_finite()).collect();
                (domain, Array::vector(reduced.values), valid, reduced.counts, reduced.sem)
            }
        };
        let values = QuantityArray::new(values, self.value_unit.clone(), self.value_display_unit.clone(), self.value_label());
        for (index, usable) in valid.iter_mut().enumerate() {
            *usable &= values.display.get_f64(index).is_finite();
        }
        let coordinates: Vec<_> = (0..domain.size()).map(|i| domain.canonical.scalar(i)).collect();
        let indices = coordinates.iter().map(|coordinate| geometry.index(coordinate)).collect::<Result<Vec<_>>>()?;
        // A mean over multiple boolean samples is a fraction, and their sum
        // is a count. Both are numerical quantities on a continuous scale.
        let boolean = self.schema.value.dtype == DType::Bool &&
            (matches!(reduction.statistic(), Reduction::First | Reduction::Min | Reduction::Max)
                || counts.iter().all(|&count| count <= 1));
        let (image, overlay, radius) = site_image(self.revision(), self.generation(), &geometry, &indices, &values, &valid)?;
        Ok(PointMapData {
            image, overlay, radius,
            revision: self.revision(), generation: self.generation().clone(), site: site.clone(),
            coordinates, geometry, indices, values,
            valid: Array::vector(valid), sigma: sigma.filter(|values| values.iter().any(|v| v.is_finite())).map(Array::vector),
            boolean,
        })
    }
}

/// Put the measured values in their calibrated apertures. The result has ordinary
/// Image axes and a validity mask; the same plane supplies the heatmap and its heights.
fn site_image(revision: u64, generation: &Arc<str>, geometry: &SiteGeometry, indices: &[usize],
              values: &QuantityArray, valid: &[bool]) -> Result<(ImageData, Arc<ImagePointOverlay>, f64)> {
    let points: Vec<_> = indices.iter().map(|&i| &geometry.points[i]).collect();
    let xy: Vec<_> = points.iter().map(|point| point.xy).collect();
    let mut low = [f64::INFINITY; 2];
    let mut high = [f64::NEG_INFINITY; 2];
    for p in &xy {
        for d in 0..2 { low[d] = low[d].min(p[d]); high[d] = high[d].max(p[d]); }
    }
    if xy.is_empty() { return Err(ProjectionError::view("No sites in the selected data")); }
    let span = (high[0] - low[0]).max(high[1] - low[1]);
    let style = crate::style::PlotStyle::default().artists;
    let radius = point_ring_radius(&xy, style.point_auto_radius_fraction, span.max(1.0) * style.point_single_radius_fraction);
    let margin = (2.5 * radius).max(if span > 0.0 { span * 0.05 } else { 0.5 });
    let longest = span + 2.0 * margin;
    // The sampled plane supplies Image statistics and Height bars. In 2D the
    // overlay paints the original value inside the same path as its ring.
    let pitch = (radius / 6.0).max(longest / 1024.0);
    if !pitch.is_finite() || pitch <= 0.0 { return Err(ProjectionError::view("Site positions have no representable image extent")); }
    let coordinates = |d: usize| -> Vec<f64> {
        let count = ((high[d] - low[d] + 2.0 * margin) / pitch).ceil() as usize;
        (0..count.clamp(2, 1024)).map(|i| low[d] - margin + (i as f64 + 0.5) * pitch).collect()
    };
    let (xs, ys) = (coordinates(0), coordinates(1));
    if xs.windows(2).chain(ys.windows(2)).any(|p| !p[0].is_finite() || p[1] <= p[0]) {
        return Err(ProjectionError::view("Site positions cannot be resolved at this coordinate precision"));
    }
    let (width, height) = (xs.len(), ys.len());
    let mut pixels = vec![f64::NAN; width * height];
    let mut nearest = vec![f64::INFINITY; pixels.len()];
    for (i, &[cx, cy]) in xy.iter().enumerate() {
        let x0 = xs.partition_point(|&x| x < cx - radius);
        let x1 = xs.partition_point(|&x| x <= cx + radius);
        let y0 = ys.partition_point(|&y| y < cy - radius);
        let y1 = ys.partition_point(|&y| y <= cy + radius);
        let sample = if valid[i] { values.canonical.get_f64(i) } else { f64::NAN };
        for row in y0..y1 {
            let dy = (ys[row] - cy) / radius;
            for col in x0..x1 {
                let distance = ((xs[col] - cx) / radius).powi(2) + dy * dy;
                let pixel = row * width + col;
                if distance <= 1.0 && distance < nearest[pixel] {
                    pixels[pixel] = sample;
                    nearest[pixel] = distance;
                }
            }
        }
    }
    let mask = pixels.iter().map(|v| v.is_finite()).collect::<Vec<_>>();
    let pixel_frame = matches!(geometry.frame.as_deref(), Some("image_pixel_xy" | "sensor_pixel_xy"));
    let unit = resolve_unit(if pixel_frame { "pixel" } else { "1" }, None)?;
    let axis = |samples, label: &str| QuantityArray::new(Array::vector(samples), unit.clone(), unit.clone(), label.into());
    let shape = vec![height, width];
    let image = ImageData {
        revision, generation: generation.clone(), x_ref: AxisRef::cell_data(&geometry.x_axis_id).map_err(ProjectionError::value)?,
        y_ref: AxisRef::cell_data(&geometry.y_axis_id).map_err(ProjectionError::value)?, x: axis(xs, "X"), y: axis(ys, "Y"),
        z: QuantityArray::new(Array::from_vec(pixels, shape.clone()), values.canonical_unit.clone(),
                              values.display_unit.clone(), values.label.clone()),
        valid: Array::from_vec(mask, shape),
    };
    let statuses = valid.iter().map(|&ok| if ok { PointStatus::Empty } else { PointStatus::Invalid }).collect();
    let mut overlay = ImagePointOverlay::new(revision.into(), OverlayArray {
        values: Some(xy.into_iter().flatten().collect()), shape: vec![points.len(), 2],
    }, Some(points.iter().map(|p| p.id.clone()).collect()), Some(points.iter().map(|p| Some(p.label.clone())).collect()),
       Some(statuses), None, None).map_err(|e| ProjectionError::view(e.message))?;
    overlay.face_values = Some(valid.iter().enumerate().map(|(i, &ok)|
        if ok { values.display.get_f64(i) } else { f64::NAN }).collect());
    Ok((image, Arc::new(overlay), radius))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::configuration::{default_spec, PlotKind, PlotSchema, PlotSpec};
    use crate::configuration::semantics::projection_scope;
    use crate::projection::plan::{Projection, ProjectionPlan, ScopeTerm, ScopeValue, SpecFacts};
    use crate::projection::output::Payload;
    use serde_json::json;
    use zlc_core::{AnyBuffer, OwnedBuffer};
    use zlc_core::dataset::{DataBlock, Validity};
    use zlc_core::runtime::{Bundle, Declaration, Input, Read, Runtime, Storage};
    use zlc_core::schema::{Axis, Coordinates, DatasetSchema, Domain, ValidityContract, ValueSchema};

    /// Input axes have only C, A from a three-site calibration, in that order.
    fn data(values: AnyBuffer, valid: Validity, sigma: Option<OwnedBuffer<f64>>) -> Read {
        let repeat = Axis::new("r".into(), "Repeat".into(), "repeat".into(),
            Coordinates::Index { origin: 0, len: 2 }, None, None, None, None).unwrap();
        let site = Axis::new("sites".into(), "Site".into(), "site".into(),
            Coordinates::Numeric(OwnedBuffer::from_vec(vec![3_i64, 1]).into()), None, None, None, None).unwrap();
        let schema = DatasetSchema::new(Domain::dense(vec![2], vec![repeat]).unwrap(), Domain::new(vec![1], vec![], vec![]).unwrap(),
            Domain::dense(vec![2], vec![site]).unwrap(),
            ValueSchema::new(values.dtype(), None, None, ValidityContract::Components(vec!["sites".into()])).unwrap()).unwrap();
        let block = DataBlock::new(schema.clone(), values, valid, sigma).unwrap();
        let runtime = Runtime::new();
        let record = json!({"image_point_overlay_geometry":{"status_axis_id":"sites","status_coordinates":[1,2,3],
            "point_ids":["A","B","C"],"labels":["1","2","3"],"coordinates_xy":[[1,10],[1,20],[4,20]],
            "coordinate_frame":"sensor_pixel_xy","x_axis_id":"x","y_axis_id":"y"}});
        let run = runtime.begin("test", vec![Declaration {signal:"signal".into(), schema, storage:Storage::Latest}], record).unwrap();
        runtime.commit(&run, &Bundle {inputs:vec![Input::Event {signal:"signal".into(),ordinal:0,data:block,origin:None,shot_time:None}],
            metadata:json!({}),parents:vec![]}).unwrap();
        Read::from_event(runtime.read_latest(&run, "signal").unwrap()).unwrap()
    }

    fn project(read: &Read, reduction: Reduction) -> PointMapData {
        let schema = PlotSchema::new(read.schema().clone());
        let Some(PlotSpec::PointMap(mut spec)) = default_spec(Some(&schema), PlotKind::PointMap).unwrap() else { panic!("admitted point map") };
        assert_eq!(spec.reduction, Reduction::Mean);
        spec.reduction = reduction;
        let scope = projection_scope(&schema, &PlotSpec::PointMap(spec.clone()), None).unwrap().into_iter().map(|(axis, value)| {
            assert!(matches!(value, crate::configuration::semantics::ScopeCoordinate::Latest));
            ScopeTerm {axis, value:ScopeValue::Latest}
        }).collect();
        let plan = ProjectionPlan { spec:SpecFacts::PointMap{site:spec.site, reduction}, scope,
            display:Default::default(), view_display:None, defaults:Default::default() };
        match Projection::new().project_read(read, &plan).unwrap().payload {
            Payload::PointMap(data) => data,
            _ => panic!("expected point map"),
        }
    }

    #[test]
    fn latest_slice_keeps_calibration_identity_validity_and_stated_sigma() {
        let read = data(OwnedBuffer::from_vec(vec![30.,10.,32.,12.]).into(),
            Validity::Mask(OwnedBuffer::from_vec(vec![true,true,true,false])), Some(OwnedBuffer::from_vec(vec![0.1,0.2,0.3,0.4])));
        let map = project(&read, Reduction::Last);
        assert_eq!(map.indices, [2, 0]);
        assert_eq!(map.geometry.points[map.indices[0]].id, "C");
        assert_eq!(map.values.display.get_f64(0), 32.);
        assert_eq!(map.valid.bools(), &[true,false]);
        assert!((map.sigma.unwrap().get_f64(0) - 0.3).abs() < 1e-12);
    }

    #[test]
    fn binary_readout_remains_discrete_until_reduced_to_a_probability_or_count() {
        let read = data(OwnedBuffer::from_vec(vec![true,false,false,false]).into(), Validity::Uniform(true), None);
        assert!(project(&read, Reduction::Last).boolean);
        assert!(project(&read, Reduction::Max).boolean);
        let probability = project(&read, Reduction::Mean);
        assert!(!probability.boolean);
        assert_eq!(probability.values.display.to_f64(), [0.5,0.0]);
        assert!(!project(&read, Reduction::Sum).boolean);
    }

    #[test]
    fn recipe_preserves_scoped_axes_and_only_a_site_axis_can_position_the_points() {
        use crate::configuration::{Value, updated_spec};
        use crate::configuration::recipe::{decode_spec, encode_spec};
        let read = data(OwnedBuffer::from_vec(vec![3., 1., 3., 1.]).into(), Validity::Uniform(true), None);
        let schema = PlotSchema::new(read.schema().clone());
        let spec = default_spec(Some(&schema), PlotKind::PointMap).unwrap().unwrap();
        assert!(updated_spec(Some(&schema), &spec, "fate:repeat:r", &Value::text("site")).is_err());
        let first_repeat = Value::list([Value::text("scope-value"), Value::Int(0)]);
        let scoped = updated_spec(Some(&schema), &spec, "fate:repeat:r", &first_repeat).unwrap();
        let document = Value::Map(encode_spec(&scoped).unwrap());
        assert_eq!(decode_spec(&document).unwrap(), scoped);
        assert_eq!(scoped.scope().len(), 1);
    }
}
