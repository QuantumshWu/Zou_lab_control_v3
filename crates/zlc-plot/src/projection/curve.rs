//! Curve, image and (curve/image) facet payloads: path selection exactly as
//! `DataView.curve`/`image`/`facet` choose (dense, factored, generic) and
//! the column/plane assembly into public payloads.

use std::sync::Arc;

use super::lattice::{FactoredPlanes, axis_coordinate_labels};
use super::nd::{Array, Plane, Scalar, product};
use super::output::{AxisValue, CurveData, CurveSeries, FacetCell, FacetData, ImageData, Payload, QuantityArray};
use super::view::{DataView, UsedDomain};
use super::{ProjectionError, Result};
use crate::kinds::{AxisRef, Reduction};

/// A facet cell spec (the projection facts of `CurvePlot`/`ImagePlot`/`HistogramPlot`).
#[derive(Clone, Debug, PartialEq)]
pub enum CellSpec {
    Curve { x: AxisRef, group: Option<AxisRef>, reduction: Reduction },
    Image { x: AxisRef, y: AxisRef, reduction: Reduction },
    Histogram { group: Option<AxisRef>, reduced: Vec<AxisRef>, reduction: Reduction },
}

fn unravel(mut flat: usize, sizes: &[usize]) -> Vec<usize> {
    let mut out = vec![0; sizes.len()];
    for axis in (0..sizes.len()).rev() { out[axis] = flat % sizes[axis]; flat /= sizes[axis]; }
    out
}

/// One `(rows, columns)` plane's column `c`.
fn column<T: Copy>(plane: &[T], columns: usize, c: usize) -> Vec<T> { plane.iter().skip(c).step_by(columns).copied().collect() }

/// `np.moveaxis(a.reshape(sizes), -1, 0).reshape(n_last, -1)`.
fn last_axis_first<T: Copy>(plane: &[T], sizes: &[usize]) -> Vec<T> {
    let last = *sizes.last().unwrap();
    let rest = product(&sizes[..sizes.len() - 1]);
    let mut out = Vec::with_capacity(plane.len());
    for x in 0..last { for r in 0..rest { out.push(plane[r * last + x]); } }
    out
}

impl DataView {
    fn key_label(&self, key: &[AxisValue]) -> String {
        if key.is_empty() { self.value_label() } else { key.iter().map(|v| v.label.as_str()).collect::<Vec<_>>().join(", ") }
    }

    /// `_series_from_columns`.
    #[allow(clippy::too_many_arguments)]
    pub fn series_from_columns(&self, x: &QuantityArray, x_labels: &Option<Vec<String>>, group_domains: &[Arc<UsedDomain>],
                               group_sizes: &[usize], y: &[f64], counts: &[i64], sem: Option<&[f64]>,
                               used: Option<&[bool]>, valid: Option<&[bool]>) -> Vec<CurveSeries> {
        let combos = product(group_sizes);
        let sizes: Vec<usize> = if group_sizes.is_empty() { vec![1] } else { group_sizes.to_vec() };
        (0..combos).map(|flat| {
            let indices = unravel(flat, &sizes);
            let key: Vec<AxisValue> = group_domains.iter().zip(&indices).map(|(domain, index)| domain.values()[*index].clone()).collect();
            let mut y_column = column(y, combos, flat);
            let mut counts_column = column(counts, combos, flat);
            let mut sem_column = sem.map(|s| column(s, combos, flat));
            let mut shown_x = x.clone();
            let mut shown_labels = x_labels.clone();
            if let Some(used) = used {
                let used_column = column(used, combos, flat);
                if !used_column.iter().all(|u| *u) {
                    shown_x = x.compress(&used_column);
                    shown_labels = x_labels.as_ref().map(|labels| labels.iter().zip(&used_column).filter(|(_, k)| **k).map(|(l, _)| l.clone()).collect());
                    let keep = |values: &[f64]| -> Vec<f64> { values.iter().zip(&used_column).filter(|(_, k)| **k).map(|(v, _)| *v).collect() };
                    y_column = keep(&y_column);
                    counts_column = counts_column.iter().zip(&used_column).filter(|(_, k)| **k).map(|(v, _)| *v).collect();
                    sem_column = sem_column.map(|s| keep(&s));
                }
            }
            let valid_column: Vec<bool> = match valid {
                Some(valid) => column(valid, combos, flat),
                None => counts_column.iter().zip(&y_column).map(|(c, v)| *c > 0 && v.is_finite()).collect(),
            };
            let label = self.key_label(&key);
            CurveSeries {
                x: shown_x, x_labels: shown_labels,
                y: QuantityArray::new(Array::vector(y_column), self.value_unit.clone(), self.value_display_unit.clone(), self.value_label()),
                valid: Array::vector(valid_column), sem: sem_column.map(Array::vector), group_key: key, label,
            }
        }).collect()
    }

    fn curve_data(&self, x: &AxisRef, groups: &[AxisRef], series: Vec<CurveSeries>) -> CurveData {
        CurveData { revision: self.revision(), generation: self.generation().clone(), x_ref: Some(x.clone()), group_by: groups.to_vec(), series, x_role: None }
    }

    /// `DataView.curve`.
    pub fn curve(&self, x: &AxisRef, groups: &[AxisRef], aggregation: Reduction, uncertainty: bool) -> Result<CurveData> {
        self.validate_curve(x, groups)?;
        let aggregation = aggregation.statistic();
        if uncertainty && aggregation != Reduction::Mean {
            return Err(ProjectionError::value(format!("uncertainty is defined for Reduction.MEAN only, not '{}'", aggregation.value())));
        }
        if let Some(curve) = self.dense_data_curve(x, groups, aggregation, uncertainty)? { return Ok(curve); }
        if let Some(curve) = self.factored_curve(x, groups, aggregation, uncertainty)? { return Ok(curve); }
        self.curve_from_axes(x, groups, aggregation, uncertainty)
    }

    fn dense_data_curve(&self, x: &AxisRef, groups: &[AxisRef], aggregation: Reduction, uncertainty: bool) -> Result<Option<CurveData>> {
        let refs: Vec<AxisRef> = std::iter::once(x.clone()).chain(groups.iter().cloned()).collect();
        let Some(dense) = self.dense_tensor_projection(&refs, aggregation, uncertainty)? else { return Ok(None) };
        let x_domain = &dense.domains[0];
        let group_domains = &dense.domains[1..];
        let group_sizes: Vec<usize> = group_domains.iter().map(|d| d.size()).collect();
        let x_quantity = self.x_quantity(x, x_domain)?;
        let labels = axis_coordinate_labels(self, x, &x_domain.canonical)?;
        let y = dense.values.to_f64();
        let series = self.series_from_columns(&x_quantity, &labels, group_domains, &group_sizes, &y, &dense.counts.to_vec(),
            dense.sem.as_deref(), None, Some(&dense.valid.to_vec()));
        Ok(Some(self.curve_data(x, groups, series)))
    }

    fn curve_from_axes(&self, x: &AxisRef, groups: &[AxisRef], aggregation: Reduction, uncertainty: bool) -> Result<CurveData> {
        let refs: Vec<AxisRef> = groups.iter().cloned().chain(std::iter::once(x.clone())).collect();
        let (domains, codes, dimensions) = self.axis_projection(&refs)?;
        let sizes: Vec<usize> = domains.iter().map(|d| d.size()).collect();
        let reduced = self.segmented_axes(&codes, &dimensions, &sizes, aggregation, uncertainty)?;
        let x_domain = domains.last().unwrap();
        let group_domains = &domains[..domains.len() - 1];
        let group_sizes: Vec<usize> = group_domains.iter().map(|d| d.size()).collect();
        let x_quantity = self.x_quantity(x, x_domain)?;
        let labels = axis_coordinate_labels(self, x, &x_domain.canonical)?;
        let series = self.series_from_columns(&x_quantity, &labels, group_domains, &group_sizes,
            &last_axis_first(&reduced.values, &sizes), &last_axis_first(&reduced.counts, &sizes),
            reduced.sem.as_ref().map(|s| last_axis_first(s, &sizes)).as_deref(),
            Some(&last_axis_first(&reduced.presence, &sizes)), None);
        Ok(self.curve_data(x, groups, series))
    }

    fn factored_curve(&self, x: &AxisRef, groups: &[AxisRef], aggregation: Reduction, uncertainty: bool) -> Result<Option<CurveData>> {
        let x_dimension = self.resolve(x)?.dimension();
        let mut carrier = Vec::new();
        for group in groups { if self.resolve(group)?.dimension() == x_dimension { carrier.push(group.clone()); } }
        let tensor: Vec<AxisRef> = groups.iter().filter(|g| !carrier.contains(g)).cloned().collect();
        let row_refs: Vec<AxisRef> = carrier.iter().cloned().chain(std::iter::once(x.clone())).collect();
        let Some(planes) = self.factored_planes(&row_refs, &tensor, aggregation, uncertainty)? else { return Ok(None) };
        if carrier.is_empty() {
            let series = self.series_from_planes(&planes, &planes.group_domains, &planes.group_sizes, &planes.y_plane,
                &planes.counts_plane, planes.sem_plane.as_deref(), None);
            return Ok(Some(self.curve_data(x, groups, series)));
        }
        let carrier_sizes = &planes.row_sizes[..planes.row_sizes.len() - 1];
        let nx = *planes.row_sizes.last().unwrap();
        let tensor_sizes = &planes.group_sizes;
        let internal: Vec<&AxisRef> = carrier.iter().chain(tensor.iter()).collect();
        let internal_sizes: Vec<usize> = carrier_sizes.iter().chain(tensor_sizes.iter()).copied().collect();
        let permutation: Vec<usize> = groups.iter().map(|g| internal.iter().position(|i| *i == g).unwrap()).collect();
        let out_sizes: Vec<usize> = permutation.iter().map(|p| internal_sizes[*p]).collect();
        let carriers = carrier_sizes.len();
        // Element (x, groups...) of the reordered columns from the (carrier.., x, tensor..) plane.
        let reorder = |plane_at: &dyn Fn(usize) -> usize| -> Vec<usize> {
            let combos = product(&out_sizes);
            let mut positions = Vec::with_capacity(nx * combos);
            for xi in 0..nx {
                for flat in 0..combos {
                    let out_index = unravel(flat, &out_sizes);
                    let mut internal_index = vec![0usize; internal.len()];
                    for (o, p) in permutation.iter().enumerate() { internal_index[*p] = out_index[o]; }
                    let mut source = 0usize;
                    for (axis, size) in carrier_sizes.iter().enumerate() { source = source * size + internal_index[axis]; }
                    source = source * nx + xi;
                    for (axis, size) in tensor_sizes.iter().enumerate() { source = source * size + internal_index[carriers + axis]; }
                    positions.push(plane_at(source));
                }
            }
            positions
        };
        let tensor_combos = product(tensor_sizes);
        let positions = reorder(&|source| source);
        let presence_positions = reorder(&|source| source / tensor_combos);
        let gather_f = |plane: &[f64]| -> Vec<f64> { positions.iter().map(|p| plane[*p]).collect() };
        let used: Vec<bool> = presence_positions.iter().map(|p| planes.row_presence[*p]).collect();
        let mut domain_by_group: Vec<(&AxisRef, Arc<UsedDomain>)> = carrier.iter().zip(&planes.row_domains[..carriers]).map(|(g, d)| (g, d.clone())).collect();
        domain_by_group.extend(tensor.iter().zip(&planes.group_domains).map(|(g, d)| (g, d.clone())));
        let group_domains: Vec<Arc<UsedDomain>> = groups.iter().map(|g| domain_by_group.iter().find(|(k, _)| *k == g).unwrap().1.clone()).collect();
        let group_sizes: Vec<usize> = group_domains.iter().map(|d| d.size()).collect();
        let series = self.series_from_columns(&planes.x_quantity, &planes.x_labels, &group_domains, &group_sizes,
            &gather_f(&planes.y_plane), &positions.iter().map(|p| planes.counts_plane[*p]).collect::<Vec<_>>(),
            planes.sem_plane.as_ref().map(|s| gather_f(s)).as_deref(), Some(&used), None);
        Ok(Some(self.curve_data(x, groups, series)))
    }

    #[allow(clippy::too_many_arguments)]
    fn series_from_planes(&self, planes: &FactoredPlanes, group_domains: &[Arc<UsedDomain>], group_sizes: &[usize],
                          y: &[f64], counts: &[i64], sem: Option<&[f64]>, x: Option<(QuantityArray, Option<Vec<String>>)>) -> Vec<CurveSeries> {
        let (x_quantity, x_labels) = x.unwrap_or_else(|| (planes.x_quantity.clone(), planes.x_labels.clone()));
        self.series_from_columns(&x_quantity, &x_labels, group_domains, group_sizes, y, counts, sem, None, None)
    }

    // ------------------------------------------------------------- image
    /// `DataView.image`.
    pub fn image(&self, x: &AxisRef, y: &AxisRef, aggregation: Reduction) -> Result<ImageData> {
        self.validate_image(x, y)?;
        let aggregation = aggregation.statistic();
        if let Some(dense) = self.dense_tensor_projection(&[y.clone(), x.clone()], aggregation, false)? {
            return self.image_from_planes(x, y, &dense.domains[1], &dense.domains[0], dense.values, dense.counts, Some(dense.valid), None, None);
        }
        if let Some(planes) = self.factored_planes(&[y.clone(), x.clone()], &[], aggregation, false)? {
            let shape = planes.row_sizes.clone();
            return self.image_from_planes(x, y, &planes.row_domains[1], &planes.row_domains[0],
                Array::from_vec(planes.y_plane, shape), Plane::Values(planes.counts_plane), None, None, None);
        }
        let (domains, codes, dimensions) = self.axis_projection(&[y.clone(), x.clone()])?;
        let sizes: Vec<usize> = domains.iter().map(|d| d.size()).collect();
        let reduced = self.segmented_axes(&codes, &dimensions, &sizes, aggregation, false)?;
        self.image_from_planes(x, y, &domains[1], &domains[0], Array::from_vec(reduced.values, sizes), Plane::Values(reduced.counts), None, None, None)
    }

    /// `_image_from_planes`.
    #[allow(clippy::too_many_arguments)]
    pub fn image_from_planes(&self, x: &AxisRef, y: &AxisRef, x_domain: &UsedDomain, y_domain: &UsedDomain, z: Array,
                             counts: Plane<i64>, valid: Option<Plane<bool>>, used_y: Option<&[bool]>, used_x: Option<&[bool]>) -> Result<ImageData> {
        let (ny, nx) = (y_domain.size(), x_domain.size());
        let mut z = z.reshape(vec![ny, nx]);
        let compressing = used_x.is_some_and(|u| !u.iter().all(|k| *k)) || used_y.is_some_and(|u| !u.iter().all(|k| *k));
        let (mut counts, mut valid) = if compressing { (Plane::Values(counts.to_vec()), valid.map(|v| Plane::Values(v.to_vec()))) } else { (counts, valid) };
        let (mut x_canonical, mut x_display) = (x_domain.canonical.clone(), x_domain.display.clone());
        let (mut y_canonical, mut y_display) = (y_domain.canonical.clone(), y_domain.display.clone());
        let (mut rows, mut columns) = (ny, nx);
        if let Some(used) = used_x.filter(|u| !u.iter().all(|k| *k)) {
            x_canonical = x_canonical.compress(used);
            x_display = x_display.compress(used);
            let keep: Vec<usize> = (0..rows * columns).filter(|p| used[p % columns]).collect();
            z = z.take_flat(&keep, vec![rows, x_canonical.len()]);
            counts = Plane::Values(keep.iter().map(|p| counts.get(*p)).collect());
            valid = valid.map(|v| Plane::Values(keep.iter().map(|p| v.get(*p)).collect()));
            columns = x_canonical.len();
        }
        if let Some(used) = used_y.filter(|u| !u.iter().all(|k| *k)) {
            y_canonical = y_canonical.compress(used);
            y_display = y_display.compress(used);
            let keep: Vec<usize> = (0..rows * columns).filter(|p| used[p / columns]).collect();
            rows = y_canonical.len();
            z = z.take_flat(&keep, vec![rows, columns]);
            counts = Plane::Values(keep.iter().map(|p| counts.get(*p)).collect());
            valid = valid.map(|v| Plane::Values(keep.iter().map(|p| v.get(*p)).collect()));
        }
        let shape = vec![rows, columns];
        let valid = match valid {
            Some(Plane::Uniform(v, _)) => Array::constant(v, shape),
            Some(Plane::Values(v)) => Array::from_vec(v, shape),
            None if counts.uniform().is_some_and(|c| c > 0) && z.all_finite() => Array::constant(true, shape),
            None => { let finite = z.finite_mask(); Array::from_vec((0..finite.len()).map(|i| counts.get(i) > 0 && finite[i]).collect(), shape) }
        };
        let xr = self.resolve(x)?;
        let yr = self.resolve(y)?;
        Ok(ImageData {
            revision: self.revision(), generation: self.generation().clone(), x_ref: x.clone(), y_ref: y.clone(),
            x: QuantityArray::with_display(x_canonical, x_display, xr.canonical_unit.clone(), xr.display_unit.clone(), xr.label.to_string()),
            y: QuantityArray::with_display(y_canonical, y_display, yr.canonical_unit.clone(), yr.display_unit.clone(), yr.label.to_string()),
            z: QuantityArray::new(z, self.value_unit.clone(), self.value_display_unit.clone(), self.value_label()),
            valid,
        })
    }

    // ------------------------------------------------------------- facet
    /// `validate_facet`.
    pub fn validate_facet(&self, facet: Option<&AxisRef>, cell: &CellSpec) -> Result<()> {
        if let Some(facet) = facet { self.resolve(facet)?; }
        match cell {
            CellSpec::Curve { x, group, .. } => self.validate_curve(x, group.as_slice()),
            CellSpec::Image { x, y, .. } => self.validate_image(x, y),
            CellSpec::Histogram { group, reduced, .. } => {
                if let Some(group) = group { self.resolve(group)?; }
                for axis in reduced { self.resolve(axis)?; }
                if facet.is_some_and(|f| reduced.contains(f)) {
                    return Err(ProjectionError::view("a facet axis cannot also be reduced: the cells it names would be collapsed into one"));
                }
                Ok(())
            }
        }
    }

    fn facet_cell(index: usize, value: &AxisValue, payload: Payload) -> FacetCell {
        FacetCell { facet_index: Some(index), facet_value_canonical: value.canonical.clone(), facet_value_display: value.display.clone(),
            label: value.label.clone(), payload }
    }
    fn facet_data(&self, cells: Vec<FacetCell>) -> FacetData { FacetData { revision: self.revision(), generation: self.generation().clone(), cells } }

    /// `DataView.facet` for curve and image cells (histogram cells: `histogram_facet`).
    pub fn facet(&self, facet: Option<&AxisRef>, cell: &CellSpec, uncertainty: bool) -> Result<FacetData> {
        self.validate_facet(facet, cell)?;
        let reduction = match cell { CellSpec::Curve { reduction, .. } | CellSpec::Image { reduction, .. } | CellSpec::Histogram { reduction, .. } => reduction.statistic() };
        if uncertainty && !matches!(cell, CellSpec::Curve { .. }) { return Err(ProjectionError::value("uncertainty is accepted only for Curve facet cells")); }
        if uncertainty && reduction != Reduction::Mean {
            return Err(ProjectionError::value(format!("uncertainty is defined for Reduction.MEAN only, not '{}'", reduction.value())));
        }
        let Some(facet) = facet else {
            let payload = match cell {
                CellSpec::Curve { x, group, reduction } => Payload::Curve(self.curve(x, group.as_slice(), *reduction, uncertainty)?),
                CellSpec::Image { x, y, reduction } => Payload::Image(self.image(x, y, *reduction)?),
                CellSpec::Histogram { .. } => unreachable!("histogram cells take histogram_facet"),
            };
            return Ok(self.facet_data(vec![FacetCell { facet_index: Some(0), facet_value_canonical: Scalar::Int(1),
                facet_value_display: Scalar::Int(1), label: "Facet 1".into(), payload }]));
        };
        if let Some(dense) = self.dense_tensor_facet(facet, cell, uncertainty)? { return Ok(dense); }
        match cell {
            CellSpec::Image { x, y, reduction } => self.factored_facet_images(facet, x, y, *reduction),
            CellSpec::Curve { x, group, reduction } => self.factored_facet_curves(facet, x, group.as_slice(), *reduction, uncertainty),
            CellSpec::Histogram { .. } => unreachable!(),
        }
    }

    fn dense_tensor_facet(&self, facet: &AxisRef, cell: &CellSpec, uncertainty: bool) -> Result<Option<FacetData>> {
        let (refs, reduction, curve_uncertainty) = match cell {
            CellSpec::Curve { x, group, reduction } =>
                (std::iter::once(facet.clone()).chain(std::iter::once(x.clone())).chain(group.iter().cloned()).collect::<Vec<_>>(), *reduction, uncertainty),
            CellSpec::Image { x, y, reduction } => (vec![facet.clone(), y.clone(), x.clone()], *reduction, false),
            CellSpec::Histogram { .. } => return Ok(None),
        };
        let Some(dense) = self.dense_tensor_projection(&refs, reduction, curve_uncertainty)? else { return Ok(None) };
        let facet_domain = &dense.domains[0];
        let facet_size = facet_domain.size();
        let per = dense.values.len() / facet_size.max(1);
        let mut cells = Vec::with_capacity(facet_size);
        match cell {
            CellSpec::Curve { x, group, .. } => {
                let x_domain = &dense.domains[1];
                let group_domains = &dense.domains[2..];
                let group_sizes: Vec<usize> = group_domains.iter().map(|d| d.size()).collect();
                let x_quantity = self.x_quantity(x, x_domain)?;
                let labels = axis_coordinate_labels(self, x, &x_domain.canonical)?;
                let y = dense.values.to_f64();
                for (index, value) in facet_domain.values().iter().enumerate() {
                    let range = index * per..(index + 1) * per;
                    let series = self.series_from_columns(&x_quantity, &labels, group_domains, &group_sizes, &y[range.clone()],
                        &dense.counts.range(range.start, range.end).to_vec(), dense.sem.as_ref().map(|s| &s[range.clone()]), None, Some(&dense.valid.range(range.start, range.end).to_vec()));
                    cells.push(Self::facet_cell(index, value, Payload::Curve(self.curve_data(x, group.as_slice(), series))));
                }
            }
            CellSpec::Image { x, y, .. } => {
                let (y_domain, x_domain) = (&dense.domains[1], &dense.domains[2]);
                let planes = dense.values.reshape(vec![facet_size, per]);
                for (index, value) in facet_domain.values().iter().enumerate() {
                    let range = index * per..(index + 1) * per;
                    let image = self.image_from_planes(x, y, x_domain, y_domain, planes.index0(index), dense.counts.range(range.start, range.end), Some(dense.valid.range(range.start, range.end)), None, None)?;
                    cells.push(Self::facet_cell(index, value, Payload::Image(image)));
                }
            }
            CellSpec::Histogram { .. } => unreachable!(),
        }
        Ok(Some(self.facet_data(cells)))
    }

    fn factored_facet_curves(&self, facet: &AxisRef, x: &AxisRef, cell_groups: &[AxisRef], reduction: Reduction, uncertainty: bool) -> Result<FacetData> {
        let row_facet = self.resolve(facet)?.dimension() == self.resolve(x)?.dimension();
        let planes = if row_facet { self.factored_planes(&[facet.clone(), x.clone()], cell_groups, reduction, uncertainty)? }
            else { self.factored_planes(&[x.clone()], &[&[facet.clone()], cell_groups].concat(), reduction, uncertainty)? };
        let Some(planes) = planes else {
            let groups: Vec<AxisRef> = std::iter::once(facet.clone()).chain(cell_groups.iter().cloned()).collect();
            let combined = self.curve_from_axes(x, &groups, reduction, uncertainty)?;
            return Ok(self.curve_groups_to_facets(x, cell_groups, combined));
        };
        let (facet_domain, facet_size, cell_domains, cell_sizes) = if row_facet {
            (planes.row_domains[0].clone(), planes.row_sizes[0], planes.group_domains.clone(), planes.group_sizes.clone())
        } else {
            (planes.group_domains[0].clone(), planes.group_sizes[0], planes.group_domains[1..].to_vec(), planes.group_sizes[1..].to_vec())
        };
        let cell_combos = product(&cell_sizes);
        let combos = planes.combos;
        let mut cells = Vec::new();
        for facet_index in 0..facet_size {
            let series = if row_facet {
                let nx = planes.row_sizes[1];
                let used = &planes.row_presence[facet_index * nx..(facet_index + 1) * nx];
                if !used.iter().any(|u| *u) { continue; }
                let rows: Vec<usize> = (0..nx).filter(|i| used[*i]).map(|i| facet_index * nx + i).collect();
                let take_rows_f = |plane: &[f64]| -> Vec<f64> { rows.iter().flat_map(|r| plane[r * combos..(r + 1) * combos].iter().copied()).collect() };
                let counts: Vec<i64> = rows.iter().flat_map(|r| planes.counts_plane[r * combos..(r + 1) * combos].iter().copied()).collect();
                let x_quantity = planes.x_quantity.compress(used);
                let x_labels = planes.x_labels.as_ref().map(|labels| labels.iter().zip(used).filter(|(_, k)| **k).map(|(l, _)| l.clone()).collect());
                self.series_from_planes(&planes, &cell_domains, &cell_sizes, &take_rows_f(&planes.y_plane), &counts,
                    planes.sem_plane.as_ref().map(|s| take_rows_f(s)).as_deref(), Some((x_quantity, x_labels)))
            } else {
                let window = facet_index * cell_combos..(facet_index + 1) * cell_combos;
                let rows = planes.y_plane.len() / combos;
                let take_cols = |plane: &[f64]| -> Vec<f64> { (0..rows).flat_map(|r| plane[r * combos + window.start..r * combos + window.end].iter().copied()).collect() };
                let counts: Vec<i64> = (0..rows).flat_map(|r| planes.counts_plane[r * combos + window.start..r * combos + window.end].iter().copied()).collect();
                self.series_from_planes(&planes, &cell_domains, &cell_sizes, &take_cols(&planes.y_plane), &counts,
                    planes.sem_plane.as_ref().map(|s| take_cols(s)).as_deref(), None)
            };
            let value = facet_domain.values()[facet_index].clone();
            let index = cells.len();
            cells.push(Self::facet_cell(index, &value, Payload::Curve(self.curve_data(x, cell_groups, series))));
        }
        Ok(self.facet_data(cells))
    }

    fn curve_groups_to_facets(&self, x: &AxisRef, cell_groups: &[AxisRef], combined: CurveData) -> FacetData {
        let mut grouped: Vec<(AxisValue, Vec<CurveSeries>)> = Vec::new();
        for series in combined.series {
            let facet_value = series.group_key[0].clone();
            if grouped.last().is_none_or(|(value, _)| *value != facet_value) { grouped.push((facet_value, Vec::new())); }
            let key: Vec<AxisValue> = series.group_key[1..].to_vec();
            let label = self.key_label(&key);
            grouped.last_mut().unwrap().1.push(CurveSeries { group_key: key, label, ..series });
        }
        let cells = grouped.into_iter().enumerate().map(|(index, (value, series))|
            Self::facet_cell(index, &value, Payload::Curve(self.curve_data(x, cell_groups, series)))).collect();
        self.facet_data(cells)
    }

    fn factored_facet_images(&self, facet: &AxisRef, x: &AxisRef, y: &AxisRef, reduction: Reduction) -> Result<FacetData> {
        let facet_dimension = self.resolve(facet)?.dimension();
        let row_facet = facet_dimension == self.resolve(x)?.dimension() && facet_dimension == self.resolve(y)?.dimension();
        let planes = if row_facet { self.factored_planes(&[facet.clone(), y.clone(), x.clone()], &[], reduction, false)? }
            else { self.factored_planes(&[y.clone(), x.clone()], &[facet.clone()], reduction, false)? };
        let mut cells = Vec::new();
        let Some(planes) = planes else {
            let refs = [facet.clone(), y.clone(), x.clone()];
            let (domains, codes, dimensions) = self.axis_projection(&refs)?;
            let sizes: Vec<usize> = domains.iter().map(|d| d.size()).collect();
            let reduced = self.segmented_axes(&codes, &dimensions, &sizes, reduction, false)?;
            let (ny, nx) = (sizes[1], sizes[2]);
            for (facet_index, value) in domains[0].values().iter().enumerate() {
                let window = facet_index * ny * nx..(facet_index + 1) * ny * nx;
                let geometry = &reduced.presence[window.clone()];
                if !geometry.iter().any(|g| *g) { continue; }
                let used_y: Vec<bool> = (0..ny).map(|r| (0..nx).any(|c| geometry[r * nx + c])).collect();
                let used_x: Vec<bool> = (0..nx).map(|c| (0..ny).any(|r| geometry[r * nx + c])).collect();
                let image = self.image_from_planes(x, y, &domains[2], &domains[1], Array::vector(reduced.values[window.clone()].to_vec()),
                    Plane::Values(reduced.counts[window].to_vec()), None, Some(&used_y), Some(&used_x))?;
                let index = cells.len();
                cells.push(Self::facet_cell(index, value, Payload::Image(image)));
            }
            return Ok(self.facet_data(cells));
        };
        let (facet_domain, facet_size, y_domain, x_domain) = if row_facet {
            (planes.row_domains[0].clone(), planes.row_sizes[0], planes.row_domains[1].clone(), planes.row_domains[2].clone())
        } else {
            (planes.group_domains[0].clone(), planes.group_sizes[0], planes.row_domains[0].clone(), planes.row_domains[1].clone())
        };
        let (ny, nx) = (y_domain.size(), x_domain.size());
        for facet_index in 0..facet_size {
            let image = if row_facet {
                let window = facet_index * ny * nx..(facet_index + 1) * ny * nx;
                let presence = &planes.row_presence[window.clone()];
                if !presence.iter().any(|p| *p) { continue; }
                let used_y: Vec<bool> = (0..ny).map(|r| (0..nx).any(|c| presence[r * nx + c])).collect();
                let used_x: Vec<bool> = (0..nx).map(|c| (0..ny).any(|r| presence[r * nx + c])).collect();
                self.image_from_planes(x, y, &x_domain, &y_domain, Array::vector(planes.y_plane[window.clone()].to_vec()),
                    Plane::Values(planes.counts_plane[window].to_vec()), None, Some(&used_y), Some(&used_x))?
            } else {
                let combos = planes.combos;
                let z: Vec<f64> = (0..ny * nx).map(|r| planes.y_plane[r * combos + facet_index]).collect();
                let counts: Vec<i64> = (0..ny * nx).map(|r| planes.counts_plane[r * combos + facet_index]).collect();
                self.image_from_planes(x, y, &x_domain, &y_domain, Array::vector(z), Plane::Values(counts), None, None, None)?
            };
            let value = facet_domain.values()[facet_index].clone();
            let index = cells.len();
            cells.push(Self::facet_cell(index, &value, Payload::Image(image)));
        }
        Ok(self.facet_data(cells))
    }
}
