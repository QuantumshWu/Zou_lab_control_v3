//! `zlc_data.snapshot_projection.indexed_history_layout`: the one reading of
//! a Runtime indexed history's Point domain, read once per schema.

use std::sync::Arc;

use super::model::{Base, Schema, PRIMARY_INDEX, SHOT_TIME};
use super::ProjectionError;

#[derive(Debug)]
pub struct IndexedHistoryLayout {
    /// Each shot's relative offset, oldest first (none above 0).
    pub cells: Arc<[i64]>,
    pub primary: usize,
    pub primary_axis_id: Arc<str>,
    /// Uniform rows per shot.
    pub inner_count: Option<usize>,
    pub times: Option<Arc<[f64]>>,
    pub time_axis: Option<usize>,
    pub time_axis_id: Option<Arc<str>>,
    pub row_count: usize,
}

impl IndexedHistoryLayout {
    pub fn shot_count(&self) -> usize { self.cells.len() }

    /// `coordinate_role(axis_id)`.
    pub fn coordinate_role(&self, axis_id: &str) -> Option<&'static str> {
        if axis_id == &*self.primary_axis_id { return Some(PRIMARY_INDEX); }
        if self.time_axis_id.as_deref() == Some(axis_id) { return Some(SHOT_TIME); }
        None
    }

    /// `window_rows(window)`: the last `window` shots as a suffix of rows.
    pub fn window_rows(&self, schema: &Schema, window: usize) -> (usize, usize) {
        let keep = window.max(1).min(self.cells.len());
        let first = self.shot_count() - keep;
        let mapping = schema.point.code_mapping(self.primary);
        let start = match &mapping.base {
            Base::Range { .. } => first,
            Base::Explicit(base) => base.partition_point(|code| *code < first as i64),
        };
        (start * mapping.inner, self.row_count)
    }
}

/// `indexed_history_layout(schema)`, cached on the schema.
pub fn layout(schema: &Schema) -> Result<Option<Arc<IndexedHistoryLayout>>, ProjectionError> {
    if let Some(found) = schema.layout.get() { return Ok(found.clone()); }
    let computed = compute(schema)?;
    let _ = schema.layout.set(computed.clone());
    Ok(computed)
}

fn compute(schema: &Schema) -> Result<Option<Arc<IndexedHistoryLayout>>, ProjectionError> {
    let domain = &schema.point;
    let primaries: Vec<usize> = (0..domain.axes.len()).filter(|i| &*domain.axes[*i].role == PRIMARY_INDEX).collect();
    if primaries.is_empty() { return Ok(None); }
    if primaries.len() != 1 { return Err(ProjectionError::value("an indexed history requires exactly one primary-index coordinate")); }
    let primary = primaries[0];
    let axis = &domain.axes[primary];
    if axis.coordinate_of.is_some() { return Err(ProjectionError::value("the primary-index coordinate must own its record axis")); }
    let values = axis.values();
    let offsets: Vec<i64> = match values.dtype().kind() {
        'i' | 'u' if values.dtype() != super::nd::DType::U64 || values.slice::<u64>().iter().all(|v| *v <= i64::MAX as u64) =>
            (0..values.len()).map(|i| values.get_f64(i) as i64).collect::<Vec<_>>(),
        _ => return Err(ProjectionError::value("primary-index coordinates must be integer relative offsets")),
    };
    let offsets: Vec<i64> = match values.dtype() {
        super::nd::DType::I64 => values.slice::<i64>().to_vec(),
        _ => offsets,
    };
    if offsets.iter().any(|v| *v > 0) { return Err(ProjectionError::value("relative primary-index coordinates are at most the latest offset 0")); }
    if offsets.windows(2).any(|pair| pair[1] <= pair[0]) { return Err(ProjectionError::value("primary-index offsets must form ordered cells")); }
    let shots = offsets.len();
    let times: Vec<usize> = (0..domain.axes.len()).filter(|i| &*domain.axes[*i].role == SHOT_TIME).collect();
    if times.len() > 1 { return Err(ProjectionError::value("an indexed history has at most one shot-time coordinate")); }
    let time_axis = times.first().copied();
    if let Some(time) = time_axis {
        if domain.axes[time].coordinate_of.as_ref().is_some_and(|of| of != &axis.id) {
            return Err(ProjectionError::value("the shot-time coordinate must correspond to the primary-index axis"));
        }
    }
    let mut counts: Option<Vec<usize>> = None;
    let mut checked = vec![primary];
    if let Some(time) = time_axis { if domain.axes[time].coordinate_of.is_none() { checked.push(time); } }
    for index in checked {
        let mapping = domain.code_mapping(index);
        let (base, inner, outer) = (&mapping.base, mapping.inner, mapping.outer);
        let mut ordered = base.at(0) == 0 && base.at(base.len() - 1) == shots as i64 - 1 && (outer == 1 || shots == 1);
        let row_counts: Vec<usize> = match base {
            Base::Range { len, step, .. } => {
                ordered = ordered && (*len == 1 || *step == 1);
                vec![if shots == 1 { domain.size() } else { inner }; shots]
            }
            Base::Explicit(codes) => {
                ordered = ordered && codes.windows(2).all(|pair| pair[1] - pair[0] == 0 || pair[1] - pair[0] == 1);
                let mut starts = vec![0usize];
                for position in 1..codes.len() { if codes[position] - codes[position - 1] > 0 { starts.push(position); } }
                starts.push(codes.len());
                starts.windows(2).map(|pair| (pair[1] - pair[0]) * inner * outer).collect()
            }
        };
        let same = counts.as_ref().is_none_or(|known| *known == row_counts);
        if !ordered || !same {
            return Err(ProjectionError::value(if index == primary { "primary-index rows must form ordered contiguous cells" }
                                              else { "shot time is one coordinate per shot, on the primary index's rows" }));
        }
        if index == primary { counts = Some(row_counts); }
    }
    let counts = counts.expect("primary checked");
    let inner_count = counts[0];
    let uniform = counts.iter().all(|count| *count == inner_count);
    let times = time_axis.map(|time| domain.axes[time].values().to_f64().into());
    Ok(Some(Arc::new(IndexedHistoryLayout {
        cells: offsets.into(), primary, primary_axis_id: axis.id.clone(),
        inner_count: uniform.then_some(inner_count), times, time_axis,
        time_axis_id: time_axis.map(|time| domain.axes[time].id.clone()), row_count: domain.size(),
    })))
}

impl IndexedHistoryLayout {
    /// `codes(rows)`: each row's shot position.
    pub fn codes(&self, schema: &Schema) -> Arc<[i64]> { schema.point.codes(self.primary) }
    pub fn codes_at(&self, schema: &Schema, rows: &[i64]) -> Vec<i64> { schema.point.codes_at(self.primary, rows) }
}
