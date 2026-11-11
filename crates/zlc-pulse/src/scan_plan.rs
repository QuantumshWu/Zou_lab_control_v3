//! A native scan plan's board segment. Expanded rows never visit a plugin.
use std::sync::Arc;
use zlc_core::units::DEFAULT_UNITS;
use zlc_core::scan::{Plan, Scope};
use crate::error::{Error, Kind, Result, value_error};
use crate::model::{FieldKind, MAXIMUM_REPEAT_COUNT, Sequence};
use crate::run::{ApiValues, RunPlan, ScanRows, plan_run};
use crate::wire::StreamerParams;

/// Conservative allowance for authored/effective/wire rows and native receipt
/// JSON, not just the first f64 table. Refused before any product is allocated.
pub const MATERIALIZATION_BUDGET_BYTES: usize = 64 * 1024 * 1024;

pub struct ScanPlanRequest { pub plan: Arc<Plan>, pub outer_index: usize }

fn invalid(error: impl ToString) -> Error { Error::new(Kind::Value, error.to_string()) }

fn unit<'a>(sequence: &'a Sequence, field: &str) -> Result<&'a str> {
    let binding = sequence.bindings.iter().find(|binding| binding.field_id() == field)
        .ok_or_else(|| invalid(format!("{field:?} is not a field of this pulse")))?;
    Ok(if binding.kind() == FieldKind::Dac { "code" } else { &binding.unit })
}

fn convert(value: f64, source: &str, target: &str) -> Result<f64> {
    let value = if source.is_empty() || source == target { value }
        else { DEFAULT_UNITS.convert(value, source, target).map_err(invalid)? };
    if !value.is_finite() { return value_error("a scan coordinate converts to a non-finite pulse value"); }
    Ok(value)
}

/// Quantization and coordinate consistency are checked before touching the
/// board, including combined LOAD/FIRE calls.
pub fn prepare_application(sequence: &Sequence, request: &ScanPlanRequest, api_values: &ApiValues, geometry: &StreamerParams) -> Result<RunPlan> {
    let table=table_for_plan(sequence,request,geometry)?;
    let prepared=plan_run(sequence,api_values,table.as_ref(),geometry)?;
    if let Some(rows)=effective_inner(&prepared.resolved,&request.plan,prepared.played.as_ref())? {
        request.plan.with_effective_inner(&rows).map_err(invalid)?;
    }
    Ok(prepared)
}

pub fn table_for_plan(sequence: &Sequence, request: &ScanPlanRequest, geometry: &StreamerParams) -> Result<Option<ScanRows>> {
    let plan = &request.plan;
    if request.outer_index >= plan.outer_count() { return value_error("the scan's outer point is outside its plan"); }
    let outer_width = plan.outer_axis_count();
    if plan.axes()[outer_width..].iter().any(|axis| !axis.port.starts_with("pulse:param:")) {
        return value_error("a board-advanced scan axis must name a pulse field");
    }
    // Inner slots first, then pulse slots linked to this host point, exactly
    // as the old control path's named table was ordered.
    let columns: Vec<_> = (outer_width..plan.axes().len()).chain(0..outer_width)
        .filter(|index| plan.axes()[*index].port.starts_with("pulse:param:")).collect();
    if columns.is_empty() { return Ok(None); }
    if columns.len() as u64 > geometry.num_slots {
        return value_error(format!("the board advances at most {} slots per cycle; this plan scans {} slots", geometry.num_slots, columns.len()));
    }
    if plan.inner_count() as u128 > u128::from(MAXIMUM_REPEAT_COUNT) { return value_error("scan table row count does not fit the 32-bit SCAN_COUNT"); }
    let sweeps = if outer_width == 0 { plan.spec().scan_repeats } else { 1 };
    if plan.inner_count() as u128 * sweeps as u128 > 1u128 << 32 { return value_error("finite scan row visits exceed the 32-bit CURSOR range"); }
    let bytes = columns.len().checked_mul(384).and_then(|columns| columns.checked_add(768))
        .and_then(|row| row.checked_mul(plan.inner_count())).ok_or_else(|| invalid("scan table materialization size overflows"))?;
    if bytes > MATERIALIZATION_BUDGET_BYTES {
        return value_error(format!("this scan table needs an estimated {bytes} bytes of native materialization, beyond the {} MiB budget; reduce the inner scan", MATERIALIZATION_BUDGET_BYTES / (1024 * 1024)));
    }
    let fields: Vec<String> = columns.iter().map(|index| plan.axes()[*index].port["pulse:param:".len()..].to_owned()).collect();
    let units = fields.iter().map(|field| unit(sequence, field)).collect::<Result<Vec<_>>>()?;
    let outer = plan.row(Scope::Outer, request.outer_index).map_err(invalid)?;
    let rows = (0..plan.inner_count()).map(|index| {
        let inner = plan.row(Scope::Inner, index).map_err(invalid)?;
        columns.iter().zip(&units).map(|(axis, native)| {
            let value = if *axis < outer_width { outer[*axis] } else { inner[*axis-outer_width] };
            convert(value, &plan.axes()[*axis].unit, native)
        }).collect()
    }).collect::<Result<Vec<_>>>()?;
    Ok(Some(ScanRows { fields, rows }))
}

/// Quantized inner coordinates back in their plan units, cached by the native
/// scan session rather than returned as a JSON table to Python.
pub fn effective_inner(sequence: &Sequence, plan: &Plan, played: Option<&ScanRows>) -> Result<Option<Vec<Vec<f64>>>> {
    let axes = &plan.axes()[plan.outer_axis_count()..];
    if axes.is_empty() { return Ok(None); }
    let played = played.ok_or_else(|| invalid("the pulse receipt is missing the native scan table"))?;
    if played.rows.len() != plan.inner_count() { return value_error("the pulse receipt has a different inner scan point count"); }
    if played.rows.iter().any(|row| row.len() != played.fields.len()) { return value_error("the pulse receipt has inconsistent scan row widths"); }
    let columns = axes.iter().map(|axis| {
        let field = axis.port.strip_prefix("pulse:param:").ok_or_else(|| invalid("a board axis does not name a pulse field"))?;
        let index = played.fields.iter().position(|name| name == field).ok_or_else(|| invalid(format!("the pulse receipt is missing scan field {field:?}")))?;
        Ok((index, unit(sequence, field)?, axis.unit.as_str()))
    }).collect::<Result<Vec<_>>>()?;
    played.rows.iter().map(|row| columns.iter().map(|(index, source, target)|
        convert(row[*index], source, if target.is_empty() { source } else { target })).collect()).collect::<Result<Vec<_>>>().map(Some)
}
