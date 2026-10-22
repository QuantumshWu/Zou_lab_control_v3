//! An authored pulse timeline (`zlc_plot.primitives.PulseTimelineData` and
//! its parts) and its time base (`_pulse_time`): the payload a pulse-timeline
//! panel draws. A timeline is validated once, where it is made; its JSON form
//! is the dataclass fields as plain JSON (what the pulse compiler and the
//! corpus write).
use std::sync::Arc;

use serde_json::Value as Json;

use crate::configuration::error::{PlotError, Result};
use crate::configuration::value::{py_strip, repr_str};
use crate::units::{prefix_for, resolve_unit, Number, Unit};

#[derive(Clone, Debug, PartialEq)]
pub struct PulseChannel { pub channel_id: String, pub label: String }

#[derive(Clone, Debug, PartialEq)]
pub struct PulseBlock { pub channel_id: String, pub start: f64, pub stop: f64, pub label: String }

/// Which kind of slot a binding interval is (`SLOT_KINDS`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SlotKind { Scan, Api }

impl SlotKind {
    /// `_slot_kind(value)`.
    pub fn parse(value: &str) -> Result<SlotKind> {
        match value {
            "scan" => Ok(SlotKind::Scan),
            "api" => Ok(SlotKind::Api),
            _ => Err(PlotError::value("slot kind must be one of ('scan', 'api')")),
        }
    }
}

/// One highlighted field interval with a short binding label.
#[derive(Clone, Debug, PartialEq)]
pub struct PulseScanRegion { pub start: f64, pub stop: f64, pub label: String, pub kind: SlotKind }

#[derive(Clone, Debug, PartialEq)]
pub struct PulseAnalogTrace {
    pub name: String,
    pub label: String,
    pub minimum: f64,
    pub maximum: f64,
    pub starts: Vec<f64>,
    pub values: Vec<f64>,
}

/// One loop bracket; `series` picks its ink (None: the Run loop).
#[derive(Clone, Debug, PartialEq)]
pub struct PulseLoopMarker { pub start: f64, pub stop: f64, pub label: String, pub series: Option<u64> }

/// One authored period; a spacer is drawn as a hatched gap.
#[derive(Clone, Debug, PartialEq)]
pub struct PulsePeriodMark { pub start: f64, pub stop: f64, pub name: String, pub spacer: bool }

#[derive(Clone, Debug, PartialEq)]
pub struct PulseDacScanSegment {
    pub trace_name: String,
    pub start: f64,
    pub stop: f64,
    pub value: f64,
    pub label: String,
    pub kind: SlotKind,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PulseTimelineData {
    pub channels: Vec<PulseChannel>,
    pub blocks: Vec<PulseBlock>,
    pub scan_regions: Vec<PulseScanRegion>,
    pub time_unit: String,
    pub total_duration: Option<f64>,
    pub analog_traces: Vec<PulseAnalogTrace>,
    pub loop_markers: Vec<PulseLoopMarker>,
    pub scan_dac_segments: Vec<PulseDacScanSegment>,
    pub periods: Vec<PulsePeriodMark>,
}

/// `_MINIMUM_SOURCE_SPAN`.
const MINIMUM_SOURCE_SPAN: f64 = 1e-12;

// ------------------------------------------------------------- validation
/// `finite_real(value, name)` (zero kept as given).
fn finite(value: f64, name: &str) -> Result<f64> {
    if !value.is_finite() { return Err(PlotError::value(format!("{name} must be finite"))); }
    Ok(value)
}

/// `_nonnegative_time(value, name)`.
fn nonnegative_time(value: f64, name: &str) -> Result<f64> {
    let value = finite(value, name)?;
    if value < 0.0 { return Err(PlotError::value(format!("{name} must be non-negative"))); }
    Ok(value)
}

fn interval(start: f64, stop: f64, name: &str, order: &str) -> Result<(f64, f64)> {
    let start = nonnegative_time(start, &format!("{name} start"))?;
    let stop = nonnegative_time(stop, &format!("{name} stop"))?;
    if stop <= start { return Err(PlotError::value(order.to_owned())); }
    Ok((start, stop))
}

impl PulseBlock {
    pub fn new(channel_id: &str, start: f64, stop: f64, label: &str) -> Result<Self> {
        let (start, stop) = interval(start, stop, "pulse", "pulse stop must be greater than start")?;
        Ok(PulseBlock { channel_id: channel_id.to_owned(), start, stop, label: label.to_owned() })
    }
}

impl PulseScanRegion {
    pub fn new(start: f64, stop: f64, label: &str, kind: SlotKind) -> Result<Self> {
        let (start, stop) = interval(start, stop, "scan-region", "scan-region stop must be greater than start")?;
        Ok(PulseScanRegion { start, stop, label: label.to_owned(), kind })
    }
}

impl PulseAnalogTrace {
    pub fn new(name: &str, label: &str, minimum: f64, maximum: f64, starts: Vec<f64>, values: Vec<f64>) -> Result<Self> {
        let minimum = finite(minimum, "analog trace minimum")?;
        let maximum = finite(maximum, "analog trace maximum")?;
        if maximum <= minimum { return Err(PlotError::value("analog trace maximum must be greater than minimum")); }
        let starts = starts.into_iter().map(|value| nonnegative_time(value, "analog trace start")).collect::<Result<Vec<_>>>()?;
        let values = values.into_iter().map(|value| finite(value, "analog trace value")).collect::<Result<Vec<_>>>()?;
        if starts.is_empty() != values.is_empty() {
            return Err(PlotError::value("analog trace starts and values must either both be empty or both be present"));
        }
        if !values.is_empty() && starts.len() != values.len() + 1 {
            return Err(PlotError::value("analog trace starts must contain one more item than values"));
        }
        if starts.windows(2).any(|pair| pair[1] <= pair[0]) {
            return Err(PlotError::value("analog trace starts must be strictly increasing"));
        }
        Ok(PulseAnalogTrace { name: name.to_owned(), label: label.to_owned(), minimum, maximum, starts, values })
    }
}

impl PulseLoopMarker {
    pub fn new(start: f64, stop: f64, label: &str, series: Option<i64>) -> Result<Self> {
        let (start, stop) = interval(start, stop, "loop", "loop stop must be greater than start")?;
        let series = match series {
            None => None,
            Some(series) if series < 0 => return Err(PlotError::value("loop series must be a non-negative bracket index")),
            Some(series) => Some(series as u64),
        };
        Ok(PulseLoopMarker { start, stop, label: label.to_owned(), series })
    }
}

impl PulsePeriodMark {
    pub fn new(start: f64, stop: f64, name: &str, spacer: bool) -> Result<Self> {
        let (start, stop) = interval(start, stop, "period", "period stop must be greater than start")?;
        Ok(PulsePeriodMark { start, stop, name: name.to_owned(), spacer })
    }
}

impl PulseDacScanSegment {
    pub fn new(trace_name: &str, start: f64, stop: f64, value: f64, label: &str, kind: SlotKind) -> Result<Self> {
        let (start, stop) = interval(start, stop, "DAC scan", "DAC scan stop must be greater than start")?;
        Ok(PulseDacScanSegment { trace_name: trace_name.to_owned(), start, stop, value: finite(value, "DAC scan value")?,
                                 label: label.to_owned(), kind })
    }
}

/// Python's `repr(sorted(names))`.
fn sorted_repr(mut names: Vec<&str>) -> String {
    names.sort_unstable();
    names.dedup();
    format!("[{}]", names.iter().map(|name| repr_str(name)).collect::<Vec<_>>().join(", "))
}

/// `max(values)`: the first of the largest.
fn py_max(values: impl Iterator<Item = f64>) -> f64 { values.reduce(|best, value| if value > best { value } else { best }).expect("values") }

/// `min(values)`: the first of the smallest.
fn py_min(values: impl Iterator<Item = f64>) -> f64 { values.reduce(|best, value| if value < best { value } else { best }).expect("values") }

impl PulseTimelineData {
    /// `PulseTimelineData(...)` with its `__post_init__` checks.
    #[allow(clippy::too_many_arguments)]
    pub fn new(channels: Vec<PulseChannel>, blocks: Vec<PulseBlock>, scan_regions: Vec<PulseScanRegion>, time_unit: &str,
               total_duration: Option<f64>, analog_traces: Vec<PulseAnalogTrace>, loop_markers: Vec<PulseLoopMarker>,
               scan_dac_segments: Vec<PulseDacScanSegment>, periods: Vec<PulsePeriodMark>) -> Result<Self> {
        if channels.is_empty() { return Err(PlotError::value("PulseTimeline requires at least one channel")); }
        let ids: Vec<&str> = channels.iter().map(|channel| channel.channel_id.as_str()).collect();
        if (1..ids.len()).any(|i| ids[..i].contains(&ids[i])) { return Err(PlotError::value("pulse channel ids must be unique")); }
        let unknown: Vec<&str> = blocks.iter().map(|block| block.channel_id.as_str()).filter(|id| !ids.contains(id)).collect();
        if !unknown.is_empty() { return Err(PlotError::value(format!("pulse blocks reference unknown channels: {}", sorted_repr(unknown)))); }
        let names: Vec<&str> = analog_traces.iter().map(|trace| trace.name.as_str()).collect();
        if (1..names.len()).any(|i| names[..i].contains(&names[i])) { return Err(PlotError::value("analog trace names must be unique")); }
        if periods.windows(2).any(|pair| pair[1].start < pair[0].stop) {
            return Err(PlotError::value("periods must follow one another without overlap"));
        }
        let unknown: Vec<&str> = scan_dac_segments.iter().map(|segment| segment.trace_name.as_str()).filter(|name| !names.contains(name)).collect();
        if !unknown.is_empty() { return Err(PlotError::value(format!("DAC scan segments reference unknown traces: {}", sorted_repr(unknown)))); }
        let total_duration = match total_duration {
            None => None,
            Some(total) => {
                let total = finite(total, "total_duration")?;
                if total <= 0.0 { return Err(PlotError::value("total_duration must be positive")); }
                let latest = py_max(std::iter::once(0.0)
                    .chain(blocks.iter().map(|item| item.stop))
                    .chain(scan_regions.iter().map(|item| item.stop))
                    .chain(loop_markers.iter().map(|item| item.stop))
                    .chain(scan_dac_segments.iter().map(|item| item.stop))
                    .chain(analog_traces.iter().filter_map(|item| item.starts.last().copied())));
                if total < latest { return Err(PlotError::value("total_duration cannot end before pulse content")); }
                Some(total)
            }
        };
        Ok(PulseTimelineData { channels, blocks, scan_regions, time_unit: time_unit.to_owned(), total_duration, analog_traces,
                               loop_markers, scan_dac_segments, periods })
    }

    /// A timeline from its JSON form (the dataclass fields as plain JSON).
    pub fn from_json(tree: &Json) -> Result<Arc<PulseTimelineData>> {
        let object = tree.as_object().ok_or_else(|| PlotError::type_error("pulse timeline must be a JSON object"))?;
        let list = |name: &str| -> Result<&[Json]> {
            match object.get(name) {
                None => Ok(&[]),
                Some(Json::Array(items)) => Ok(items.as_slice()),
                Some(_) => Err(PlotError::type_error(format!("{name} must be a list"))),
            }
        };
        let channels = list("channels")?.iter().map(|item| {
            let item = Fields::of(item)?;
            Ok(PulseChannel { channel_id: item.text("channel_id", "channel_id")?, label: item.text("label", "channel label")? })
        }).collect::<Result<Vec<_>>>()?;
        let blocks = list("blocks")?.iter().map(|item| {
            let item = Fields::of(item)?;
            let channel = item.text("channel_id", "channel_id")?;
            let (start, stop) = (item.real("start", "pulse start")?, item.real("stop", "pulse stop")?);
            PulseBlock::new(&channel, start, stop, &item.text_or("label", "pulse label", "")?)
        }).collect::<Result<Vec<_>>>()?;
        let scan_regions = list("scan_regions")?.iter().map(|item| {
            let item = Fields::of(item)?;
            let (start, stop) = (item.real("start", "scan-region start")?, item.real("stop", "scan-region stop")?);
            PulseScanRegion::new(start, stop, &item.text("label", "binding label")?, item.slot()?)
        }).collect::<Result<Vec<_>>>()?;
        let time_unit = match object.get("time_unit") {
            None => "s".to_owned(),
            Some(Json::String(unit)) => unit.clone(),
            Some(_) => return Err(PlotError::type_error("time_unit must be text")),
        };
        let total_duration = match object.get("total_duration") {
            None | Some(Json::Null) => None,
            Some(value) => Some(real(value, "total_duration")?),
        };
        let analog_traces = list("analog_traces")?.iter().map(|item| {
            let item = Fields::of(item)?;
            let name = item.text("name", "analog trace name")?;
            let label = item.text("label", "analog trace label")?;
            let (minimum, maximum) = (item.real("minimum", "analog trace minimum")?, item.real("maximum", "analog trace maximum")?);
            let starts = item.reals("starts", "analog trace start")?;
            let values = item.reals("values", "analog trace value")?;
            PulseAnalogTrace::new(&name, &label, minimum, maximum, starts, values)
        }).collect::<Result<Vec<_>>>()?;
        let loop_markers = list("loop_markers")?.iter().map(|item| {
            let item = Fields::of(item)?;
            let (start, stop) = (item.real("start", "loop start")?, item.real("stop", "loop stop")?);
            let label = item.text("label", "loop label")?;
            let series = match item.0.get("series") {
                None | Some(Json::Null) => None,
                Some(Json::Bool(flag)) => Some(i64::from(*flag)),
                Some(value) => Some(value.as_i64().or_else(|| value.as_f64().filter(|v| v.is_finite()).map(|v| v.trunc() as i64))
                    .ok_or_else(|| PlotError::type_error("loop series must be an integer"))?),
            };
            PulseLoopMarker::new(start, stop, &label, series)
        }).collect::<Result<Vec<_>>>()?;
        let scan_dac_segments = list("scan_dac_segments")?.iter().map(|item| {
            let item = Fields::of(item)?;
            let trace = item.text("trace_name", "DAC trace name")?;
            let (start, stop) = (item.real("start", "DAC scan start")?, item.real("stop", "DAC scan stop")?);
            let value = item.real("value", "DAC scan value")?;
            PulseDacScanSegment::new(&trace, start, stop, value, &item.text_or("label", "binding label", "")?, item.slot()?)
        }).collect::<Result<Vec<_>>>()?;
        let periods = list("periods")?.iter().map(|item| {
            let item = Fields::of(item)?;
            let (start, stop) = (item.real("start", "period start")?, item.real("stop", "period stop")?);
            let spacer = match item.0.get("spacer") {
                None => false,
                Some(Json::Bool(flag)) => *flag,
                Some(_) => return Err(PlotError::type_error("period spacer flag must be a bool")),
            };
            PulsePeriodMark::new(start, stop, &item.text("name", "period name")?, spacer)
        }).collect::<Result<Vec<_>>>()?;
        Ok(Arc::new(PulseTimelineData::new(channels, blocks, scan_regions, &time_unit, total_duration, analog_traces, loop_markers,
                                           scan_dac_segments, periods)?))
    }

    /// The JSON form [`PulseTimelineData::from_json`] reads.
    pub fn to_json(&self) -> Json {
        let kind = |kind: SlotKind| if kind == SlotKind::Api { "api" } else { "scan" };
        serde_json::json!({
            "channels": self.channels.iter().map(|c| serde_json::json!({"channel_id": c.channel_id, "label": c.label})).collect::<Vec<_>>(),
            "blocks": self.blocks.iter().map(|b| serde_json::json!({"channel_id": b.channel_id, "start": b.start, "stop": b.stop, "label": b.label}))
                .collect::<Vec<_>>(),
            "scan_regions": self.scan_regions.iter()
                .map(|r| serde_json::json!({"start": r.start, "stop": r.stop, "label": r.label, "kind": kind(r.kind)})).collect::<Vec<_>>(),
            "time_unit": self.time_unit,
            "total_duration": self.total_duration,
            "analog_traces": self.analog_traces.iter().map(|t| serde_json::json!({"name": t.name, "label": t.label, "minimum": t.minimum,
                "maximum": t.maximum, "starts": t.starts, "values": t.values})).collect::<Vec<_>>(),
            "loop_markers": self.loop_markers.iter()
                .map(|m| serde_json::json!({"start": m.start, "stop": m.stop, "label": m.label, "series": m.series})).collect::<Vec<_>>(),
            "scan_dac_segments": self.scan_dac_segments.iter().map(|s| serde_json::json!({"trace_name": s.trace_name, "start": s.start,
                "stop": s.stop, "value": s.value, "label": s.label, "kind": kind(s.kind)})).collect::<Vec<_>>(),
            "periods": self.periods.iter()
                .map(|p| serde_json::json!({"start": p.start, "stop": p.stop, "name": p.name, "spacer": p.spacer})).collect::<Vec<_>>(),
        })
    }

    // ------------------------------------------------------------- time base
    /// `pulse_content_bounds(payload)`: the authored bounds in the source unit.
    pub fn content_bounds(&self) -> (f64, f64) {
        let starts = std::iter::once(0.0)
            .chain(self.blocks.iter().map(|item| item.start))
            .chain(self.scan_regions.iter().map(|item| item.start))
            .chain(self.analog_traces.iter().filter_map(|trace| trace.starts.first().copied()))
            .chain(self.loop_markers.iter().map(|item| item.start))
            .chain(self.scan_dac_segments.iter().map(|item| item.start));
        let stops = std::iter::once(MINIMUM_SOURCE_SPAN)
            .chain(self.blocks.iter().map(|item| item.stop))
            .chain(self.scan_regions.iter().map(|item| item.stop))
            .chain(self.analog_traces.iter().filter_map(|trace| trace.starts.last().copied()))
            .chain(self.loop_markers.iter().map(|item| item.stop))
            .chain(self.scan_dac_segments.iter().map(|item| item.stop))
            .chain(self.total_duration);
        let start = py_min(starts);
        let mut stop = py_max(stops);
        if stop - start < MINIMUM_SOURCE_SPAN { stop = start + MINIMUM_SOURCE_SPAN; }
        (start, stop)
    }

    /// `pulse_time_scale(payload, display_unit, source_span=)`: the factor
    /// from a source value to the displayed one, and the displayed unit.
    pub fn time_scale(&self, display_unit: Option<&str>, source_span: Option<f64>) -> Result<(f64, String)> {
        let Some(source) = time_unit(&self.time_unit) else {
            if display_unit.is_none_or(|unit| time_unit(unit).is_none()) { return Ok((1.0, self.time_unit.clone())); }
            return Err(PlotError::value(format!("cannot convert pulse source unit {} to {}", repr_str(&self.time_unit),
                                                repr_str(display_unit.unwrap_or("")))));
        };
        let target = match display_unit {
            None => {
                let span = match source_span {
                    None => { let (start, stop) = self.content_bounds(); stop - start }
                    Some(span) => span,
                };
                if !span.is_finite() || span <= 0.0 { return Err(PlotError::value("pulse source span must be positive and finite")); }
                let seconds = span.abs() * source.scale()?;
                let base = resolve_unit("s", None)?;
                resolve_unit(&format!("{}s", prefix_for(&[Number::Float(seconds)], &base).symbol), None)?
            }
            Some(unit) => time_unit(unit)
                .ok_or_else(|| PlotError::value(format!("pulse display unit must be a time unit: {}", repr_str(unit))))?,
        };
        Ok((source.scale()? / target.scale()?, target.symbol.clone()))
    }
}

/// `_time_unit(unit)`: one time unit, or None for a source that cannot rescale.
fn time_unit(unit: &str) -> Option<Arc<Unit>> {
    resolve_unit(py_strip(unit), None).ok().filter(|unit| unit.dimension == "time")
}

/// A real number of the JSON form (`finite_real`'s type rule; a bool is not one).
fn real(value: &Json, name: &str) -> Result<f64> {
    match value {
        Json::Number(number) => Ok(number.as_f64().expect("a JSON number")),
        _ => Err(PlotError::type_error(format!("{name} must be a real number"))),
    }
}

/// One JSON item's fields.
struct Fields<'a>(&'a serde_json::Map<String, Json>);

impl<'a> Fields<'a> {
    fn of(item: &'a Json) -> Result<Self> {
        item.as_object().map(Fields).ok_or_else(|| PlotError::type_error("pulse timeline items must be JSON objects"))
    }

    fn get(&self, key: &str) -> Result<&'a Json> {
        self.0.get(key).ok_or_else(|| PlotError::type_error(format!("missing required argument: {}", repr_str(key))))
    }

    fn text(&self, key: &str, name: &str) -> Result<String> {
        match self.get(key)? { Json::String(text) => Ok(text.clone()), _ => Err(PlotError::type_error(format!("{name} must be text"))) }
    }

    fn text_or(&self, key: &str, name: &str, default: &str) -> Result<String> {
        if self.0.contains_key(key) { self.text(key, name) } else { Ok(default.to_owned()) }
    }

    fn real(&self, key: &str, name: &str) -> Result<f64> { real(self.get(key)?, name) }

    fn reals(&self, key: &str, name: &str) -> Result<Vec<f64>> {
        match self.get(key)? {
            Json::Array(items) => items.iter().map(|item| real(item, name)).collect(),
            _ => Err(PlotError::type_error(format!("{key} must be a list"))),
        }
    }

    fn slot(&self) -> Result<SlotKind> {
        match self.0.get("kind") {
            None => Ok(SlotKind::Scan),
            Some(Json::String(kind)) => SlotKind::parse(kind),
            Some(other) => SlotKind::parse(&other.to_string()),
        }
    }
}
