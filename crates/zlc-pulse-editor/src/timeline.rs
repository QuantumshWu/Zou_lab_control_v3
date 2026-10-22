//! The pulse as a drawable timeline (`timeline_of`, `preview_candidate`): in
//! seconds, from the durations the board plays and the DAC levels its own
//! compiler emits, so the preview is a drawing of this pulse and no other.
use std::sync::Arc;

use zlc_plot::layout::{Kind, LayoutConfig, PANEL_PRESETS, SurfaceOptions, resolve_surface};
use zlc_plot::pulse::{
    PulseAnalogTrace, PulseBlock, PulseChannel, PulseDacScanSegment, PulseLoopMarker, PulsePeriodMark, PulseScanRegion, PulseTimelineData,
    SlotKind,
};
use zlc_plot::style::FontStyle;
use zlc_pulse::compile::analog_levels;
use zlc_pulse::model::{FieldKind, PortKind, Sequence, Source};
use zlc_pulse::{Error, Kind as ErrorKind, Result};

use crate::text::nanoseconds;

/// How a preview writes "this scan point plays until Stop".
pub const RUN_FOREVER_LABEL: &str = "×∞";

fn plot(error: zlc_plot::configuration::error::PlotError) -> Error { Error::new(ErrorKind::Value, error.to_string()) }

/// `timeline_of(sequence, include_off=)`.
pub fn timeline_of(sequence: &Sequence, include_off: bool) -> Result<PulseTimelineData> {
    let target = &sequence.target;
    let lengths: Vec<f64> = sequence.periods.iter().map(|p| nanoseconds(p.duration, &p.unit) * 1e-9).collect();
    let mut starts = Vec::with_capacity(lengths.len());
    let mut elapsed = 0.0;
    for length in &lengths { starts.push(elapsed); elapsed += length; }
    let total = elapsed;
    let mut marks = Vec::new();
    for ((start, length), period) in starts.iter().zip(&lengths).zip(&sequence.periods) {
        if nanoseconds(period.duration, &period.unit) > 0.0 {
            marks.push(PulsePeriodMark::new(*start, start + length, &sequence.period_label(&period.period_id), period.is_spacer()).map_err(plot)?);
        }
    }
    let mut channels = Vec::new();
    let mut blocks = Vec::new();
    for port in target.ports.iter().filter(|p| p.kind == PortKind::Digital) {
        let lane = target.lane_index(&port.lanes[0]).expect("a lane");
        let spans: Vec<(f64, f64)> = sequence.periods.iter().enumerate().filter(|(_, p)| p.states[lane] != 0)
            .map(|(index, _)| (starts[index], starts[index] + lengths[index])).collect();
        if spans.is_empty() && !include_off { continue; }
        channels.push(PulseChannel { channel_id: port.key.clone(), label: port.display().to_owned() });
        for (start, stop) in spans { blocks.push(PulseBlock::new(&port.key, start, stop, "").map_err(plot)?); }
    }
    let levels = analog_levels(sequence)?;
    let tick_seconds = sequence.time_step_ns * 1e-9;
    let mut traces = Vec::new();
    for port in target.ports.iter().filter(|p| p.kind == PortKind::Dac) {
        let (low, high) = port.signed_range().expect("a DAC");
        let points: Vec<(f64, f64)> = levels.iter().find(|(key, _)| *key == port.key).map(|(_, points)| points.as_slice()).unwrap_or(&[])
            .iter().map(|(tick, value)| (*tick as f64 * tick_seconds, *value as f64)).filter(|(at, _)| *at < total).collect();
        if !points.iter().any(|(_, value)| *value != 0.0) && !include_off { continue; }
        let mut times: Vec<f64> = points.iter().map(|(at, _)| *at).collect();
        times.push(total);
        traces.push(PulseAnalogTrace::new(&port.key, port.display(), low as f64, high as f64, times,
                                          points.iter().map(|(_, value)| *value).collect()).map_err(plot)?);
    }
    if channels.is_empty() {
        let first = target.ports.iter().find(|p| p.kind == PortKind::Digital)
            .ok_or_else(|| Error::new(ErrorKind::Value, "this target has no digital port to draw"))?;
        channels.push(PulseChannel { channel_id: first.key.clone(), label: first.display().to_owned() });
    }
    // Innermost first: the model keeps brackets outermost first.
    let mut markers = Vec::new();
    for (series, (bracket, (first, stop_gap))) in sequence.brackets.iter().zip(sequence.bracket_bounds()).enumerate().rev() {
        if stop_gap == first { continue; }
        let last = stop_gap - 1;
        markers.push(PulseLoopMarker::new(starts[*first], starts[last] + lengths[last], &format!("×{}", bracket.count), Some(series as i64))
            .map_err(plot)?);
    }
    let run_label = if sequence.run_repeats == 0 { RUN_FOREVER_LABEL.to_owned() } else { format!("×{}", sequence.run_repeats) };
    if total > 0.0 { markers.push(PulseLoopMarker::new(0.0, total, &run_label, None).map_err(plot)?); }
    let mut regions = Vec::new();
    let mut segments = Vec::new();
    for binding in &sequence.bindings {
        if !binding.scan && binding.source != Source::Api { continue; }
        let kind = if binding.scan { SlotKind::Scan } else { SlotKind::Api };
        let label = [if binding.scan { "S" } else { "" }, match binding.source { Source::Api => "A", Source::Config => "C", Source::Default => "" }]
            .iter().filter(|part| !part.is_empty()).copied().collect::<Vec<_>>().join("+");
        let Some(index) = binding.field_ref.period_id.as_deref().and_then(|id| sequence.period_index(id)) else { continue };
        let (start, stop) = (starts[index], starts[index] + lengths[index]);
        if stop <= start { continue; }
        match binding.field_ref.kind {
            FieldKind::Duration => regions.push(PulseScanRegion::new(start, stop, &label, kind).map_err(plot)?),
            FieldKind::Dac => {
                let port = binding.field_ref.port.as_deref().unwrap_or("");
                if traces.iter().any(|trace| trace.name == port) {
                    let held = sequence.periods[index].step(port).map_or(0.0, |step| step.value as f64);
                    segments.push(PulseDacScanSegment::new(port, start, stop, held, &label, kind).map_err(plot)?);
                }
            }
            _ => {}
        }
    }
    PulseTimelineData::new(channels, blocks, regions, "s", Some(total), traces, markers, segments, marks).map_err(plot)
}

/// `recommended_pulse_preset(rows, periods)`: the smallest preset whose data
/// region draws a row and a period legibly; the largest when nothing fits.
pub fn recommended_pulse_preset(channel_count: usize, period_count: usize) -> &'static str {
    let layout = LayoutConfig::default();
    let fonts = FontStyle::default();
    let rows = channel_count.max(1) as f64;
    let periods = period_count.max(1) as f64;
    let mut by_area: Vec<(&'static str, u32, u32)> = PANEL_PRESETS.to_vec();
    by_area.sort_by_key(|(_, rows, columns)| rows * columns);
    for (name, _, _) in &by_area {
        let Ok(plan) = resolve_surface(Some(name), Kind::PulseTimeline, &SurfaceOptions::default(), &layout, &fonts) else { continue };
        let Some(main) = plan.axes.iter().find(|axes| axes.role == "main") else { continue };
        let (width, height) = (plan.logical_size.0 as f64, plan.logical_size.1 as f64);
        let data_w = (main.r#box.right - main.r#box.left) * width;
        let data_h = (main.r#box.bottom - main.r#box.top) * height;
        if data_h >= rows * layout.pulse_row_min_px as f64 && data_w >= periods * layout.pulse_period_min_px as f64 { return name; }
    }
    by_area.last().expect("presets").0
}

/// One drawing of a pulse: its timeline, the size it is drawn at, its rows,
/// periods and how long the board plays it (ns, brackets expanded).
pub struct Preview { pub data: Arc<PulseTimelineData>, pub size: String, pub rows: usize, pub periods: usize, pub total_ns: f64 }

/// `preview_candidate(sequence, include_off, pinned_size)`.
pub fn preview_candidate(sequence: &Sequence, include_off: bool, pinned_size: Option<&str>) -> Result<Preview> {
    let data = timeline_of(sequence, include_off)?;
    let rows = data.channels.len() + data.analog_traces.len();
    let size = pinned_size.map_or_else(|| recommended_pulse_preset(rows, sequence.periods.len()).to_owned(), str::to_owned);
    Ok(Preview { data: Arc::new(data), size, rows, periods: sequence.periods.len(), total_ns: sequence.played_nanoseconds() })
}
