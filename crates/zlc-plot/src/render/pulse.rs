//! The PulseTimeline kind (`_rendering/pulse.update_pulse_timeline` and
//! `MatplotlibRenderer._update_pulse_timeline`): one row per channel and
//! analog trace, blocks with their names where they fit on screen, scan and
//! API slots, the authored periods named in a band above the rows, spacers
//! hatched, and loop brackets whose feet are sized on screen -- all updated
//! in place, artists made once per group and hidden when unused.
use std::collections::HashMap;
use std::sync::Arc;

use zlc_figure::ir::{self, CapStyle, HAlign, VAlign};

use super::core::{Kind, Renderer};
use super::labels::{has_explicit_unit_suffix, state_label};
use super::mpl::axes::{Badge, Child, ChildId, LineArtist, RectangleArtist, RichTextArtist, StyledLineArtist, TextArtist, TextFrame,
                       TickPolicy};
use crate::configuration::error::PlotError;
use crate::configuration::{DisplayState, Value};
use crate::projection::output::Payload;
use crate::pulse::{PulseTimelineData, SlotKind};

/// The artist groups, in the order `update_pulse_timeline` first makes their
/// `_artists` entries: the order the live front composes them in.
#[derive(Clone, Copy, Debug)]
enum Group {
    Baselines, Blocks, BlockLabels, AnalogZero, AnalogValues, ScanRegions, ScanLabels, ScanDac, ScanDacLabels,
    PeriodBounds, Spacers, PeriodLabels, LoopLeft, LoopRight, LoopLabels,
}

const GROUPS: usize = 15;

/// The pulse renderer's tables: every group's artists in the order made.
#[derive(Clone, Debug, Default)]
pub struct PulseTables {
    axes: usize,
    groups: [Vec<ChildId>; GROUPS],
    /// The render's refusal (the Python renderer's exception), until taken.
    pub failure: Option<PlotError>,
}

impl PulseTables {
    /// Every artist, group by group.
    pub fn artists(&self) -> impl Iterator<Item = (usize, ChildId)> + '_ {
        self.groups.iter().flatten().map(move |&id| (self.axes, id))
    }
}

pub struct PulseKind;

impl Kind for PulseKind {
    fn render(&self, renderer: &mut Renderer, payload: &Arc<Payload>, state: &DisplayState, axes: usize, _key: &str) {
        let Payload::Pulse(data) = &**payload else { panic!("pulse timeline requires PulseTimelineData") };
        if let Err(error) = renderer.update_pulse_timeline(axes, &data.data, state) { renderer.pulse.failure = Some(error); }
    }
    fn key(&self) -> &'static str { "pulse_timeline" }
}

/// `np.clip(value, 0.0, 1.0)`.
fn clip01(value: f64) -> f64 { value.max(0.0).min(1.0) }

/// Python's `max(a, b)`: `a` unless `b` is larger.
fn max2(a: f64, b: f64) -> f64 { if b > a { b } else { a } }

/// Python's `min(a, b)`: `a` unless `b` is smaller.
fn min2(a: f64, b: f64) -> f64 { if b < a { b } else { a } }

/// `_loop_depths(markers)`: a loop stands one step outside the deepest loop it contains.
pub fn loop_depths(spans: &[(f64, f64)]) -> Vec<usize> {
    let contains = |outer: usize, inner: usize| {
        if outer == inner { return false; }
        let ((o_start, o_stop), (i_start, i_stop)) = (spans[outer], spans[inner]);
        if !(o_start <= i_start && i_stop <= o_stop) { return false; }
        (o_stop - o_start) > (i_stop - i_start) || outer > inner
    };
    let mut order: Vec<usize> = (0..spans.len()).collect();
    order.sort_by(|&a, &b| (spans[a].1 - spans[a].0).partial_cmp(&(spans[b].1 - spans[b].0)).expect("finite spans").then(a.cmp(&b)));
    let mut depths = vec![0usize; spans.len()];
    for outer in order {
        depths[outer] = (0..spans.len()).filter(|&inner| contains(outer, inner)).map(|inner| depths[inner] + 1).max().unwrap_or(0);
    }
    depths
}

impl Renderer {
    /// `_sync_*(axis, artists, key, count)`: at least `count` artists of a
    /// group (new ones made by `make`), the first `count` visible.
    fn sync_group(&mut self, axes: usize, group: Group, count: usize, make: impl Fn(&Renderer) -> Child) -> Vec<ChildId> {
        while self.pulse.groups[group as usize].len() < count {
            let child = make(self);
            let id = self.fig.axes[axes].add(child);
            self.pulse.groups[group as usize].push(id);
        }
        let ids = self.pulse.groups[group as usize].clone();
        for (index, &id) in ids.iter().enumerate() {
            self.fig.axes[axes].child_mut(id).expect("pulse artist").set_visible(index < count);
        }
        ids
    }

    /// `axis.plot([], [])` (then clipped).
    fn plotted_line(&self) -> LineArtist {
        LineArtist {
            visible: true, zorder: 2.0, alpha: None, color: String::new(), linewidth: self.rc.lines_linewidth, linestyle: "-".into(),
            marker: "None".into(), markersize: self.rc.lines_markersize, markeredgewidth: self.rc.lines_markeredgewidth,
            markevery: None, marker_colors: None, label: String::new(), x: Vec::new(), y: Vec::new(),
        }
    }

    fn styled(&self, line: LineArtist) -> Child {
        Child::StyledLine(StyledLineArtist { line, dashes: None, solid_capstyle: CapStyle::Projecting, bracket: None })
    }

    /// A text made on transData (`axis.text`, `SpanLabel`, `axis.annotate`).
    fn rich_text(&self, anchor: Option<(f64, f64)>) -> Child {
        Child::RichText(RichTextArtist {
            text: TextArtist {
                visible: true, zorder: 3.0, x: 0.0, y: 0.0, text: String::new(), color: self.rc.text_color.clone(), size: self.rc.font_size,
                ha: HAlign::Left, va: VAlign::Baseline, ma: None, clip_on: true, frame: TextFrame::Data, family: None,
            },
            alpha: None, span: None, badge: None, anchor,
        })
    }

    fn rectangle(&self) -> Child {
        Child::Rectangle(RectangleArtist {
            visible: true, zorder: 1.0, alpha: None, xy: (0.0, 0.0), width: 0.0, height: 0.0, facecolor: "none".into(),
            edgecolor: "none".into(), linewidth: self.rc.patch_linewidth, hatch: None,
        })
    }

    fn styled_mut(&mut self, axes: usize, id: ChildId) -> &mut StyledLineArtist {
        match self.fig.axes[axes].child_mut(id) { Some(Child::StyledLine(line)) => line, _ => panic!("pulse line expected") }
    }

    fn rich_mut(&mut self, axes: usize, id: ChildId) -> &mut RichTextArtist {
        match self.fig.axes[axes].child_mut(id) { Some(Child::RichText(text)) => text, _ => panic!("pulse text expected") }
    }

    fn rectangle_mut(&mut self, axes: usize, id: ChildId) -> &mut RectangleArtist {
        match self.fig.axes[axes].child_mut(id) { Some(Child::Rectangle(rectangle)) => rectangle, _ => panic!("pulse rectangle expected") }
    }

    /// `_room_above(axis, room_pt, below)`: axis units that leave `room_pt`
    /// points above a line `below` units over the bottom limit.
    fn room_above(&self, axes: usize, room_pt: f64, below: f64) -> f64 {
        let dpi = self.fig.dpi;
        let position = self.fig.axes[axes].active;
        let room_pixels = room_pt * dpi / 72.0;
        let axes_pixels = (position[3] - position[1]) * self.fig.size_inches[1] * dpi;
        let room_pixels = min2(room_pixels, 0.45 * axes_pixels);
        room_pixels * below / (axes_pixels - room_pixels)
    }

    /// `_update_pulse_timeline(axes, payload, state, key)`.
    pub fn update_pulse_timeline(&mut self, axes: usize, payload: &PulseTimelineData, state: &DisplayState) -> Result<(), PlotError> {
        self.pulse.axes = axes;
        let style = self.style.clone();
        let pulse = &style.pulse;
        let palette = &style.palette;
        let analog_keys: Vec<String> = payload.analog_traces.iter().enumerate().map(|(index, trace)| format!("analog:{index}:{}", trace.name)).collect();
        let row_keys: Vec<String> = payload.channels.iter().map(|channel| channel.channel_id.clone()).chain(analog_keys.iter().cloned()).collect();
        let row_count = row_keys.len();
        let mut row_index: HashMap<&str, usize> = HashMap::new();
        let mut row_colors: HashMap<&str, &'static str> = HashMap::new();
        for (index, key) in row_keys.iter().enumerate() {
            row_index.insert(key, row_count - 1 - index);
            row_colors.insert(key, palette.pulse_cycle[index % palette.pulse_cycle.len()]);
        }
        let row_height = if row_count <= pulse.dense_row_threshold { pulse.row_height }
            else { max2(pulse.dense_min_row_height, pulse.dense_total_height / row_count as f64) };
        let baseline_offset = row_height / 2.0;
        let (start_min, stop_max) = payload.content_bounds();
        let span = stop_max - start_min;
        let margin = max2(span * pulse.x_margin_fraction, f64::EPSILON * span);
        let (left_limit, right_limit) = (start_min - margin, stop_max + margin);

        // Baselines: one per channel, in the channel's ink.
        let mut baseline_y: Vec<(String, f64)> = Vec::new();
        let set_baseline = |baseline_y: &mut Vec<(String, f64)>, key: &str, y: f64| match baseline_y.iter_mut().find(|(k, _)| k == key) {
            Some(entry) => entry.1 = y,
            None => baseline_y.push((key.to_owned(), y)),
        };
        let lines = self.sync_group(axes, Group::Baselines, payload.channels.len(), |r| r.styled(r.plotted_line()));
        for (index, channel) in payload.channels.iter().enumerate() {
            let y = row_index[channel.channel_id.as_str()] as f64 - baseline_offset;
            set_baseline(&mut baseline_y, &channel.channel_id, y);
            let line = &mut self.styled_mut(axes, lines[index]).line;
            line.x = vec![left_limit, right_limit];
            line.y = vec![y, y];
            line.color = row_colors[channel.channel_id.as_str()].into();
            line.linewidth = pulse.trace_linewidth;
            line.alpha = Some(1.0);
            line.linestyle = "-".into();
            line.zorder = pulse.base_zorder;
        }
        let baseline_of = |baseline_y: &Vec<(String, f64)>, key: &str| baseline_y.iter().find(|(k, _)| k == key).map(|(_, y)| *y).expect("a row");

        // Blocks and their names.
        let blocks = self.sync_group(axes, Group::Blocks, payload.blocks.len(), |r| r.rectangle());
        let block_labels = self.sync_group(axes, Group::BlockLabels, payload.blocks.len(), |r| r.rich_text(None));
        for (index, block) in payload.blocks.iter().enumerate() {
            let baseline = baseline_of(&baseline_y, &block.channel_id);
            let colour = row_colors[block.channel_id.as_str()];
            let rectangle = self.rectangle_mut(axes, blocks[index]);
            rectangle.xy = (block.start, baseline);
            rectangle.width = block.stop - block.start;
            rectangle.height = row_height;
            rectangle.facecolor = colour.into();
            rectangle.edgecolor = "none".into();
            rectangle.linewidth = 0.0;
            rectangle.alpha = Some(1.0);
            rectangle.zorder = pulse.base_zorder;
            let label = self.rich_mut(axes, block_labels[index]);
            label.text.x = (block.start + block.stop) / 2.0;
            label.text.y = row_index[block.channel_id.as_str()] as f64;
            label.text.text = block.label.clone();
            label.text.ha = HAlign::Center;
            label.text.va = VAlign::Center;
            label.text.clip_on = true;
            label.text.zorder = pulse.base_zorder + 1.0;
            label.text.size = style.fonts.pulse_bar_label_pt;
            label.text.color = palette.pulse_name.into();
            label.span = Some(ir::SpanLabel { span: [block.start, block.stop], pad_pt: pulse.label_fit_pad_pt });
            label.text.visible = !block.label.is_empty();
        }

        // Analog traces: a dashed zero line and the stepped value.
        let zeros = self.sync_group(axes, Group::AnalogZero, payload.analog_traces.len(), |r| r.styled(r.plotted_line()));
        let values = self.sync_group(axes, Group::AnalogValues, payload.analog_traces.len(), |r| r.styled(r.plotted_line()));
        let mut analog_ranges: HashMap<&str, (f64, f64, f64)> = HashMap::new();
        for (index, (key, trace)) in analog_keys.iter().zip(&payload.analog_traces).enumerate() {
            let row_base = row_index[key.as_str()] as f64 - baseline_offset;
            set_baseline(&mut baseline_y, key, row_base);
            let value_span = trace.maximum - trace.minimum;
            let zero_y = row_base + row_height * clip01((0.0 - trace.minimum) / value_span);
            analog_ranges.insert(&trace.name, (row_base, trace.minimum, value_span));
            let colour = row_colors[key.as_str()];
            let zero = self.styled_mut(axes, zeros[index]);
            zero.line.x = vec![left_limit, right_limit];
            zero.line.y = vec![zero_y, zero_y];
            zero.line.color = colour.into();
            zero.line.linewidth = pulse.trace_linewidth;
            zero.line.alpha = Some(pulse.analog_zero_alpha);
            zero.dashes = Some((pulse.analog_zero_dash.0, pulse.analog_zero_dash.1.to_vec()));
            zero.line.zorder = pulse.base_zorder + 1.0;
            let count = trace.values.len().min(trace.starts.len().saturating_sub(1));
            let value = self.styled_mut(axes, values[index]);
            if count > 0 {
                value.line.x = trace.starts[..count + 1].iter().flat_map(|&x| [x, x]).skip(1).take(2 * count).collect();
                value.line.y = trace.values[..count].iter()
                    .flat_map(|&v| { let y = row_base + row_height * clip01((v - trace.minimum) / value_span); [y, y] }).collect();
            } else {
                value.line.x = vec![left_limit, right_limit];
                value.line.y = vec![zero_y, zero_y];
            }
            value.line.visible = true;
            value.line.color = colour.into();
            value.line.linewidth = pulse.trace_linewidth;
            value.line.alpha = Some(1.0);
            value.dashes = None;
            value.line.linestyle = "-".into();
            value.line.zorder = pulse.base_zorder + 2.0;
        }

        // Scan and API slots: a translucent band over the rows, a badge with the binding.
        let show_scan = matches!(state.get("show_scan_regions"), Some(Value::Bool(true)));
        let regions = self.sync_group(axes, Group::ScanRegions, payload.scan_regions.len(), |r| r.rectangle());
        let region_labels = self.sync_group(axes, Group::ScanLabels, payload.scan_regions.len(), |r| r.rich_text(None));
        let area_bottom = baseline_y.iter().map(|(_, y)| *y).reduce(min2).unwrap_or(-baseline_offset);
        let area_top = baseline_y.iter().map(|(_, y)| *y).reduce(max2).unwrap_or(0.0) + row_height;
        let slot_color = |kind: SlotKind| if kind == SlotKind::Api { style.artists.pulse_api_region_color } else { style.artists.pulse_scan_region_color };
        let pad: f64 = crate::pyfmt::format_g(pulse.scan_badge_pad, 6).parse().expect("a badge pad");
        let badge = |label: &str, kind: SlotKind| Badge { circle: label.chars().count() == 1, pad, facecolor: slot_color(kind).into() };
        for (index, region) in payload.scan_regions.iter().enumerate() {
            let rectangle = self.rectangle_mut(axes, regions[index]);
            rectangle.xy = (region.start, area_bottom);
            rectangle.width = region.stop - region.start;
            rectangle.height = area_top - area_bottom;
            rectangle.facecolor = slot_color(region.kind).into();
            rectangle.edgecolor = "none".into();
            rectangle.alpha = Some(pulse.scan_region_alpha);
            rectangle.zorder = pulse.scan_region_zorder;
            rectangle.visible = show_scan;
            let text = self.rich_mut(axes, region_labels[index]);
            text.text.x = (region.start + region.stop) / 2.0;
            text.text.y = area_top - row_height / 2.0;
            text.text.text = region.label.clone();
            text.text.ha = HAlign::Center;
            text.text.va = VAlign::Center;
            text.text.zorder = pulse.annotation_zorder;
            text.badge = Some(badge(&region.label, region.kind));
            text.text.size = style.fonts.pulse_scan_annotation_pt;
            text.text.color = style.artists.pulse_scan_annotation_color.into();
            text.text.visible = show_scan;
        }
        let dac_lines = self.sync_group(axes, Group::ScanDac, payload.scan_dac_segments.len(), |r| r.styled(r.plotted_line()));
        let dac_labels = self.sync_group(axes, Group::ScanDacLabels, payload.scan_dac_segments.len(), |r| r.rich_text(None));
        for (index, segment) in payload.scan_dac_segments.iter().enumerate() {
            let (row_base, minimum, value_span) = analog_ranges[segment.trace_name.as_str()];
            let y = row_base + row_height * clip01((segment.value - minimum) / value_span);
            let line = self.styled_mut(axes, dac_lines[index]);
            line.line.x = vec![segment.start, segment.stop];
            line.line.y = vec![y, y];
            line.line.color = slot_color(segment.kind).into();
            line.line.linewidth = pulse.scan_dac_linewidth;
            line.line.alpha = Some(pulse.scan_dac_alpha);
            line.solid_capstyle = CapStyle::Butt;
            line.line.zorder = pulse.scan_dac_zorder;
            line.line.visible = show_scan;
            let text = self.rich_mut(axes, dac_labels[index]);
            text.text.x = (segment.start + segment.stop) / 2.0;
            text.text.y = row_base + row_height / 2.0;
            text.text.text = segment.label.clone();
            text.text.ha = HAlign::Center;
            text.text.va = VAlign::Center;
            text.text.zorder = pulse.annotation_zorder;
            text.badge = Some(badge(&segment.label, segment.kind));
            text.text.size = style.fonts.pulse_scan_annotation_pt;
            text.text.color = style.artists.pulse_scan_annotation_color.into();
            text.text.visible = show_scan && !segment.label.is_empty();
        }

        // The periods, named in a band above the rows, ruled at every
        // boundary; a spacer hatched across rows and band.
        let periods = &payload.periods;
        let band = if periods.is_empty() { 0.0 } else { pulse.period_band_height };
        let row_top = (row_count as f64 - 1.0) + row_height / 2.0;
        let frame_low = pulse.repeat_bottom;
        let frame_high = (row_count as f64 + band) + pulse.repeat_top_offset;
        let edges: Vec<f64> = periods.iter().map(|mark| mark.start).chain(periods.last().map(|mark| mark.stop)).collect();
        let bounds = self.sync_group(axes, Group::PeriodBounds, edges.len(), |r| r.styled(r.plotted_line()));
        for (index, &edge) in edges.iter().enumerate() {
            let line = self.styled_mut(axes, bounds[index]);
            line.line.x = vec![edge, edge];
            line.line.y = vec![frame_low, frame_high];
            line.line.color = palette.pulse_period.into();
            line.line.linewidth = pulse.period_boundary_linewidth;
            line.line.alpha = Some(pulse.period_boundary_alpha);
            line.dashes = Some((pulse.period_boundary_dash.0, pulse.period_boundary_dash.1.to_vec()));
            line.line.zorder = pulse.base_zorder - 1.0;
        }
        let spacers: Vec<_> = periods.iter().filter(|mark| mark.spacer).collect();
        let hatches = self.sync_group(axes, Group::Spacers, spacers.len(), |r| r.rectangle());
        for (index, mark) in spacers.iter().enumerate() {
            let rectangle = self.rectangle_mut(axes, hatches[index]);
            rectangle.xy = (mark.start, frame_low);
            rectangle.width = mark.stop - mark.start;
            rectangle.height = frame_high - frame_low;
            rectangle.facecolor = "none".into();
            rectangle.edgecolor = palette.pulse_period.into();
            rectangle.linewidth = 0.0;
            rectangle.hatch = Some(pulse.spacer_hatch.into());
            rectangle.alpha = Some(pulse.spacer_alpha);
            rectangle.zorder = pulse.base_zorder - 1.0;
        }
        let period_labels = self.sync_group(axes, Group::PeriodLabels, periods.len(), |r| r.rich_text(None));
        for (index, mark) in periods.iter().enumerate() {
            let text = self.rich_mut(axes, period_labels[index]);
            text.text.x = (mark.start + mark.stop) / 2.0;
            text.text.y = row_top + band / 2.0;
            text.text.text = mark.name.clone();
            text.text.ha = HAlign::Center;
            text.text.va = VAlign::Center;
            text.text.clip_on = true;
            text.text.zorder = pulse.base_zorder + 1.0;
            text.text.size = style.fonts.pulse_bar_label_pt;
            text.text.color = palette.pulse_period.into();
            text.span = Some(ir::SpanLabel { span: [mark.start, mark.stop], pad_pt: pulse.label_fit_pad_pt });
            text.text.visible = !mark.name.is_empty();
        }

        // Loop brackets: depth from what each contains, ink from its series.
        let markers = &payload.loop_markers;
        let bracket_line = |r: &Renderer| r.styled(r.plotted_line());
        let lefts = self.sync_group(axes, Group::LoopLeft, markers.len(), bracket_line);
        let rights = self.sync_group(axes, Group::LoopRight, markers.len(), bracket_line);
        let notes = self.sync_group(axes, Group::LoopLabels, markers.len(), |r| r.rich_text(Some((0.0, 0.0))));
        let depths = loop_depths(&markers.iter().map(|marker| (marker.start, marker.stop)).collect::<Vec<_>>());
        for (index, marker) in markers.iter().enumerate() {
            let colour = match marker.series { None => palette.pulse_run, Some(series) => palette.bracket_color(series as usize) };
            let outer_depth = depths[index] as f64;
            let y_low = frame_low - pulse.repeat_bottom_step * outer_depth;
            let y_high = frame_high;
            let lift_pt = pulse.repeat_top_step_pt * outer_depth;
            for (id, at_start) in [(lefts[index], true), (rights[index], false)] {
                let side = self.styled_mut(axes, id);
                side.bracket = Some(ir::Bracket {
                    start: marker.start, stop: marker.stop, at_start, y_low, y_high, lift_pt,
                    foot_fraction: pulse.repeat_foot_axes_fraction, max_foot_fraction: pulse.repeat_max_foot_fraction,
                });
                side.line.color = colour.into();
                side.line.alpha = Some(pulse.repeat_alpha);
                side.line.linewidth = pulse.repeat_linewidth;
                side.solid_capstyle = CapStyle::Round;
                side.line.zorder = pulse.repeat_bracket_zorder + index as f64;
            }
            let note = self.rich_mut(axes, notes[index]);
            note.anchor = Some((marker.stop, y_high));
            let (offset_x, offset_y) = pulse.repeat_label_offset_pt;
            note.text.x = offset_x;
            note.text.y = offset_y + lift_pt;
            note.text.text = marker.label.clone();
            note.text.ha = HAlign::Right;
            note.text.va = VAlign::Bottom;
            note.alpha = Some(if marker.series.is_none() { 1.0 } else { pulse.repeat_alpha });
            note.text.clip_on = true;
            note.text.zorder = pulse.repeat_label_zorder + index as f64;
            note.text.size = style.fonts.pulse_repeat_pt;
            note.text.color = colour.into();
            note.text.visible = !marker.label.is_empty();
        }

        // The limits: every loop's foot inside the axes, room above for the
        // stacked top rails and the outermost label.
        self.fig.axes[axes].set_xlim(left_limit, right_limit);
        let mut top_limit = (row_count as f64 + band) + pulse.ylim_top_offset;
        let mut bottom_limit = pulse.ylim_bottom;
        if !markers.is_empty() {
            let deepest = *depths.iter().max().expect("loops") as f64;
            let lowest_foot = pulse.repeat_bottom - pulse.repeat_bottom_step * deepest;
            let margin = pulse.repeat_bottom - pulse.repeat_bottom_step - pulse.ylim_bottom;
            bottom_limit = min2(bottom_limit, lowest_foot - margin);
            let room_pt = pulse.repeat_top_step_pt * deepest + pulse.repeat_ylim_room_pt;
            top_limit = frame_high + self.room_above(axes, room_pt, frame_high - bottom_limit);
        }
        self.fig.axes[axes].set_ylim(bottom_limit, top_limit);

        // The rows: a fixed tick each, the name in the text ink, the trace's
        // ink on the mark.
        let ticks: Vec<f64> = row_keys.iter().map(|key| row_index[key.as_str()] as f64).collect();
        let names: Vec<String> = payload.channels.iter().map(|channel| channel.label.clone())
            .chain(payload.analog_traces.iter().map(|trace| trace.label.clone())).collect();
        let fontsize = max2(pulse.ytick_font_floor_pt, style.fonts.tick_pt - pulse.ytick_font_delta_pt);
        let ink = self.fig.axes[axes].xaxis.label_color.clone();
        {
            let yaxis = &mut self.fig.axes[axes].yaxis;
            yaxis.policy = TickPolicy::Named { ticks: ticks.clone(), texts: names };
            for tick in yaxis.tick_objects(ticks.len()) { tick.label_size = [Some(fontsize), Some(fontsize)]; }
            for (tick, key) in yaxis.tick_objects(ticks.len()).iter_mut().zip(&row_keys) {
                tick.mark_color = Some(row_colors[key.as_str()].into());
                tick.label_color[0] = Some(ink.clone());
            }
        }

        // The time axis.
        let display_unit = text_value(state.get("x_display_unit"));
        let (factor, unit) = payload.time_scale(display_unit.as_deref(), Some(span))?;
        let model = &mut self.fig.axes[axes];
        model.xaxis.label = format!("Time ({unit})");
        model.xaxis.label_size = style.fonts.axis_label_pt;
        model.yaxis.label = String::new();
        model.xaxis.policy = TickPolicy::PulseTime { nbins: pulse.xtick_count, factor };
        for params in [&mut model.xaxis.major, &mut model.xaxis.minor] {
            params.tick1on = true;
            params.tick2on = false;
            params.label1on = true;
            params.label2on = false;
            params.pad = pulse.xtick_pad_pt;
        }
        model.axis_zorder = Some(0.5);
        model.yaxis.major.grid_on = false;
        model.spine_visible = [true, false, true, false];

        // The operator's labels over the time base.
        if let Some(mut x_label) = state_label(text_value(state.get("x_label")).as_deref(), None) {
            if !x_label.is_empty() {
                let (_, unit) = payload.time_scale(display_unit.as_deref(), None)?;
                if !has_explicit_unit_suffix(&x_label) { x_label = format!("{x_label} ({unit})"); }
            }
            self.fig.axes[axes].xaxis.label = x_label;
        }
        if let Some(y_label) = state_label(text_value(state.get("y_label")).as_deref(), None) {
            self.fig.axes[axes].yaxis.label = y_label;
        }
        Ok(())
    }

    /// `_effective_labels` of a pulse timeline: the time label with its unit.
    pub(crate) fn pulse_labels(&self, payload: &PulseTimelineData, state: &DisplayState) -> (String, String) {
        let display_unit = text_value(state.get("x_display_unit"));
        let unit = payload.time_scale(display_unit.as_deref(), None).map(|(_, unit)| unit).expect("the time base its render resolved");
        let mut x_label = state_label(text_value(state.get("x_label")).as_deref(), Some(&format!("Time ({unit})"))).unwrap_or_default();
        if !x_label.is_empty() && !has_explicit_unit_suffix(&x_label) { x_label = format!("{x_label} ({unit})"); }
        let y_label = state_label(text_value(state.get("y_label")).as_deref(), Some("")).unwrap_or_default();
        (x_label, y_label)
    }

    /// The factor a pulse timeline's selector readouts print source time by.
    pub(crate) fn pulse_label_factor(&self) -> f64 {
        match (&self.spec, self.last_payload.as_deref(), &self.last_state) {
            (crate::configuration::PlotSpec::PulseTimeline(_), Some(Payload::Pulse(data)), Some(state)) => {
                let display_unit = text_value(state.get("x_display_unit"));
                data.data.time_scale(display_unit.as_deref(), None).map(|(factor, _)| factor).expect("the time base its render resolved")
            }
            _ => 1.0,
        }
    }
}

fn text_value(value: Option<&Value>) -> Option<String> { super::core::text_value(value) }
