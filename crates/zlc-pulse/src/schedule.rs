//! Pure finite timing queries of a compiled program. Every query walks one
//! table in hardware order: each row holds for `run_repeats` complete Pulse
//! timelines, then the table replays `scan_repeats` times. An infinite count
//! has no finite projection.
use std::collections::BTreeMap;

use crate::compile::CompiledProgram;
use crate::error::{Error, Kind, Result, repr, type_error, value_error};
use crate::exact::{BigUint, big_to_f64};
use crate::model::MAXIMUM_REPEAT_COUNT;

fn finite_repeat_count(value: u64, field_name: &str) -> Result<u64> {
    if !(1..=MAXIMUM_REPEAT_COUNT).contains(&value) {
        return value_error(format!("{field_name} must be in the finite hardware range [1, 2^32-1]"));
    }
    Ok(value)
}

/// Finite execution rows in `scan -> row -> run` order.
fn scan_points(program: &CompiledProgram, table: Option<&[Vec<u64>]>, run_repeats: u64, scan_repeats: u64) -> Result<Vec<Vec<u64>>> {
    let run_repeats = finite_repeat_count(run_repeats, "run_repeats")?;
    let scan_repeats = finite_repeat_count(scan_repeats, "scan_repeats")?;
    let rows: Vec<Vec<u64>> = match table {
        None => {
            if program.slot_count() > 0 { return value_error("a slotted program requires an explicit value table"); }
            if scan_repeats != 1 { return value_error("scan_repeats must be 1 when there is no scan table"); }
            vec![Vec::new()]
        }
        Some(rows) => {
            if rows.iter().any(|row| row.len() != program.slot_count()) { return value_error("table width differs from the program slot count"); }
            if rows.is_empty() { return value_error("table must contain at least one row"); }
            rows.to_vec()
        }
    };
    let sweep: Vec<Vec<u64>> = rows.iter().flat_map(|row| std::iter::repeat_n(row.clone(), run_repeats as usize)).collect();
    Ok((0..scan_repeats).flat_map(|_| sweep.iter().cloned()).collect())
}

type Shapes = BTreeMap<Vec<u64>, (Vec<(usize, u128)>, u128)>;

/// Each distinct point's row visits and Pulse length, derived once.
fn point_shapes(program: &CompiledProgram, points: &[Vec<u64>], bracket_bodies: Option<&dyn Fn(u128) -> u64>) -> Result<Shapes> {
    let mut shapes = Shapes::new();
    for point in points {
        if !shapes.contains_key(point) {
            shapes.insert(point.clone(), (program.frame_visits(point, bracket_bodies)?, program.frame_ticks(point)?));
        }
    }
    Ok(shapes)
}

fn overflow() -> Error { Error::new(Kind::Value, "this run lasts longer than the schedule can count") }

/// The tick of every edge each named lane plays (rise and fall alike), the
/// lane's output delay included. A level carried across a Pulse seam does not
/// toggle; only the end of a finite run lowers every lane.
fn channel_edges(program: &CompiledProgram, channels: &[&str], table: Option<&[Vec<u64>]>, run_repeats: u64, scan_repeats: u64,
                 bracket_bodies: Option<&dyn Fn(u128) -> u64>) -> Result<Vec<Vec<u128>>> {
    let mut bits = Vec::new();
    for channel in channels {
        let physical = program.logical_digital_outputs.iter().find(|(key, _)| key == channel).map_or(*channel, |(_, lane)| lane.as_str());
        let Some(bit) = program.channels.iter().position(|lane| lane == physical) else { return value_error(format!("unknown channel {}", repr(channel))); };
        if program.clk_enable & (1 << bit) != 0 { return value_error("clock lanes do not have digital trigger results"); }
        bits.push(bit);
    }
    let points = scan_points(program, table, run_repeats, scan_repeats)?;
    let shapes = point_shapes(program, &points, bracket_bodies)?;
    let mut streams = Vec::new();
    for bit in bits {
        let delay = u128::from(program.channel_delays[bit]);
        let levels: Vec<bool> = program.masks.iter().map(|mask| (mask >> bit) & 1 == 1).collect();
        let mut ticks = Vec::new();
        let mut previous = false;
        let mut offset: u128 = 0;
        for point in &points {
            let (visits, total) = &shapes[point];
            for &(row, tick) in visits {
                let level = levels[row];
                if level != previous { previous = level; ticks.push(offset + tick + delay); }
            }
            offset = offset.checked_add(*total).ok_or_else(overflow)?;
        }
        if previous { ticks.push(offset + delay); }
        streams.push(ticks);
    }
    Ok(streams)
}

/// `trigger_edge_ticks(program, channels, table, run_repeats=, scan_repeats=, bracket_bodies=)`.
pub fn trigger_edge_ticks(program: &CompiledProgram, channels: &[&str], table: Option<&[Vec<u64>]>, run_repeats: u64, scan_repeats: u64,
                          bracket_bodies: Option<&dyn Fn(u128) -> u64>) -> Result<Vec<(String, Vec<u128>)>> {
    let streams = channel_edges(program, channels, table, run_repeats, scan_repeats, bracket_bodies)?;
    Ok(channels.iter().map(|name| name.to_string()).zip(streams).collect())
}

/// `trigger_windows_by_channel`: the exposure windows of several lanes from one walk.
pub fn trigger_windows_by_channel(program: &CompiledProgram, channels: &[&str], table: Option<&[Vec<u64>]>, run_repeats: u64,
                                  scan_repeats: u64) -> Result<Vec<(String, Vec<(u128, u128)>)>> {
    let streams = channel_edges(program, channels, table, run_repeats, scan_repeats, None)?;
    channels.iter().zip(streams).map(|(name, edges)| {
        let windows: Vec<(u128, u128)> = edges.chunks(2).filter(|pair| pair.len() == 2).map(|pair| (pair[0], pair[1])).collect();
        if windows.iter().any(|(start, end)| end <= start) { return value_error("program contains a non-positive trigger window"); }
        Ok((name.to_string(), windows))
    }).collect()
}

/// `bus_action_ticks`: the tick every DAC bus is handed an action, per bus,
/// in playback order (undelayed: the FIFO fills at that tick).
pub fn bus_action_ticks(program: &CompiledProgram, table: Option<&[Vec<u64>]>, run_repeats: u64, scan_repeats: u64,
                        bracket_bodies: Option<&dyn Fn(u128) -> u64>) -> Result<Vec<Vec<u128>>> {
    let mut by_row: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for action in &program.bus_actions { by_row.entry(action.row as usize).or_default().push(action.bus_index as usize); }
    let mut ticks = vec![Vec::new(); program.bus_names.len()];
    let points = scan_points(program, table, run_repeats, scan_repeats)?;
    let shapes = point_shapes(program, &points, bracket_bodies)?;
    let mut offset: u128 = 0;
    for point in &points {
        let (visits, total) = &shapes[point];
        for &(row, tick) in visits {
            for &bus in by_row.get(&row).map(Vec::as_slice).unwrap_or(&[]) {
                ticks.get_mut(bus).ok_or_else(|| Error::new(Kind::Value, "a bus action names a bus the program does not have"))?.push(offset + tick);
            }
        }
        offset = offset.checked_add(*total).ok_or_else(overflow)?;
    }
    Ok(ticks)
}

/// `run_duration_seconds`: the exact finite playback duration of one run.
pub fn run_duration_seconds(program: &CompiledProgram, table: Option<&[Vec<u64>]>, run_repeats: u64, scan_repeats: u64) -> Result<f64> {
    let mut total = BigUint::zero();
    for point in scan_points(program, table, run_repeats, scan_repeats)? {
        total = total.add(&BigUint::from_u128(program.frame_ticks(&point)?));
    }
    Ok(big_to_f64(&total) / program.clock_hz)
}

/// A table whose values are not integers is refused as v2 refused it.
pub fn integer_table(rows: &[Vec<f64>]) -> Result<Vec<Vec<u64>>> {
    rows.iter().map(|row| row.iter().map(|value| {
        if value.fract() != 0.0 || *value < 0.0 { type_error("table values must be integers") } else { Ok(*value as u64) }
    }).collect()).collect()
}

/// Refuse an application whose delayed events overflow the frozen FIFOs
/// (`PulseStreamer._validate_delay_capacity`). A queue entry stays live through
/// the edge it is emitted on, so capacity counts a closed `delay` window.
///
/// The walk is bounded: a delay window overlaps at most `delay / T + 2`
/// consecutive Pulses no shorter than `T` ticks, and after `depth` identical
/// ones a FIFO has overflowed or reached its periodic state, so each row's Run
/// repeats are cut to the tighter of the two around the final complete sweep;
/// brackets are cut the same way at their true ticks.
pub fn validate_delay_capacity(program: &CompiledProgram, rows: &[Vec<u64>], params: &crate::wire::StreamerParams, run_repeats: u64,
                               scan_repeats: u64) -> Result<()> {
    let ttl: Vec<(usize, u64)> = program.channel_delays.iter().enumerate().filter(|(_, delay)| **delay >= 2).map(|(bit, delay)| (bit, *delay)).collect();
    let bus_delays: BTreeMap<u64, u64> = program.bus_delays.iter().filter(|item| item.delay_ticks > 0).map(|item| (item.bus_index, item.delay_ticks)).collect();
    if ttl.is_empty() && bus_delays.is_empty() { return Ok(()); }
    let empty = [Vec::new()];
    let table: &[Vec<u64>] = if rows.is_empty() { &empty } else { rows };
    let depth = u128::from(params.evt_fifo_depth.max(params.bus_evt_fifo_depth));
    let longest = u128::from(ttl.iter().map(|(_, delay)| *delay).chain(bus_delays.values().copied()).max().unwrap_or(0));
    let mut shortest = u128::MAX;
    for point in table { shortest = shortest.min(program.frame_ticks(point)?); }
    let span = (depth + 1).min(longest / shortest + 2) as usize;
    let (execution_rows, finite_completion): (Vec<Vec<u64>>, bool) = if run_repeats == 0 {
        (vec![table[0].clone(); span], false)
    } else {
        let bounded = (run_repeats as usize).min(span);
        let one_sweep: Vec<Vec<u64>> = table.iter().flat_map(|row| std::iter::repeat_n(row.clone(), bounded)).collect();
        let preceding = if scan_repeats == 0 { span - 1 } else { (span - 1).min((scan_repeats as usize - 1).saturating_mul(one_sweep.len())) };
        let warmup: Vec<Vec<u64>> = if preceding > 0 {
            let copies = (preceding + one_sweep.len() - 1) / one_sweep.len();
            let repeated: Vec<Vec<u64>> = (0..copies).flat_map(|_| one_sweep.iter().cloned()).collect();
            repeated[repeated.len() - preceding..].to_vec()
        } else { Vec::new() };
        (warmup.into_iter().chain(one_sweep).collect(), scan_repeats != 0)
    };
    let kept_bodies = move |body: u128| -> u64 { ((depth + 2).min(longest / body.max(1) + 3)) as u64 };
    let physical_to_logical: BTreeMap<&str, &str> = program.logical_digital_outputs.iter().map(|(logical, physical)| (physical.as_str(), logical.as_str())).collect();
    for (bit, _) in &ttl {
        if *bit >= program.channels.len() { return value_error(format!("channel delay index {bit} is outside the program")); }
    }
    let asked: Vec<&str> = ttl.iter().map(|(bit, _)| physical_to_logical.get(program.channels[*bit].as_str()).copied().unwrap_or(&program.channels[*bit])).collect();
    if !ttl.is_empty() {
        let edges = trigger_edge_ticks(program, &asked, Some(&execution_rows), 1, 1, Some(&kept_bodies))?;
        for ((bit, delay), (_, events)) in ttl.iter().zip(&edges) {
            check_delay_window(events, u128::from(*delay), params.evt_fifo_depth, &format!("channel {}", repr(&program.channels[*bit])))?;
        }
    }
    if !bus_delays.is_empty() {
        let actions = bus_action_ticks(program, Some(&execution_rows), 1, 1, Some(&kept_bodies))?;
        let mut run_end: u128 = 0;
        for point in &execution_rows { run_end = run_end.checked_add(program.frame_ticks(point)?).ok_or_else(overflow)?; }
        for (bus, delay) in &bus_delays {
            let mut events = actions.get(*bus as usize).cloned().unwrap_or_default();
            if finite_completion { events.push(run_end); }
            check_delay_window(&events, u128::from(*delay), params.bus_evt_fifo_depth, &format!("DAC bus {bus}"))?;
        }
    }
    Ok(())
}

fn check_delay_window(events: &[u128], delay: u128, capacity: u64, label: &str) -> Result<()> {
    let mut first = 0;
    for (last, tick) in events.iter().enumerate() {
        while first < last && events[first] + delay < *tick { first += 1; }
        let required = last - first + 1;
        if required as u64 > capacity {
            return value_error(format!("{label} needs {required} delayed events in flight but the connected geometry holds {capacity}"));
        }
    }
    Ok(())
}
