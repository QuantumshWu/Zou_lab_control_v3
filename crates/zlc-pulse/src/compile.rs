//! Lower a sequence to the frozen period-table program.
//!
//! The board plays a PERIOD TABLE: one row per authored period holding how
//! long the row lasts (a tick count, or which scan slot supplies it), the TTL
//! levels it holds, and what each DAC bus does when the row is entered.
//! Brackets are a separate loop table the board's row walker follows. A scan
//! point is the plain vector of slot values the rows read at run time.
use std::collections::{BTreeMap, BTreeSet};

use crate::error::{Error, Kind, Result, repr, repr_tuple, type_error, value_error};
use crate::exact::{BigUint, Decimal, big_to_f64};
use crate::json::Json;
use crate::loops::{Loop, frame_ticks, frame_ticks_big, frame_visits, loop_nesting_depth};
use crate::model::{AnalogMode, FieldKind, FieldRef, MAXIMUM_REPEAT_COUNT, MINIMUM_BRACKET_COUNT, PortKind, Sequence, exact_ticks};
use crate::wire::{StreamerParams, build_fingerprint};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TargetBusDelay { pub bus_index: u64, pub delay_ticks: u64 }

/// What one DAC bus does when one row is entered: `edge` takes the value at the
/// row's first tick; `ramp` walks from the held level to the value over the
/// row. `value` is the offset-binary code unless `value_select` names a scan
/// slot (`k` = slot `k-1`) that supplies it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TargetBusAction { pub row: u64, pub bus_index: u64, pub bus_name: String, pub mode: AnalogMode, pub value: u64, pub value_select: u64 }

/// The compiled program: everything the board plays, and nothing a document
/// says that the board does not (a renamed period is the same program).
#[derive(Clone, Debug, PartialEq)]
pub struct CompiledProgram {
    pub clock_hz: f64,
    pub target_abi_fingerprint: String,
    pub geometry_fingerprint: u32,
    pub channels: Vec<String>,
    /// Per row: how many ticks it lasts when no slot supplies the duration.
    pub durations: Vec<u64>,
    /// Per row: `0` plays `durations[row]`; `k` plays scan slot `k-1`.
    pub duration_slots: Vec<u64>,
    /// Per row: the TTL levels held for the whole row (raw-lane bits).
    pub masks: Vec<u128>,
    /// `(first_row, last_row, count)` per bracket, outermost first.
    pub loops: Vec<Loop>,
    pub duration_seconds: f64,
    pub slot_kinds: Vec<FieldKind>,
    pub bus_names: Vec<String>,
    pub bus_actions: Vec<TargetBusAction>,
    pub bus_delays: Vec<TargetBusDelay>,
    pub channel_delays: Vec<u64>,
    pub clk_enable: u128,
    pub logical_digital_outputs: Vec<(String, String)>,
    pub bus_safe_values: Vec<u64>,
}

impl CompiledProgram {
    /// `CompiledProgram.__post_init__`: what any program, however it was made
    /// (compiled here or received from a client), must satisfy.
    pub fn validate(&self) -> Result<()> {
        if !self.clock_hz.is_finite() || self.clock_hz <= 0.0 { return value_error("clock_hz must be positive and finite"); }
        if self.channels.is_empty() || self.channels.iter().collect::<BTreeSet<_>>().len() != self.channels.len() {
            return value_error("program channels must be unique and non-empty");
        }
        let rows = self.durations.len();
        if rows == 0 || self.duration_slots.len() != rows || self.masks.len() != rows {
            return value_error("program rows must be non-empty and of equal lengths");
        }
        if self.durations.iter().any(|value| !(1..=MAXIMUM_REPEAT_COUNT).contains(value)) {
            return value_error("every row duration must be from 1 through 2^32-1 ticks");
        }
        for &selector in &self.duration_slots {
            if selector > self.slot_kinds.len() as u64 { return value_error("a row names a duration slot the program does not have"); }
            if selector > 0 && self.slot_kinds[selector as usize - 1] != FieldKind::Duration {
                return value_error("a row duration can only come from a duration slot");
            }
        }
        for &(start, end, count) in &self.loops {
            if !(start <= end && end < rows) { return value_error("a loop lies outside the row table"); }
            if !(MINIMUM_BRACKET_COUNT..=MAXIMUM_REPEAT_COUNT).contains(&count) {
                return value_error(format!("a loop count must be from {MINIMUM_BRACKET_COUNT} through 2^32-1"));
            }
        }
        for (index, &(start, end, _)) in self.loops.iter().enumerate() {
            for &(other_start, other_end, _) in &self.loops[index + 1..] {
                let nested = (start <= other_start && other_end <= end) || (other_start <= start && end <= other_end);
                if !(nested || end < other_start || other_end < start) {
                    return value_error("loops overlap without one lying inside the other");
                }
            }
        }
        let mut sorted = self.loops.clone();
        sorted.sort_by_key(|&(start, end, _)| (start, std::cmp::Reverse(end)));
        if sorted != self.loops { return value_error("loops must be stored outermost first"); }
        if self.channel_delays.len() != self.channels.len() { return value_error("channel delay vector must match channels"); }
        let mut placed = BTreeSet::new();
        for action in &self.bus_actions {
            if action.row >= rows as u64 { return value_error("a bus action names a row outside the table"); }
            if self.bus_names.get(action.bus_index as usize) != Some(&action.bus_name) {
                return value_error("a bus action names a bus the program does not have");
            }
            if action.value_select > self.slot_kinds.len() as u64 { return value_error("a bus action names a slot the program does not have"); }
            if action.value_select > 0 && self.slot_kinds[action.value_select as usize - 1] != FieldKind::Dac {
                return value_error("a bus value can only come from a DAC slot");
            }
            if !placed.insert((action.row, action.bus_index)) { return value_error("a row has at most one action per DAC bus"); }
        }
        if self.bus_safe_values.len() != self.bus_names.len() { return value_error("one safe value per DAC bus"); }
        if !self.duration_seconds.is_finite() || self.duration_seconds < 0.0 {
            return value_error("duration_seconds must be a finite non-negative number");
        }
        Ok(())
    }

    pub fn slot_count(&self) -> usize { self.slot_kinds.len() }
    pub fn row_count(&self) -> usize { self.durations.len() }
    pub fn loop_depth(&self) -> usize { loop_nesting_depth(&self.loops) }

    /// How long each row lasts for one scan point, in ticks.
    pub fn resolved_durations(&self, point: &[u64]) -> Result<Vec<u64>> {
        if point.len() != self.slot_count() {
            return value_error(format!("a scan point has one value per slot: {} slot(s), {} value(s)", self.slot_count(), point.len()));
        }
        Ok(self.durations.iter().zip(&self.duration_slots)
            .map(|(&literal, &selector)| if selector > 0 { point[selector as usize - 1] } else { literal }).collect())
    }

    /// The offset-binary code one action drives for one scan point.
    pub fn resolved_bus_value(&self, action: &TargetBusAction, point: &[u64]) -> u64 {
        if action.value_select > 0 { point[action.value_select as usize - 1] } else { action.value }
    }

    pub fn frame_visits(&self, point: &[u64], bracket_bodies: Option<&dyn Fn(u128) -> u64>) -> Result<Vec<(usize, u128)>> {
        frame_visits(&self.resolved_durations(point)?, &self.loops, bracket_bodies)
    }

    pub fn frame_ticks(&self, point: &[u64]) -> Result<u128> { frame_ticks(&self.resolved_durations(point)?, &self.loops) }

    /// The canonical tree `canonical_digest` names (field order is irrelevant: keys sort).
    pub fn canonical_tree(&self) -> Json {
        Json::object(vec![
            ("clock_hz", Json::Float(self.clock_hz)),
            ("target_abi_fingerprint", Json::str(&self.target_abi_fingerprint)),
            ("geometry_fingerprint", Json::Int(self.geometry_fingerprint.into())),
            ("channels", Json::strings(&self.channels)),
            ("durations", Json::ints(&self.durations)),
            ("duration_slots", Json::ints(&self.duration_slots)),
            ("masks", Json::Array(self.masks.iter().map(|mask| Json::Int(*mask as i128)).collect())),
            ("loops", Json::Array(self.loops.iter().map(|&(start, end, count)| Json::Array(vec![
                Json::Int(start as i128), Json::Int(end as i128), Json::Int(count.into())])).collect())),
            ("duration_seconds", Json::Float(self.duration_seconds)),
            ("slot_kinds", Json::Array(self.slot_kinds.iter().map(|kind| Json::str(kind.name())).collect())),
            ("bus_names", Json::strings(&self.bus_names)),
            ("bus_actions", Json::Array(self.bus_actions.iter().map(|action| Json::object(vec![
                ("row", Json::Int(action.row.into())), ("bus_index", Json::Int(action.bus_index.into())),
                ("bus_name", Json::str(&action.bus_name)), ("mode", Json::str(action.mode.name())),
                ("value", Json::Int(action.value.into())), ("value_select", Json::Int(action.value_select.into())),
            ])).collect())),
            ("bus_delays", Json::Array(self.bus_delays.iter().map(|delay| Json::object(vec![
                ("bus_index", Json::Int(delay.bus_index.into())), ("delay_ticks", Json::Int(delay.delay_ticks.into())),
            ])).collect())),
            ("channel_delays", Json::ints(&self.channel_delays)),
            ("clk_enable", Json::Int(self.clk_enable as i128)),
            ("logical_digital_outputs", Json::Array(self.logical_digital_outputs.iter()
                .map(|(key, lane)| Json::Array(vec![Json::str(key), Json::str(lane)])).collect())),
            ("bus_safe_values", Json::ints(&self.bus_safe_values)),
        ])
    }

    /// One short name for exactly what this program plays.
    pub fn digest(&self) -> String { crate::canonical::digest(&self.canonical_tree()) }
}

fn slot_index(sequence: &Sequence) -> BTreeMap<FieldRef, u64> {
    sequence.scan_bindings().iter().enumerate().map(|(index, binding)| (binding.field_ref.clone(), index as u64)).collect()
}

/// The authored value of every scan slot as the wire carries it: a duration
/// slot holds the period's tick count, a DAC slot the step's offset-binary code.
pub fn nominal_slot_values(sequence: &Sequence) -> Result<Vec<u64>> {
    let mut values = Vec::new();
    for slot in sequence.scan_bindings() {
        let reference = &slot.field_ref;
        let period_id = reference.period_id.as_deref().unwrap_or("");
        match reference.kind {
            FieldKind::Duration => {
                let period = sequence.period(period_id).expect("validated slot");
                let ticks = exact_ticks(period.duration, &period.unit, sequence.time_step_ns, &format!("period {} duration", period.period_id), Some(1))?;
                values.push(u64::try_from(ticks).map_err(|_| Error::new(Kind::Value, "every row duration must be from 1 through 2^32-1 ticks"))?);
            }
            FieldKind::Dac => {
                let period = sequence.period(period_id).expect("validated slot");
                let port_key = reference.port.as_deref().unwrap_or("");
                let Some(step) = period.step(port_key) else {
                    return value_error(format!(
                        "scan slot {} names the DAC field {} of period {}, which has no step on that port",
                        repr(&slot.field_id()), repr(port_key), repr(period_id)));
                };
                let low = sequence.target.port(port_key).and_then(|port| port.signed_range()).expect("a DAC").0;
                values.push((step.value - low) as u64);
            }
            other => return value_error(format!("unsupported scan slot kind {}", repr(other.name()))),
        }
    }
    Ok(values)
}

/// The level each DAC port holds from every tick it changes, over one pass,
/// in signed codes, opening at `(0, 0)`. A ramp is the engine's staircase from
/// the level carried in to the step's value at the period end.
pub fn analog_levels(sequence: &Sequence) -> Result<Vec<(String, Vec<(i128, i64)>)>> {
    let mut boundaries = vec![0i128];
    for period in &sequence.periods {
        let ticks = exact_ticks(period.duration, &period.unit, sequence.time_step_ns, &format!("period {} duration", period.period_id), Some(1))?;
        boundaries.push(boundaries.last().unwrap() + ticks);
    }
    let mut levels = Vec::new();
    for port in sequence.target.ports.iter().filter(|port| port.kind == PortKind::Dac) {
        let mut points: Vec<(i128, i64)> = vec![(0, 0)];
        let mut change = |tick: i128, value: i64| {
            let last = points.last_mut().expect("opened");
            if last.0 == tick { *last = (tick, value); } else if last.1 != value { points.push((tick, value)); }
        };
        let mut held: i64 = 0;
        for (index, period) in sequence.periods.iter().enumerate() {
            let Some(step) = period.step(&port.key) else { continue };
            let (start, stop) = (boundaries[index], boundaries[index + 1]);
            let target = step.value;
            if step.mode == AnalogMode::Edge { change(start, target); } else {
                let span = stop - start;
                let distance = i128::from((target - held).abs());
                let direction: i64 = if target >= held { 1 } else { -1 };
                if distance <= span {
                    for level in 1..=distance {
                        // `start - (-level * span // distance)`: the ceiling of level * span / distance.
                        change(start + (level * span + distance - 1) / distance, held + direction * level as i64);
                    }
                } else {
                    for k in 1..=span { change(start + k, held + direction * (k * distance / span) as i64); }
                }
            }
            held = target;
        }
        levels.push((port.key.clone(), points));
    }
    Ok(levels)
}

fn delay_values(sequence: &Sequence) -> Result<(Vec<u64>, Vec<TargetBusDelay>)> {
    let target = &sequence.target;
    let mut raw_lane: BTreeMap<usize, i128> = BTreeMap::new();
    let mut raw_bus: BTreeMap<u64, i128> = BTreeMap::new();
    for delay in &sequence.delays {
        let ticks = exact_ticks(delay.value, &delay.unit, sequence.time_step_ns, "output delay", None)?;
        let port = target.port(&delay.port).expect("validated delay port");
        if port.kind == PortKind::Dac { raw_bus.insert(port.bus_index.expect("a DAC"), ticks); }
        else { for lane in &port.lanes { raw_lane.insert(target.lane_index(lane).expect("a lane"), ticks); } }
    }
    for period in &sequence.periods {
        for (index, state) in period.states.iter().enumerate() { if *state != 0 { raw_lane.entry(index).or_insert(0); } }
        for step in &period.analog_steps {
            raw_bus.entry(target.port(&step.port).and_then(|port| port.bus_index).expect("a DAC")).or_insert(0);
        }
    }
    let shift = raw_lane.values().chain(raw_bus.values()).min().map_or(0, |least| (-least).max(0));
    let to_ticks = |value: i128| u64::try_from(value + shift).map_err(|_| Error::new(Kind::Value, "output delay is too long for the device clock"));
    let mut channels = vec![0u64; target.raw_lanes.len()];
    for (&index, &value) in &raw_lane { channels[index] = to_ticks(value)?; }
    let mut buses = Vec::new();
    for (&bus_index, &value) in &raw_bus {
        let delay_ticks = to_ticks(value)?;
        if delay_ticks != 0 { buses.push(TargetBusDelay { bus_index, delay_ticks }); }
    }
    Ok((channels, buses))
}

/// `compile_sequence(sequence, geom, clock_hz)`: compile once; slot rows are
/// data written later by the device.
pub fn compile_sequence(sequence: &Sequence, params: &StreamerParams, clock_hz: f64) -> Result<CompiledProgram> {
    let api = sequence.api_bindings();
    if !api.is_empty() {
        let declared: Vec<String> = api.iter().map(|binding| binding.field_id()).collect();
        let declared: Vec<&str> = declared.iter().map(String::as_str).collect();
        return value_error(format!("pulse API parameters must be resolved before compile: {}", repr_tuple(&declared)));
    }
    let target = &sequence.target;
    if target.raw_lanes.len() as u64 > params.channel_count { return value_error("sequence has more lanes than the streamer geometry"); }
    let dacs: Vec<_> = target.ports.iter().filter(|port| port.kind == PortKind::Dac).collect();
    if dacs.len() as u64 > params.bus_count { return value_error("sequence has more DAC buses than the streamer geometry"); }
    if dacs.iter().any(|port| port.width > params.bus_width) { return value_error("a DAC port is wider than the streamer geometry"); }
    if !clock_hz.is_finite() || clock_hz <= 0.0 { return value_error("clock_hz must be positive and finite"); }
    let tick = Decimal::from_f64(sequence.time_step_ns).mul(&Decimal::from_f64(clock_hz));
    if tick != Decimal::from_i128(1_000_000_000) { return value_error("sequence time_step_ns does not match the compiler clock_hz"); }
    if sequence.slot_count() as u64 > params.num_slots {
        return value_error(format!("sequence binds {} scan slot(s) but the streamer geometry holds {}", sequence.slot_count(), params.num_slots));
    }
    if sequence.periods.len() as u64 > params.max_rows {
        return value_error(format!("sequence has {} periods but the streamer geometry holds {} rows", sequence.periods.len(), params.max_rows));
    }
    let loops = sequence.loops();
    if loops.len() as u64 > params.max_loops {
        return value_error(format!("sequence loops {} brackets but the streamer geometry holds {} loops", loops.len(), params.max_loops));
    }
    let depth = loop_nesting_depth(&loops);
    if depth as u64 > params.loop_depth {
        return value_error(format!("brackets nest {depth} deep but the streamer geometry holds {} levels", params.loop_depth));
    }
    let binding = slot_index(sequence);
    let nominal = nominal_slot_values(sequence)?;
    let mut clk_enable: u128 = 0;
    for port in target.ports.iter().filter(|port| port.kind == PortKind::Clock) {
        clk_enable |= 1u128 << target.lane_index(&port.lanes[0]).expect("a lane");
    }
    let mut buses = dacs.clone();
    buses.sort_by_key(|port| port.bus_index);
    let bus_names: Vec<String> = buses.iter().map(|port| port.key.clone()).collect();
    let (mut durations, mut duration_slots, mut masks, mut actions) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut wide = false;
    for (row, period) in sequence.periods.iter().enumerate() {
        let ticks = exact_ticks(period.duration, &period.unit, sequence.time_step_ns, &format!("period {} duration", period.period_id), Some(1))?;
        wide |= ticks > i128::from(MAXIMUM_REPEAT_COUNT);
        durations.push(u64::try_from(ticks).unwrap_or(u64::MAX));
        duration_slots.push(binding.get(&FieldRef::duration(&period.period_id)).map_or(0, |slot| slot + 1));
        let mask = period.states.iter().enumerate().filter(|(_, state)| **state != 0).fold(0u128, |mask, (index, _)| mask | (1 << index));
        masks.push(mask & !clk_enable);
        for port in &buses {
            let Some(step) = period.step(&port.key) else { continue };
            let selector = binding.get(&FieldRef::dac(&period.period_id, &port.key));
            let low = port.signed_range().expect("a DAC").0;
            actions.push(TargetBusAction {
                row: row as u64, bus_index: port.bus_index.expect("a DAC"), bus_name: port.key.clone(), mode: step.mode,
                value: if selector.is_some() { 0 } else { (step.value - low) as u64 },
                value_select: selector.map_or(0, |slot| slot + 1),
            });
        }
    }
    let (channel_delays, bus_delays) = delay_values(sequence)?;
    if wide { return value_error("every row duration must be from 1 through 2^32-1 ticks"); }
    let mut logical: Vec<(String, String)> = target.ports.iter().filter(|port| port.kind == PortKind::Digital)
        .map(|port| (port.key.clone(), port.lanes[0].clone())).collect();
    logical.sort();
    let nominal_durations: Vec<BigUint> = durations.iter().zip(&duration_slots)
        .map(|(&literal, &selector)| BigUint::from_u128(u128::from(if selector > 0 { nominal[selector as usize - 1] } else { literal })))
        .collect();
    let program = CompiledProgram {
        clock_hz,
        target_abi_fingerprint: target.abi_fingerprint().to_owned(),
        geometry_fingerprint: build_fingerprint(params)?,
        channels: target.raw_lanes.clone(),
        durations,
        duration_slots,
        masks,
        duration_seconds: big_to_f64(&frame_ticks_big(&nominal_durations, &loops)) / clock_hz,
        loops,
        slot_kinds: sequence.scan_bindings().iter().map(|binding| binding.kind()).collect(),
        bus_names,
        bus_actions: actions,
        bus_delays,
        channel_delays,
        clk_enable,
        logical_digital_outputs: logical,
        bus_safe_values: buses.iter().map(|port| port.safe_value).collect(),
    };
    program.validate()?;
    Ok(program)
}

/// `TargetBusAction.__post_init__` on decoded values.
pub fn bus_action(row: i128, bus_index: i128, bus_name: &str, mode: &str, value: i128, value_select: i128) -> Result<TargetBusAction> {
    let check = |item: i128, name: &str| u64::try_from(item).map_err(|_| Error::new(Kind::Value, format!("{name} must be a non-negative integer")));
    let (row, bus_index, value, value_select) = (check(row, "row")?, check(bus_index, "bus_index")?, check(value, "value")?, check(value_select, "value_select")?);
    if bus_name.is_empty() { return value_error("bus_name must be non-empty text"); }
    let Some(mode) = AnalogMode::parse(mode) else { return value_error("bus action mode must be 'edge' or 'ramp'"); };
    Ok(TargetBusAction { row, bus_index, bus_name: bus_name.to_owned(), mode, value, value_select })
}

/// `TargetBusDelay.__post_init__` on decoded values.
pub fn bus_delay(bus_index: i128, delay_ticks: i128) -> Result<TargetBusDelay> {
    let bus_index = u64::try_from(bus_index).map_err(|_| Error::new(Kind::Value, "bus_index must be a non-negative integer"))?;
    let delay_ticks = u64::try_from(delay_ticks).map_err(|_| Error::new(Kind::Value, "delay_ticks must be a non-negative integer"))?;
    Ok(TargetBusDelay { bus_index, delay_ticks })
}

/// The type refusal the remote decoder gives a field that is not what it should be.
pub fn field_type<T>(name: &str) -> Result<T> { type_error(format!("{name} has the wrong type")) }
