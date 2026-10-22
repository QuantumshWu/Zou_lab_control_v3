//! Device-facing pulse model: physical lanes, periods, output actions and the
//! one binding per physical field. Every value is validated where it is made,
//! with the v2 refusals, so an editor can never hold a pulse the compiler would
//! refuse for a reason the model could have stated.
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::error::{Error, Kind, Result, is_identifier, repr, repr_tuple, type_error, value_error};
use crate::exact::{BigUint, Decimal, Ratio};
use crate::json::Json;
use crate::loops::frame_ticks_big;
use crate::num::Num;
use crate::units::{canonical_time_unit, nanoseconds_exponent};

pub const MINIMUM_BRACKET_COUNT: u64 = 1;
pub const MAXIMUM_REPEAT_COUNT: u64 = (1 << 32) - 1;
pub const DAC_OFFSET_BINARY: &str = "offset_binary";
pub const ANALOG_MODE_CHOICES: [&str; 2] = ["edge", "ramp"];

// ------------------------------------------------------------------ helpers
pub(crate) fn text<'a>(value: &'a str, field_name: &str) -> Result<&'a str> {
    if value.trim().is_empty() { return type_error(format!("{field_name} must be non-empty text")); }
    Ok(value)
}

pub(crate) fn identifier(value: &str, field_name: &str) -> Result<String> {
    text(value, field_name)?;
    if !is_identifier(value) { return value_error(format!("{field_name} must be an identifier")); }
    Ok(value.to_owned())
}

pub fn config_parameter_key(value: &str) -> Result<String> { identifier(value, "Config parameter name") }

/// `_nonnegative_int(value, field_name)` of an already integral value.
pub(crate) fn nonnegative(value: i128, field_name: &str) -> Result<u64> {
    if value < 0 { return value_error(format!("{field_name} must be non-negative")); }
    u64::try_from(value).map_err(|_| Error::new(Kind::Value, format!("{field_name} is outside the supported range")))
}

/// How many device-clock ticks `value unit` is, exactly (`_tick_ratio`).
fn tick_ratio(value: Num, unit: &str, time_step_ns: f64) -> Result<(bool, Ratio)> {
    if time_step_ns <= 0.0 { return value_error("time_step_ns must be positive"); }
    let nanoseconds = value.decimal().scale10(nanoseconds_exponent(unit)?);
    Ok((nanoseconds.negative, Ratio::of(&nanoseconds, &Decimal::from_f64(time_step_ns))))
}

fn ticks_from(negative: bool, magnitude: &BigUint, field_name: &str) -> Result<i128> {
    let value = magnitude.to_u128().filter(|value| *value < (1 << 126))
        .ok_or_else(|| Error::new(Kind::Value, format!("{field_name} is too long for the device clock")))? as i128;
    Ok(if negative { -value } else { value })
}

/// `exact_ticks(value, unit, time_step_ns, field_name, minimum=)`.
pub fn exact_ticks(value: Num, unit: &str, time_step_ns: f64, field_name: &str, minimum: Option<i128>) -> Result<i128> {
    let (negative, ratio) = tick_ratio(value, unit, time_step_ns)?;
    let Some(integer) = ratio.integer() else { return value_error(format!("{field_name} is not on the device clock grid")); };
    let ticks = ticks_from(negative, &integer, field_name)?;
    if let Some(minimum) = minimum {
        if ticks < minimum { return value_error(format!("{field_name} must be at least {minimum} tick(s)")); }
    }
    Ok(ticks)
}

/// `align_to_grid(value, unit, time_step_ns, field_name, minimum=)`: the
/// nearest value on the clock grid, in the unit it arrived in.
pub fn align_to_grid(value: Num, unit: &str, time_step_ns: f64, field_name: &str, minimum: Option<i128>) -> Result<f64> {
    let (negative, ratio) = tick_ratio(value, unit, time_step_ns)?;
    let rounded = ratio.round_half_up();
    let mut ticks = ticks_from(negative, &rounded, field_name)?;
    if let Some(minimum) = minimum { if ticks < minimum { ticks = minimum; } }
    let nanoseconds = Decimal::from_i128(ticks).mul(&Decimal::from_f64(time_step_ns));
    Ok(nanoseconds.scale10(-nanoseconds_exponent(unit)?).to_f64())
}

/// `nanoseconds_per(unit)` as a float (exact: a power of ten up to 1e9).
pub fn nanoseconds_per(unit: &str) -> Result<f64> { Ok(10f64.powi(nanoseconds_exponent(unit)? as i32)) }

// ------------------------------------------------------------------ ports
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PortKind { Digital, Dac, Clock }

impl PortKind {
    pub fn parse(text: &str) -> Option<PortKind> {
        match text { "digital" => Some(PortKind::Digital), "dac" => Some(PortKind::Dac), "clock" => Some(PortKind::Clock), _ => None }
    }
    pub fn name(self) -> &'static str { match self { PortKind::Digital => "digital", PortKind::Dac => "dac", PortKind::Clock => "clock" } }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PortSpec {
    pub key: String,
    pub kind: PortKind,
    pub lanes: Vec<String>,
    pub label: String,
    pub bus_index: Option<u64>,
    pub width: u64,
    pub encoding: String,
    pub safe_value: u64,
    pub latch_clock: Option<String>,
}

impl PortSpec {
    /// `PulsePortSpec(key, kind, lanes, label, bus_index, width, encoding, safe_value, latch_clock)`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(key: &str, kind: &str, lanes: &[String], label: &str, bus_index: Option<i128>, width: Option<i128>,
               encoding: Option<&str>, safe_value: Option<i128>, latch_clock: Option<&str>) -> Result<PortSpec> {
        let key = identifier(key, "port key")?;
        text(kind, "port kind")?;
        let Some(kind) = PortKind::parse(kind) else { return value_error("port kind must be one of ['clock', 'dac', 'digital']"); };
        let lanes = lanes.iter().map(|lane| identifier(lane, "raw lane")).collect::<Result<Vec<_>>>()?;
        if lanes.is_empty() || lanes.iter().collect::<BTreeSet<_>>().len() != lanes.len() {
            return value_error("port lanes must be unique and non-empty");
        }
        let width = match width { None => lanes.len() as u64, Some(width) => nonnegative(width, "port width")? };
        if width != lanes.len() as u64 { return value_error("port width differs from lane count"); }
        let encoding = encoding.filter(|text| !text.is_empty());
        let (bus_index, encoding, safe_value) = if kind == PortKind::Dac {
            if width < 2 { return value_error("DAC ports require at least two lanes"); }
            let bus_index = nonnegative(bus_index.unwrap_or(0), "DAC bus_index")?;
            let encoding = encoding.unwrap_or(DAC_OFFSET_BINARY);
            let mid_scale = 1u64 << (width - 1).min(63);
            let safe_value = match safe_value { None => mid_scale, Some(value) => nonnegative(value, "DAC safe_value")? };
            if encoding != DAC_OFFSET_BINARY { return value_error("only offset-binary DAC encoding is supported"); }
            if safe_value != mid_scale { return value_error(format!("a DAC port rests at its mid-scale code {mid_scale} (0 V)")); }
            (Some(bus_index), encoding, safe_value)
        } else {
            if width != 1 { return value_error("digital and clock ports require one lane"); }
            if bus_index.is_some() { return value_error("digital and clock ports cannot have a bus index"); }
            let encoding = encoding.unwrap_or("binary");
            let safe_value = match safe_value { None => 0, Some(value) => nonnegative(value, "safe_value")? };
            if encoding != "binary" || safe_value != 0 { return value_error("digital and clock ports require the low safe state"); }
            if latch_clock.is_some() { return value_error("digital and clock ports cannot have a latch clock"); }
            (None, encoding, 0)
        };
        let latch_clock = latch_clock.map(|clock| identifier(clock, "latch clock")).transpose()?;
        Ok(PortSpec { key, kind, lanes, label: label.to_owned(), bus_index, width, encoding: encoding.to_owned(), safe_value, latch_clock })
    }

    /// The signed code range of a DAC port.
    pub fn signed_range(&self) -> Option<(i64, i64)> {
        if self.kind != PortKind::Dac { return None; }
        let half = 1i64 << (self.width - 1).min(62);
        Some((-half, half - 1))
    }

    pub fn display(&self) -> &str { if self.label.is_empty() { &self.key } else { &self.label } }
}

/// A board's lanes and logical ports. Two targets are equal when their lanes
/// and ports are; the package pins describe wiring, not what a pulse drives.
#[derive(Debug)]
pub struct Target {
    pub raw_lanes: Vec<String>,
    pub ports: Vec<PortSpec>,
    /// Package pin per raw lane, in the order it was given; empty for a logical target.
    pub package_pins: Vec<(String, String)>,
    abi_fingerprint: String,
    by_key: BTreeMap<String, usize>,
}

impl PartialEq for Target {
    fn eq(&self, other: &Self) -> bool { self.raw_lanes == other.raw_lanes && self.ports == other.ports }
}

impl Target {
    pub fn new(raw_lanes: &[String], ports: Vec<PortSpec>, package_pins: Vec<(String, String)>) -> Result<Arc<Target>> {
        let raw = raw_lanes.iter().map(|lane| identifier(lane, "target lane")).collect::<Result<Vec<_>>>()?;
        if raw.is_empty() || raw.iter().collect::<BTreeSet<_>>().len() != raw.len() {
            return value_error("target lanes must be unique and non-empty");
        }
        if ports.iter().map(|port| &port.key).collect::<BTreeSet<_>>().len() != ports.len() {
            return value_error("target port keys must be unique");
        }
        let mut owner: BTreeMap<&str, &str> = BTreeMap::new();
        for port in &ports {
            for lane in &port.lanes {
                if !raw.contains(lane) { return value_error(format!("port {} owns unknown lane {}", repr(&port.key), repr(lane))); }
                if owner.insert(lane, &port.key).is_some() { return value_error(format!("lane {} belongs to two ports", repr(lane))); }
            }
        }
        let missing: Vec<&str> = raw.iter().map(String::as_str).filter(|lane| !owner.contains_key(lane)).collect();
        if !missing.is_empty() { return value_error(format!("target lanes have no logical owner: {}", repr_tuple(&missing))); }
        let clocks: BTreeSet<&str> = ports.iter().filter(|port| port.kind == PortKind::Clock).map(|port| port.key.as_str()).collect();
        for port in &ports {
            if let Some(clock) = &port.latch_clock {
                if !clocks.contains(clock.as_str()) { return value_error(format!("DAC port {} references a missing clock", repr(&port.key))); }
            }
        }
        let mut buses: Vec<u64> = ports.iter().filter(|port| port.kind == PortKind::Dac).filter_map(|port| port.bus_index).collect();
        buses.sort_unstable();
        if buses.iter().enumerate().any(|(index, bus)| *bus != index as u64) {
            return value_error("DAC bus indices must be contiguous from zero");
        }
        if !package_pins.is_empty() {
            let pinned: BTreeSet<&str> = package_pins.iter().map(|(lane, _)| lane.as_str()).collect();
            if pinned != raw.iter().map(String::as_str).collect() || pinned.len() != package_pins.len() {
                return value_error("package_pins must contain exactly one pin for every target lane");
            }
            if package_pins.iter().any(|(_, pin)| pin.trim().is_empty()) {
                return type_error("package_pins must map lane names to non-empty pin names");
            }
        }
        let abi = Json::object(vec![
            ("schema", Json::str("zlc_pulse.PulseTargetABI")),
            ("lanes", Json::strings(&raw)),
            ("ports", Json::Array(ports.iter().map(|port| Json::object(vec![
                ("key", Json::str(&port.key)),
                ("kind", Json::str(port.kind.name())),
                ("lanes", Json::strings(&port.lanes)),
                ("bus_index", port.bus_index.map_or(Json::Null, |bus| Json::Int(bus.into()))),
                ("width", Json::Int(port.width.into())),
                ("encoding", Json::str(&port.encoding)),
                ("safe_value", Json::Int(port.safe_value.into())),
                ("latch_clock", Json::opt_str(port.latch_clock.as_deref())),
            ])).collect())),
        ]);
        let by_key = ports.iter().enumerate().map(|(index, port)| (port.key.clone(), index)).collect();
        Ok(Arc::new(Target { raw_lanes: raw, ports, package_pins, abi_fingerprint: crate::canonical::digest(&abi), by_key }))
    }

    pub fn abi_fingerprint(&self) -> &str { &self.abi_fingerprint }
    pub fn port(&self, key: &str) -> Option<&PortSpec> { self.by_key.get(key).map(|index| &self.ports[*index]) }
    pub fn lane_index(&self, lane: &str) -> Option<usize> { self.raw_lanes.iter().position(|item| item == lane) }
    pub fn pin(&self, lane: &str) -> Option<&str> {
        self.package_pins.iter().find(|(name, _)| name == lane).map(|(_, pin)| pin.as_str())
    }
}

// ------------------------------------------------------------------ fields
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FieldKind { Duration, Dac, Delay, BracketCount }

impl FieldKind {
    pub fn parse(text: &str) -> Option<FieldKind> {
        match text {
            "duration" => Some(FieldKind::Duration), "dac" => Some(FieldKind::Dac), "delay" => Some(FieldKind::Delay),
            "bracket_count" => Some(FieldKind::BracketCount), _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self { FieldKind::Duration => "duration", FieldKind::Dac => "dac", FieldKind::Delay => "delay", FieldKind::BracketCount => "bracket_count" }
    }
    /// Whether a scan slot can supply this field.
    pub fn scannable(self) -> bool { matches!(self, FieldKind::Duration | FieldKind::Dac) }
}

/// One physical field: the stable identity a binding, a scan column and an
/// API value all name, independent of display names.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FieldRef {
    pub kind: FieldKind,
    pub period_id: Option<String>,
    pub port: Option<String>,
    pub bracket_id: Option<String>,
}

impl FieldRef {
    pub fn new(kind: &str, period_id: Option<&str>, port: Option<&str>, bracket_id: Option<&str>) -> Result<FieldRef> {
        text(kind, "field kind")?;
        let Some(kind) = FieldKind::parse(kind) else { return value_error(format!("unsupported field kind {}", repr(kind))); };
        if kind != FieldKind::BracketCount && bracket_id.is_some() {
            return value_error("only bracket count fields may name a bracket_id");
        }
        let id = |value: Option<&str>, name: &str| value.map(|value| identifier(value, name)).transpose();
        Ok(match kind {
            FieldKind::BracketCount => {
                if bracket_id.is_none() || period_id.is_some() || port.is_some() { return value_error("bracket count fields require bracket_id only"); }
                FieldRef { kind, period_id: None, port: None, bracket_id: id(bracket_id, "count bracket_id")? }
            }
            FieldKind::Duration => {
                if period_id.is_none() || port.is_some() { return value_error("duration fields require period_id only"); }
                FieldRef { kind, period_id: id(period_id, "duration period_id")?, port: None, bracket_id: None }
            }
            FieldKind::Dac => {
                if period_id.is_none() || port.is_none() { return value_error("DAC fields require period_id and port"); }
                FieldRef { kind, period_id: id(period_id, "DAC period_id")?, port: id(port, "DAC port")?, bracket_id: None }
            }
            FieldKind::Delay => {
                if period_id.is_some() || port.is_none() { return value_error("delay fields require port only"); }
                FieldRef { kind, period_id: None, port: id(port, "delay port")?, bracket_id: None }
            }
        })
    }

    pub fn duration(period_id: &str) -> FieldRef { FieldRef { kind: FieldKind::Duration, period_id: Some(period_id.into()), port: None, bracket_id: None } }
    pub fn dac(period_id: &str, port: &str) -> FieldRef { FieldRef { kind: FieldKind::Dac, period_id: Some(period_id.into()), port: Some(port.into()), bracket_id: None } }
    pub fn delay(port: &str) -> FieldRef { FieldRef { kind: FieldKind::Delay, period_id: None, port: Some(port.into()), bracket_id: None } }
    pub fn bracket_count(bracket_id: &str) -> FieldRef { FieldRef { kind: FieldKind::BracketCount, period_id: None, port: None, bracket_id: Some(bracket_id.into()) } }

    /// The stable physical identity: `duration:p`, `dac:p:port`, `delay:port`, `bracket_count:b`.
    pub fn key(&self) -> String {
        match self.kind {
            FieldKind::BracketCount => format!("bracket_count:{}", self.bracket_id.as_deref().unwrap_or("")),
            FieldKind::Delay => format!("delay:{}", self.port.as_deref().unwrap_or("")),
            FieldKind::Duration => format!("duration:{}", self.period_id.as_deref().unwrap_or("")),
            FieldKind::Dac => format!("dac:{}:{}", self.period_id.as_deref().unwrap_or(""), self.port.as_deref().unwrap_or("")),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Source { Default, Api, Config }

impl Source {
    pub fn parse(text: &str) -> Option<Source> {
        match text { "default" => Some(Source::Default), "api" => Some(Source::Api), "config" => Some(Source::Config), _ => None }
    }
    pub fn name(self) -> &'static str { match self { Source::Default => "default", Source::Api => "api", Source::Config => "config" } }
}

/// One field's independent scan capability and its Default/API/Config source.
/// `config_key` is read only while the source is Config; Default or API may
/// keep the name so switching back finds it again.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Binding {
    pub field_ref: FieldRef,
    pub unit: String,
    pub scan: bool,
    pub source: Source,
    pub config_key: String,
}

/// `_unit(value, field_name)`: `value`, `count`, or a canonical time unit.
pub fn binding_unit(value: &str, field_name: &str) -> Result<String> {
    text(value, field_name)?;
    if value == "value" || value == "count" { return Ok(value.to_owned()); }
    Ok(canonical_time_unit(value, field_name)?.to_owned())
}

impl Binding {
    pub fn new(field_ref: FieldRef, unit: &str, scan: bool, source: &str, config_key: &str) -> Result<Binding> {
        if scan && !field_ref.kind.scannable() { return value_error("this field cannot be scanned"); }
        let Some(source) = Source::parse(source) else { return value_error("binding source must be default, api or config"); };
        let unit = binding_unit(unit, "binding unit")?;
        let numeric = match field_ref.kind { FieldKind::Dac => Some("value"), FieldKind::BracketCount => Some("count"), _ => None };
        if numeric.is_some_and(|expected| unit != expected) || (numeric.is_none() && (unit == "value" || unit == "count")) {
            return value_error("DAC bindings use 'value'; bracket counts use 'count'; time bindings use a time unit");
        }
        if !config_key.is_empty() { config_parameter_key(config_key)?; }
        Ok(Binding { field_ref, unit, scan, source, config_key: config_key.to_owned() })
    }

    pub fn field_id(&self) -> String { self.field_ref.key() }
    pub fn kind(&self) -> FieldKind { self.field_ref.kind }
}

// ------------------------------------------------------------------ periods
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnalogMode { Edge, Ramp }

impl AnalogMode {
    pub fn parse(text: &str) -> Option<AnalogMode> { match text { "edge" => Some(AnalogMode::Edge), "ramp" => Some(AnalogMode::Ramp), _ => None } }
    pub fn name(self) -> &'static str { match self { AnalogMode::Edge => "edge", AnalogMode::Ramp => "ramp" } }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AnalogStep { pub port: String, pub mode: AnalogMode, pub value: i64 }

impl AnalogStep {
    pub fn new(port: &str, mode: &str, value: Num) -> Result<AnalogStep> {
        let port = identifier(port, "analog port")?;
        text(mode, "analog mode")?;
        let Some(mode) = AnalogMode::parse(mode) else { return value_error(format!("unsupported analog mode {}", repr(mode))); };
        let Num::Int(value) = value else { return value_error("analog value must be an integral code"); };
        let value = i64::try_from(value).map_err(|_| Error::new(Kind::Value, "analog value is outside the supported range"))?;
        Ok(AnalogStep { port, mode, value })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PeriodKind { Period, Spacer }

impl PeriodKind {
    pub fn parse(text: &str) -> Option<PeriodKind> { match text { "period" => Some(PeriodKind::Period), "spacer" => Some(PeriodKind::Spacer), _ => None } }
    pub fn name(self) -> &'static str { match self { PeriodKind::Period => "period", PeriodKind::Spacer => "spacer" } }
}

/// One authored period; a spacer is time given to a slow device: it holds
/// every DAC, is never scanned and takes no API value.
#[derive(Clone, Debug, PartialEq)]
pub struct Period {
    pub period_id: String,
    pub duration: Num,
    pub unit: String,
    pub states: Vec<u8>,
    pub analog_steps: Vec<AnalogStep>,
    pub name: String,
    pub kind: PeriodKind,
}

impl Period {
    pub fn new(period_id: &str, duration: Num, unit: &str, states: Vec<u8>, analog_steps: Vec<AnalogStep>, name: &str,
               kind: &str) -> Result<Period> {
        let period_id = identifier(period_id, "period_id")?;
        text(kind, "period kind")?;
        let Some(kind) = PeriodKind::parse(kind) else {
            return value_error(format!("period kind must be one of ('period', 'spacer'), got {}", repr(kind)));
        };
        if duration.to_f64() <= 0.0 { return value_error("period duration must be positive"); }
        let unit = canonical_time_unit(unit, "period unit")?.to_owned();
        if states.iter().any(|state| *state > 1) { return value_error("period states must contain only 0/1 values"); }
        if analog_steps.iter().map(|step| &step.port).collect::<BTreeSet<_>>().len() != analog_steps.len() {
            return value_error("a period has at most one analog step per port");
        }
        if kind == PeriodKind::Spacer && !analog_steps.is_empty() { return value_error("a spacer holds every DAC; it has no analog steps"); }
        Ok(Period { period_id, duration, unit, states, analog_steps, name: name.to_owned(), kind })
    }

    pub fn step(&self, port: &str) -> Option<&AnalogStep> { self.analog_steps.iter().find(|step| step.port == port) }
    pub fn is_spacer(&self) -> bool { self.kind == PeriodKind::Spacer }
    pub fn display_name(&self) -> &str { if self.name.is_empty() { &self.period_id } else { &self.name } }
}

#[derive(Clone, Debug, PartialEq)]
pub struct OutputDelay { pub port: String, pub value: Num, pub unit: String }

impl OutputDelay {
    pub fn new(port: &str, value: Num, unit: &str) -> Result<OutputDelay> {
        Ok(OutputDelay { port: identifier(port, "delay port")?, value, unit: canonical_time_unit(unit, "delay unit")?.to_owned() })
    }
}

/// Loop one continuous range of periods a whole number of times. A missing
/// start anchor means "after the last period", a missing end "before the
/// first", so an empty bracket can wait at either edge of the timeline.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bracket { pub bracket_id: String, pub start_period_id: Option<String>, pub end_period_id: Option<String>, pub count: u64 }

impl Bracket {
    pub fn new(bracket_id: &str, start: Option<&str>, end: Option<&str>, count: i128) -> Result<Bracket> {
        let bracket_id = identifier(bracket_id, "bracket_id")?;
        let start_period_id = start.map(|id| identifier(id, "bracket start")).transpose()?;
        let end_period_id = end.map(|id| identifier(id, "bracket end")).transpose()?;
        let count = nonnegative(count, "bracket count").or_else(|error| {
            if count > 0 { value_error("bracket count does not fit the hardware 32-bit count") } else { Err(error) }
        })?;
        if count < MINIMUM_BRACKET_COUNT { return value_error(format!("a bracket loops at least {MINIMUM_BRACKET_COUNT} time(s)")); }
        if count > MAXIMUM_REPEAT_COUNT { return value_error("bracket count does not fit the hardware 32-bit count"); }
        Ok(Bracket { bracket_id, start_period_id, end_period_id, count })
    }
}

fn bracket_gap_bounds(bracket: &Bracket, ids: &[&str]) -> (usize, usize) {
    let index = |id: &str| ids.iter().position(|item| *item == id).expect("validated bracket anchor");
    (bracket.start_period_id.as_deref().map_or(ids.len(), index), bracket.end_period_id.as_deref().map_or(0, |id| index(id) + 1))
}

/// Whether `inner` lies inside `outer` (half-open gap bounds). An empty
/// bracket at a bracket's own boundary gap sits beside it, not within it.
pub fn bracket_contains(outer: (usize, usize), inner: (usize, usize)) -> bool {
    let (start, end) = inner;
    if !(outer.0 <= start && end <= outer.1) { return false; }
    start < end || (outer.0 < start && start < outer.1)
}

pub fn brackets_disjoint(first: (usize, usize), second: (usize, usize)) -> bool { first.1 <= second.0 || second.1 <= first.0 }

// ------------------------------------------------------------------ components
/// One named, continuous group of periods; its waveform still lives in the sequence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Component {
    pub component_id: String,
    pub name: String,
    pub period_ids: Vec<String>,
    pub bracket_ids: Vec<String>,
    pub source: Source,
    pub config_path: String,
    pub period_id_map: Vec<(String, String)>,
    pub bracket_id_map: Vec<(String, String)>,
}

impl Component {
    #[allow(clippy::too_many_arguments)]
    pub fn new(component_id: &str, name: &str, period_ids: &[String], bracket_ids: &[String], source: &str, config_path: &str,
               period_id_map: &[Vec<String>], bracket_id_map: &[Vec<String>]) -> Result<Component> {
        let component_id = identifier(component_id, "component id")?;
        text(name, "component name")?;
        let ids = period_ids.iter().map(|id| identifier(id, "component period id")).collect::<Result<Vec<_>>>()?;
        if ids.is_empty() || ids.iter().collect::<BTreeSet<_>>().len() != ids.len() {
            return value_error("component period ids must be non-empty and unique");
        }
        let brackets = bracket_ids.iter().map(|id| identifier(id, "component bracket id")).collect::<Result<Vec<_>>>()?;
        if brackets.iter().collect::<BTreeSet<_>>().len() != brackets.len() { return value_error("component bracket ids must be unique"); }
        let source = match Source::parse(source) { Some(source @ (Source::Default | Source::Config)) => source,
                                                  _ => return value_error("Subpulse source must be default or config") };
        let pairs = |items: &[Vec<String>], name: &str| -> Result<Vec<(String, String)>> {
            let mut mapping = Vec::new();
            for pair in items {
                let ids = pair.iter().map(|id| identifier(id, "Subpulse source id")).collect::<Result<Vec<_>>>()?;
                if ids.len() != 2 { return value_error(format!("{name} must contain source/instance id pairs")); }
                mapping.push((ids[0].clone(), ids[1].clone()));
            }
            if mapping.iter().map(|(source, _)| source).collect::<BTreeSet<_>>().len() != mapping.len() {
                return value_error(format!("{name} source ids must be unique"));
            }
            Ok(mapping)
        };
        Ok(Component { component_id, name: name.to_owned(), period_ids: ids, bracket_ids: brackets, source,
                       config_path: config_path.to_owned(), period_id_map: pairs(period_id_map, "period_id_map")?,
                       bracket_id_map: pairs(bracket_id_map, "bracket_id_map")? })
    }

    /// Whether this instance plays a Subpulse file: Config with a file chosen.
    pub fn from_file(&self) -> bool { self.source == Source::Config && !self.config_path.is_empty() }
}

// ------------------------------------------------------------------ sequence
/// Everything a sequence is made from: what `dataclasses.replace` edits.
#[derive(Clone, Debug)]
pub struct SequenceParts {
    pub name: String,
    pub target: Arc<Target>,
    pub time_step_ns: f64,
    pub periods: Vec<Period>,
    pub bindings: Vec<Binding>,
    pub delays: Vec<OutputDelay>,
    pub brackets: Vec<Bracket>,
    pub run_repeats: u64,
    pub components: Vec<Component>,
    pub config_enabled: bool,
}

impl SequenceParts {
    pub fn new(target: Arc<Target>, periods: Vec<Period>) -> Self {
        SequenceParts { name: "sequence".into(), target, time_step_ns: 20.0, periods, bindings: Vec::new(), delays: Vec::new(),
                        brackets: Vec::new(), run_repeats: 0, components: Vec::new(), config_enabled: false }
    }
}

/// An immutable, validated pulse. Brackets are kept outermost first and
/// components in timeline order, whatever order they were authored in.
#[derive(Clone, Debug)]
pub struct Sequence { parts: SequenceParts, bracket_bounds: Vec<(usize, usize)> }

impl PartialEq for Sequence {
    fn eq(&self, other: &Self) -> bool {
        let (a, b) = (&self.parts, &other.parts);
        a.name == b.name && (Arc::ptr_eq(&a.target, &b.target) || *a.target == *b.target) && a.time_step_ns == b.time_step_ns
            && a.periods == b.periods && a.bindings == b.bindings && a.delays == b.delays && a.brackets == b.brackets
            && a.run_repeats == b.run_repeats && a.components == b.components && a.config_enabled == b.config_enabled
    }
}

impl std::ops::Deref for Sequence {
    type Target = SequenceParts;
    fn deref(&self) -> &SequenceParts { &self.parts }
}

impl Sequence {
    pub fn new(parts: SequenceParts) -> Result<Sequence> {
        let SequenceParts { name, target, time_step_ns, periods, bindings, delays, brackets, run_repeats, components, config_enabled } = parts;
        if periods.is_empty() { return value_error("periods must contain PulsePeriod values"); }
        if !(time_step_ns > 0.0) || !time_step_ns.is_finite() { return value_error("time_step_ns must be positive and finite"); }
        let ids: Vec<&str> = periods.iter().map(|period| period.period_id.as_str()).collect();
        if ids.iter().collect::<BTreeSet<_>>().len() != ids.len() { return value_error("period ids must be unique"); }
        if components.iter().map(|c| &c.component_id).collect::<BTreeSet<_>>().len() != components.len() {
            return value_error("component ids must be unique");
        }
        if components.iter().map(|c| &c.name).collect::<BTreeSet<_>>().len() != components.len() {
            return value_error("component names must be unique");
        }
        let mut component_owner: BTreeMap<&str, &str> = BTreeMap::new();
        let mut component_bounds: BTreeMap<String, (usize, usize)> = BTreeMap::new();
        for component in &components {
            if component.period_ids.iter().any(|id| !ids.contains(&id.as_str())) {
                return value_error(format!("component {} references a missing period", repr(&component.name)));
            }
            let start = ids.iter().position(|id| *id == component.period_ids[0]).expect("checked");
            let end = start + component.period_ids.len();
            if end > ids.len() || ids[start..end].iter().zip(&component.period_ids).any(|(a, b)| *a != b) {
                return value_error(format!("component {} must contain continuous periods in order", repr(&component.name)));
            }
            if component.period_ids.iter().any(|id| component_owner.contains_key(id.as_str())) {
                return value_error("components cannot overlap or nest");
            }
            for id in &component.period_ids { component_owner.insert(id, &component.component_id); }
            component_bounds.insert(component.component_id.clone(), (start, end));
        }
        let mut display: BTreeSet<(Option<&str>, &str)> = BTreeSet::new();
        for period in &periods {
            if !display.insert((component_owner.get(period.period_id.as_str()).copied(), period.display_name())) {
                return value_error("period names must be unique within their component (empty Name uses the period ID)");
            }
        }
        let lane_owner: BTreeMap<&str, &PortSpec> = target.ports.iter().flat_map(|port| port.lanes.iter().map(move |lane| (lane.as_str(), port))).collect();
        for period in &periods {
            if period.states.len() != target.raw_lanes.len() { return value_error("every period state vector must match target lanes"); }
            exact_ticks(period.duration, &period.unit, time_step_ns, &format!("period {} duration", period.period_id), Some(1))?;
            for (index, state) in period.states.iter().enumerate() {
                if *state != 0 && lane_owner[target.raw_lanes[index].as_str()].kind != PortKind::Digital {
                    return value_error("non-digital lanes cannot carry digital states");
                }
            }
            for step in &period.analog_steps {
                let Some(port) = target.port(&step.port).filter(|port| port.kind == PortKind::Dac) else {
                    return value_error(format!("analog step references unknown DAC {}", repr(&step.port)));
                };
                let (low, high) = port.signed_range().expect("a DAC");
                if !(low <= step.value && step.value <= high) {
                    return value_error(format!("analog value for {} is outside its range", repr(&step.port)));
                }
            }
        }
        if delays.iter().map(|delay| &delay.port).collect::<BTreeSet<_>>().len() != delays.len() {
            return value_error("at most one delay per logical port");
        }
        for delay in &delays {
            if !target.port(&delay.port).is_some_and(|port| matches!(port.kind, PortKind::Digital | PortKind::Dac)) {
                return value_error(format!("delay references unsupported port {}", repr(&delay.port)));
            }
            exact_ticks(delay.value, &delay.unit, time_step_ns, &format!("delay {}", delay.port), None)?;
        }
        if bindings.iter().map(|binding| &binding.field_ref).collect::<BTreeSet<_>>().len() != bindings.len() {
            return value_error("each physical field has one binding declaration");
        }
        let by_period: BTreeMap<&str, &Period> = periods.iter().map(|period| (period.period_id.as_str(), period)).collect();
        for binding in &bindings {
            let reference = &binding.field_ref;
            if matches!(reference.kind, FieldKind::Duration | FieldKind::Dac) {
                let id = reference.period_id.as_deref().expect("a period field");
                let Some(period) = by_period.get(id) else { return value_error(format!("binding references missing period {}", repr(id))); };
                if period.is_spacer() {
                    if reference.kind == FieldKind::Dac {
                        return value_error(format!("spacer {} holds its DACs; nothing there can be bound", repr(id)));
                    }
                    if binding.scan || binding.source == Source::Api {
                        return value_error(format!("spacer {} duration can take a Config value but is never scanned or set by API", repr(id)));
                    }
                }
            }
            if reference.kind == FieldKind::Dac {
                let port = reference.port.as_deref().expect("a DAC field");
                if !target.port(port).is_some_and(|spec| spec.kind == PortKind::Dac) {
                    return value_error(format!("binding references missing DAC {}", repr(port)));
                }
            }
            if reference.kind == FieldKind::Delay {
                let port = reference.port.as_deref().expect("a delay field");
                if !target.port(port).is_some_and(|spec| matches!(spec.kind, PortKind::Digital | PortKind::Dac)) {
                    return value_error(format!("binding references missing delay port {}", repr(port)));
                }
            }
        }
        if brackets.iter().map(|bracket| &bracket.bracket_id).collect::<BTreeSet<_>>().len() != brackets.len() {
            return value_error("bracket ids must be unique");
        }
        for binding in &bindings {
            if binding.kind() == FieldKind::BracketCount {
                let id = binding.field_ref.bracket_id.as_deref().expect("a bracket field");
                if !brackets.iter().any(|bracket| bracket.bracket_id == id) {
                    return value_error(format!("binding references missing bracket {}", repr(id)));
                }
            }
        }
        for bracket in &brackets {
            let missing = |id: &Option<String>| id.as_deref().is_some_and(|id| !by_period.contains_key(id));
            if missing(&bracket.start_period_id) || missing(&bracket.end_period_id) {
                return value_error(format!("bracket {} references a missing period", repr(&bracket.bracket_id)));
            }
        }
        let bounds: BTreeMap<&str, (usize, usize)> = brackets.iter().map(|bracket| (bracket.bracket_id.as_str(), bracket_gap_bounds(bracket, &ids))).collect();
        for component in &components {
            if component.bracket_ids.iter().any(|id| !bounds.contains_key(id.as_str())) {
                return value_error(format!("component {} references a missing bracket", repr(&component.name)));
            }
        }
        for bracket in &brackets {
            let (start, end) = bounds[bracket.bracket_id.as_str()];
            if start > end { return value_error(format!("bracket {} end precedes its start", repr(&bracket.bracket_id))); }
            for component in &components {
                let extent = component_bounds[component.component_id.as_str()];
                if component.bracket_ids.contains(&bracket.bracket_id) {
                    if !(extent.0 <= start && start <= end && end <= extent.1) {
                        return value_error(format!("component {} must contain its complete bracket {}", repr(&component.name), repr(&bracket.bracket_id)));
                    }
                    continue;
                }
                if !(brackets_disjoint((start, end), extent) || bracket_contains((start, end), extent)) {
                    return value_error(format!("bracket {} partially crosses component {}", repr(&bracket.bracket_id), repr(&component.name)));
                }
            }
        }
        for (index, outer) in brackets.iter().enumerate() {
            for inner in &brackets[index + 1..] {
                let (first, second) = (bounds[outer.bracket_id.as_str()], bounds[inner.bracket_id.as_str()]);
                if !(brackets_disjoint(first, second) || bracket_contains(first, second) || bracket_contains(second, first)) {
                    return value_error(format!("brackets {} and {} overlap without one lying inside the other",
                                               repr(&outer.bracket_id), repr(&inner.bracket_id)));
                }
            }
        }
        let mut ordered = brackets.clone();
        // Outer first; a stable sort keeps the authored order for identical bounds.
        ordered.sort_by_key(|bracket| { let (start, end) = bounds[bracket.bracket_id.as_str()]; (start, std::cmp::Reverse(end)) });
        let bracket_bounds = ordered.iter().map(|bracket| bounds[bracket.bracket_id.as_str()]).collect();
        if run_repeats > MAXIMUM_REPEAT_COUNT { return value_error("run_repeats does not fit the hardware 32-bit count"); }
        text(&name, "sequence name")?;
        let mut components = components;
        components.sort_by_key(|component| component_bounds[component.component_id.as_str()].0);
        Ok(Sequence {
            parts: SequenceParts { name, target, time_step_ns, periods, bindings, delays, brackets: ordered, run_repeats, components, config_enabled },
            bracket_bounds,
        })
    }

    /// A copy of the parts, to edit and rebuild (`dataclasses.replace`).
    pub fn parts(&self) -> SequenceParts { self.parts.clone() }

    pub fn with(&self, edit: impl FnOnce(&mut SequenceParts)) -> Result<Sequence> {
        let mut parts = self.parts();
        edit(&mut parts);
        Sequence::new(parts)
    }

    pub fn period(&self, period_id: &str) -> Option<&Period> { self.periods.iter().find(|period| period.period_id == period_id) }
    pub fn period_index(&self, period_id: &str) -> Option<usize> { self.periods.iter().position(|period| period.period_id == period_id) }
    pub fn period_ids(&self) -> Vec<&str> { self.periods.iter().map(|period| period.period_id.as_str()).collect() }
    pub fn bracket_bounds(&self) -> &[(usize, usize)] { &self.bracket_bounds }

    pub fn component_for_period(&self, period_id: &str) -> Option<&Component> {
        self.components.iter().find(|component| component.period_ids.iter().any(|id| id == period_id))
    }

    pub fn component_for_field(&self, reference: &FieldRef) -> Option<&Component> {
        if let Some(bracket) = &reference.bracket_id {
            return self.components.iter().find(|component| component.bracket_ids.contains(bracket));
        }
        reference.period_id.as_deref().and_then(|id| self.component_for_period(id))
    }

    pub fn period_label(&self, period_id: &str) -> String {
        let label = self.period(period_id).map_or(period_id, Period::display_name);
        match self.component_for_period(period_id) { None => label.to_owned(), Some(component) => format!("{} / {label}", component.name) }
    }

    /// The row loop table: `(first_row, last_row, count)` per non-empty bracket, outermost first.
    pub fn loops(&self) -> Vec<(usize, usize, u64)> {
        self.brackets.iter().zip(&self.bracket_bounds).filter(|(_, (start, end))| end > start)
            .map(|(bracket, (start, end))| (*start, end - 1, bracket.count)).collect()
    }

    /// How long the board plays one Pulse, in nanoseconds: the periods as the
    /// loop table expands them, exactly, then rounded once.
    pub fn played_nanoseconds(&self) -> f64 {
        let durations: Vec<Decimal> = self.periods.iter()
            .map(|period| Decimal::from_f64(period.duration.to_f64()).scale10(nanoseconds_exponent(&period.unit).expect("canonical unit")))
            .collect();
        let exponent = durations.iter().map(|value| value.exponent).min().unwrap_or(0).min(0);
        let integers: Vec<BigUint> = durations.iter().map(|value| {
            if value.is_zero() { BigUint::zero() } else { value.mantissa.mul(&BigUint::pow10((value.exponent - exponent) as u32)) }
        }).collect();
        let total = frame_ticks_big(&integers, &self.loops());
        Decimal { negative: false, mantissa: total, exponent }.to_f64()
    }

    pub fn scan_bindings(&self) -> Vec<&Binding> { self.bindings.iter().filter(|binding| binding.scan).collect() }
    pub fn api_bindings(&self) -> Vec<&Binding> { self.bindings.iter().filter(|binding| binding.source == Source::Api).collect() }
    pub fn config_bindings(&self) -> Vec<&Binding> { self.bindings.iter().filter(|binding| binding.source == Source::Config).collect() }
    pub fn slot_count(&self) -> usize { self.bindings.iter().filter(|binding| binding.scan).count() }
    pub fn binding(&self, reference: &FieldRef) -> Option<&Binding> { self.bindings.iter().find(|binding| &binding.field_ref == reference) }

    /// The authored unit of one physical field.
    pub fn field_unit(&self, reference: &FieldRef) -> Result<String> {
        Ok(match reference.kind {
            FieldKind::BracketCount => "count".into(),
            FieldKind::Dac => "value".into(),
            FieldKind::Delay => self.delays.iter().find(|delay| reference.port.as_deref() == Some(&delay.port))
                .map_or_else(|| "ns".to_owned(), |delay| delay.unit.clone()),
            FieldKind::Duration => {
                let id = reference.period_id.as_deref().unwrap_or("");
                self.period(id).ok_or_else(|| Error::new(Kind::Value, format!("no period exists with id {}", repr(id))))?.unit.clone()
            }
        })
    }
}

/// An editable fragment: no output delays, no execution repeat count.
#[derive(Clone, Debug, PartialEq)]
pub struct Subpulse(Sequence);

impl Subpulse {
    pub fn new(name: &str, target: Arc<Target>, time_step_ns: f64, periods: Vec<Period>, bindings: Vec<Binding>,
               brackets: Vec<Bracket>) -> Result<Subpulse> {
        let sequence = Sequence::new(SequenceParts { name: name.to_owned(), time_step_ns, bindings, brackets,
                                                     ..SequenceParts::new(target, periods) })?;
        if sequence.bindings.iter().any(|binding| binding.kind() == FieldKind::Delay) {
            return value_error("subpulses cannot bind global output delays");
        }
        Ok(Subpulse(sequence))
    }

    /// The fragment as a sequence for the existing editor, every id retained.
    pub fn to_sequence(&self) -> &Sequence { &self.0 }
}

pub(crate) fn available_id(base: &str, used: &mut BTreeSet<String>) -> String {
    let mut candidate = base.to_owned();
    let mut suffix = 2;
    while used.contains(&candidate) { candidate = format!("{base}_{suffix}"); suffix += 1; }
    used.insert(candidate.clone());
    candidate
}

pub(crate) fn available_name(base: &str, used: &mut BTreeSet<String>) -> String {
    let mut candidate = base.to_owned();
    let mut suffix = 2;
    while used.contains(&candidate) { candidate = format!("{base} ({suffix})"); suffix += 1; }
    used.insert(candidate.clone());
    candidate
}
