//! The wiring as the Target page shows and edits it: which ports a pulse can
//! drive, the page's records, the target its records describe, and the pulse
//! carried output by output onto a re-wired target.
use std::collections::BTreeMap;
use std::sync::Arc;

use serde_json::{Value, json};
use zlc_pulse::binding::field_label;
use zlc_pulse::error::repr;
use zlc_pulse::model::{OutputDelay, Period, PortKind, PortSpec, Sequence, Target};
use zlc_pulse::{Error, Kind, Result};

/// `programmable_ports`: digital and DAC ports in catalog order, each DAC
/// owning its latch clock; a clock no DAC latches with is still listed.
pub fn programmable_ports(target: &Target) -> Vec<&PortSpec> {
    let owned: Vec<&str> = target.ports.iter().filter(|p| p.kind == PortKind::Dac).filter_map(|p| p.latch_clock.as_deref()).collect();
    target.ports.iter().filter(|p| !(p.kind == PortKind::Clock && owned.contains(&p.key.as_str()))).collect()
}

/// `_pins_of`: the package pin per lane, the board's answer else the target's own.
pub fn pins_of(target: &Target, pins: Option<&[(String, String)]>) -> BTreeMap<String, String> {
    let given = pins.filter(|pins| !pins.is_empty()).unwrap_or(&target.package_pins);
    given.iter().cloned().collect()
}

/// One row of the Target page (`TargetPortRecord`).
#[derive(Clone, Debug, PartialEq)]
pub struct TargetRecord {
    pub key: String,
    pub kind: String,
    pub signal: String,
    pub endpoints: Vec<String>,
    pub clock_key: Option<String>,
    pub clock_endpoint: Option<String>,
}

impl TargetRecord {
    pub fn to_json(&self) -> Value {
        json!({"key": self.key, "kind": self.kind, "signal": self.signal, "endpoints": self.endpoints,
               "clock_key": self.clock_key, "clock_endpoint": self.clock_endpoint})
    }

    pub fn from_json(value: &Value) -> Result<TargetRecord> {
        let text = |name: &str| -> Result<String> {
            Ok(value.get(name).map(crate::args::py_str).ok_or_else(|| Error::new(Kind::Key, repr(name)))?)
        };
        let optional = |name: &str| value.get(name).and_then(crate::args::py_optional_str);
        let endpoints = value.get("endpoints").and_then(Value::as_array)
            .ok_or_else(|| Error::new(Kind::Type, "a target record's endpoints must be a list"))?
            .iter().map(crate::args::py_str).collect();
        Ok(TargetRecord { key: text("key")?, kind: text("kind")?, signal: text("signal").unwrap_or_default(), endpoints,
                          clock_key: optional("clock_key").filter(|key| !key.is_empty()), clock_endpoint: optional("clock_endpoint") })
    }
}

/// `project_target`: the page's records, one per digital or DAC output; a
/// DAC carries its latch clock's wire as its own field.
pub fn project_target(target: &Target, pins: Option<&[(String, String)]>) -> Vec<TargetRecord> {
    let pin_by_lane = pins_of(target, pins);
    let clock_lane: BTreeMap<&str, &str> = target.ports.iter().filter(|p| p.kind == PortKind::Clock)
        .map(|p| (p.key.as_str(), p.lanes[0].as_str())).collect();
    let endpoint = |lane: &str| pin_by_lane.get(lane).filter(|pin| !pin.is_empty()).cloned().unwrap_or_else(|| lane.to_owned());
    programmable_ports(target).into_iter().filter(|p| p.kind != PortKind::Clock).map(|port| {
        let clock_key = if port.kind == PortKind::Dac { port.latch_clock.clone() } else { None };
        TargetRecord {
            key: port.key.clone(), kind: port.kind.name().into(), signal: port.display().to_owned(),
            endpoints: port.lanes.iter().map(|lane| endpoint(lane)).collect(),
            clock_endpoint: clock_key.as_deref().map(|key| endpoint(clock_lane.get(key).copied().unwrap_or(key))),
            clock_key,
        }
    }).collect()
}

fn prefixed(key: &str, error: Error) -> Error {
    if error.kind == Kind::Value { Error::new(Kind::Value, format!("{key}: {}", error.message)) } else { error }
}

/// `_target_from_records`: the target the page's records describe. A wire
/// keeps the lane it has (found by its pin, or by its output and bit); lanes
/// and ports keep their order, new ones after them; DAC buses keep their numbers.
pub fn target_from_records(target: &Target, records: &[TargetRecord]) -> Result<Arc<Target>> {
    let pins: BTreeMap<&str, &str> = target.package_pins.iter().map(|(lane, pin)| (lane.as_str(), pin.as_str())).collect();
    let lane_of_pin: BTreeMap<&str, &str> = target.package_pins.iter().map(|(lane, pin)| (pin.as_str(), lane.as_str())).collect();
    let old: BTreeMap<&str, &PortSpec> = target.ports.iter().map(|p| (p.key.as_str(), p)).collect();
    let carried: Vec<&PortSpec> = programmable_ports(target).into_iter().filter(|p| p.kind == PortKind::Clock).collect();
    let mut clock_on: Vec<(String, String)> = carried.iter().map(|p| (p.lanes[0].clone(), p.key.clone())).collect();
    let wire_of_clock: BTreeMap<String, String> = clock_on.iter().map(|(lane, key)| (key.clone(), lane.clone())).collect();
    let mut lanes: Vec<String> = Vec::new();
    let mut new_pins: Vec<(String, String)> = Vec::new();

    let wire = |endpoint: &str, key: &str, bit: usize, fallback: &str, lanes: &mut Vec<String>, new_pins: &mut Vec<(String, String)>,
                clock_on: &[(String, String)]| -> Result<String> {
        let text = endpoint.trim();
        if text.is_empty() { return Err(Error::new(Kind::Value, format!("{key}: every wire needs an endpoint"))); }
        let lane = if !pins.is_empty() {
            let lane = match lane_of_pin.get(text) {
                Some(lane) => (*lane).to_owned(),
                None => match old.get(key) {
                    Some(previous) if bit < previous.lanes.len() => previous.lanes[bit].clone(),
                    _ => fallback.to_owned(),
                },
            };
            match new_pins.iter_mut().find(|(existing, _)| *existing == lane) {
                Some(slot) => slot.1 = text.to_owned(),
                None => new_pins.push((lane.clone(), text.to_owned())),
            }
            lane
        } else { text.to_owned() };
        if lanes.contains(&lane) {
            return Err(Error::new(Kind::Value, match clock_on.iter().find(|(wired, _)| *wired == lane) {
                Some((_, clock)) => format!("{key}: wire {} is held by clock {clock}, which no DAC latches with", repr(text)),
                None => format!("{key}: wire {} is already used by another output", repr(text)),
            }));
        }
        lanes.push(lane.clone());
        Ok(lane)
    };
    let unclaimed = |key: &str, clock_on: &[(String, String)]| -> Result<String> {
        if let Some(lane) = wire_of_clock.get(key) {
            let shown = pins.get(lane.as_str()).copied().unwrap_or(lane.as_str());
            let tail = if clock_on.iter().any(|(wired, _)| wired == lane) {
                "which no DAC latches with; a DAC without a latch clock takes it by naming that wire as its latch endpoint"
            } else { "which another DAC latches with in this apply" };
            return Err(Error::new(Kind::Value, format!("{key}: that name is taken by the clock on wire {}, {tail}", repr(shown))));
        }
        Ok(key.to_owned())
    };

    let dac_records: Vec<&TargetRecord> = records.iter().filter(|r| r.kind == "dac").collect();
    let mut dac_order: Vec<usize> = (0..dac_records.len()).collect();
    dac_order.sort_by_key(|index| {
        let bus = old.get(dac_records[*index].key.as_str()).filter(|p| p.kind == PortKind::Dac).and_then(|p| p.bus_index);
        (bus.unwrap_or(old.len() as u64), *index)
    });
    let mut bus_of: BTreeMap<&str, usize> = BTreeMap::new();
    for (bus, index) in dac_order.iter().enumerate() { bus_of.insert(dac_records[*index].key.as_str(), bus); }

    for port in &carried {
        let lane = &port.lanes[0];
        let endpoint = pins.get(lane.as_str()).copied().unwrap_or(lane.as_str());
        wire(endpoint, &port.key, 0, lane, &mut lanes, &mut new_pins, &clock_on)?;
    }
    let mut ports: Vec<PortSpec> = carried.iter().map(|p| (*p).clone()).collect();
    for record in records {
        let key = unclaimed(&record.key, &clock_on)?;
        let label = record.signal.trim().to_owned();
        let previous = old.get(key.as_str()).copied();
        match record.kind.as_str() {
            "digital" => {
                if record.endpoints.len() != 1 {
                    return Err(Error::new(Kind::Value, format!("{key}: a digital output has one wire, not {}", record.endpoints.len())));
                }
                let lane = wire(&record.endpoints[0], &key, 0, &key, &mut lanes, &mut new_pins, &clock_on)?;
                ports.push(PortSpec::new(&key, "digital", &[lane], &label, None, None, None, None, None).map_err(|e| prefixed(&key, e))?);
            }
            "dac" => {
                let mut data = Vec::new();
                for (bit, endpoint) in record.endpoints.iter().enumerate() {
                    data.push(wire(endpoint, &key, bit, &format!("{key}_{bit}"), &mut lanes, &mut new_pins, &clock_on)?);
                }
                let clock_text = record.clock_endpoint.clone().unwrap_or_default().trim().to_owned();
                let taken = if record.clock_key.is_some() || clock_text.is_empty() { None } else {
                    let lane = if pins.is_empty() { Some(clock_text.as_str()) } else { lane_of_pin.get(clock_text.as_str()).copied() };
                    lane.and_then(|lane| clock_on.iter().position(|(wired, _)| wired == lane)).map(|at| clock_on.remove(at).1)
                };
                let clock_key = match (&taken, &record.clock_key) {
                    (Some(taken), _) => Some(taken.clone()),
                    (None, Some(clock)) => Some(unclaimed(clock, &clock_on)?),
                    (None, None) if !clock_text.is_empty() => Some(unclaimed(&format!("{key}_clock"), &clock_on)?),
                    _ => None,
                };
                let clock_lane = match (&clock_key, &taken) {
                    (Some(clock), None) => {
                        let endpoint = record.clock_endpoint.clone().unwrap_or_else(|| "None".into());
                        Some(wire(&endpoint, clock, 0, clock, &mut lanes, &mut new_pins, &clock_on)?)
                    }
                    _ => None,
                };
                let same_width = previous.is_some_and(|p| p.kind == PortKind::Dac && p.lanes.len() == data.len());
                let (encoding, safe) = if same_width { (Some(previous.unwrap().encoding.as_str()), Some(previous.unwrap().safe_value as i128)) } else { (None, None) };
                ports.push(PortSpec::new(&key, "dac", &data, &label, Some(bus_of[record.key.as_str()] as i128), None, encoding, safe,
                                         clock_key.as_deref()).map_err(|e| prefixed(&key, e))?);
                if let (Some(clock), Some(lane)) = (&clock_key, clock_lane) {
                    let clock_label = old.get(clock.as_str()).map_or("", |p| p.label.as_str());
                    ports.push(PortSpec::new(clock, "clock", &[lane], clock_label, None, None, None, None, None).map_err(|e| prefixed(clock, e))?);
                }
            }
            other => return Err(Error::new(Kind::Value, format!("{key}: unknown output kind {}", repr(other)))),
        }
    }
    let position: BTreeMap<&str, usize> = target.ports.iter().enumerate().map(|(index, p)| (p.key.as_str(), index)).collect();
    ports.sort_by_key(|p| position.get(p.key.as_str()).copied().unwrap_or(position.len()));
    let mut raw: Vec<String> = target.raw_lanes.iter().filter(|lane| lanes.contains(lane)).cloned().collect();
    raw.extend(lanes.iter().filter(|lane| !target.raw_lanes.contains(lane)).cloned());
    Target::new(&raw, ports, new_pins)
}

/// `_carried_onto`: the pulse's periods and delays on a re-wired target, or
/// why it cannot go -- an output still driven, stepped, delayed or bound is
/// refused by name, never dropped.
pub fn carried_onto(sequence: &Sequence, target: &Target) -> Result<(Vec<Period>, Vec<OutputDelay>)> {
    let old = &sequence.target;
    let kept: Vec<&PortSpec> = target.ports.iter().filter(|p| old.port(&p.key).is_some_and(|o| o.kind == p.kind)).collect();
    let is_kept = |key: &str| kept.iter().any(|p| p.key == key);
    let mut blocked = Vec::new();
    for port in &old.ports {
        if is_kept(&port.key) || port.kind == PortKind::Clock { continue; }
        let mut uses = Vec::new();
        if port.kind == PortKind::Digital {
            let lane = old.lane_index(&port.lanes[0]).expect("a lane");
            let high: Vec<&str> = sequence.periods.iter().filter(|p| p.states[lane] != 0).map(|p| p.period_id.as_str()).collect();
            if !high.is_empty() { uses.push(format!("high in {}", high.join(", "))); }
        } else {
            let stepped: Vec<&str> = sequence.periods.iter().filter(|p| p.step(&port.key).is_some()).map(|p| p.period_id.as_str()).collect();
            if !stepped.is_empty() { uses.push(format!("stepped in {}", stepped.join(", "))); }
        }
        if sequence.delays.iter().any(|d| d.port == port.key) { uses.push("delayed".into()); }
        let bound = sequence.bindings.iter().filter(|b| b.field_ref.port.as_deref() == Some(&port.key))
            .map(|b| field_label(sequence, &b.field_ref)).collect::<Result<Vec<_>>>()?;
        if !bound.is_empty() { uses.push(format!("bound as {}", bound.join(", "))); }
        if !uses.is_empty() { blocked.push(format!("{} is {}", port.display(), uses.join("; "))); }
    }
    if !blocked.is_empty() {
        return Err(Error::new(Kind::Value, format!("clear these outputs before removing them: {}", blocked.join(" | "))));
    }
    let mut periods = Vec::new();
    for period in &sequence.periods {
        let mut states = vec![0u8; target.raw_lanes.len()];
        for port in &kept {
            if port.kind == PortKind::Digital {
                let from = old.lane_index(&old.port(&port.key).expect("kept").lanes[0]).expect("a lane");
                states[target.lane_index(&port.lanes[0]).expect("a lane")] = period.states[from];
            }
        }
        let mut copy = period.clone();
        copy.states = states;
        copy.analog_steps.retain(|step| is_kept(&step.port));
        periods.push(Period::new(&copy.period_id, copy.duration, &copy.unit, copy.states, copy.analog_steps, &copy.name, copy.kind.name())?);
    }
    let delays = sequence.delays.iter().filter(|d| is_kept(&d.port)).cloned().collect();
    Ok((periods, delays))
}

/// `_retarget_labels`: the same target with new display labels, or None when nothing changes.
pub fn retarget_labels(target: &Target, labels: &[(String, String)]) -> Result<Option<Arc<Target>>> {
    let mut changed = false;
    let ports = target.ports.iter().map(|port| {
        match labels.iter().find(|(key, _)| *key == port.key).map(|(_, label)| label) {
            Some(wanted) if !wanted.is_empty() && *wanted != port.label => { changed = true; PortSpec { label: wanted.clone(), ..port.clone() } }
            _ => port.clone(),
        }
    }).collect();
    if !changed { return Ok(None); }
    Ok(Some(Target::new(&target.raw_lanes, ports, target.package_pins.clone())?))
}
