//! The authoring operations, independent of which view (or command) asked:
//! each builds a candidate sequence for the model to accept or refuse
//! (`_sequence_edited`, `_reordered_sequence`). Arguments are the v2
//! presenter's positional ones, as JSON values.
use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Map, Value};
use zlc_pulse::binding::prune_orphaned_bindings;
use zlc_pulse::components::extract_subpulse;
use zlc_pulse::error::repr;
use zlc_pulse::model::{
    ANALOG_MODE_CHOICES, AnalogStep, Binding, Bracket, FieldRef, OutputDelay, Period, PortKind, Sequence, SequenceParts,
    align_to_grid, config_parameter_key,
};
use zlc_pulse::num::Num;
use zlc_pulse::{Error, Kind, Result};

use crate::args::{at, item, items, py_count, py_float, py_int_of_float, py_optional_str, py_str, py_truthy};
use crate::items::{BRACKET, COMPONENT, Item, PERIOD, bracket_post_key, post_bracket, sequence_item_order, unique_id};

/// What a period with no step for a DAC means: it holds the level carried in.
pub const HOLD_MODE: &str = "hold";
/// `NEW_PULSE_PERIOD_NS`: each period of a brand-new pulse.
pub const NEW_PULSE_PERIOD_NS: f64 = 1000.0;

fn key_error(key: &str) -> Error { Error::new(Kind::Key, repr(key)) }

/// `replace(period, ...)`: the period rebuilt through its own checks.
fn rebuilt(period: &Period, edit: impl FnOnce(&mut Period)) -> Result<Period> {
    let mut copy = period.clone();
    edit(&mut copy);
    Period::new(&copy.period_id, copy.duration, &copy.unit, copy.states, copy.analog_steps, &copy.name, copy.kind.name())
}

/// `_held_value`: the last level set before this period, else rest (0).
pub fn held_value(sequence: &Sequence, period_id: &str, port: &str) -> i64 {
    let mut held = 0;
    for earlier in &sequence.periods {
        if earlier.period_id == period_id { break; }
        if let Some(step) = earlier.step(port) { held = step.value; }
    }
    held
}

/// `_reordered_sequence(sequence, order, periods=)`: one edit of the flat timeline.
pub fn reordered_sequence(sequence: &Sequence, order: &[Item], periods: Option<&[Period]>) -> Result<Sequence> {
    let by_id: Vec<Period> = periods.map_or_else(|| sequence.periods.clone(), <[Period]>::to_vec);
    if by_id.is_empty() { return Err(Error::new(Kind::Value, "a sequence needs at least one period")); }
    let mut expected: BTreeSet<Item> = by_id.iter().map(|p| (PERIOD.to_owned(), p.period_id.clone())).collect();
    for bracket in &sequence.brackets {
        expected.insert((BRACKET.to_owned(), bracket_post_key(&bracket.bracket_id, "start")));
        expected.insert((BRACKET.to_owned(), bracket_post_key(&bracket.bracket_id, "end")));
    }
    let given: BTreeSet<&Item> = order.iter().collect();
    if order.len() != expected.len() || given.len() != expected.len() || given.iter().any(|item| !expected.contains(*item)) {
        return Err(Error::new(Kind::Value, "schedule order must contain each current item exactly once"));
    }
    let position = |wanted: &Item| order.iter().position(|item| item == wanted).expect("checked");
    let mut brackets = Vec::new();
    for bracket in &sequence.brackets {
        let start = position(&(BRACKET.to_owned(), bracket_post_key(&bracket.bracket_id, "start")));
        let end = position(&(BRACKET.to_owned(), bracket_post_key(&bracket.bracket_id, "end")));
        if end < start { return Err(Error::new(Kind::Value, format!("bracket {} end precedes its start", bracket.bracket_id))); }
        let first = order[start + 1..].iter().find(|(kind, _)| kind == PERIOD).map(|(_, key)| key.as_str());
        let last = order[..end].iter().rev().find(|(kind, _)| kind == PERIOD).map(|(_, key)| key.as_str());
        brackets.push(Bracket::new(&bracket.bracket_id, first, last, bracket.count.into())?);
    }
    let bracket_ids: BTreeSet<&str> = brackets.iter().map(|b| b.bracket_id.as_str()).collect();
    let ids: Vec<&str> = order.iter().filter(|(kind, _)| kind == PERIOD).map(|(_, key)| key.as_str()).collect();
    let id_set: BTreeSet<&str> = ids.iter().copied().collect();
    let mut members: BTreeMap<&str, BTreeSet<&str>> = sequence.components.iter()
        .map(|c| (c.component_id.as_str(), c.period_ids.iter().map(String::as_str).filter(|id| id_set.contains(id)).collect())).collect();
    // An inserted card strictly between two members belongs to that group.
    for (index, key) in ids.iter().enumerate() {
        if sequence.period(key).is_some() || index == 0 || index + 1 >= ids.len() { continue; }
        let left = sequence.component_for_period(ids[index - 1]);
        let right = sequence.component_for_period(ids[index + 1]);
        if let (Some(left), Some(right)) = (left, right) {
            if left == right { members.get_mut(left.component_id.as_str()).expect("a component").insert(key); }
        }
    }
    let period_of: BTreeMap<&str, &Period> = by_id.iter().map(|p| (p.period_id.as_str(), p)).collect();
    let periods: Vec<Period> = ids.iter().map(|key| period_of[key].clone()).collect();
    let bindings = sequence.bindings.iter().filter(|b| {
        b.field_ref.period_id.as_deref().is_none_or(|id| period_of.contains_key(id))
            && b.field_ref.bracket_id.as_deref().is_none_or(|id| bracket_ids.contains(id))
    }).cloned().collect();
    let components = sequence.components.iter().filter(|c| !members[c.component_id.as_str()].is_empty()).map(|c| {
        let mut copy = c.clone();
        copy.period_ids = ids.iter().filter(|key| members[c.component_id.as_str()].contains(*key)).map(|key| (*key).to_owned()).collect();
        copy.bracket_ids = c.bracket_ids.iter().filter(|key| bracket_ids.contains(key.as_str())).cloned().collect();
        copy
    }).collect();
    sequence.with(|parts| {
        parts.periods = periods;
        parts.brackets = brackets;
        parts.bindings = bindings;
        parts.components = components;
    })
}

/// The field one `binding` edit names (`PulseFieldRef(...)` from its kind).
fn binding_reference(kind: &str, period_id: &Value, port_key: &Value) -> Result<FieldRef> {
    let field_kind = if kind == "analog" { "dac" } else { kind };
    let period = matches!(kind, "duration" | "analog" | "dac").then(|| py_str(period_id));
    let port = matches!(kind, "delay" | "analog" | "dac").then(|| py_str(port_key));
    let bracket = (kind == "bracket_count").then(|| py_str(period_id));
    FieldRef::new(field_kind, period.as_deref(), port.as_deref(), bracket.as_deref())
}

/// `_sequence_edited(sequence, action, args)`.
pub fn sequence_edited(sequence: &Sequence, action: &str, args: &[Value]) -> Result<Sequence> {
    match action {
        "document_name" => {
            let name = py_str(at(args, 0)?);
            sequence.with(|parts| parts.name = name)
        }
        "run_repeats" => {
            let repeats = py_count(at(args, 0)?, "run_repeats")?;
            if repeats < 0 { return Err(Error::new(Kind::Value, "run_repeats must be non-negative")); }
            let repeats = u64::try_from(repeats).map_err(|_| Error::new(Kind::Value, "run_repeats does not fit the hardware 32-bit count"))?;
            sequence.with(|parts| parts.run_repeats = repeats)
        }
        "period_name" | "duration" | "digital" | "analog" => {
            let key = py_str(at(args, 0)?);
            let period = sequence.period(&key).ok_or_else(|| key_error(&key))?;
            let edited = match action {
                "period_name" => { let name = py_str(at(args, 1)?); rebuilt(period, |p| p.name = name)? }
                "duration" => {
                    let value = py_float(at(args, 1)?)?;
                    let unit = py_str(at(args, 2)?);
                    let aligned = align_to_grid(Num::Float(value), &unit, sequence.time_step_ns, "duration", Some(1))?;
                    let duration = Num::new(aligned, "period duration")?;
                    rebuilt(period, |p| { p.duration = duration; p.unit = unit; })?
                }
                "digital" => {
                    let port_key = py_str(at(args, 1)?);
                    let high = py_truthy(at(args, 2)?);
                    let port = sequence.target.port(&port_key).ok_or_else(|| key_error(&port_key))?;
                    let lane = sequence.target.lane_index(&port.lanes[0]).expect("a target lane");
                    rebuilt(period, |p| p.states[lane] = u8::from(high))?
                }
                _ => {
                    let port_key = py_str(at(args, 1)?);
                    let mode = py_str(at(args, 2)?);
                    let value = at(args, 3).cloned().unwrap_or(Value::Null);
                    let mut steps: Vec<AnalogStep> = period.analog_steps.iter().filter(|s| s.port != port_key).cloned().collect();
                    let text = if value.is_null() { String::new() } else { py_str(&value).trim().to_owned() };
                    if !text.is_empty() && mode != HOLD_MODE {
                        let code = py_int_of_float(py_float(&Value::String(text))?)?;
                        steps.push(AnalogStep::new(&port_key, &mode, Num::Int(code))?);
                    }
                    rebuilt(period, |p| p.analog_steps = steps)?
                }
            };
            sequence.with(|parts| for item in &mut parts.periods { if item.period_id == key { *item = edited.clone(); } })
        }
        "binding" => {
            let kind = py_str(at(args, 0)?);
            let reference = binding_reference(&kind, at(args, 1)?, at(args, 2)?)?;
            let scan = match at(args, 3)? { Value::Bool(flag) => *flag, _ => return Err(Error::new(Kind::Type, "binding scan must be a bool")) };
            let source = py_str(at(args, 4)?);
            let current = sequence.binding(&reference);
            let unit = match current { Some(binding) => binding.unit.clone(), None => sequence.field_unit(&reference)? };
            let config_key = current.map_or_else(String::new, |binding| binding.config_key.clone());
            let binding = Binding::new(reference.clone(), &unit, scan, &source, &config_key)?;
            let live = scan || source != "default";
            let mut bindings: Vec<Binding> = sequence.bindings.iter()
                .filter(|b| b.field_ref != reference || live || !binding.config_key.is_empty())
                .map(|b| if b.field_ref == reference { binding.clone() } else { b.clone() }).collect();
            if current.is_none() && live { bindings.push(binding.clone()); }
            let mut periods = sequence.periods.clone();
            if live && reference.kind == zlc_pulse::model::FieldKind::Dac {
                let id = reference.period_id.clone().unwrap_or_default();
                let port = reference.port.clone().unwrap_or_default();
                if let Some(index) = sequence.period_index(&id) {
                    if periods[index].step(&port).is_none() {
                        let step = AnalogStep::new(&port, ANALOG_MODE_CHOICES[0], Num::Int(held_value(sequence, &id, &port).into()))?;
                        periods[index] = rebuilt(&periods[index], |p| p.analog_steps.push(step))?;
                    }
                }
            }
            sequence.with(|parts| { parts.periods = periods; parts.bindings = bindings; })
        }
        "config_binding" => {
            let field_id = py_str(at(args, 0)?);
            let mut key = py_str(at(args, 1)?).trim().to_owned();
            if !key.is_empty() { key = config_parameter_key(&key)?; }
            let current = sequence.bindings.iter().find(|b| b.field_id() == field_id);
            if !current.is_some_and(|b| b.source == zlc_pulse::model::Source::Config || !b.config_key.is_empty()) {
                return Err(Error::new(Kind::Value, "the field is not a Config parameter"));
            }
            let mut bindings = Vec::new();
            for b in &sequence.bindings {
                if b.field_id() != field_id { bindings.push(b.clone()); continue; }
                if key.is_empty() && !b.scan && b.source == zlc_pulse::model::Source::Default { continue; }
                bindings.push(Binding::new(b.field_ref.clone(), &b.unit, b.scan, b.source.name(), &key)?);
            }
            sequence.with(|parts| parts.bindings = bindings)
        }
        "delay" => {
            let port = py_str(at(args, 0)?);
            let value = at(args, 1)?;
            let unit = py_str(at(args, 2)?);
            let value = if py_truthy(value) { py_float(value)? } else { 0.0 };
            let amount = align_to_grid(Num::Float(value), &unit, sequence.time_step_ns, "delay", None)?;
            let mut delays: Vec<OutputDelay> = sequence.delays.iter().filter(|d| d.port != port).cloned().collect();
            if amount != 0.0 { delays.push(OutputDelay::new(&port, Num::new(amount, "delay value")?, &unit)?); }
            sequence.with(|parts| parts.delays = delays)
        }
        "insert_period" | "insert_spacer" => {
            let before = at(args, 0)?;
            let before = if before.is_null() { None } else { Some(item(before)?) };
            let periods = &sequence.periods;
            let mut order = sequence_item_order(sequence);
            let position = match &before {
                None => order.len(),
                Some(wanted) => order.iter().position(|item| item == wanted)
                    .ok_or_else(|| Error::new(Kind::Value, format!("{} is not in list", crate::items::item_repr(wanted))))?,
            };
            let index = order[..position].iter().filter(|(kind, _)| kind == PERIOD).count();
            let spacer = action == "insert_spacer";
            let model = if spacer { periods.iter().rev().find(|p| p.is_spacer()) } else { Some(&periods[index.saturating_sub(1)]) };
            let neighbours: Vec<&Period> = [index.checked_sub(1), Some(index)].into_iter().flatten().filter_map(|i| periods.get(i)).collect();
            let lanes = sequence.target.raw_lanes.len();
            let states = if spacer {
                (0..lanes).map(|lane| u8::from(neighbours.iter().all(|p| p.states[lane] != 0))).collect()
            } else { model.expect("a neighbour").states.clone() };
            let existing = periods.iter().map(|p| p.period_id.as_str()).chain(periods.iter().map(|p| p.name.as_str()));
            let key = unique_id(existing, if spacer { "spacer" } else { "period" });
            let (duration, unit) = model.map_or((Num::Int(1), "ms".to_owned()), |m| (m.duration, m.unit.clone()));
            let added = Period::new(&key, duration, &unit, states, Vec::new(), "", if spacer { "spacer" } else { "period" })?;
            order.insert(position, (PERIOD.to_owned(), key));
            let mut all = periods.clone();
            all.push(added);
            reordered_sequence(sequence, &order, Some(&all))
        }
        "reorder_items" => reordered_sequence(sequence, &items(at(args, 0)?)?, None),
        "remove_items" => {
            let selected = items(at(args, 0)?)?;
            if selected.is_empty() { return Err(Error::new(Kind::Value, "select the timeline items to remove")); }
            let mut periods: BTreeSet<String> = selected.iter().filter(|(kind, _)| kind == PERIOD).map(|(_, key)| key.clone()).collect();
            let mut brackets: BTreeSet<String> = selected.iter().filter(|(kind, _)| kind == BRACKET)
                .map(|(_, key)| post_bracket(key).to_owned()).collect();
            for (kind, key) in &selected {
                if kind == COMPONENT {
                    let fragment = extract_subpulse(sequence, key)?;
                    periods.extend(fragment.to_sequence().periods.iter().map(|p| p.period_id.clone()));
                    brackets.extend(fragment.to_sequence().brackets.iter().map(|b| b.bracket_id.clone()));
                }
            }
            let retained = sequence.with(|parts| {
                parts.brackets.retain(|b| !brackets.contains(&b.bracket_id));
                parts.bindings.retain(|b| b.field_ref.bracket_id.as_ref().is_none_or(|id| !brackets.contains(id)));
                for component in &mut parts.components { component.bracket_ids.retain(|id| !brackets.contains(id)); }
            })?;
            let order: Vec<Item> = sequence_item_order(&retained).into_iter().filter(|(kind, key)| kind != PERIOD || !periods.contains(key)).collect();
            let kept: Vec<Period> = retained.periods.iter().filter(|p| !periods.contains(&p.period_id)).cloned().collect();
            reordered_sequence(&retained, &order, Some(&kept))
        }
        "bracket" | "bracket_add" => {
            let (key, rest) = if action == "bracket_add" {
                (unique_id(sequence.brackets.iter().map(|b| b.bracket_id.as_str()), "bracket"), args)
            } else { (py_str(at(args, 0)?), &args[1.min(args.len())..]) };
            let start = py_optional_str(at(rest, 0)?);
            let end = py_optional_str(at(rest, 1)?);
            let count = py_count(at(rest, 2)?, "bracket count")?;
            let changed = Bracket::new(&key, start.as_deref(), end.as_deref(), count)?;
            sequence.with(|parts| {
                if action == "bracket_add" { parts.brackets.insert(0, changed); }
                else { for b in &mut parts.brackets { if b.bracket_id == key { *b = changed.clone(); } } }
            })
        }
        other => Err(Error::new(Kind::Value, format!("unknown Pulse edit {}", repr(other)))),
    }
}

/// An edit and the bindings it orphaned (`prune_orphaned_bindings`).
pub fn edited(sequence: &Sequence, action: &str, args: &[Value]) -> Result<(Sequence, Vec<String>)> {
    prune_orphaned_bindings(&sequence_edited(sequence, action, args)?)
}

/// `_control_edit_arguments`: the flat named arguments of a command as the
/// editor's positional intent.
pub fn control_edit_arguments(action: &str, values: &Map<String, Value>) -> Result<(String, Vec<Value>)> {
    let get = |name: &str| values.get(name).cloned().ok_or_else(|| key_error(name));
    let opt = |name: &str| values.get(name).cloned().unwrap_or(Value::Null);
    Ok(match action {
        "period_name" => (action.into(), vec![get("period_id")?, get("name")?]),
        "duration" => (action.into(), vec![get("period_id")?, get("value")?, get("unit")?]),
        "digital" => {
            let high = get("high")?;
            if !high.is_boolean() { return Err(Error::new(Kind::Type, "high must be a boolean")); }
            (action.into(), vec![get("period_id")?, get("port_key")?, high])
        }
        "analog" => (action.into(), vec![get("period_id")?, get("port_key")?, get("mode")?, opt("value")]),
        "binding" => {
            let kind = get("field_kind")?;
            let id = if kind.as_str() == Some("bracket_count") { get("bracket_id")? } else { opt("period_id") };
            (action.into(), vec![kind, id, opt("port_key"), get("scan")?, get("source")?])
        }
        "config_binding" => (action.into(), vec![get("field_id")?, get("key")?]),
        "bracket" => (action.into(), vec![get("bracket_id")?, get("start")?, get("end")?, get("count")?]),
        "bracket_add" => (action.into(), vec![get("start")?, get("end")?, get("count")?]),
        "period_add" | "spacer_add" => {
            let name = if action == "period_add" { "insert_period" } else { "insert_spacer" };
            (name.into(), vec![opt("before_item")])
        }
        "period_remove" => ("remove_items".into(), vec![Value::Array(vec![Value::Array(vec![PERIOD.into(), get("period_id")?])])]),
        "bracket_remove" => {
            let id = py_str(&get("bracket_id")?);
            ("remove_items".into(), vec![Value::Array(vec![Value::Array(vec![BRACKET.into(), bracket_post_key(&id, "start").into()])])])
        }
        "remove_items" => (action.into(), vec![get("items")?]),
        "reorder" | "reorder_items" => ("reorder_items".into(), vec![get("order")?]),
        "delay" => (action.into(), vec![get("port_key")?, get("value")?, get("unit")?]),
        "run_repeats" => (action.into(), vec![get("repeats")?]),
        "document_name" => (action.into(), vec![get("name")?]),
        other => return Err(Error::new(Kind::Value, format!("unknown Pulse edit {}; inspect the pulse_edit command's description", repr(other)))),
    })
}

/// `start_new_pulse`: the first digital output high in period1, every lane
/// safe in period2, each 1 µs -- the smallest thing that is a pulse.
pub fn new_pulse(target: std::sync::Arc<zlc_pulse::model::Target>, time_step_ns: f64) -> Result<Sequence> {
    let safe = vec![0u8; target.raw_lanes.len()];
    let mut driven = safe.clone();
    if let Some(first) = target.ports.iter().find(|p| p.kind == PortKind::Digital) {
        driven[target.lane_index(&first.lanes[0]).expect("a target lane")] = 1;
    }
    let length = Num::new(NEW_PULSE_PERIOD_NS, "period duration")?;
    let periods = vec![Period::new("period1", length, "ns", driven, Vec::new(), "", "period")?,
                       Period::new("period2", length, "ns", safe, Vec::new(), "", "period")?];
    Sequence::new(SequenceParts { name: "untitled".into(), time_step_ns, ..SequenceParts::new(target, periods) })
}

/// `clear_all`: one safe tick-long period on the same target, name and clock.
pub fn cleared(sequence: &Sequence) -> Result<Sequence> {
    let safe = vec![0u8; sequence.target.raw_lanes.len()];
    let period = Period::new("period1", Num::new(sequence.time_step_ns, "period duration")?, "ns", safe, Vec::new(), "", "period")?;
    Sequence::new(SequenceParts { name: sequence.name.clone(), time_step_ns: sequence.time_step_ns,
                                  ..SequenceParts::new(sequence.target.clone(), vec![period]) })
}

/// `_set_port_everywhere`: one digital port high or low in every period, or
/// one DAC's steps gone; the port's delay stays.
pub fn port_everywhere(sequence: &Sequence, port_key: &str, high: bool) -> Result<Sequence> {
    let port = sequence.target.port(port_key).ok_or_else(|| Error::new(Kind::Value, format!("{port_key} is not a port on this target")))?;
    let mut periods = Vec::new();
    if port.kind == PortKind::Digital {
        let lane = sequence.target.lane_index(&port.lanes[0]).expect("a target lane");
        for period in &sequence.periods { periods.push(rebuilt(period, |p| p.states[lane] = u8::from(high))?); }
    } else if high {
        return Err(Error::new(Kind::Value, format!("{port_key} is an analog port; there is no level to fill it with")));
    } else {
        for period in &sequence.periods { periods.push(rebuilt(period, |p| p.analog_steps.retain(|s| s.port != port_key))?); }
    }
    sequence.with(|parts| parts.periods = periods)
}
