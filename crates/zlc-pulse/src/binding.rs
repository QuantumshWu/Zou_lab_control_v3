//! Resolve named pulse inputs into the physical fields they own: reading and
//! writing one field in a binding's unit, labels, API values, Config values and
//! Subpulse file sources.
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use crate::codec::{ConfigValues, read_subpulse};
use crate::error::{Error, Kind, Result, repr, value_error};
use crate::exact::{Decimal, Ratio};
use crate::model::{Binding, Bracket, FieldKind, FieldRef, OutputDelay, Sequence, Source, align_to_grid, available_id,
                   config_parameter_key};
use crate::num::{Num, number_for};
use crate::units::nanoseconds_exponent;

/// `convert_time(value, source_unit, target_unit)`: one duration in another unit.
pub fn convert_time(value: f64, source_unit: &str, target_unit: &str) -> Result<f64> {
    if !value.is_finite() { return value_error("time value must be finite"); }
    let source = nanoseconds_exponent(source_unit)?;
    if source_unit == target_unit { return Ok(value); }
    let target = nanoseconds_exponent(target_unit)?;
    if source == target { return Ok(value); }
    Ok(Decimal::from_f64(value).scale10(source - target).to_f64())
}

/// `pulse_field_value(sequence, reference, unit)`: one field in its binding's unit.
pub fn pulse_field_value(sequence: &Sequence, reference: &FieldRef, unit: &str) -> Result<f64> {
    match reference.kind {
        FieldKind::BracketCount => {
            if unit != "count" { return value_error("bracket count fields use unit 'count'"); }
            let id = reference.bracket_id.as_deref().unwrap_or("");
            sequence.brackets.iter().find(|bracket| bracket.bracket_id == id).map(|bracket| bracket.count as f64)
                .ok_or_else(|| Error::new(Kind::Value, format!("no bracket exists with id {}", repr(id))))
        }
        FieldKind::Delay => match sequence.delays.iter().find(|delay| reference.port.as_deref() == Some(&delay.port)) {
            None => Ok(0.0),
            Some(delay) => convert_time(delay.value.to_f64(), &delay.unit, unit),
        },
        FieldKind::Duration | FieldKind::Dac => {
            let id = reference.period_id.as_deref().unwrap_or("");
            let period = sequence.period(id).ok_or_else(|| Error::new(Kind::Value, format!("no period exists with id {}", repr(id))))?;
            if reference.kind == FieldKind::Duration { return convert_time(period.duration.to_f64(), &period.unit, unit); }
            let port = reference.port.as_deref().unwrap_or("");
            let step = period.step(port).ok_or_else(|| Error::new(Kind::Value,
                format!("period {} has no DAC step on port {}", repr(id), repr(port))))?;
            if unit != "value" { return value_error("DAC fields use unit 'value'"); }
            Ok(step.value as f64)
        }
    }
}

/// One field write: `(reference, value, unit, field_name)`.
pub type FieldWrite = (FieldRef, Num, String, String);

/// `replace_pulse_field`: one physical field replaced, durations on the grid.
pub fn replace_pulse_field(sequence: &Sequence, reference: &FieldRef, value: Num, unit: &str, field_name: &str) -> Result<Sequence> {
    replace_pulse_fields(sequence, &[(reference.clone(), value, unit.to_owned(), field_name.to_owned())], false)
}

/// `_replace_pulse_fields`: normalize all edits, then validate one final pulse.
pub fn replace_pulse_fields(sequence: &Sequence, fields: &[FieldWrite], remove_api_parameters: bool) -> Result<Sequence> {
    let mut period_durations: BTreeMap<String, Num> = BTreeMap::new();
    let mut period_steps: BTreeMap<String, Vec<crate::model::AnalogStep>> = BTreeMap::new();
    let mut delay_changes: Vec<OutputDelay> = Vec::new();
    let mut bracket_changes: BTreeMap<String, i128> = BTreeMap::new();
    for (reference, value, unit, field_name) in fields {
        match reference.kind {
            FieldKind::BracketCount => {
                let current = pulse_field_value(sequence, reference, unit)?;
                let number = value.to_f64();
                if number.fract() != 0.0 { return value_error(format!("{} must be an integer count", repr(field_name))); }
                if number != current { bracket_changes.insert(reference.bracket_id.clone().unwrap_or_default(), number as i128); }
            }
            FieldKind::Delay => {
                let port_key = reference.port.as_deref().unwrap_or("");
                if !sequence.target.port(port_key).is_some_and(|port| matches!(port.kind, crate::model::PortKind::Digital | crate::model::PortKind::Dac)) {
                    return value_error(format!("no delay output exists with port {}", repr(port_key)));
                }
                let previous = sequence.delays.iter().find(|delay| delay.port == port_key);
                let target_unit = previous.map_or(unit.as_str(), |delay| delay.unit.as_str());
                let held = previous.map_or(0.0, |delay| delay.value.to_f64());
                let authored = convert_time(value.to_f64(), unit, target_unit)?;
                if authored == held { continue; }
                let authored = number_for(align_to_grid(Num::new(authored, field_name)?, target_unit, sequence.time_step_ns, field_name, None)?);
                if !authored.same_value(held) {
                    delay_changes.retain(|delay| delay.port != port_key);
                    delay_changes.push(OutputDelay::new(port_key, authored, target_unit)?);
                }
            }
            FieldKind::Duration | FieldKind::Dac => {
                let id = reference.period_id.as_deref().unwrap_or("");
                let Some(period) = sequence.period(id) else { return value_error(format!("{} names no period on this sequence", repr(field_name))); };
                if reference.kind == FieldKind::Duration {
                    let authored = convert_time(value.to_f64(), unit, &period.unit)?;
                    if period.duration.same_value(authored) { continue; }
                    if authored <= 0.0 {
                        return value_error(format!("{} must be a positive duration, not {} {unit}", repr(field_name), value.py_str()));
                    }
                    let authored = number_for(align_to_grid(Num::new(authored, field_name)?, &period.unit, sequence.time_step_ns, field_name, Some(1))?);
                    if authored != period.duration { period_durations.insert(id.to_owned(), authored); }
                    continue;
                }
                if unit != "value" { return value_error("DAC fields use unit 'value'"); }
                let port = reference.port.as_deref().unwrap_or("");
                let Some(at) = period.analog_steps.iter().position(|step| step.port == port) else {
                    return value_error(format!("{} names no DAC step in period {}", repr(field_name), repr(id)));
                };
                let authored = value.to_f64().round_ties_even() as i64;
                let steps = period_steps.entry(id.to_owned()).or_insert_with(|| period.analog_steps.clone());
                if authored != steps[at].value { steps[at].value = authored; }
            }
        }
    }
    // A DAC write that put a value back to what it was is no change.
    period_steps.retain(|id, steps| sequence.period(id).is_some_and(|period| period.analog_steps != *steps));
    if bracket_changes.is_empty() && period_durations.is_empty() && period_steps.is_empty() && delay_changes.is_empty()
        && !(remove_api_parameters && !sequence.api_bindings().is_empty()) {
        return Ok(sequence.clone());
    }
    // A count written through a field still obeys the bracket's own rule, refused
    // before the pulse is rebuilt.
    let mut counts: BTreeMap<String, u64> = BTreeMap::new();
    for bracket in &sequence.brackets {
        if let Some(count) = bracket_changes.get(&bracket.bracket_id) {
            let checked = Bracket::new(&bracket.bracket_id, bracket.start_period_id.as_deref(), bracket.end_period_id.as_deref(), *count)?;
            counts.insert(bracket.bracket_id.clone(), checked.count);
        }
    }
    sequence.with(|parts| {
        for bracket in &mut parts.brackets {
            if let Some(count) = counts.get(&bracket.bracket_id) { bracket.count = *count; }
        }
        for period in &mut parts.periods {
            if let Some(duration) = period_durations.get(&period.period_id) { period.duration = *duration; }
            if let Some(steps) = period_steps.get(&period.period_id) { period.analog_steps = steps.clone(); }
        }
        let mut fresh = delay_changes.clone();
        for delay in &mut parts.delays {
            if let Some(at) = fresh.iter().position(|change| change.port == delay.port) { *delay = fresh.remove(at); }
        }
        parts.delays.extend(fresh);
        if remove_api_parameters {
            for binding in &mut parts.bindings { if binding.source == Source::Api { binding.source = Source::Default; } }
        }
    })
}

/// `prune_orphaned_bindings`: drop every binding whose DAC field the pulse no longer has.
pub fn prune_orphaned_bindings(sequence: &Sequence) -> Result<(Sequence, Vec<String>)> {
    let held = |reference: &FieldRef| reference.kind != FieldKind::Dac
        || reference.period_id.as_deref().and_then(|id| sequence.period(id))
            .is_some_and(|period| period.step(reference.port.as_deref().unwrap_or("")).is_some());
    let dropped: Vec<String> = sequence.bindings.iter().filter(|binding| !held(&binding.field_ref)).map(Binding::field_id).collect();
    if dropped.is_empty() { return Ok((sequence.clone(), dropped)); }
    let pruned = sequence.with(|parts| parts.bindings.retain(|binding| held(&binding.field_ref)))?;
    Ok((pruned, dropped))
}

/// The API parameters as the pulse plays them: by period, a duration before
/// that period's DAC fields, bracket counts at their start, delays last.
pub fn api_bindings_in_period_order(sequence: &Sequence) -> Vec<&Binding> {
    let rank = |kind: FieldKind| match kind { FieldKind::Duration => 0, FieldKind::Dac => 1, FieldKind::BracketCount => 2, FieldKind::Delay => 3 };
    let played_at = |binding: &Binding| {
        let reference = &binding.field_ref;
        let position = match reference.kind {
            FieldKind::Delay => sequence.periods.len(),
            FieldKind::BracketCount => {
                let id = reference.bracket_id.as_deref().unwrap_or("");
                sequence.brackets.iter().zip(sequence.bracket_bounds()).find(|(bracket, _)| bracket.bracket_id == id).map_or(0, |(_, bounds)| bounds.0)
            }
            _ => reference.period_id.as_deref().and_then(|id| sequence.period_index(id)).unwrap_or(sequence.periods.len()),
        };
        (position, rank(reference.kind), reference.port.clone().unwrap_or_default())
    };
    let mut bindings = sequence.api_bindings();
    bindings.sort_by_key(|binding| played_at(binding));
    bindings
}

fn port_text(sequence: &Sequence, port: Option<&str>) -> String {
    let key = port.unwrap_or("None");
    sequence.target.port(key).map_or(key.to_owned(), |spec| spec.display().to_owned())
}

/// `field_label(sequence, reference)`: one physical field said the way it reads on the pulse.
pub fn field_label(sequence: &Sequence, reference: &FieldRef) -> Result<String> {
    Ok(match reference.kind {
        FieldKind::BracketCount => format!("{}.repeats", bracket_label(sequence, reference.bracket_id.as_deref().unwrap_or(""))?),
        FieldKind::Delay => format!("{}.delay", port_text(sequence, reference.port.as_deref())),
        FieldKind::Duration => format!("{}.duration", sequence.period_label(reference.period_id.as_deref().unwrap_or(""))),
        FieldKind::Dac => format!("{}.{}", sequence.period_label(reference.period_id.as_deref().unwrap_or("")),
                                  port_text(sequence, reference.port.as_deref())),
    })
}

/// `bracket_label`: "Bracket N" among the pulse's brackets, under its Subpulse's name.
pub fn bracket_label(sequence: &Sequence, bracket_id: &str) -> Result<String> {
    let Some(ordinal) = sequence.brackets.iter().position(|bracket| bracket.bracket_id == bracket_id) else {
        return value_error(format!("no bracket exists with id {}", repr(bracket_id)));
    };
    let label = format!("Bracket {}", ordinal + 1);
    Ok(match sequence.component_for_field(&FieldRef::bracket_count(bracket_id)) {
        None => label,
        Some(component) => format!("{} / {label}", component.name),
    })
}

/// `authored_api_entries`: every API parameter as `(value, unit)`, both the pulse's own.
pub fn authored_api_entries(sequence: &Sequence) -> Result<Vec<(String, (f64, String))>> {
    sequence.api_bindings().iter().map(|parameter| {
        let value = pulse_field_value(sequence, &parameter.field_ref, &parameter.unit).map_err(|error| Error::new(Kind::Value,
            format!("API parameter {} has no field: {error}", repr(&parameter.field_id()))))?;
        Ok((parameter.field_id(), (value, parameter.unit.clone())))
    }).collect()
}

/// Which declared bindings a name space addresses: `scan`, or a source name.
fn declared<'a>(sequence: &'a Sequence, source: &str) -> Vec<&'a Binding> {
    if source == "scan" { sequence.scan_bindings() } else { sequence.bindings.iter().filter(|binding| binding.source.name() == source).collect() }
}

/// `normalize_binding_values`: stable field ids or exact displayed paths,
/// resolved once at a public boundary; the entries keep their order.
pub fn normalize_binding_values<T: Clone>(sequence: &Sequence, entries: &[(String, T)], source: &str) -> Result<Vec<(String, T)>> {
    let declared = declared(sequence, source);
    let ids: BTreeSet<String> = declared.iter().map(|binding| binding.field_id()).collect();
    let mut labels: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for binding in &declared { labels.entry(field_label(sequence, &binding.field_ref)?).or_default().push(binding.field_id()); }
    let mut resolved: Vec<(String, T)> = Vec::new();
    for (name, value) in entries {
        let key = if ids.contains(name) { name.clone() } else {
            match labels.get(name).map(Vec::as_slice) {
                Some([only]) => only.clone(),
                _ => return value_error(format!("unknown or ambiguous {source} field {}", repr(name))),
            }
        };
        if resolved.iter().any(|(existing, _)| *existing == key) { return value_error(format!("duplicate {source} field {}", repr(&key))); }
        resolved.push((key, value.clone()));
    }
    Ok(resolved)
}

fn check_value_unit(declared: &str, supplied: &str, label: &str) -> Result<()> {
    let numeric = |unit: &str| unit == "value" || unit == "count";
    if declared != supplied && (numeric(declared) || numeric(supplied)) {
        return value_error(format!("{label} is in {} where the pulse declares {}", repr(supplied), repr(declared)));
    }
    Ok(())
}

/// `apply_config_values(sequence, entries, current=)`: saved names into every
/// matching non-scanned Config field; `(filled, applied keys, unknown keys)`.
pub fn apply_config_values(sequence: &Sequence, entries: &ConfigValues, current: Option<&Sequence>) -> Result<(Sequence, Vec<String>, Vec<String>)> {
    let entries: &[(String, (f64, String))] = if sequence.config_enabled { entries } else { &[] };
    for (key, _) in entries { config_parameter_key(key)?; }
    let mut restore = current.is_some();
    let mut destination = current.cloned().unwrap_or_else(|| sequence.clone());
    if destination.config_enabled != sequence.config_enabled {
        destination = destination.with(|parts| parts.config_enabled = sequence.config_enabled)?;
        restore = true;
    }
    let declared: Vec<Binding> = destination.config_bindings().into_iter().filter(|binding| !binding.scan).cloned().collect();
    let names: BTreeSet<&str> = declared.iter().map(|binding| binding.config_key.as_str()).filter(|key| !key.is_empty()).collect();
    let mut fields = Vec::new();
    let mut applied: Vec<String> = Vec::new();
    for binding in &declared {
        let key = &binding.config_key;
        if let Some((_, (number, unit))) = entries.iter().find(|(name, _)| !key.is_empty() && name == key) {
            check_value_unit(&binding.unit, unit, &format!("Config value {}", repr(key)))?;
            fields.push((binding.field_ref.clone(), number_for(*number), unit.clone(), key.clone()));
            if !applied.contains(key) { applied.push(key.clone()); }
        } else if restore {
            let value = pulse_field_value(sequence, &binding.field_ref, &binding.unit)?;
            fields.push((binding.field_ref.clone(), number_for(value), binding.unit.clone(), binding.field_id()));
        }
    }
    let unknown = entries.iter().map(|(key, _)| key.clone()).filter(|key| !names.contains(key.as_str())).collect();
    Ok((replace_pulse_fields(&destination, &fields, false)?, applied, unknown))
}

/// `resolve_subpulse_sources`: enabled file sources read into a playback copy;
/// a relative path is resolved against `base_directory` (else the working one).
pub fn resolve_subpulse_sources(sequence: &Sequence, base_directory: Option<&Path>) -> Result<Sequence> {
    if !sequence.config_enabled { return Ok(sequence.clone()); }
    let mut resolved = sequence.clone();
    for component in &sequence.components {
        if component.source != Source::Config { continue; }
        if !component.from_file() {
            resolved = resolved.with(|parts| for item in &mut parts.components {
                if item.component_id == component.component_id { item.source = Source::Default; }
            })?;
            continue;
        }
        let mut path = PathBuf::from(&component.config_path);
        if !path.is_absolute() { path = base_directory.map_or_else(|| std::env::current_dir().unwrap_or_default(), Path::to_path_buf).join(path); }
        let path = crate::codec::absolute(&path);
        let shown = path.display().to_string();
        let attempt = || -> Result<Sequence> {
            let fragment = read_subpulse(&path)?;
            let fragment_sequence = fragment.to_sequence();
            let current = resolved.components.iter().find(|item| item.component_id == component.component_id).expect("present");
            let mut mappings: Vec<Vec<(String, String)>> = Vec::new();
            let groups: [(BTreeSet<String>, &Vec<String>, Vec<String>, &Vec<(String, String)>); 2] = [
                (resolved.periods.iter().map(|p| p.period_id.clone()).collect(), &current.period_ids,
                 fragment_sequence.periods.iter().map(|p| p.period_id.clone()).collect(), &component.period_id_map),
                (resolved.brackets.iter().map(|b| b.bracket_id.clone()).collect(), &current.bracket_ids,
                 fragment_sequence.brackets.iter().map(|b| b.bracket_id.clone()).collect(), &component.bracket_id_map),
            ];
            for (existing, members, source_ids, known) in groups {
                let mut used: BTreeSet<String> = existing.into_iter().filter(|id| !members.contains(id)).collect();
                let mut mapped = Vec::new();
                for source_id in source_ids {
                    let known_target = known.iter().find(|(source, _)| *source == source_id).map(|(_, target)| target.clone())
                        .unwrap_or_else(|| if members.contains(&source_id) { source_id.clone() } else { String::new() });
                    let target_id = if !known_target.is_empty() {
                        if used.contains(&known_target) {
                            return value_error(format!("Subpulse source id {} conflicts with {}", repr(&source_id), repr(&known_target)));
                        }
                        used.insert(known_target.clone());
                        known_target
                    } else { available_id(&format!("{}_{source_id}", component.component_id), &mut used) };
                    mapped.push((source_id, target_id));
                }
                mappings.push(mapped);
            }
            let candidate = crate::components::replace_component(&resolved, &component.component_id, &fragment,
                                                                Some(&mappings[0]), Some(&mappings[1]))?;
            let members = candidate.components.iter().find(|item| item.component_id == component.component_id).expect("present").period_ids.clone();
            let unset: Vec<Binding> = candidate.bindings.iter().filter(|binding| binding.kind() == FieldKind::Dac
                && binding.field_ref.period_id.as_ref().is_some_and(|id| members.contains(id))
                && candidate.period(binding.field_ref.period_id.as_deref().unwrap()).is_some_and(|period|
                    period.step(binding.field_ref.port.as_deref().unwrap_or("")).is_none())).cloned().collect();
            let candidate = if unset.is_empty() { candidate } else { candidate.with(|parts| parts.bindings.retain(|binding| !unset.contains(binding)))? };
            candidate.with(|parts| for item in &mut parts.components {
                if item.component_id == component.component_id {
                    item.source = Source::Default;
                    item.config_path = shown.clone();
                    item.period_id_map = mappings[0].clone();
                    item.bracket_id_map = mappings[1].clone();
                }
            })
        };
        resolved = attempt().map_err(|error| Error::new(Kind::Value,
            format!("Cannot resolve Subpulse {} from {}: {error}", repr(&component.name), repr(&shown))))?;
    }
    Ok(resolved)
}

/// `stale_subpulse_bindings`: Scan/API marks inside a file-backed Subpulse
/// that its file no longer offers.
pub fn stale_subpulse_bindings(authored: &Sequence, resolved: &Sequence) -> Vec<Binding> {
    if !authored.config_enabled { return Vec::new(); }
    authored.bindings.iter().filter(|binding| {
        if !(binding.scan || binding.source == Source::Api) { return false; }
        if !authored.component_for_field(&binding.field_ref).is_some_and(|component| component.from_file()) { return false; }
        match resolved.bindings.iter().find(|item| item.field_ref == binding.field_ref) {
            None => true,
            Some(current) => (binding.scan && !current.scan) || (binding.source == Source::Api && current.source != Source::Api),
        }
    }).cloned().collect()
}

/// `apply_api_values`: explicit API values `(number, unit)`; omitted inputs keep their defaults.
pub fn apply_api_values(sequence: &Sequence, entries: &[(String, (Num, String))]) -> Result<(Sequence, Vec<String>)> {
    let sequence = resolve_subpulse_sources(sequence, None)?;
    let normalized = normalize_binding_values(&sequence, entries, "api")?;
    let mut fields = Vec::new();
    for binding in sequence.api_bindings() {
        if let Some((_, (number, unit))) = normalized.iter().find(|(key, _)| *key == binding.field_id()) {
            check_value_unit(&binding.unit, unit, &format!("API value {}", repr(&binding.field_id())))?;
            fields.push((binding.field_ref.clone(), *number, unit.clone(), binding.field_id()));
        }
    }
    Ok((replace_pulse_fields(&sequence, &fields, false)?, normalized.into_iter().map(|(key, _)| key).collect()))
}

/// `resolve_api_parameters(sequence, values=)`: supplied API inputs (in each
/// binding's unit) or the authored values baked in; the API source cleared.
pub fn resolve_api_parameters(sequence: &Sequence, values: Option<&[(String, Num)]>) -> Result<Sequence> {
    let sequence = resolve_subpulse_sources(sequence, None)?;
    let mut resolved: Vec<(String, Num)> = authored_api_entries(&sequence)?.into_iter().map(|(key, (value, _))| (key, number_for(value))).collect();
    if let Some(values) = values {
        for (key, value) in normalize_binding_values(&sequence, values, "api")? {
            match resolved.iter_mut().find(|(existing, _)| *existing == key) { Some(slot) => slot.1 = value, None => resolved.push((key, value)) }
        }
    }
    let fields: Vec<FieldWrite> = sequence.api_bindings().iter().map(|binding| {
        let value = resolved.iter().find(|(key, _)| *key == binding.field_id()).map(|(_, value)| *value).expect("authored");
        (binding.field_ref.clone(), value, binding.unit.clone(), binding.field_id())
    }).collect();
    replace_pulse_fields(&sequence, &fields, true)
}

/// One tick in a field's unit: `float(Fraction(str(step)) / nanoseconds_per(unit))`.
pub fn tick_in(sequence: &Sequence, unit: &str) -> Result<f64> {
    Ok(Decimal::from_f64(sequence.time_step_ns).scale10(-nanoseconds_exponent(unit)?).to_f64())
}

/// How many ticks one of `unit` is: `float(nanoseconds_per(unit) / Fraction(str(step)))`.
pub fn ticks_per(sequence: &Sequence, unit: &str) -> Result<f64> {
    Ok(Ratio::of(&Decimal::from_i128(1).scale10(nanoseconds_exponent(unit)?), &Decimal::from_f64(sequence.time_step_ns)).to_f64())
}
