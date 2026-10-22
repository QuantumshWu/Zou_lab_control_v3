//! Subpulse components: grouping consecutive periods without changing what
//! plays, copying a group out as a `zlc.subpulse` fragment, inserting a
//! fragment as an independent instance (its ids remapped), and replacing or
//! removing an instance while enclosing brackets keep their gaps.
use std::collections::BTreeSet;

use crate::error::{Result, repr, value_error};
use crate::model::{Binding, Bracket, Component, FieldRef, Period, Sequence, SequenceParts, Subpulse, available_id, available_name};

fn component<'a>(sequence: &'a Sequence, component_id: &str) -> Result<&'a Component> {
    sequence.components.iter().find(|item| item.component_id == component_id)
        .ok_or_else(|| crate::error::Error::new(crate::error::Kind::Value, format!("no component exists with id {}", repr(component_id))))
}

fn internal_brackets(sequence: &Sequence, component: &Component) -> Vec<Bracket> {
    sequence.brackets.iter().filter(|bracket| component.bracket_ids.contains(&bracket.bracket_id)).cloned().collect()
}

/// `group_component`: group the selected periods and brackets.
pub fn group_component(sequence: &Sequence, period_ids: &[String], name: &str, component_id: Option<&str>, bracket_ids: &[String]) -> Result<Sequence> {
    if period_ids.iter().collect::<BTreeSet<_>>().len() != period_ids.len() || period_ids.iter().any(|id| sequence.period(id).is_none()) {
        return value_error("component selection must name existing unique periods");
    }
    let members: Vec<String> = sequence.periods.iter().filter(|period| period_ids.contains(&period.period_id)).map(|period| period.period_id.clone()).collect();
    let mut used: BTreeSet<String> = sequence.components.iter().map(|item| item.component_id.clone()).collect();
    let component_id = component_id.map_or_else(|| available_id("component1", &mut used), str::to_owned);
    let component = Component::new(&component_id, name, &members, bracket_ids, "default", "", &[], &[])?;
    sequence.with(|parts| parts.components.push(component))
}

/// `ungroup_component`: remove a group, qualifying only names that would collide.
pub fn ungroup_component(sequence: &Sequence, component_id: &str) -> Result<Sequence> {
    let component = component(sequence, component_id)?.clone();
    let mut used: BTreeSet<String> = sequence.periods.iter().filter(|period| sequence.component_for_period(&period.period_id).is_none())
        .map(|period| period.display_name().to_owned()).collect();
    let mut periods = Vec::new();
    for period in &sequence.periods {
        let mut period = period.clone();
        if component.period_ids.contains(&period.period_id) {
            let label = period.display_name().to_owned();
            if used.contains(&label) { period.name = available_name(&sequence.period_label(&period.period_id), &mut used); }
            else { used.insert(label); }
        }
        periods.push(period);
    }
    sequence.with(|parts| { parts.periods = periods; parts.components.retain(|item| *item != component); })
}

/// `extract_subpulse`: copy one group's contents; its parent's loops and delays stay.
pub fn extract_subpulse(sequence: &Sequence, component_id: &str) -> Result<Subpulse> {
    let component = component(sequence, component_id)?;
    let periods = component.period_ids.iter().map(|id| sequence.period(id).expect("member").clone()).collect();
    let bindings = sequence.bindings.iter().filter(|binding| {
        binding.field_ref.period_id.as_ref().is_some_and(|id| component.period_ids.contains(id))
            || binding.field_ref.bracket_id.as_ref().is_some_and(|id| component.bracket_ids.contains(id))
    }).cloned().collect();
    let brackets = internal_brackets(sequence, component).into_iter().map(|mut bracket| {
        if !bracket.start_period_id.as_ref().is_some_and(|id| component.period_ids.contains(id)) { bracket.start_period_id = None; }
        if !bracket.end_period_id.as_ref().is_some_and(|id| component.period_ids.contains(id)) { bracket.end_period_id = None; }
        bracket
    }).collect();
    Subpulse::new(&component.name, sequence.target.clone(), sequence.time_step_ns, periods, bindings, brackets)
}

#[allow(clippy::too_many_arguments)]
fn remap(subpulse: &Subpulse, component_id: &str, used_periods: &mut BTreeSet<String>, used_brackets: &mut BTreeSet<String>,
         retained_periods: &[String], retained_brackets: &[String], previous: Option<&str>, next: Option<&str>,
         period_map: Option<&[(String, String)]>, bracket_map: Option<&[(String, String)]>) -> (Vec<Period>, Vec<Binding>, Vec<Bracket>) {
    let fragment = subpulse.to_sequence();
    let lookup = |map: &[(String, String)], id: &str| map.iter().find(|(source, _)| source == id).map(|(_, target)| target.clone()).expect("mapped id");
    let period_ids: Vec<(String, String)> = fragment.periods.iter().map(|period| {
        let target = match period_map {
            Some(map) => lookup(map, &period.period_id),
            None if retained_periods.contains(&period.period_id) => period.period_id.clone(),
            None => available_id(&format!("{component_id}_{}", period.period_id), used_periods),
        };
        (period.period_id.clone(), target)
    }).collect();
    let mapped = |pairs: &[(String, String)], id: &str| pairs.iter().find(|(source, _)| source == id).map(|(_, target)| target.clone());
    let periods = fragment.periods.iter().map(|period| {
        let id = mapped(&period_ids, &period.period_id).expect("mapped");
        let mut copy = period.clone();
        if id != period.period_id && copy.name.is_empty() { copy.name = period.period_id.clone(); }
        copy.period_id = id;
        copy
    }).collect();
    let bracket_ids: Vec<(String, String)> = fragment.brackets.iter().map(|bracket| {
        let target = match bracket_map {
            Some(map) => lookup(map, &bracket.bracket_id),
            None if retained_brackets.contains(&bracket.bracket_id) => bracket.bracket_id.clone(),
            None => available_id(&format!("{component_id}_{}", bracket.bracket_id), used_brackets),
        };
        (bracket.bracket_id.clone(), target)
    }).collect();
    let bindings = fragment.bindings.iter().map(|binding| {
        let mut copy = binding.clone();
        copy.field_ref = FieldRef {
            period_id: binding.field_ref.period_id.as_deref().and_then(|id| mapped(&period_ids, id)),
            bracket_id: binding.field_ref.bracket_id.as_deref().and_then(|id| mapped(&bracket_ids, id)),
            ..binding.field_ref.clone()
        };
        copy
    }).collect();
    let brackets = fragment.brackets.iter().map(|bracket| Bracket {
        bracket_id: mapped(&bracket_ids, &bracket.bracket_id).expect("mapped"),
        start_period_id: bracket.start_period_id.as_deref().and_then(|id| mapped(&period_ids, id)).or_else(|| next.map(str::to_owned)),
        end_period_id: bracket.end_period_id.as_deref().and_then(|id| mapped(&period_ids, id)).or_else(|| previous.map(str::to_owned)),
        count: bracket.count,
    }).collect();
    (periods, bindings, brackets)
}

/// `insert_subpulse`: an independent embedded instance; no file is followed at run time.
pub fn insert_subpulse(sequence: Option<&Sequence>, subpulse: &Subpulse, before_period_id: Option<&str>, name: Option<&str>) -> Result<Sequence> {
    let fragment = subpulse.to_sequence();
    if let Some(sequence) = sequence {
        if sequence.target.abi_fingerprint() != fragment.target.abi_fingerprint() { return value_error("subpulse target ABI differs from the Pulse"); }
        if sequence.time_step_ns != fragment.time_step_ns { return value_error("subpulse time step differs from the Pulse"); }
    }
    let ids: Vec<String> = sequence.map_or_else(Vec::new, |sequence| sequence.periods.iter().map(|period| period.period_id.clone()).collect());
    if let Some(before) = before_period_id { if !ids.iter().any(|id| id == before) { return value_error(format!("no period exists with id {}", repr(before))); } }
    let position = before_period_id.map_or(ids.len(), |before| ids.iter().position(|id| id == before).expect("checked"));
    let components: Vec<Component> = sequence.map_or_else(Vec::new, |sequence| sequence.components.clone());
    if components.iter().any(|item| before_period_id.is_some_and(|before| item.period_ids[1..].iter().any(|id| id == before))) {
        return value_error("a component cannot be inserted inside another component");
    }
    let component_id = available_id("component1", &mut components.iter().map(|item| item.component_id.clone()).collect());
    let name = name.map_or_else(|| available_name(&fragment.name, &mut components.iter().map(|item| item.name.clone()).collect()), str::to_owned);
    let mut used_periods: BTreeSet<String> = ids.iter().cloned().collect();
    let mut used_brackets: BTreeSet<String> = sequence.map_or_else(BTreeSet::new, |sequence| sequence.brackets.iter().map(|b| b.bracket_id.clone()).collect());
    let (periods, bindings, brackets) = remap(subpulse, &component_id, &mut used_periods, &mut used_brackets, &[], &[],
                                              if position > 0 { Some(ids[position - 1].as_str()) } else { None },
                                              ids.get(position).map(String::as_str), None, None);
    let component = Component::new(&component_id, &name, &periods.iter().map(|p| p.period_id.clone()).collect::<Vec<_>>(),
                                   &brackets.iter().map(|b| b.bracket_id.clone()).collect::<Vec<_>>(), "default", "",
                                   &fragment.periods.iter().zip(&periods).map(|(a, b)| vec![a.period_id.clone(), b.period_id.clone()]).collect::<Vec<_>>(),
                                   &fragment.brackets.iter().zip(&brackets).map(|(a, b)| vec![a.bracket_id.clone(), b.bracket_id.clone()]).collect::<Vec<_>>())?;
    let Some(sequence) = sequence else {
        return Sequence::new(SequenceParts { name: fragment.name.clone(), time_step_ns: fragment.time_step_ns, bindings, brackets,
                                             components: vec![component], ..SequenceParts::new(fragment.target.clone(), periods) });
    };
    let first = periods[0].period_id.clone();
    let existing: Vec<Bracket> = sequence.brackets.iter().zip(sequence.bracket_bounds()).map(|(bracket, bounds)| {
        let mut bracket = bracket.clone();
        if *bounds == (position, position) { bracket.start_period_id = Some(first.clone()); }
        bracket
    }).collect();
    sequence.with(|parts| {
        let tail = parts.periods.split_off(position);
        parts.periods.extend(periods);
        parts.periods.extend(tail);
        parts.bindings.extend(bindings);
        parts.brackets = existing.into_iter().chain(brackets).collect();
        parts.components.push(component);
    })
}

fn parent_brackets_after_splice(sequence: &Sequence, component: &Component, periods: &[Period]) -> Vec<Bracket> {
    let original: Vec<&str> = sequence.period_ids();
    let start = original.iter().position(|id| *id == component.period_ids[0]).expect("member");
    let end = start + component.period_ids.len();
    let internal = internal_brackets(sequence, component);
    let ids: Vec<String> = original[..start].iter().map(|id| id.to_string()).chain(periods.iter().map(|p| p.period_id.clone()))
        .chain(original[end..].iter().map(|id| id.to_string())).collect();
    let delta = periods.len() as i64 - component.period_ids.len() as i64;
    sequence.brackets.iter().zip(sequence.bracket_bounds()).filter(|(bracket, _)| !internal.contains(bracket)).map(|(bracket, &(first, last))| {
        let shift = |gap: usize| if gap >= end { (gap as i64 + delta) as usize } else { gap };
        let (first, last) = (shift(first), shift(last));
        Bracket { start_period_id: ids.get(first).cloned(), end_period_id: if last > 0 { ids.get(last - 1).cloned() } else { None }, ..bracket.clone() }
    }).collect()
}

/// `replace_component`: adopt edited contents, keeping member and parent-loop ids.
pub fn replace_component(sequence: &Sequence, component_id: &str, subpulse: &Subpulse, period_map: Option<&[(String, String)]>,
                         bracket_map: Option<&[(String, String)]>) -> Result<Sequence> {
    let component = component(sequence, component_id)?.clone();
    let fragment = subpulse.to_sequence();
    if sequence.target.abi_fingerprint() != fragment.target.abi_fingerprint() { return value_error("subpulse target ABI differs from the Pulse"); }
    if sequence.time_step_ns != fragment.time_step_ns { return value_error("subpulse time step differs from the Pulse"); }
    let internal = internal_brackets(sequence, &component);
    let ids = sequence.period_ids();
    let start = ids.iter().position(|id| *id == component.period_ids[0]).expect("member");
    let end = start + component.period_ids.len();
    let (periods, bindings, brackets) = remap(subpulse, component_id,
        &mut sequence.periods.iter().map(|p| p.period_id.clone()).collect(), &mut sequence.brackets.iter().map(|b| b.bracket_id.clone()).collect(),
        &component.period_ids, &internal.iter().map(|b| b.bracket_id.clone()).collect::<Vec<_>>(),
        if start > 0 { Some(ids[start - 1]) } else { None }, ids.get(end).copied(), period_map, bracket_map);
    let mut fresh: Vec<Binding> = bindings;
    let mut merged = Vec::new();
    for binding in &sequence.bindings {
        let inside = binding.field_ref.period_id.as_ref().is_some_and(|id| component.period_ids.contains(id))
            || binding.field_ref.bracket_id.as_ref().is_some_and(|id| component.bracket_ids.contains(id));
        if !inside { merged.push(binding.clone()); }
        else if let Some(at) = fresh.iter().position(|item| item.field_ref == binding.field_ref) { merged.push(fresh.remove(at)); }
    }
    merged.extend(fresh);
    let parent = parent_brackets_after_splice(sequence, &component, &periods);
    let period_ids: Vec<String> = periods.iter().map(|p| p.period_id.clone()).collect();
    let bracket_ids: Vec<String> = brackets.iter().map(|b| b.bracket_id.clone()).collect();
    sequence.with(|parts| {
        let tail = parts.periods.split_off(end);
        parts.periods.truncate(start);
        parts.periods.extend(periods);
        parts.periods.extend(tail);
        parts.bindings = merged;
        parts.brackets = parent.into_iter().chain(brackets).collect();
        for item in &mut parts.components {
            if *item == component {
                item.period_id_map.retain(|(_, target)| period_ids.contains(target));
                item.bracket_id_map.retain(|(_, target)| bracket_ids.contains(target));
                item.period_ids = period_ids.clone();
                item.bracket_ids = bracket_ids.clone();
            }
        }
    })
}

/// `set_subpulse_source`: choose an instance's source; `path=None` keeps the file.
pub fn set_subpulse_source(sequence: &Sequence, component_id: &str, source: &str, path: Option<&str>) -> Result<Sequence> {
    let current = component(sequence, component_id)?.clone();
    let selected = Component::new(&current.component_id, &current.name, &current.period_ids, &current.bracket_ids, source,
                                  path.unwrap_or(&current.config_path),
                                  &current.period_id_map.iter().map(|(a, b)| vec![a.clone(), b.clone()]).collect::<Vec<_>>(),
                                  &current.bracket_id_map.iter().map(|(a, b)| vec![a.clone(), b.clone()]).collect::<Vec<_>>())?;
    sequence.with(|parts| for item in &mut parts.components { if *item == current { *item = selected.clone(); } })
}

/// `remove_component`: delete this instance and its contents, keeping any enclosing bracket.
pub fn remove_component(sequence: &Sequence, component_id: &str) -> Result<Sequence> {
    let component = component(sequence, component_id)?.clone();
    if sequence.periods.iter().all(|period| component.period_ids.contains(&period.period_id)) {
        return value_error("removing the only component leaves no periods; use Clear for the whole Pulse");
    }
    let brackets = parent_brackets_after_splice(sequence, &component, &[]);
    sequence.with(|parts| {
        parts.periods.retain(|period| !component.period_ids.contains(&period.period_id));
        parts.bindings.retain(|binding| !(binding.field_ref.period_id.as_ref().is_some_and(|id| component.period_ids.contains(id))
            || binding.field_ref.bracket_id.as_ref().is_some_and(|id| component.bracket_ids.contains(id))));
        parts.brackets = brackets;
        parts.components.retain(|item| *item != component);
    })
}
