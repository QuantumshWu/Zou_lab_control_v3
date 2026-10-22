//! zlc-pulse-editor against v2's pulse editor, on the 595 decodable golden
//! pulses (research/pulse-golden/make_editor_golden.py). Env-gated:
//! `ZLC_PULSE_EDITOR_GOLDEN` = that script's output, `ZLC_PULSE_GOLDEN` =
//! make_golden.py's (the timelines and the case indices).
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use serde_json::{Value, json};
use zlc_plot::pulse::PulseTimelineData;
use zlc_pulse::binding::apply_config_values;
use zlc_pulse::codec::{delay_to_tree, period_to_tree, sequence_from_tree, sequence_to_tree, target_to_tree};
use zlc_pulse::model::{AnalogMode, AnalogStep, PortKind, Sequence};
use zlc_pulse_editor::edits::edited;
use zlc_pulse_editor::items::sequence_item_order;
use zlc_pulse_editor::project::{Cards, Context, ScheduleRequest, project_board, project_schedule};
use zlc_pulse_editor::target::{TargetRecord, carried_onto, project_target, target_from_records};
use zlc_pulse_editor::timeline::{preview_candidate, timeline_of};
use zlc_pulse_editor::{from_value, to_value};

fn outcome<T>(result: zlc_pulse::Result<T>, ok: impl FnOnce(T) -> Value) -> Value {
    match result { Ok(value) => json!({"ok": ok(value)}), Err(error) => json!({"error": [error.kind.python_name(), error.message]}) }
}

/// The first place two trees differ, for a readable failure.
fn first_difference(path: &str, ours: &Value, theirs: &Value) -> Option<String> {
    match (ours, theirs) {
        (Value::Object(a), Value::Object(b)) => {
            let keys: BTreeSet<&String> = a.keys().chain(b.keys()).collect();
            keys.into_iter().find_map(|key| match (a.get(key), b.get(key)) {
                (Some(x), Some(y)) => first_difference(&format!("{path}.{key}"), x, y),
                (x, y) => Some(format!("{path}.{key}: ours {x:?} v2 {y:?}")),
            })
        }
        (Value::Array(a), Value::Array(b)) if a.len() == b.len() => {
            a.iter().zip(b).enumerate().find_map(|(index, (x, y))| first_difference(&format!("{path}[{index}]"), x, y))
        }
        _ if ours == theirs => None,
        _ => Some(format!("{path}: ours {ours} v2 {theirs}")),
    }
}

#[derive(Default)]
struct Tally { checked: usize, differing: usize, shown: Vec<String> }

#[derive(Default)]
struct Tallies(BTreeMap<&'static str, Tally>);

impl Tallies {
    fn check(&mut self, stage: &'static str, case: &str, ours: &Value, theirs: &Value) {
        let tally = self.0.entry(stage).or_default();
        tally.checked += 1;
        if let Some(difference) = first_difference("", ours, theirs) {
            tally.differing += 1;
            if tally.shown.len() < 10 { tally.shown.push(format!("{case}: {difference}")); }
        }
    }
}

fn sequence_of(tree: &Value) -> Sequence { sequence_from_tree(&from_value(tree).unwrap()).unwrap() }

fn strings(value: &Value) -> BTreeSet<String> { value.as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_owned()).collect() }

fn order_of(sequence: &Sequence) -> Value { Value::Array(sequence_item_order(sequence).into_iter().map(|(k, v)| json!([k, v])).collect()) }

#[test]
fn the_editor_layer_equals_v2() {
    let (Some(editor), Some(compiler)) = (std::env::var_os("ZLC_PULSE_EDITOR_GOLDEN"), std::env::var_os("ZLC_PULSE_GOLDEN")) else { return };
    let mut tallies = Tallies::default();
    for group in ["examples", "diff", "synthetic", "fuzz_small", "fuzz_deployed"] {
        let read = |dir: &std::ffi::OsStr| -> Value {
            serde_json::from_str(&std::fs::read_to_string(PathBuf::from(dir).join(format!("{group}.json"))).unwrap()).unwrap()
        };
        let compiled = read(&compiler);
        let index_of: BTreeMap<&str, (usize, &Value)> = compiled["cases"].as_array().unwrap().iter().enumerate()
            .map(|(index, case)| (case["name"].as_str().unwrap(), (index, case))).collect();
        let golden = read(&editor);
        for case in golden["cases"].as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            let (index, compiled_case) = index_of[name];
            let sequence = sequence_of(&case["tree"]);
            tallies.check("order", name, &order_of(&sequence), &case["order"]);

            for (off, key) in [(false, "timeline"), (true, "timeline_off")] {
                let theirs = match compiled_case[key].get("ok") {
                    Some(tree) => json!({"ok": PulseTimelineData::from_json(tree).unwrap().to_json()}),
                    None => compiled_case[key].clone(),
                };
                tallies.check("timeline", name, &outcome(timeline_of(&sequence, off), |data| data.to_json()), &theirs);
                let preview = outcome(preview_candidate(&sequence, off, None), |p| json!([p.size, p.rows, p.periods, p.total_ns]));
                tallies.check("preview", name, &preview, &case["preview"][if off { "True" } else { "False" }]);
            }

            let schedule = &case["schedule"];
            tallies.check("schedule", name, &json!({"ok": project_schedule(&sequence, &ScheduleRequest::default())}), &schedule["plain"]);
            tallies.check("schedule_board", name, &json!({"ok": project_board(&sequence.target, sequence.time_step_ns, 3, None)}), &schedule["board"]);
            let config: Vec<(String, (f64, String))> = case.get("config_values").and_then(Value::as_object).map(|values| values.iter()
                .map(|(key, pair)| (key.clone(), (pair[0].as_f64().unwrap(), pair[1].as_str().unwrap().to_owned()))).collect()).unwrap_or_default();
            let visible = strings(&case["visible"]);
            let scanned = strings(&case["scanned"]);
            let cards = std::cell::RefCell::new(Cards::default());
            let configured = (|| -> zlc_pulse::Result<Value> {
                let enabled = sequence.with(|parts| parts.config_enabled = true)?;
                let effective = if config.is_empty() { enabled.clone() } else { apply_config_values(&enabled, &config, None)?.0 };
                let request = ScheduleRequest { path: "C:/pulses/case.json", generation: 2, revision: 7, pins: None, scan_points: 3,
                    config_error: if index % 5 == 0 { "bad file" } else { "" },
                    context: Context { visible_ports: Some(&visible), config: &config, scanned: Some(&scanned), authored: Some(&enabled) },
                    cards: None };
                let made = project_schedule(&effective, &request);
                // Every port shown, its cards sent and back, one period edited (a DAC's level, which each later card
                // holding it shows, else a digital state): the next schedule makes the cards the edit changes and
                // takes back the rest -- the schedule a projection from nothing makes.
                let every = ScheduleRequest { context: Context { visible_ports: None, ..request.context }, cards: Some(&cards), ..request.clone() };
                if let Value::Array(sent) = project_schedule(&effective, &every)["periods"].take() { cards.borrow_mut().recycle(sent); }
                let dac = effective.target.ports.iter().find(|port| port.kind == PortKind::Dac).map(|port| port.key.clone());
                let edited = effective.with(|parts| {
                    let Some(period) = parts.periods.iter_mut().find(|period| !period.is_spacer()) else { return };
                    match dac.as_ref().map(|dac| (period.analog_steps.iter().position(|step| step.port == *dac), dac)) {
                        Some((Some(at), _)) => period.analog_steps[at].value = i64::from(period.analog_steps[at].value == 0),
                        Some((None, dac)) => period.analog_steps.push(AnalogStep { port: dac.clone(), mode: AnalogMode::Edge, value: 1 }),
                        None => if let Some(state) = period.states.first_mut() { *state ^= 1 },
                    }
                });
                if let Ok(edited) = edited {
                    let fresh = project_schedule(&edited, &ScheduleRequest { cards: None, ..every.clone() });
                    tallies.check("schedule_cards_taken_back", name, &project_schedule(&edited, &every), &fresh);
                }
                Ok(made)
            })();
            tallies.check("schedule_configured", name, &outcome(configured, |v| v), &schedule["configured"]);

            let records = project_target(&sequence.target, None);
            tallies.check("target_records", name, &Value::Array(records.iter().map(TargetRecord::to_json).collect()), &case["target"]["records"]);
            let same = outcome(target_from_records(&sequence.target, &records), |t| to_value(&target_to_tree(&t)));
            tallies.check("target_same", name, &same, &case["target"]["same"]);
            let edited_records: Vec<TargetRecord> = case["target"]["edited_records"].as_array().unwrap().iter()
                .map(|r| TargetRecord::from_json(r).unwrap()).collect();
            let moved = (|| -> zlc_pulse::Result<Value> {
                let target = target_from_records(&sequence.target, &edited_records)?;
                let (periods, delays) = carried_onto(&sequence, &target)?;
                Ok(json!({"target": to_value(&target_to_tree(&target)),
                          "periods": periods.iter().map(|p| to_value(&period_to_tree(p))).collect::<Vec<_>>(),
                          "delays": delays.iter().map(|d| to_value(&delay_to_tree(d))).collect::<Vec<_>>()}))
            })();
            tallies.check("target_edited", name, &outcome(moved, |v| v), &case["target"]["edited"]);

            let mut current = sequence.clone();
            for (step_index, step) in case["edits"].as_array().unwrap().iter().enumerate() {
                let action = step["action"].as_str().unwrap();
                let label = format!("{name} step {step_index} {action} {}", step["args"]);
                let theirs = json!({"ok": step.get("ok"), "dropped": step.get("dropped"), "order": step.get("order"), "error": step.get("error")});
                let ours = match edited(&current, action, step["args"].as_array().unwrap()) {
                    Ok((candidate, dropped)) => {
                        let ours = json!({"ok": to_value(&sequence_to_tree(&candidate)), "dropped": dropped, "order": order_of(&candidate), "error": null});
                        current = candidate;
                        ours
                    }
                    Err(error) => {
                        if step.get("ok").is_some() { current = sequence_of(&step["ok"]); }
                        json!({"ok": null, "dropped": null, "order": null, "error": [error.kind.python_name(), error.message]})
                    }
                };
                tallies.check("edits", &label, &ours, &theirs);
            }
        }
    }
    let mut report = String::new();
    for (stage, tally) in &tallies.0 {
        report.push_str(&format!("{stage}: {} checked, {} differ\n", tally.checked, tally.differing));
        for shown in &tally.shown { report.push_str(&format!("    {shown}\n")); }
    }
    println!("{report}");
    assert!(tallies.0.values().all(|tally| tally.differing == 0), "{report}");
}
