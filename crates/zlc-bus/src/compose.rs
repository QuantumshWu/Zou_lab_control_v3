//! Catalogue commands the MCP endpoint answers itself, from other commands or
//! its own facts: `guide`, `health`, `wait_for`, `batch`. The instrument never
//! receives them; each inner command goes through the same validation and
//! control check as a client's own call.

use std::sync::mpsc;
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::{Value, json};

use crate::catalog::{self, Batch, GuideTopic, PathStep, WaitFor};
use crate::mcp::{Caller, ToolOutput};

/// The operator guide: `## <topic>` sections.
const GUIDE: &str = include_str!("guide.md");

fn sections() -> impl Iterator<Item = (&'static str, &'static str)> {
    GUIDE.split("\n## ").skip(1).map(|section| {
        let (name, body) = section.split_once('\n').unwrap_or((section, ""));
        (name.trim(), body.trim())
    })
}

pub fn run(caller: &Caller, name: &str, arguments: Value) -> Result<ToolOutput, String> {
    let json = match name {
        "guide" => guide(caller, parse(arguments)?)?,
        "health" => health(caller),
        "wait_for" => wait_for(caller, parse(arguments)?)?,
        "batch" => batch(caller, parse(arguments)?)?,
        _ => return Err(format!("{name} is not composed by the endpoint")),
    };
    Ok(ToolOutput { json, png: None })
}

fn parse<T: for<'de> Deserialize<'de>>(arguments: Value) -> Result<T, String> {
    T::deserialize(arguments).map_err(|error| error.to_string())
}

fn guide(caller: &Caller, request: GuideTopic) -> Result<Value, String> {
    let topic = request.topic.unwrap_or_else(|| "overview".into());
    let topics: Vec<&str> = sections().map(|(name, _)| name).collect();
    // A plugin type's own words are its descriptor (v2 `node:<api_name>`).
    if let Some(type_id) = topic.strip_prefix("type:").or_else(|| topic.strip_prefix("node:")) {
        let descriptor = caller.instrument("list_plugins", json!({"type_id": type_id}))?.json;
        return Ok(json!({"topic": topic, "descriptor": descriptor, "topics": topics}));
    }
    let (_, content) = sections().find(|(name, _)| *name == topic)
        .ok_or_else(|| format!("No guide topic {topic:?}; topics: {}, or type:<type_id>", topics.join(", ")))?;
    // A checkout may carry CRLF; the text is the same.
    Ok(json!({"topic": topic, "content": content.replace('\r', ""), "topics": topics}))
}

fn health(caller: &Caller) -> Value {
    let mut facts = json!({"service": "running", "pid": std::process::id(), "control_enabled": caller.control,
        "calls_in_flight": caller.calls_in_flight(), "seconds_since_instrument_answer": caller.seconds_since_answer()});
    if let (Some(facts), Value::Object(identity)) = (facts.as_object_mut(), caller.host.identity()) { facts.extend(identity); }
    facts
}

/// Whether a command only reads (a catalogue read, or a device command declared read-only).
fn reads(caller: &Caller, name: &str) -> bool {
    match catalog::find(name) {
        Some(spec) => spec.read_only,
        None => caller.host.tools().iter().any(|tool| tool["name"] == name && tool["annotations"]["readOnlyHint"] == true),
    }
}

fn field<'a>(value: &'a Value, path: &[PathStep]) -> Option<&'a Value> {
    path.iter().try_fold(value, |value, step| match step {
        PathStep::Key(key) => value.get(key.as_str()),
        PathStep::Index(index) => value.get(*index as usize),
    })
}

fn wait_for(caller: &Caller, request: WaitFor) -> Result<Value, String> {
    let timeout = request.timeout_seconds.unwrap_or(30.0);
    if !(0.0..=1800.0).contains(&timeout) { return Err("timeout_seconds must be between 0 and 1800".into()); }
    if matches!(request.tool.as_str(), "wait" | "wait_for" | "batch") || !reads(caller, &request.tool) {
        return Err(format!("wait_for re-reads a command that only reads; {} is not one (use wait for runs and frames)", request.tool));
    }
    let arguments = Value::Object(request.arguments.unwrap_or_default());
    let deadline = Instant::now() + Duration::from_secs_f64(timeout);
    // Each read is a whole projection on the instrument's turn: quick at first, then backing off.
    let mut delay = Duration::from_millis(20);
    loop {
        let result = caller.dispatch(&request.tool, arguments.clone())?.json;
        let value = field(&result, &request.path)
            .ok_or_else(|| format!("{} has no field {} in its reply", request.tool, serde_json::to_string(&path_json(&request.path)).unwrap_or_default()))?;
        if *value == request.equals { return Ok(json!({"matched": true, "timed_out": false, "result": result})); }
        let now = Instant::now();
        if now >= deadline { return Ok(json!({"matched": false, "timed_out": true, "result": result})); }
        std::thread::sleep(delay.min(deadline - now));
        delay = (delay * 3 / 2).min(Duration::from_millis(250));
    }
}

fn path_json(path: &[PathStep]) -> Value {
    Value::Array(path.iter().map(|step| match step { PathStep::Key(key) => json!(key), PathStep::Index(index) => json!(index) }).collect())
}

/// Whether one reply reports the step done: not a refusal (those are errors), not an unmet wait, not a
/// run that stopped or failed while it was waited on.
pub fn succeeded(result: &Value) -> bool {
    let Some(object) = result.as_object() else { return *result != Value::Bool(false) };
    if ["accepted", "matched", "reached"].iter().any(|key| object.get(*key) == Some(&Value::Bool(false))) { return false; }
    if object.get("timed_out") == Some(&Value::Bool(true)) { return false; }
    !(object.contains_key("timed_out") && matches!(object.get("run").and_then(|run| run["state"].as_str()), Some("stopped" | "failed")))
}

/// `{"$result": [i, key, index, ...]}` is that field of an earlier completed call's reply.
fn bind(value: &Value, completed: &[Value]) -> Result<Value, String> {
    match value {
        Value::Object(object) if object.len() == 1 && object.contains_key("$result") => {
            let path = object["$result"].as_array().filter(|path| !path.is_empty())
                .ok_or("$result is [call index, then the keys and indices of its reply]")?;
            let index = path[0].as_u64().filter(|index| (*index as usize) < completed.len())
                .ok_or("$result must name an earlier completed call")?;
            path[1..].iter().try_fold(&completed[index as usize], |value, step| match step {
                Value::String(key) => value.get(key.as_str()),
                Value::Number(number) => number.as_u64().and_then(|index| value.get(index as usize)),
                _ => None,
            }).cloned().ok_or_else(|| format!("Call {index}'s reply has no field {}; read it, do not replay the writes", Value::Array(path[1..].to_vec())))
        }
        Value::Object(object) => object.iter().map(|(key, item)| Ok((key.clone(), bind(item, completed)?))).collect::<Result<_, String>>().map(Value::Object),
        Value::Array(items) => items.iter().map(|item| bind(item, completed)).collect::<Result<_, _>>().map(Value::Array),
        other => Ok(other.clone()),
    }
}

const BATCH_SECONDS: f64 = 30.0;

fn batch(caller: &Caller, request: Batch) -> Result<Value, String> {
    if !(1..=16).contains(&request.calls.len()) { return Err("batch takes 1 to 16 calls".into()); }
    for call in &request.calls {
        if call.tool == "batch" { return Err("A batch cannot contain a batch".into()); }
        if catalog::find(&call.tool).is_none() && !caller.host.tools().iter().any(|tool| tool["name"] == call.tool.as_str()) {
            return Err(format!("unknown tool {:?}", call.tool));
        }
    }
    let deadline = Instant::now() + Duration::from_secs_f64(BATCH_SECONDS);
    let (mut results, mut replies) = (Vec::new(), Vec::new());
    for (index, call) in request.calls.iter().enumerate() {
        let outcome = bind(&Value::Object(call.arguments.clone().unwrap_or_default()), &replies).and_then(|arguments| {
            // On its own thread: an expired deadline answers the batch while the call finishes.
            let (sender, answer) = mpsc::channel();
            let (inner, tool) = (caller.clone(), call.tool.clone());
            std::thread::Builder::new().name("zlc-mcp-batch".into())
                .spawn(move || { let _ = sender.send(inner.dispatch(&tool, arguments)); }).map_err(|error| error.to_string())?;
            match answer.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
                Ok(reply) => reply.map(|output| output.json),
                Err(_) => Err("The batch's 30 s ran out; this call may have started. Read the state before any retry.".into()),
            }
        });
        let completed = match outcome {
            Ok(result) => {
                let mut ok = succeeded(&result);
                if let Some(expect) = &call.expect {
                    ok &= expect.iter().all(|(key, value)| result.get(key) == Some(value));
                }
                results.push(json!({"tool": call.tool, "result": result, "completed": ok}));
                replies.push(result);
                ok
            }
            Err(error) => { results.push(json!({"tool": call.tool, "error": error, "completed": false})); false }
        };
        if !completed {
            return Ok(json!({"completed": false, "stopped_at": index, "results": results,
                "note": "Earlier effects remain. Read this reply and the state; do not replay the whole batch."}));
        }
    }
    Ok(json!({"completed": true, "results": results}))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_guide_topic_the_catalogue_names_exists() {
        let topics: Vec<&str> = sections().map(|(name, _)| name).collect();
        let named = catalog::find("guide").unwrap().input_schema()["properties"]["topic"]["description"].as_str().unwrap().to_owned();
        for topic in &topics { assert!(named.contains(topic), "guide topic {topic} is not named in the guide command"); }
        assert_eq!(topics[0], "overview");
        assert!(sections().all(|(_, body)| !body.is_empty()));
    }

    #[test]
    fn a_reply_succeeds_unless_it_says_otherwise() {
        assert!(succeeded(&json!({"panels": []})));
        assert!(!succeeded(&json!({"matched": false, "timed_out": true})));
        assert!(!succeeded(&json!({"timed_out": false, "reached": false, "run": {"state": "running"}})));
        assert!(!succeeded(&json!({"timed_out": false, "run": {"state": "failed"}})));
        assert!(succeeded(&json!({"timed_out": false, "run": {"state": "completed"}})));
        assert!(succeeded(&json!({"run": {"state": "stopped"}, "node_id": "x"})));
    }

    #[test]
    fn a_result_reference_reads_an_earlier_reply() {
        let earlier = [json!({"run": {"run": "occupancy#1"}, "rows": [10, 20]})];
        assert_eq!(bind(&json!({"node": "x", "run": {"$result": [0, "run", "run"]}, "n": [{"$result": [0, "rows", 1]}]}), &earlier).unwrap(),
                   json!({"node": "x", "run": "occupancy#1", "n": [20]}));
        assert!(bind(&json!({"$result": [1, "run"]}), &earlier).is_err());
        assert!(bind(&json!({"$result": [0, "missing"]}), &earlier).unwrap_err().contains("do not replay"));
    }
}
