//! The application's state in the flight journal (`zlc_core::journal`): one more reader of the state
//! messages the windows are sent, told what every page is told -- the change since the last message
//! (delta.rs), and with it the request a message answers, its result and its error.
//!
//! So whatever a window can show is in the journal without anyone deciding to record it, and a command's
//! reply is the state line that carries its `request`. A segment of the journal opens with the whole
//! state (`state_snapshot`), from which the changes that follow are applied as a page applies them:
//! `[path, value]` sets, `[path]` deletes a key, `[path, items, start, removed]` splices an array.
use std::sync::Mutex;

use serde_json::{Map, Value};

use crate::delta::Sent;

static TOLD: Mutex<Option<Sent>> = Mutex::new(None);

/// Begin: every new segment of the journal opens with the state as it stands.
pub fn open() {
    *TOLD.lock().unwrap_or_else(|error| error.into_inner()) = Some(Sent::default());
    zlc_core::journal::at_segment_start(|say| {
        // Held while the snapshot is said, so no change is stamped between it and the state it is of.
        let told = TOLD.lock().unwrap_or_else(|error| error.into_inner());
        if let Some(sent) = told.as_ref().filter(|sent| !sent.sections().is_empty()) {
            let mut fields = Map::new();
            fields.insert("state".into(), Value::Object(sent.sections().clone()));
            say("app", "state_snapshot", &Value::Object(fields).to_string());
        }
    });
}

/// A state message as it goes to the windows: its changes, and the reply it carries, are one line.
pub fn say(state: &Value) {
    if !zlc_core::journal::is_open() { return; }
    let Some(object) = state.as_object().filter(|object| object.get("type").and_then(Value::as_str) == Some("state")) else { return };
    let mut told = TOLD.lock().unwrap_or_else(|error| error.into_inner());
    let Some(sent) = told.as_mut() else { return };
    if let Some(text) = line(sent, object) { zlc_core::journal::note_fields("app", "state", &text); }
}

/// The journal's line for a state message, or `None` when it says nothing new: every section's
/// changes, and every reply that says something -- a request answered with a result (a read-only
/// command's reply is all the journal has of it) or an error. A bare "done" of a request that
/// changed nothing (a hover answered) says nothing the journal does not have; a message that
/// answers no request (0 is none) and changed nothing is not a line at all.
fn line(sent: &mut Sent, object: &Map<String, Value>) -> Option<String> {
    let says = object.get("result").is_some_and(|result| !result.is_null()) || object.get("error").is_some_and(|error| !error.is_null());
    sent.encode(object, |_| true, |_| says)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_reader_arrives_at_the_state_by_applying_what_was_said() {
        let mut sent = Sent::default();
        let first = json!({"type": "state", "logic": [{"node": "camera", "status": "idle"}], "report": {"text": ""}});
        let text = line(&mut sent, first.as_object().unwrap()).unwrap();
        let said: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(said["changes"].as_array().unwrap().len(), 2, "a section first said is said whole");
        // One field changes, and the message answers request 7 with a result.
        let second = json!({"type": "state", "request": 7, "result": {"node": "camera"}, "logic": [{"node": "camera", "status": "running"}], "report": {"text": ""}});
        let text = line(&mut sent, second.as_object().unwrap()).unwrap();
        let said: Value = serde_json::from_str(&text).unwrap();
        assert_eq!((&said["request"], &said["result"]["node"]), (&json!(7), &json!("camera")), "the reply rides with the change");
        assert_eq!(said["changes"], json!([[["logic", 0, "status"], "running"]]), "only what changed, by its path");
        assert_eq!(sent.sections()["logic"][0]["status"], "running", "what a snapshot would say now");
        // Nothing changed and nothing is answered: nothing is said.
        let third = json!({"type": "state", "logic": [{"node": "camera", "status": "running"}], "report": {"text": ""}});
        assert!(line(&mut sent, third.as_object().unwrap()).is_none());
        // Request 0 is no request: a broadcast that changed nothing is no line.
        let broadcast = json!({"type": "state", "request": 0, "error": null, "logic": [{"node": "camera", "status": "running"}]});
        assert!(line(&mut sent, broadcast.as_object().unwrap()).is_none());
        // A request answered with nothing (a hover) is no line; one answered with a result, or refused, is.
        let done = json!({"type": "state", "request": 8, "result": null, "logic": [{"node": "camera", "status": "running"}]});
        assert!(line(&mut sent, done.as_object().unwrap()).is_none());
        let read = json!({"type": "state", "request": 9, "result": {"windows": []}, "logic": [{"node": "camera", "status": "running"}]});
        assert_eq!(serde_json::from_str::<Value>(&line(&mut sent, read.as_object().unwrap()).unwrap()).unwrap()["result"], json!({"windows": []}));
        let refused = json!({"type": "state", "request": 10, "error": "no", "logic": [{"node": "camera", "status": "running"}]});
        assert!(line(&mut sent, refused.as_object().unwrap()).is_some());
        // An error of no request is said once, with its message.
        let error = json!({"type": "state", "error": "worker stopped", "logic": [{"node": "camera", "status": "running"}]});
        assert!(line(&mut sent, error.as_object().unwrap()).is_some());
        assert!(line(&mut sent, third.as_object().unwrap()).is_none(), "and not again with the next message");
    }
}
