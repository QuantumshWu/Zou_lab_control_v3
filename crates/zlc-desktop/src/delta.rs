//! What one page was last told, and a state message as the change since.
//!
//! A page keeps the state sections it was sent and applies `changes` to them
//! (transport.ts): `[path, value]` sets, `[path]` deletes a key, `[path,
//! items, start, removed]` splices an array. A one-checkbox edit of a
//! 500-period pulse is then a few hundred bytes instead of the whole document
//! re-serialized, re-parsed and re-walked on every edit; the page also keeps
//! the identity of everything that did not change, so views skip it.
use serde_json::{Map, Value};

/// Keys of a state message that belong to the message, not to the document.
const ENVELOPE: [&str; 4] = ["type", "request", "error", "result"];

#[derive(Default)]
pub struct Sent { sections: Map<String, Value>, error: Value }

impl Sent {
    /// The page was (re)loaded: it holds nothing.
    pub fn reset(&mut self) { *self = Sent::default(); }

    /// The message for `state` on a page that reads the sections `reads`
    /// accepts, or `None` when it would tell the page nothing: no section it
    /// reads changed, the error is the one it already shows, and the request
    /// is not its own (`owns`).
    pub fn encode(&mut self, state: &Map<String, Value>, reads: impl Fn(&str) -> bool, owns: impl Fn(&Value) -> bool) -> Option<String> {
        let mut changes = Changes(String::new());
        let mut path = Vec::new();
        for (key, value) in state {
            if ENVELOPE.contains(&key.as_str()) || !reads(key) { continue; }
            match self.sections.get_mut(key) {
                Some(previous) => { path.push(Step::Key(key)); diff(&mut path, previous, value, &mut changes); path.pop(); }
                None => { changes.set(&path, Some(key), value); self.sections.insert(key.clone(), value.clone()); }
            }
        }
        let error = state.get("error").cloned().unwrap_or(Value::Null);
        let request = state.get("request").cloned().unwrap_or(Value::Null);
        if changes.0.is_empty() && error == self.error && !owns(&request) { return None; }
        self.error = error;
        let envelope: Map<String, Value> = ENVELOPE.iter().filter_map(|key| state.get(*key).map(|value| ((*key).to_owned(), value.clone()))).collect();
        let mut text = Value::Object(envelope).to_string();
        text.pop();
        if text.len() > 1 { text.push(','); }
        text.push_str("\"changes\":[");
        text.push_str(&changes.0);
        text.push_str("]}");
        Some(text)
    }
}

/// One step of a path into the state, written out only where a change is said.
#[derive(Clone, Copy)]
enum Step<'a> { Key(&'a str), Index(usize) }

/// The changes, written as they are found: the message copies no value.
struct Changes(String);

impl Changes {
    fn open(&mut self, path: &[Step], last: Option<&str>) {
        if !self.0.is_empty() { self.0.push(','); }
        self.0.push_str("[[");
        let steps = path.iter().copied().chain(last.map(Step::Key));
        for (at, step) in steps.enumerate() {
            if at > 0 { self.0.push(','); }
            match step { Step::Key(key) => self.0.push_str(&Value::from(key).to_string()), Step::Index(index) => self.0.push_str(&index.to_string()) }
        }
        self.0.push(']');
    }
    fn set(&mut self, path: &[Step], last: Option<&str>, value: &Value) {
        self.open(path, last);
        self.0.push(',');
        self.0.push_str(&value.to_string());
        self.0.push(']');
    }
    fn delete(&mut self, path: &[Step], key: &str) { self.open(path, Some(key)); self.0.push(']'); }
    fn splice(&mut self, path: &[Step], items: &[Value], start: usize, removed: usize) {
        self.open(path, None);
        self.0.push(',');
        self.0.push_str(&serde_json::to_string(items).expect("JSON values serialize"));
        self.0.push_str(&format!(",{start},{removed}]"));
    }
}

/// Bring `old` to `new`, recording each change at the deepest place it can be said.
fn diff<'a>(path: &mut Vec<Step<'a>>, old: &mut Value, new: &'a Value, changes: &mut Changes) {
    match (&mut *old, new) {
        // The same keys in the same order, as a record made again always has them: walked side by side.
        (Value::Object(before), Value::Object(after)) if before.len() == after.len() && before.keys().zip(after.keys()).all(|(a, b)| a == b) => {
            for ((_, previous), (key, value)) in before.iter_mut().zip(after) {
                path.push(Step::Key(key));
                diff(path, previous, value, changes);
                path.pop();
            }
        }
        (Value::Object(before), Value::Object(after)) => {
            let gone: Vec<String> = before.keys().filter(|key| !after.contains_key(*key)).cloned().collect();
            for key in gone {
                before.remove(&key);
                changes.delete(path, &key);
            }
            for (key, value) in after {
                match before.get_mut(key) {
                    Some(previous) => { path.push(Step::Key(key)); diff(path, previous, value, changes); path.pop(); }
                    None => { changes.set(path, Some(key), value); before.insert(key.clone(), value.clone()); }
                }
            }
        }
        (Value::Array(before), Value::Array(after)) if before.len() == after.len() => {
            for (index, (previous, value)) in before.iter_mut().zip(after).enumerate() {
                path.push(Step::Index(index));
                diff(path, previous, value, changes);
                path.pop();
            }
        }
        (Value::Array(before), Value::Array(after)) => {
            // An insertion or a removal: keep the equal ends, splice the middle.
            let head = before.iter().zip(after).take_while(|(a, b)| a == b).count();
            let room = before.len().min(after.len()) - head;
            let tail = before.iter().rev().zip(after.iter().rev()).take(room).take_while(|(a, b)| a == b).count();
            let items = &after[head..after.len() - tail];
            let removed = before.len() - tail - head;
            changes.splice(path, items, head, removed);
            before.splice(head..head + removed, items.iter().cloned());
        }
        (previous, value) => {
            if previous != value {
                changes.set(path, None, value);
                *previous = value.clone();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// The page's side (transport.ts `applyChanges`), for checking: applying the changes gives the new state.
    fn apply(doc: &mut Value, changes: &[Value]) {
        fn step<'a>(at: &'a mut Value, key: &Value) -> &'a mut Value {
            match key {
                Value::String(key) => at.as_object_mut().unwrap().entry(key.clone()).or_insert(Value::Null),
                index => &mut at.as_array_mut().unwrap()[index.as_u64().unwrap() as usize],
            }
        }
        for change in changes {
            let change = change.as_array().unwrap();
            let (last, parents) = change[0].as_array().unwrap().split_last().unwrap();
            let mut at = &mut *doc;
            for key in parents { at = step(at, key); }
            match change.len() {
                1 => { at.as_object_mut().unwrap().remove(last.as_str().unwrap()); }
                2 => *step(at, last) = change[1].clone(),
                _ => {
                    let (start, removed) = (change[2].as_u64().unwrap() as usize, change[3].as_u64().unwrap() as usize);
                    step(at, last).as_array_mut().unwrap().splice(start..start + removed, change[1].as_array().unwrap().iter().cloned());
                }
            }
        }
    }

    fn round(sent: &mut Sent, page: &mut Value, state: Value) -> Option<Value> {
        let text = sent.encode(state.as_object().unwrap(), |_| true, |request| request.as_u64().is_some_and(|id| id > 0))?;
        let message: Value = serde_json::from_str(&text).unwrap();
        apply(page, message["changes"].as_array().unwrap());
        for (key, value) in state.as_object().unwrap().iter().filter(|(key, _)| !ENVELOPE.contains(&key.as_str())) {
            assert_eq!(page.get(key), Some(value), "section {key}");
        }
        Some(message)
    }

    #[test]
    fn a_page_holds_what_it_was_sent() {
        let (mut sent, mut page) = (Sent::default(), json!({}));
        let first = round(&mut sent, &mut page, json!({"type": "state", "pulse": {"periods": [{"id": "a", "on": [1, 0]}, {"id": "b", "on": [0, 0]}], "dirty": false}})).unwrap();
        assert_eq!(first["changes"].as_array().unwrap().len(), 1);
        // One flag: one change, at the flag.
        let toggle = round(&mut sent, &mut page, json!({"type": "state", "request": 3, "pulse": {"periods": [{"id": "a", "on": [1, 1]}, {"id": "b", "on": [0, 0]}], "dirty": true}})).unwrap();
        let mut said: Vec<String> = toggle["changes"].as_array().unwrap().iter().map(Value::to_string).collect();
        said.sort();
        assert_eq!(said, [r#"[["pulse","dirty"],true]"#, r#"[["pulse","periods",0,"on",1],1]"#]);
        assert_eq!(toggle["request"], 3);
        // Insertion in the middle: one splice of the new element.
        let insert = round(&mut sent, &mut page, json!({"type": "state", "pulse": {"periods": [{"id": "a", "on": [1, 1]}, {"id": "n"}, {"id": "b", "on": [0, 0]}], "dirty": true}})).unwrap();
        assert_eq!(insert["changes"], json!([[["pulse", "periods"], [{"id": "n"}], 1, 0]]));
        // Removal at the end, a key gone, a kind change.
        round(&mut sent, &mut page, json!({"type": "state", "pulse": {"periods": [{"id": "a", "on": "x"}]}})).unwrap();
        // Nothing new, no request of its own: nothing is sent.
        assert!(round(&mut sent, &mut page, json!({"type": "state", "pulse": {"periods": [{"id": "a", "on": "x"}]}})).is_none());
        // Its own request is answered even when nothing changed.
        assert_eq!(round(&mut sent, &mut page, json!({"type": "state", "request": 9, "pulse": {"periods": [{"id": "a", "on": "x"}]}})).unwrap()["changes"], json!([]));
        // A new error is told; a second section arrives whole.
        let error = round(&mut sent, &mut page, json!({"type": "state", "error": "no", "board": [1], "pulse": {"periods": [{"id": "a", "on": "x"}]}})).unwrap();
        assert_eq!(error["changes"], json!([[["board"], [1]]]));
    }
}
