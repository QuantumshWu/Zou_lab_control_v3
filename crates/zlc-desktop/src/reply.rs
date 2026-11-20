//! A reply belongs to the one message that carries it.
//!
//! A state message is the application's state (sections every page reads, kept from turn to
//! turn) and, beside it, the reply that message is: the `request` it answers, with its `result`
//! or its `error`. The state is long-lived; a reply written into it would ride on every later
//! message too, and each page, the journal and the remote clients would be told the same answer
//! again with each of them. So a reply is put on the message as it goes out and taken off again
//! (`Reply::send`): the state never holds one. Request 0, or none, answers nothing; an error of no
//! request is said to every page once, with the message that carries it.
use serde_json::{Map, Value};

/// The keys of a state message that are its reply, not its state.
pub const KEYS: [&str; 3] = ["request", "result", "error"];

/// What one message answers: nothing (`Reply::default()`), or a request's result or error.
#[derive(Default, Debug, Clone, PartialEq)]
pub struct Reply(Map<String, Value>);

impl Reply {
    /// `request` answered with `outcome`. A request of 0 (or none) is no request: only an error is then said.
    pub fn to(request: &Value, outcome: Result<Value, String>) -> Self {
        let mut reply = Map::new();
        if request.as_u64().is_some_and(|request| request > 0) { reply.insert("request".into(), request.clone()); }
        match outcome {
            Ok(result) => { if reply.contains_key("request") { reply.insert("result".into(), result); } }
            Err(error) => { reply.insert("error".into(), Value::String(error)); }
        }
        Self(reply)
    }

    /// `request` answered with nothing to say beyond that it is done.
    pub fn done(request: &Value) -> Self { Self::to(request, Ok(Value::Null)) }

    /// The reply a state made elsewhere carries (the scientific worker's answer), taken out of it.
    pub fn take(state: &mut Value) -> Self {
        let Some(object) = state.as_object_mut() else { return Self::default() };
        let request = object.shift_remove("request").unwrap_or(Value::Null);
        let result = object.shift_remove("result").unwrap_or(Value::Null);
        let outcome = match object.shift_remove("error") { Some(Value::String(error)) => Err(error), _ => Ok(result) };
        Self::to(&request, outcome)
    }

    /// `state` delivered as the message that is this reply; the state is given back as it was.
    pub fn send<T>(self, state: &mut Value, deliver: impl FnOnce(&Value) -> T) -> T {
        let Some(object) = state.as_object_mut().filter(|_| !self.0.is_empty()) else { return deliver(state) };
        debug_assert!(KEYS.iter().all(|key| !object.contains_key(*key)), "a state holds no reply");
        let keys: Vec<String> = self.0.keys().cloned().collect();
        object.extend(self.0);
        let delivered = deliver(state);
        if let Some(object) = state.as_object_mut() { for key in keys.iter().rev() { object.shift_remove(key); } }
        delivered
    }
}

/// Whether a message answers a request (a positive id): request 0, or none, is no request.
pub fn answers(request: &Value) -> Option<u64> { request.as_u64().filter(|request| *request > 0) }

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_reply_rides_on_its_message_and_leaves_the_state_as_it_was() {
        let mut state = json!({"type": "state", "panels": [1], "acquisition": {"state": "idle"}});
        let before = state.clone();
        let said = Reply::to(&json!(7), Ok(json!({"node": "camera"}))).send(&mut state, |message| message.clone());
        assert_eq!((&said["request"], &said["result"]["node"], said.get("error")), (&json!(7), &json!("camera"), None));
        assert_eq!(Reply::done(&json!(4)).send(&mut state, |message| message.clone())["result"], Value::Null, "a done request's result is null");
        assert_eq!(state, before, "the state holds no reply after it is sent");
        assert_eq!(state.as_object().unwrap().keys().collect::<Vec<_>>(), ["type", "panels", "acquisition"], "and keeps its order");
        // Request 0 answers nothing; its error is still said.
        assert_eq!(Reply::done(&json!(0)), Reply::default());
        let error = Reply::to(&json!(0), Err("no".into())).send(&mut state, |message| message.clone());
        assert_eq!((error.get("request"), &error["error"]), (None, &json!("no")));
        // A worker's state gives its reply up.
        let mut worker = json!({"type": "state", "request": 3, "panels": [], "error": null, "result": null, "saved": null});
        assert_eq!(Reply::take(&mut worker), Reply::done(&json!(3)));
        assert_eq!(worker, json!({"type": "state", "panels": [], "saved": null}));
    }
}
