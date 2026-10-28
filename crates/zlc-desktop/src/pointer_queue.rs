//! Lossless wheel travel batching while a native render lane is occupied.
//! Call only on the queue's final entry: any intervening gesture or edit is
//! an ordering boundary. UI wheel notifications explicitly opt in; commands
//! with independent remote replies must always complete individually.
use serde_json::Value;

pub fn merge_scroll(pending: &mut Value, incoming: &Value) -> bool {
    let eligible = |command: &Value| command["action"] == "pointer" && command["event"] == "scroll"
        && command["_coalesce_scroll"] == true && command.get("_origin").is_none()
        && command["id"].as_u64().is_some_and(|id| id < crate::remote::FIRST_REMOTE_REQUEST);
    if !eligible(pending) || !eligible(incoming) { return false; }
    // Bounds/limits may come from successive fronts; role and cell identify
    // the same axes. The most recent snapshot supplies the final routing.
    for field in ["panel", "document", "modifiers", "buttons"] {
        if pending[field] != incoming[field] { return false; }
    }
    if !pending["axes"].is_object() || !incoming["axes"].is_object() { return false; }
    for field in ["role", "cell_index"] {
        if pending["axes"][field] != incoming["axes"][field] { return false; }
    }
    let (Some(before), Some(after)) = (pending["steps"].as_f64(), incoming["steps"].as_f64()) else { return false; };
    let total = before + after;
    if before == 0.0 || after == 0.0 || before.signum() != after.signum() || !total.is_finite() { return false; }
    *pending = incoming.clone();
    pending["steps"] = Value::from(total);
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn scroll(id: u64, steps: f64) -> Value {
        json!({"id":id,"action":"pointer","event":"scroll","panel":"camera","document":1,
            "steps":steps,"axes":{"role":"image","cell_index":null,"x_limits":[0,10]},
            "modifiers":[],"buttons":[],"_coalesce_scroll":true})
    }
    #[test]
    fn pending_wheel_preserves_all_travel_and_latest_front() {
        let mut pending = scroll(1, -0.08);
        for id in 2..=100 {
            let mut incoming = scroll(id, -0.08);
            incoming["axes"]["x_limits"] = json!([id, id + 10]);
            assert!(merge_scroll(&mut pending, &incoming));
        }
        assert!((pending["steps"].as_f64().unwrap() + 8.0).abs() < 1e-12);
        assert_eq!(pending["id"], 100);
        assert_eq!(pending["axes"]["x_limits"], json!([100,110]));
    }
    #[test]
    fn gesture_direction_routing_and_reply_boundaries_are_never_merged() {
        let patches = [json!({"event":"move"}), json!({"event":"press"}), json!({"action":"configure"}),
            json!({"steps":0.08}), json!({"steps":0}), json!({"panel":"other"}), json!({"document":2}),
            json!({"modifiers":["shift"]}), json!({"buttons":["middle"]}), json!({"axes":null}),
            json!({"axes":{"role":"distribution","cell_index":null}}),
            json!({"axes":{"role":"image","cell_index":2}}), json!({"_coalesce_scroll":false}),
            json!({"_origin":"remote"}), json!({"id":crate::remote::FIRST_REMOTE_REQUEST})];
        for patch in patches {
            let original = scroll(1, -0.08);
            let mut pending = original.clone();
            let mut incoming = scroll(2, -0.08);
            incoming.as_object_mut().unwrap().extend(patch.as_object().unwrap().clone());
            assert!(!merge_scroll(&mut pending, &incoming), "{patch}");
            assert_eq!(pending, original, "failed merge must leave the queue untouched");
            assert!(!merge_scroll(&mut incoming, &original), "{patch}");
        }
    }
}