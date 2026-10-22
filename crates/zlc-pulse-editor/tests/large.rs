//! Time of an edit and of the window's state on a large pulse (line I's probe;
//! ignored by default): `ZLC_LARGE_PULSE=<pulse.json> cargo test --release -p
//! zlc-pulse-editor --test large -- --ignored --nocapture`.
use std::time::Instant;
use serde_json::{Map, Value, json};
use zlc_pulse_editor::editor::{Deployed, Editor};

fn args(value: Value) -> Map<String, Value> { value.as_object().cloned().unwrap() }

#[test]
#[ignore]
fn edit_and_state_of_a_large_pulse() {
    let path = std::env::var("ZLC_LARGE_PULSE").expect("ZLC_LARGE_PULSE");
    let folder = std::path::PathBuf::from(&path).parent().unwrap().to_owned();
    let mut editor = Editor::new(Some(Deployed::load().unwrap()), folder);
    let started = Instant::now();
    editor.command("pulse_open", &args(json!({"path": path, "discard_changes": true})), true).unwrap();
    println!("open {:.1} ms", started.elapsed().as_secs_f64() * 1e3);
    let started = Instant::now();
    let mut state = editor.state();
    println!("state {:.1} ms ({} chars)", started.elapsed().as_secs_f64() * 1e3, state.to_string().len());
    let first = state["schedule"]["periods"][0]["period_id"].as_str().unwrap().to_owned();
    // Sent: its cards come back (the host does this once a state is told to the window).
    editor.recycle(&mut state);
    for round in 0..4 {
        let started = Instant::now();
        editor.command("pulse_edit", &args(json!({"edit": "digital", "period_id": first, "port_key": "repump", "high": round % 2 == 0})), true).unwrap();
        let edited = started.elapsed().as_secs_f64() * 1e3;
        let started = Instant::now();
        let mut state = editor.state();
        editor.recycle(&mut state);
        drop(state);
        println!("digital edit {edited:.1} ms, state {:.1} ms", started.elapsed().as_secs_f64() * 1e3);
    }
    for round in 0..2 {
        let started = Instant::now();
        editor.command("pulse_edit", &args(json!({"edit": "duration", "period_id": first, "value": if round % 2 == 0 { "200" } else { "100" }, "unit": "µs"})), true).unwrap();
        let edited = started.elapsed().as_secs_f64() * 1e3;
        let started = Instant::now();
        let mut state = editor.state();
        editor.recycle(&mut state);
        drop(state);
        println!("duration edit {edited:.1} ms, state {:.1} ms", started.elapsed().as_secs_f64() * 1e3);
    }
}
