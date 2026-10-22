//! The Pulse Editor's flows end to end on the virtual board, with the host's
//! lanes run in line: author, save, reopen; connect; On Pulse, Stop; a
//! finite run's completion; a scan table and Hold; Value Config. No hardware.
use serde_json::{Map, Value, json};
use zlc_pulse_editor::editor::{Deployed, Editor};
use zlc_pulse::board::{Board, Fate};
use zlc_pulse_editor::session::{Job, Outcome};

fn args(value: Value) -> Map<String, Value> { value.as_object().cloned().unwrap() }

/// The host's lanes, run here one after another.
fn settle(editor: &mut Editor) {
    loop {
        let jobs = editor.take_jobs();
        if jobs.is_empty() { return; }
        for job in jobs {
            let outcome = match job {
                Job::Device { work, done } => { let outcome = work(); done.set(); outcome }
                Job::Safe { work } | Job::Completion { work } => work(),
                Job::Program { source, fields } => Outcome::Program { source, fields, table: Err("no interpreter in this test".into()) },
            };
            editor.deliver(outcome);
        }
    }
}

fn command(editor: &mut Editor, action: &str, value: Value) -> Value {
    let answer = editor.command(action, &args(value), true);
    settle(editor);
    answer.unwrap_or_else(|error| panic!("{action}: {error}"))
}

fn temp(name: &str) -> std::path::PathBuf {
    let folder = std::env::temp_dir().join(format!("zlc_pulse_editor_{}_{name}", std::process::id()));
    std::fs::create_dir_all(&folder).unwrap();
    folder
}

#[test]
fn author_save_reopen_connect_fire_and_stop() {
    let folder = temp("flow");
    let mut editor = Editor::new(Some(Deployed::load().unwrap()), folder.clone());
    assert!(editor.start_new_pulse());
    let state = editor.state();
    assert_eq!(state["schedule"]["periods"].as_array().unwrap().len(), 2);
    assert_eq!(state["title"], "PulseGUI · untitled");
    assert_eq!(state["status"], "idle", "no board: nothing runs yet");
    command(&mut editor, "pulse_edit", json!({"edit": "duration", "period_id": "period1", "value": "2.5", "unit": "µs"}));
    command(&mut editor, "pulse_edit", json!({"edit": "digital", "period_id": "period2", "port_key": "probe", "high": true}));
    command(&mut editor, "pulse_edit", json!({"edit": "period_add", "before_item": null}));
    let state = editor.state();
    assert_eq!(state["schedule"]["periods"][0]["duration"]["text"], "2.5");
    assert_eq!(state["schedule"]["periods"].as_array().unwrap().len(), 3);
    assert_eq!(state["dirty"], true);
    // A refused edit is said, and the pulse is kept.
    assert!(editor.command("pulse_edit", &args(json!({"edit": "duration", "period_id": "nope", "value": 1, "unit": "ns"})), true).is_err());
    assert_eq!(editor.state()["message"]["kind"], "warning");

    let path = folder.join("first.json");
    command(&mut editor, "pulse_save", json!({"path": path.display().to_string()}));
    let state = editor.state();
    assert_eq!(state["dirty"], false);
    assert_eq!(state["title"], "PulseGUI · first", "the file names the pulse");
    let mut reopened = Editor::new(Some(Deployed::load().unwrap()), folder.clone());
    command(&mut reopened, "pulse_open", json!({"path": path.display().to_string()}));
    let without_revision = |mut schedule: Value| { schedule["revision"] = Value::Null; schedule };
    assert_eq!(without_revision(reopened.state()["schedule"].clone()), without_revision(editor.state()["schedule"].clone()));

    command(&mut editor, "pulse_connect", json!({"mode": "virtual"}));
    let state = editor.state();
    assert!(state["connection"]["status"].as_str().unwrap().starts_with("virtual · "), "{}", state["connection"]);
    assert_eq!(state["control"]["can_run"], true);
    // Run repeats 0: the pulse plays until Stop.
    command(&mut editor, "pulse_fire", json!({}));
    let state = editor.state();
    assert_eq!(state["board"]["firing"], true, "{}", state["message"]);
    assert_eq!(state["status"], "running-synced");
    command(&mut editor, "pulse_edit", json!({"edit": "digital", "period_id": "period2", "port_key": "probe", "high": false}));
    assert_eq!(editor.state()["status"], "running-stale", "an edit makes the board older than the screen");
    command(&mut editor, "pulse_stop", json!({}));
    let state = editor.state();
    assert_eq!(state["board"]["firing"], false);
    assert_eq!(state["summary"], "Stopped");
    assert_eq!(state["status"], "dirty-ready");

    // Finite: two runs, then the completion wait reports.
    command(&mut editor, "pulse_edit", json!({"edit": "run_repeats", "repeats": 2}));
    command(&mut editor, "pulse_fire", json!({}));
    let state = editor.state();
    assert_eq!(state["board"]["firing"], false, "the completion lane waited the two runs out");
    assert!(state["message"].is_null() || state["message"]["kind"] != "warning" || state["message"]["text"] != "", "{}", state["message"]);
    command(&mut editor, "pulse_connect", json!({"mode": "offline"}));
    assert_eq!(editor.state()["connection"]["status"], "edit only");
    let _ = std::fs::remove_dir_all(&folder);
}

#[test]
fn scan_table_hold_and_value_config() {
    let folder = temp("scan");
    let mut editor = Editor::new(Some(Deployed::load().unwrap()), folder.clone());
    assert!(editor.start_new_pulse());
    command(&mut editor, "pulse_edit", json!({"edit": "binding", "field_kind": "duration", "period_id": "period1", "scan": true, "source": "default"}));
    let table = folder.join("points.csv");
    std::fs::write(&table, "# a sweep\nduration:period1\n1000\n2000\n3000\n").unwrap();
    command(&mut editor, "pulse_scan", json!({"op": "load_array", "path": table.display().to_string()}));
    let state = editor.state();
    assert_eq!(state["scan"]["points"], 3, "{}", state["scan"]);
    assert!(state["scan"]["table_text"].as_str().unwrap().contains("2000"), "{}", state["scan"]["table_text"]);
    assert_eq!(state["schedule"]["scan_summary_text"], "1 slot · 3 points");
    command(&mut editor, "pulse_connect", json!({"mode": "virtual"}));
    command(&mut editor, "pulse_step", json!({"delta": 2}));
    let state = editor.state();
    assert_eq!(state["control"]["held_point"], 2);
    assert_eq!(state["board"]["firing"], true);
    assert!(state["scan"]["progress_text"].as_str().unwrap().starts_with("Held at scan point 3 of 3"), "{}", state["scan"]["progress_text"]);
    assert_eq!(state["status"], "running-synced", "a held row is this pulse");
    command(&mut editor, "pulse_stop", json!({}));
    // Saved under the Pulse's folder: one named column per driven field.
    let saved = command(&mut editor, "pulse_scan", json!({"op": "save_array"}));
    let path = saved["path"].as_str().unwrap().to_owned();
    command(&mut editor, "pulse_scan", json!({"op": "clear"}));
    command(&mut editor, "pulse_scan", json!({"op": "load_array", "path": path}));
    assert_eq!(editor.state()["scan"]["points"], 3);

    // Value Config: a saved value feeds a Config-bound field while Config is on.
    command(&mut editor, "pulse_edit", json!({"edit": "binding", "field_kind": "duration", "period_id": "period2", "scan": false, "source": "config"}));
    command(&mut editor, "pulse_config", json!({"op": "binding", "field_id": "duration:period2", "key": "wait"}));
    command(&mut editor, "pulse_config", json!({"op": "edit", "rows": [["wait", "4", "µs"]]}));
    let values = folder.join("config_values").join("current.json");
    command(&mut editor, "pulse_config", json!({"op": "save", "path": values.display().to_string()}));
    command(&mut editor, "pulse_config", json!({"op": "mode", "enabled": true}));
    let state = editor.state();
    let period = &state["schedule"]["periods"][1];
    assert_eq!(period["duration"]["source"], "config");
    assert_eq!(period["duration"]["config_active"], true, "{period}");
    assert_eq!(period["duration"]["text"], "4000", "the effective value is shown");
    assert_eq!(state["config"]["bindings"][0][5], "Applied");
    let validated = command(&mut editor, "pulse_validate", json!({}));
    assert_eq!(validated["valid"], true);
    let _ = std::fs::remove_dir_all(&folder);
}

fn connected(name: &str) -> (std::path::PathBuf, Editor) {
    let folder = temp(name);
    let mut editor = Editor::new(Some(Deployed::load().unwrap()), folder.clone());
    assert!(editor.start_new_pulse());
    command(&mut editor, "pulse_connect", json!({"mode": "virtual"}));
    (folder, editor)
}

/// A table of three points on period1's duration.
fn with_scan(editor: &mut Editor, folder: &std::path::Path) {
    command(editor, "pulse_edit", json!({"edit": "binding", "field_kind": "duration", "period_id": "period1", "scan": true, "source": "default"}));
    let table = folder.join("points.csv");
    std::fs::write(&table, "duration:period1\n1000\n2000\n3000\n").unwrap();
    command(editor, "pulse_scan", json!({"op": "load_array", "path": table.display().to_string()}));
}

/// v2 `_run_device_command` / `_drive_program`: a Stop superseding an On Pulse still on the
/// device lane ends the busy that command set when it delivers, the command fires nothing, and
/// the Stop is heard after it (its answer is held while the command is out).
#[test]
fn a_stop_superseding_on_pulse_ends_its_busy_and_fires_nothing() {
    let (folder, mut editor) = connected("supersede");
    command(&mut editor, "pulse_fire", json!({}));
    let board = editor.board().unwrap().clone();
    let (first, _) = board.last_fire().unwrap();
    editor.command("pulse_fire", &args(json!({})), true).unwrap();
    editor.command("pulse_stop", &args(json!({})), true).unwrap();
    let (mut drove, mut stopped) = (None, None);
    for job in editor.take_jobs() {
        match job {
            Job::Device { work, done } => { drove = Some(work()); done.set(); }
            Job::Safe { work } => stopped = Some(work),
            _ => {}
        }
    }
    editor.deliver(stopped.expect("the Stop's SAFE")());
    assert_eq!(editor.state()["stopping"], true, "held until the superseded command delivers");
    editor.deliver(drove.expect("the On Pulse"));
    let state = editor.state();
    assert!(state["busy"].is_null(), "{}", state["busy"]);
    assert_eq!(state["stopping"], false);
    assert_eq!(state["summary"], "Stopped");
    assert_eq!(state["control"]["can_run"], true);
    assert_eq!(board.last_fire(), Some((first, Fate::Safed)), "the superseded On Pulse fired nothing");
    command(&mut editor, "pulse_sync", json!({}));
    assert!(editor.state()["busy"].is_null());
    let _ = std::fs::remove_dir_all(&folder);
}

/// v2: the digest the board reports after On Pulse is what this revision plays, its Config
/// filled in; the editor's own compile does not have the Config values.
#[test]
fn on_pulse_of_a_config_pulse_is_synchronized() {
    let (folder, mut editor) = connected("config_digest");
    command(&mut editor, "pulse_edit", json!({"edit": "binding", "field_kind": "duration", "period_id": "period2", "scan": false, "source": "config"}));
    command(&mut editor, "pulse_config", json!({"op": "binding", "field_id": "duration:period2", "key": "wait"}));
    command(&mut editor, "pulse_config", json!({"op": "edit", "rows": [["wait", "4", "µs"]]}));
    command(&mut editor, "pulse_config", json!({"op": "save", "path": folder.join("config_values").join("current.json").display().to_string()}));
    command(&mut editor, "pulse_config", json!({"op": "mode", "enabled": true}));
    command(&mut editor, "pulse_fire", json!({}));
    let state = editor.state();
    assert_eq!(state["control"]["synchronized"], true, "{}", state["board"]);
    assert_eq!(state["status"], "running-synced");
    command(&mut editor, "pulse_stop", json!({}));
    let _ = std::fs::remove_dir_all(&folder);
}

/// v2 `_adopt_applied`: a board holding one row of this draft's scan (Hold, Step) keeps the
/// draft and its table.
#[test]
fn sync_while_holding_a_point_keeps_the_draft_and_its_table() {
    let (folder, mut editor) = connected("hold_sync");
    with_scan(&mut editor, &folder);
    command(&mut editor, "pulse_step", json!({"delta": 2}));
    command(&mut editor, "pulse_sync", json!({}));
    let state = editor.state();
    assert_eq!(state["scan"]["points"], 3);
    assert_eq!(state["message"]["text"], "The board is holding scan point 3 of this pulse; the draft and its scan table are kept.");
    assert_eq!(editor.control_state()["authored"]["bindings"][0]["scan"], true);
    command(&mut editor, "pulse_stop", json!({}));
    let _ = std::fs::remove_dir_all(&folder);
}

/// v2 `_render_scan_progress`: while the board plays the table and the Scan page is shown,
/// its line says which point the board is on.
#[test]
fn the_scan_page_says_where_the_board_is() {
    let (folder, mut editor) = connected("progress");
    with_scan(&mut editor, &folder);
    command(&mut editor, "pulse_edit", json!({"edit": "run_repeats", "repeats": 1}));
    command(&mut editor, "pulse_fire", json!({}));
    command(&mut editor, "pulse_page", json!({"page": "Scan"}));
    editor.poll();
    settle(&mut editor);
    let progress = editor.state()["scan"]["progress_text"].as_str().unwrap().to_owned();
    assert!(progress.starts_with("Scan: point ") && progress.contains(" / 3  "), "{progress}");
    command(&mut editor, "pulse_stop", json!({}));
    let _ = std::fs::remove_dir_all(&folder);
}

/// v2 device use: while a run holds the sequencer the editor's Stop is refused; the node's Stop
/// makes the board safe.
#[test]
fn stop_is_refused_while_a_run_holds_the_board() {
    let (folder, mut editor) = connected("claimed_stop");
    editor.set_claimed_by(Some("Calibration".into()));
    assert_eq!(editor.command("pulse_stop", &args(json!({})), true).unwrap_err(), "device is in use by Calibration");
    let _ = std::fs::remove_dir_all(&folder);
}

/// v2 confirm_pulse_discard: Sync asks before the board's pulse replaces unsaved work; yes syncs again.
#[test]
fn sync_asks_before_replacing_unsaved_work() {
    let (folder, mut editor) = connected("sync_discard");
    command(&mut editor, "pulse_fire", json!({}));
    command(&mut editor, "pulse_edit", json!({"edit": "duration", "period_id": "period1", "value": "2.5", "unit": "µs"}));
    command(&mut editor, "pulse_sync", json!({}));
    let state = editor.state();
    assert_eq!(state["message"]["kind"], "sync_discard");
    assert_eq!(state["schedule"]["periods"][0]["duration"]["text"], "2.5", "nothing replaced yet");
    command(&mut editor, "pulse_sync", json!({"discard_changes": true}));
    let state = editor.state();
    assert!(state["message"]["text"].as_str().unwrap().starts_with("Synced from the board"), "{}", state["message"]);
    assert_ne!(state["schedule"]["periods"][0]["duration"]["text"], "2.5", "the board's pulse");
    command(&mut editor, "pulse_stop", json!({}));
    let _ = std::fs::remove_dir_all(&folder);
}

/// v2 PE:1306: the session's board brings its own Config; the editor shows it and pushes none of its own.
#[test]
fn the_session_board_keeps_its_own_config() {
    let folder = temp("given_config");
    let mut editor = Editor::new(Some(Deployed::load().unwrap()), folder.clone());
    command(&mut editor, "pulse_config", json!({"op": "edit", "rows": [["other", "1", "µs"]]}));
    command(&mut editor, "pulse_config", json!({"op": "save", "path": folder.join("mine.json").display().to_string()}));
    let board = std::sync::Arc::new(Board::virtual_board(None).unwrap());
    board.config().load_values(&[("wait".into(), (4.0, "µs".into()))], "device row").unwrap();
    editor.attach_given(board.clone(), "sequencer");
    assert_eq!(board.config().values(), vec![("wait".to_owned(), (4.0, "µs".to_owned()))]);
    assert_eq!(editor.state()["config"]["active_path"], "device row");
    let _ = std::fs::remove_dir_all(&folder);
}
