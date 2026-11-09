//! A bracket keeps the name it is given; no edit makes two fields read alike; a
//! Subpulse's Edit opens it, and its Save and Save as write its own file.
use serde_json::{Map, Value, json};
use zlc_pulse_editor::editor::{Deployed, Editor};

fn args(value: Value) -> Map<String, Value> { value.as_object().cloned().unwrap() }

fn command(editor: &mut Editor, action: &str, value: Value) -> Value {
    editor.command(action, &args(value), true).unwrap_or_else(|error| panic!("{action}: {error}"))
}

fn refused(editor: &mut Editor, action: &str, value: Value) -> String {
    editor.command(action, &args(value), true).expect_err("refused")
}

fn edit(editor: &mut Editor, value: Value) -> Value { command(editor, "pulse_edit", value) }

fn temp(name: &str) -> std::path::PathBuf {
    let folder = std::env::temp_dir().join(format!("zlc_pulse_brackets_{}_{name}", std::process::id()));
    std::fs::create_dir_all(&folder).unwrap();
    folder
}

/// Four periods, period1..period4.
fn pulse(name: &str) -> (std::path::PathBuf, Editor) {
    let folder = temp(name);
    let mut editor = Editor::new(Some(Deployed::load().unwrap()), folder.clone());
    assert!(editor.start_new_pulse());
    edit(&mut editor, json!({"edit": "period_add", "before_item": null}));
    edit(&mut editor, json!({"edit": "period_add", "before_item": null}));
    let ids: Vec<String> = editor.state()["schedule"]["periods"].as_array().unwrap().iter().map(|p| p["period_id"].as_str().unwrap().to_owned()).collect();
    assert_eq!(ids.len(), 4, "{ids:?}");
    (folder, editor)
}

fn period_ids(editor: &mut Editor) -> Vec<String> {
    editor.state()["schedule"]["periods"].as_array().unwrap().iter().map(|p| p["period_id"].as_str().unwrap().to_owned()).collect()
}

/// (bracket id, name) in the schedule's order.
fn brackets(editor: &mut Editor) -> Vec<(String, String)> {
    editor.state()["schedule"]["brackets"].as_array().unwrap().iter()
        .map(|b| (b["bracket_id"].as_str().unwrap().to_owned(), b["name"].as_str().unwrap().to_owned())).collect()
}

fn name_of(editor: &mut Editor, id: &str) -> String { brackets(editor).into_iter().find(|(bracket, _)| bracket == id).unwrap().1 }

#[test]
fn a_bracket_keeps_its_name_through_adds_and_reopening() {
    let (folder, mut editor) = pulse("names");
    let ids = period_ids(&mut editor);
    edit(&mut editor, json!({"edit": "bracket_add", "start": ids[0], "end": ids[1], "count": 3}));
    let first = brackets(&mut editor)[0].0.clone();
    assert_eq!(name_of(&mut editor, &first), "Bracket 1", "a new bracket is named once, by its place");
    edit(&mut editor, json!({"edit": "bracket_name", "bracket_id": first, "name": "load_loop"}));
    // A bracket added before it, and one inside it, take free names; the first keeps its own.
    edit(&mut editor, json!({"edit": "bracket_add", "start": ids[2], "end": ids[3], "count": 2}));
    edit(&mut editor, json!({"edit": "bracket_add", "start": ids[0], "end": ids[0], "count": 4}));
    let names: Vec<String> = brackets(&mut editor).into_iter().map(|(_, name)| name).collect();
    assert_eq!(name_of(&mut editor, &first), "load_loop");
    assert_eq!(names.iter().collect::<std::collections::BTreeSet<_>>().len(), 3, "{names:?}");
    // Recounting it keeps its name.
    edit(&mut editor, json!({"edit": "bracket", "bracket_id": first, "start": ids[0], "end": ids[1], "count": 5}));
    assert_eq!(name_of(&mut editor, &first), "load_loop");
    let path = folder.join("named.json");
    command(&mut editor, "pulse_save", json!({"path": path.display().to_string()}));
    let mut reopened = Editor::new(Some(Deployed::load().unwrap()), folder.clone());
    command(&mut reopened, "pulse_open", json!({"path": path.display().to_string()}));
    assert_eq!(brackets(&mut reopened), brackets(&mut editor), "the names are saved");
}

#[test]
fn a_file_without_names_reads_as_it_did() {
    let (folder, mut editor) = pulse("legacy");
    let ids = period_ids(&mut editor);
    edit(&mut editor, json!({"edit": "bracket_add", "start": ids[0], "end": ids[1], "count": 3}));
    edit(&mut editor, json!({"edit": "bracket_add", "start": ids[2], "end": ids[3], "count": 2}));
    let path = folder.join("legacy.json");
    command(&mut editor, "pulse_save", json!({"path": path.display().to_string()}));
    // As written before brackets had names.
    let mut tree: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    for bracket in tree["brackets"].as_array_mut().unwrap() { bracket.as_object_mut().unwrap().remove("name"); }
    std::fs::write(&path, serde_json::to_string(&tree).unwrap()).unwrap();
    let mut reopened = Editor::new(Some(Deployed::load().unwrap()), folder.clone());
    command(&mut reopened, "pulse_open", json!({"path": path.display().to_string()}));
    let names: Vec<String> = brackets(&mut reopened).into_iter().map(|(_, name)| name).collect();
    assert_eq!(names, ["Bracket 1", "Bracket 2"], "by their places, as they read before");
}

#[test]
fn no_rename_makes_two_fields_read_alike() {
    let (_folder, mut editor) = pulse("clash");
    let ids = period_ids(&mut editor);
    edit(&mut editor, json!({"edit": "bracket_add", "start": ids[0], "end": ids[1], "count": 3}));
    edit(&mut editor, json!({"edit": "bracket_add", "start": ids[2], "end": ids[3], "count": 2}));
    let [(a, _), (b, _)] = <[(String, String); 2]>::try_from(brackets(&mut editor)).unwrap();
    edit(&mut editor, json!({"edit": "bracket_name", "bracket_id": a, "name": "loop"}));
    let error = refused(&mut editor, "pulse_edit", json!({"edit": "bracket_name", "bracket_id": b, "name": "loop"}));
    assert!(error.contains("unique"), "{error}");
    // A period may share a bracket's name: its fields read `loop.duration`, the bracket's `loop.repeats`.
    edit(&mut editor, json!({"edit": "period_name", "period_id": ids[3], "name": "loop"}));
    // ...until a DAC port would read `repeats` too.
    let dac = editor.state()["schedule"]["ports"].as_array().unwrap().iter().find(|p| p["kind"] == "dac").unwrap()["key"].as_str().unwrap().to_owned();
    let error = refused(&mut editor, "pulse_edit", json!({"edit": "port_label", "port_key": dac, "label": "repeats"}));
    assert!(error.contains("loop.repeats"), "{error}");
    // Two periods named alike are refused as before.
    let error = refused(&mut editor, "pulse_edit", json!({"edit": "period_name", "period_id": ids[2], "name": "loop"}));
    assert!(error.contains("unique"), "{error}");
}

#[test]
fn a_subpulse_edit_opens_it_and_save_writes_its_own_file() {
    let (folder, mut editor) = pulse("subpulse");
    let ids = period_ids(&mut editor);
    command(&mut editor, "pulse_subpulse", json!({"op": "group", "items": [["period", ids[1]], ["period", ids[2]]]}));
    let component = editor.state()["subpulse"]["contexts"].as_array().unwrap().iter().find_map(|c| c[0].as_str().filter(|id| !id.is_empty()).map(str::to_owned)).unwrap();
    command(&mut editor, "pulse_subpulse", json!({"op": "select", "component_id": component}));
    let state = editor.state();
    assert_eq!(state["page"], "Subpulse Config", "Edit opens the Subpulse");
    assert_eq!(state["subpulse"]["context_id"], component.as_str());
    assert_eq!(state["subpulse"]["dirty"], true, "a block with no file of its own is unsaved");
    // Save of a block with no file asks for one; Save as writes it, and the block names it.
    let error = refused(&mut editor, "pulse_subpulse", json!({"op": "save", "component_id": component}));
    assert!(error.contains("path"), "{error}");
    let file = folder.join("block.subpulse.json");
    command(&mut editor, "pulse_subpulse", json!({"op": "save_as", "component_id": component, "path": file.display().to_string()}));
    let state = editor.state();
    assert!(file.is_file());
    assert_eq!(state["subpulse"]["path"], file.display().to_string());
    assert_eq!(state["subpulse"]["dirty"], false);
    // An edit makes it unsaved; Save writes the same file.
    command(&mut editor, "pulse_subpulse", json!({"op": "edit", "component_id": component, "edit": "duration", "period_id": ids[1], "value": "7", "unit": "µs"}));
    assert_eq!(editor.state()["subpulse"]["dirty"], true);
    command(&mut editor, "pulse_subpulse", json!({"op": "save", "component_id": component}));
    assert_eq!(editor.state()["subpulse"]["dirty"], false);
    let written: Value = serde_json::from_str(&std::fs::read_to_string(&file).unwrap()).unwrap();
    assert!(written["periods"].as_array().unwrap().iter().any(|p| p["duration"] == json!(7) && p["unit"] == "µs"), "{written}");
}
