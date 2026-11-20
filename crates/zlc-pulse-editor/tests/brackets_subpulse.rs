//! A bracket keeps the name it is given; no edit makes two fields read alike; a
//! Subpulse's Edit opens it, and its Save and Save as write its own file.
use serde_json::{Map, Value, json};
use zlc_pulse_editor::editor::{Deployed, Editor};

fn args(value: Value) -> Map<String, Value> { value.as_object().cloned().unwrap() }

fn command(editor: &mut Editor, action: &str, value: Value) -> Value {
    let answer = editor.command(action, &args(value), true);
    written(editor, answer).unwrap_or_else(|error| panic!("{action}: {error}"))
}

/// As the host does: the files a command wrote put on the disk, then its answer settled.
fn written(editor: &mut Editor, answer: zlc_pulse_editor::editor::Answer) -> zlc_pulse_editor::editor::Answer {
    let Some(writes) = editor.take_writes() else { return answer };
    let result = writes.files.iter().try_for_each(|(path, bytes)| zlc_pulse::codec::write_file(path, bytes).map_err(|error| error.message));
    editor.written(writes, result, answer)
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
    // Ungrouped, its id free again: a new block under that id has saved nothing.
    command(&mut editor, "pulse_subpulse", json!({"op": "ungroup", "component_id": component}));
    let ids = period_ids(&mut editor);
    command(&mut editor, "pulse_subpulse", json!({"op": "group", "items": [["period", ids[1]], ["period", ids[2]]]}));
    command(&mut editor, "pulse_subpulse", json!({"op": "select", "component_id": component}));
    assert_eq!(editor.state()["subpulse"]["context_id"], component.as_str(), "the id is reused");
    assert_eq!(editor.state()["subpulse"]["dirty"], true, "a new block is not the old one's save");
}

#[test]
fn ungrouping_a_block_qualifies_a_bracket_name_taken_outside() {
    let (_folder, mut editor) = pulse("ungroup");
    let ids = period_ids(&mut editor);
    edit(&mut editor, json!({"edit": "bracket_add", "start": ids[0], "end": ids[0], "count": 2}));
    edit(&mut editor, json!({"edit": "bracket_add", "start": ids[1], "end": ids[1], "count": 3}));
    let inner = editor.state()["schedule"]["brackets"].as_array().unwrap().iter().find(|b| b["start_period_id"] == ids[1]).unwrap()["bracket_id"].as_str().unwrap().to_owned();
    let outer_name = brackets(&mut editor).into_iter().find(|(id, _)| *id != inner).unwrap().1;
    command(&mut editor, "pulse_subpulse", json!({"op": "group", "items": [["bracket", format!("{inner}:start")], ["period", ids[1]],
        ["bracket", format!("{inner}:end")], ["period", ids[2]]]}));
    let component = editor.state()["subpulse"]["contexts"].as_array().unwrap().iter().find_map(|c| c[0].as_str().filter(|id| !id.is_empty()).map(str::to_owned)).unwrap();
    // Inside the block the name may repeat the outer one's.
    edit(&mut editor, json!({"edit": "bracket_name", "bracket_id": inner, "name": outer_name}));
    command(&mut editor, "pulse_subpulse", json!({"op": "ungroup", "component_id": component}));
    let after: Vec<String> = brackets(&mut editor).into_iter().map(|(_, name)| name).collect();
    assert_eq!(after.iter().collect::<std::collections::BTreeSet<_>>().len(), 2, "{after:?}");
    assert!(after.contains(&outer_name), "{after:?}");
}

#[test]
fn ports_labelled_alike_do_not_stop_adding_or_renaming_periods() {
    let (folder, mut editor) = pulse("ports_alike");
    let path = folder.join("alike.json");
    command(&mut editor, "pulse_save", json!({"path": path.display().to_string()}));
    let mut tree: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let dacs: Vec<usize> = tree["target"]["ports"].as_array().unwrap().iter().enumerate().filter(|(_, p)| p["kind"] == "dac").map(|(i, _)| i).take(2).collect();
    for index in &dacs { tree["target"]["ports"][*index]["label"] = json!("Coil"); }
    std::fs::write(&path, serde_json::to_string(&tree).unwrap()).unwrap();
    let mut reopened = Editor::new(Some(Deployed::load().unwrap()), folder.clone());
    command(&mut reopened, "pulse_open", json!({"path": path.display().to_string()}));
    edit(&mut reopened, json!({"edit": "period_add", "before_item": null}));
    let ids = period_ids(&mut reopened);
    edit(&mut reopened, json!({"edit": "period_name", "period_id": ids[0], "name": "load"}));
}

#[test]
fn the_boards_copy_reads_under_the_drafts_bracket_names() {
    let (folder, mut editor) = pulse("sync_names");
    let ids = period_ids(&mut editor);
    edit(&mut editor, json!({"edit": "bracket_add", "start": ids[0], "end": ids[1], "count": 3}));
    let first = brackets(&mut editor)[0].0.clone();
    edit(&mut editor, json!({"edit": "bracket_name", "bracket_id": first, "name": "load_loop"}));
    let path = folder.join("sync.json");
    command(&mut editor, "pulse_save", json!({"path": path.display().to_string()}));
    let (draft, _) = zlc_pulse::codec::read_pulse_document(&path).unwrap();
    let draft = std::sync::Arc::new(draft);
    // The board's wire carries no names: its copy reads as "Bracket 1" until named as the draft's.
    let copy = std::sync::Arc::new(zlc_pulse::remote::sequence_from_remote(&zlc_pulse::remote::sequence_tree(&draft)).unwrap());
    assert_eq!(copy.brackets[0].name, "Bracket 1");
    assert_ne!(*copy, *draft);
    assert_eq!(*copy.bracket_names_of(&draft), *draft);
}

/// A Subpulse source is read once per change of its file: views made meanwhile read nothing (a
/// file rewritten with its stamp kept is not read again), and a changed file, once the poll has
/// found it, is read again.
#[test]
fn a_subpulse_source_is_read_again_only_when_its_file_changes() {
    let (folder, mut editor) = pulse("source_cache");
    let ids = period_ids(&mut editor);
    command(&mut editor, "pulse_subpulse", json!({"op": "group", "items": [["period", ids[1]], ["period", ids[2]]]}));
    let component = editor.state()["subpulse"]["contexts"].as_array().unwrap().iter().find_map(|c| c[0].as_str().filter(|id| !id.is_empty()).map(str::to_owned)).unwrap();
    command(&mut editor, "pulse_subpulse", json!({"op": "select", "component_id": component}));
    let file = folder.join("source.subpulse.json");
    command(&mut editor, "pulse_subpulse", json!({"op": "save_as", "component_id": component, "path": file.display().to_string()}));
    command(&mut editor, "pulse_config", json!({"op": "mode", "enabled": true}));
    command(&mut editor, "pulse_subpulse", json!({"op": "source", "component_id": component, "source": "config", "path": file.display().to_string()}));
    let error = |editor: &mut Editor| { editor.state(); editor.control_state()["config"]["error"].as_str().unwrap_or("").to_owned() };
    assert_eq!(error(&mut editor), "", "the file resolves");
    // The same stamp, other bytes: not read again, even for a new revision's view.
    let stamp = std::fs::metadata(&file).unwrap().modified().unwrap();
    let length = std::fs::metadata(&file).unwrap().len() as usize;
    std::fs::write(&file, " ".repeat(length)).unwrap();
    std::fs::File::options().write(true).open(&file).unwrap().set_modified(stamp).unwrap();
    edit(&mut editor, json!({"edit": "duration", "period_id": ids[0], "value": "3", "unit": "µs"}));
    assert_eq!(error(&mut editor), "", "an unchanged file is not read again");
    let source = zlc_pulse::codec::absolute(&file);
    assert!(!zlc_pulse::codec::changed_subpulses().contains(&source));
    // Changed: the poll finds it, and the view reads it again.
    std::fs::File::options().write(true).open(&file).unwrap().set_modified(stamp + std::time::Duration::from_secs(2)).unwrap();
    assert!(zlc_pulse::codec::changed_subpulses().contains(&source));
    editor.subpulse_files_changed();
    assert!(error(&mut editor).contains("Cannot resolve Subpulse"), "the changed file is read again");
}
