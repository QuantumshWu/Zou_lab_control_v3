//! The Pulse Editor (v2 `PulseEditorPresenter`): one authored pulse, its
//! pages and the board it runs on. Every intent builds a candidate and lets
//! the model accept or refuse it; a refusal is said and the previous pulse
//! kept. Board work is handed out as [`Job`]s for the host to run off its
//! window thread (`session.rs`); their outcomes come back through
//! [`Editor::deliver`]. The window renders [`Editor::state`] and nothing else.
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::{Map, Value, json};
use zlc_pulse::binding::{apply_config_values, field_label, prune_orphaned_bindings, pulse_field_value, resolve_api_parameters,
                         resolve_subpulse_sources, stale_subpulse_bindings};
use zlc_pulse::codec::{ConfigValues, EditorSection, absolute, atomic_write, document_to_tree, read_config_values, read_pulse_document,
                       read_subpulse, write_config_values, write_subpulse};
use zlc_pulse::compile::{CompiledProgram, compile_sequence};
use zlc_pulse::components::{extract_subpulse, group_component, insert_subpulse, remove_component, replace_component,
                            set_subpulse_source, ungroup_component};
use zlc_pulse::device::BoardDescription;
use zlc_pulse::manifest::target_from_config;
use zlc_pulse::model::{FieldKind, FieldRef, Period, PortKind, Sequence, SequenceParts, Source, Subpulse, Target};
use zlc_pulse::scan::{ScanTable, scan_columns_for, scan_table_for, scan_table_template};
use zlc_pulse::units::TIME_UNIT_CHOICES;
use zlc_pulse::wire::{StreamerParams, load_streamer_config, require_streamer_config};
use zlc_pulse::{Error, Kind, Result};

use crate::args::{py_str, py_truthy};
use crate::edits::{self, NEW_PULSE_PERIOD_NS, cleared, control_edit_arguments, new_pulse, port_everywhere, reordered_sequence};
use crate::items::{BRACKET, COMPONENT, Item, PERIOD, bracket_post_key, post_bracket, sequence_item_order, unique_id};
use crate::project::{Context, ScheduleRequest, empty_schedule, period_counts, project_board, project_schedule};
use crate::session::{BoardLink, Job};
use crate::target::{TargetRecord, carried_onto, programmable_ports, project_target, retarget_labels, target_from_records};
use crate::text::{SEPARATOR, counted, number, readable, sentence};

pub const PREVIEW_PAGE: &str = "Preview";
pub const SCAN_PAGE: &str = "Scan";
/// The pages, in the window's order.
pub const PAGES: [&str; 6] = ["Edit", "Preview", "Scan", "Value Config", "Subpulse Config", "Target"];
pub const CONNECTION_VIRTUAL: &str = "virtual";
pub const CONNECTION_REMOTE: &str = "remote";
pub const CONNECTION_OFFLINE: &str = "offline";
pub const CONNECTION_GIVEN: &str = "given";
/// The message kind of Sync's question before the board's pulse replaces
/// unsaved work (v2 confirm_pulse_discard): the window asks it, and a yes is
/// `pulse_sync {discard_changes: true}`.
pub const SYNC_DISCARD: &str = "sync_discard";
pub use zlc_pulse::codec::{CONFIG_VALUES_DIRECTORY, CURRENT_CONFIG_VALUES};

/// The deployed board as this computer's files describe it: what a pulse is
/// authored against and compiled for when no board is attached.
#[derive(Clone)]
pub struct Deployed { pub target: Arc<Target>, pub geometry: StreamerParams, pub clock_hz: f64 }

impl Deployed {
    pub fn load() -> Result<Deployed> {
        let found = load_streamer_config(None)?;
        let source = found.source.clone().ok_or_else(|| Error::new(Kind::Runtime,
            "no streamer config was found, so the deployed board geometry is unknown; refusing to compile against built-in defaults"))?;
        let config = require_streamer_config(&source)?;
        Ok(Deployed { target: target_from_config(&config, None, None)?, geometry: config.params, clock_hz: config.clock_hz })
    }
}

/// Everything authored in one editor, independent of its file path (`PulseEditorState`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EditorState { pub sequence: Option<Arc<Sequence>>, pub section: EditorSection }

/// One sentence for the operator: a refusal (`warning`) or a confirmation (`info`).
#[derive(Clone, Debug, PartialEq)]
pub struct Message { pub id: u64, pub kind: &'static str, pub text: String }

/// What a command answered (for a remote caller) beside the state it left.
pub type Answer = std::result::Result<Value, String>;

pub struct Editor {
    pub(crate) state: EditorState,
    pub(crate) saved: EditorState,
    pub path: String,
    pub(crate) pulses_directory: PathBuf,
    pub revision: u64,
    pub(crate) deployed: Option<Deployed>,
    // ---- the board (session.rs drives it)
    pub(crate) link: BoardLink,
    // ---- Value Config
    pub(crate) config_path: String,
    pub(crate) config_rows: Vec<[String; 3]>,
    pub(crate) config_saved_rows: Vec<[String; 3]>,
    pub(crate) config_loaded_path: String,
    pub(crate) config_loaded_values: ConfigValues,
    pub(crate) config_read_error: String,
    pub(crate) config_error: String,
    // ---- Subpulse Config
    pub(crate) component_context: String,
    pub(crate) selected_subpulse: String,
    pub(crate) subpulse_draft: Option<Subpulse>,
    pub(crate) subpulse_saved: Option<Subpulse>,
    pub(crate) subpulse_path: String,
    pub(crate) subpulse_visible: Option<BTreeSet<String>>,
    // ---- Scan, Target, Preview
    pub(crate) scan_progress: (String, &'static str),
    pub(crate) scan_running: bool,
    pub(crate) target_feedback: (String, &'static str),
    /// The revision the Target page was told it at: v2 says it until the page is next refreshed.
    pub(crate) target_feedback_at: u64,
    pub(crate) include_off: bool,
    pub(crate) pinned_size: Option<String>,
    pub(crate) selectors: bool,
    pub(crate) page: String,
    pub(crate) preview_status: (String, &'static str),
    // ---- words
    pub(crate) message: Option<Message>,
    pub(crate) messages: u64,
    pub last_warning: String,
    pub(crate) summary_note: Option<String>,
    pub(crate) jobs: Vec<Job>,
    /// Whether the command running came from the window (v2 `present`): a confirmation is then said in a dialog.
    pub(crate) operator: bool,
    /// The schedule's cards last sent, to take back the unchanged ones (`project::Cards`).
    cards: std::cell::RefCell<crate::project::Cards>,
}

fn arg<'a>(args: &'a Map<String, Value>, name: &str) -> std::result::Result<&'a Value, String> {
    args.get(name).ok_or_else(|| format!("missing argument {name}"))
}

fn text_arg(args: &Map<String, Value>, name: &str) -> std::result::Result<String, String> { arg(args, name).map(py_str) }

fn optional_text(args: &Map<String, Value>, name: &str) -> Option<String> {
    args.get(name).filter(|value| !value.is_null()).map(py_str).filter(|text| !text.is_empty())
}

fn flag(args: &Map<String, Value>, name: &str) -> bool { args.get(name).is_some_and(|value| value.as_bool() == Some(true)) }

impl Editor {
    pub fn new(deployed: Option<Deployed>, pulses_directory: PathBuf) -> Editor {
        Editor {
            state: EditorState::default(), saved: EditorState::default(), path: String::new(), pulses_directory, revision: 0, deployed,
            // v2 build(): the window offers where a pulse server usually listens while it dials its own board.
            link: BoardLink { endpoint: format!("{}:{}", zlc_pulse::remote::DEFAULT_HOST, zlc_pulse::remote::DEFAULT_PORT), ..BoardLink::default() },
            config_path: String::new(), config_rows: Vec::new(), config_saved_rows: Vec::new(),
            config_loaded_path: String::new(), config_loaded_values: Vec::new(), config_read_error: String::new(), config_error: String::new(),
            component_context: String::new(), selected_subpulse: String::new(), subpulse_draft: None, subpulse_saved: None,
            subpulse_path: String::new(), subpulse_visible: None, scan_progress: (String::new(), "task"), scan_running: false,
            target_feedback: (String::new(), "idle"), target_feedback_at: 0, include_off: false, pinned_size: None, selectors: true, page: "Edit".into(),
            preview_status: (String::new(), "task"), message: None, messages: 0, last_warning: String::new(), summary_note: None,
            jobs: Vec::new(), operator: false, cards: Default::default(),
        }
    }

    pub fn sequence(&self) -> Option<&Arc<Sequence>> { self.state.sequence.as_ref() }
    pub fn dirty(&self) -> bool { self.state != self.saved }
    pub fn page(&self) -> &str { &self.page }
    /// Whether the operator may drag on the preview (its Selectors switch).
    pub fn preview_selectors(&self) -> bool { self.selectors }
    /// Where a file written beside this pulse goes (the pulse's folder, else the workspace's pulses).
    pub fn state_folder(&self) -> String { self.save_folder().display().to_string() }

    // ------------------------------------------------------------- words
    pub fn warn(&mut self, text: impl Into<String>) {
        let text = text.into();
        self.last_warning = text.clone();
        self.messages += 1;
        self.message = Some(Message { id: self.messages, kind: "warning", text: sentence(&text) });
    }

    /// Sync's question (`SYNC_DISCARD`), for the window to ask.
    pub(crate) fn ask_sync_discard(&mut self) {
        self.messages += 1;
        self.message = Some(Message { id: self.messages, kind: SYNC_DISCARD, text: "Discard unsaved Pulse edits?".into() });
    }

    pub(crate) fn done(&mut self, text: impl Into<String>) {
        self.messages += 1;
        self.message = Some(Message { id: self.messages, kind: "info", text: sentence(&text.into()) });
    }

    // ------------------------------------------------------------- state
    /// Replace authoring atomically (`_accept_state`).
    pub(crate) fn accept_state(&mut self, candidate: EditorState) {
        if candidate == self.state { return; }
        if candidate.section.scan_rows != self.state.section.scan_rows { self.link.held_point = None; }
        self.state = candidate;
        self.revision += 1;
    }

    /// `_edit_state`: one change, with the scan table and the shown ports kept truthful.
    pub(crate) fn edit_state(&mut self, sequence: Option<Arc<Sequence>>, edit: impl FnOnce(&mut EditorSection)) {
        let mut candidate = self.state.clone();
        let before_rows = candidate.section.scan_rows.clone();
        edit(&mut candidate.section);
        let rows_changed = candidate.section.scan_rows != before_rows;
        if let Some(sequence) = &sequence {
            if !rows_changed && !candidate.section.scan_rows.is_empty() {
                let before: BTreeSet<(String, String)> = self.state.sequence.as_ref()
                    .map(|s| s.scan_bindings().iter().map(|b| (b.field_id(), b.unit.clone())).collect()).unwrap_or_default();
                let after: BTreeSet<(String, String)> = sequence.scan_bindings().iter().map(|b| (b.field_id(), b.unit.clone())).collect();
                if candidate.section.scan_fields.iter().any(|pair| before.contains(pair) && !after.contains(pair)) {
                    candidate.section.scan_rows.clear();
                    candidate.section.scan_fields.clear();
                    candidate.section.scan_source_dirty = !candidate.section.scan_source.is_empty();
                }
            }
            if let (Some(visible), Some(previous)) = (candidate.section.visible_ports.clone(), self.state.sequence.as_ref()) {
                candidate.section.visible_ports = Some(sequence.target.ports.iter()
                    .filter(|p| visible.contains(&p.key) || previous.target.port(&p.key).is_none()).map(|p| p.key.clone()).collect());
            }
            candidate.sequence = Some(sequence.clone());
        }
        if candidate.section.scan_rows.is_empty() { candidate.section.scan_fields.clear(); }
        self.accept_state(candidate);
    }

    pub(crate) fn visible(&self) -> Option<BTreeSet<String>> { self.state.section.visible_ports.as_ref().map(|ports| ports.iter().cloned().collect()) }

    /// The geometry and clock every compile and scan conversion uses: the
    /// attached board's, else the deployed board's files'; never defaults.
    pub(crate) fn compiler_target(&self) -> Result<(StreamerParams, f64)> {
        if let Some(facts) = &self.link.facts { return Ok((facts.description.geometry, facts.description.clock_hz)); }
        match &self.deployed {
            Some(deployed) => Ok((deployed.geometry, deployed.clock_hz)),
            None => Err(Error::new(Kind::Runtime, "no streamer config was found, so the deployed board geometry is unknown; \
                                                   refusing to compile against built-in defaults")),
        }
    }

    pub(crate) fn compile(&self, sequence: &Sequence) -> Result<CompiledProgram> {
        let (geometry, clock) = self.compiler_target()?;
        compile_sequence(sequence, &geometry, clock)
    }

    pub(crate) fn current_target(&self) -> Option<Arc<Target>> {
        self.state.sequence.as_ref().map(|s| s.target.clone()).or_else(|| self.link.board_target.clone())
    }

    pub(crate) fn base_directory(&self) -> Option<PathBuf> {
        if self.path.is_empty() { None } else { Path::new(&self.path).parent().map(Path::to_path_buf) }
    }

    pub(crate) fn save_folder(&self) -> PathBuf {
        self.base_directory().unwrap_or_else(|| if self.pulses_directory.as_os_str().is_empty() { std::env::current_dir().unwrap_or_default() }
                                              else { self.pulses_directory.clone() })
    }

    pub(crate) fn active_config_values(&self) -> ConfigValues {
        match &self.link.board { Some(board) => board.config().values(), None => self.config_loaded_values.clone() }
    }

    pub(crate) fn active_config_path(&self) -> String {
        match &self.link.board { Some(board) => board.config().source(), None => self.config_loaded_path.clone() }
    }

    /// `_scanned_fields`: the fields the armed table drives.
    pub(crate) fn scanned_fields(&self) -> BTreeSet<String> {
        if !self.scan_armed() { return BTreeSet::new(); }
        self.state.section.scan_fields.iter().map(|(field, _)| field.clone()).collect()
    }

    /// `_scan_armed`: a table AND a field it drives.
    pub(crate) fn scan_armed(&self) -> bool {
        let Some(sequence) = &self.state.sequence else { return false };
        if self.state.section.scan_rows.is_empty() { return false; }
        let held: BTreeSet<&str> = self.state.section.scan_fields.iter().map(|(field, _)| field.as_str()).collect();
        resolve_subpulse_sources(sequence, self.base_directory().as_deref())
            .is_ok_and(|resolved| resolved.scan_bindings().iter().any(|b| held.contains(b.field_id().as_str())))
    }

    /// `_played_scan`: Scan only on the fields the armed table drives.
    pub(crate) fn played_scan(&self, source: &Sequence) -> Result<Sequence> {
        let scanned = self.scanned_fields();
        if !source.bindings.iter().any(|b| b.scan && !scanned.contains(&b.field_id())) { return Ok(source.clone()); }
        source.with(|parts| for binding in &mut parts.bindings { if binding.scan && !scanned.contains(&binding.field_id()) { binding.scan = false; } })
    }

    /// `_effective_sequence`: what the pages show -- the saved Config values
    /// and Subpulse files in force, never the Config draft.
    pub(crate) fn effective_sequence(&self, sequence: &Sequence) -> Result<Sequence> {
        if sequence.config_enabled && !self.config_read_error.is_empty() { return Err(Error::new(Kind::Value, self.config_read_error.clone())); }
        let resolved = resolve_subpulse_sources(sequence, self.base_directory().as_deref())?;
        let base = self.played_scan(&resolved)?;
        let effective = apply_config_values(&resolved, &self.active_config_values(), Some(&base))?.0;
        if effective.bindings == resolved.bindings { return Ok(effective); }
        let bindings = resolved.bindings.clone();
        effective.with(|parts| parts.bindings = bindings)
    }

    /// `_execution_sequence`: the document On Pulse plays and the digest compares.
    pub(crate) fn execution_sequence(&self) -> Result<Sequence> {
        let sequence = self.state.sequence.as_ref().ok_or_else(|| Error::new(Kind::Runtime, "no pulse is open"))?;
        let resolved = resolve_subpulse_sources(sequence, self.base_directory().as_deref())?;
        self.played_scan(&resolve_api_parameters(&resolved, None)?)
    }

    // ------------------------------------------------------------- documents
    /// `start_new_pulse`: on the board this bench has.
    pub fn start_new_pulse(&mut self) -> bool {
        let target = self.link.board_target.clone().or_else(|| self.deployed.as_ref().map(|d| d.target.clone()));
        let step = self.link.board_step_ns.or_else(|| self.compiler_target().ok().map(|(_, clock)| 1e9 / clock));
        let candidate = match (target, step) {
            (Some(target), Some(step)) => new_pulse(target, step),
            _ => Err(Error::new(Kind::Runtime, "no streamer config was found, so the deployed board is unknown")),
        };
        match candidate {
            Ok(sequence) => {
                self.accept_state(EditorState { sequence: Some(Arc::new(sequence)), section: EditorSection::default() });
                self.path.clear();
                true
            }
            Err(error) => { self.warn(format!("Cannot start a pulse for this board: {error}")); false }
        }
    }

    /// `open_pulse(path)`.
    pub fn open_pulse(&mut self, path: &str, discard_changes: bool) -> Answer {
        if self.dirty() && !discard_changes {
            return Err("Pulse has unsaved changes; save it or explicitly pass discard_changes=true to replace the draft".into());
        }
        let location = absolute(Path::new(path));
        let (sequence, section) = read_pulse_document(&location).map_err(|error| format!("Cannot open {}: {error}", location.display()))?;
        let state = EditorState { sequence: Some(Arc::new(sequence)), section };
        self.accept_state(state.clone());
        self.revision += 1;
        self.saved = state;
        self.path = location.display().to_string();
        self.component_context.clear();
        self.selected_subpulse.clear();
        if let Some(facts) = self.link.facts.clone() { self.align_sequence_to(&facts.description); }
        Ok(json!({"path": self.path}))
    }

    /// A pulse a saved run played, as its board recorded it (a Figure's Devices
    /// row): an independent unsaved draft with no save path, the archive and the
    /// original file untouched (v2 `FigureViewerPresenter.open_pulse`).
    pub fn open_recorded(&mut self, document: &str, discard_changes: bool) -> Answer {
        if self.dirty() && !discard_changes {
            return Err("Pulse has unsaved changes; save it or explicitly pass discard_changes=true to replace the draft".into());
        }
        let (tree, section) = zlc_pulse::codec::parse_pulse_json(document).and_then(|tree| zlc_pulse::codec::split_document(&tree))
            .map_err(|error| format!("Cannot open the recorded pulse: {}", error.message))?;
        let sequence = zlc_pulse::codec::sequence_from_tree(&tree).map_err(|error| format!("Cannot open the recorded pulse: {}", error.message))?;
        self.accept_state(EditorState { sequence: Some(Arc::new(sequence)), section });
        self.revision += 1;
        self.saved = EditorState::default();
        self.path.clear();
        self.component_context.clear();
        self.selected_subpulse.clear();
        if let Some(facts) = self.link.facts.clone() { self.align_sequence_to(&facts.description); }
        Ok(json!({"path": null}))
    }

    /// `save_pulse(path)`: the draft as a `zlc.pulse` document; the file names the pulse.
    pub fn save_pulse(&mut self, path: &str) -> Answer {
        let Some(sequence) = self.state.sequence.clone() else { return Err("There is no pulse to save".into()) };
        // A name's own dots stay ("scan_1.5us"): a name not ending in .json gets it added.
        let mut target = PathBuf::from(path);
        if !target.extension().is_some_and(|e| e.eq_ignore_ascii_case("json")) { target.as_mut_os_string().push(".json"); }
        let target = absolute(&target);
        let stem = target.file_stem().map_or_else(String::new, |s| s.to_string_lossy().into_owned());
        if sequence.name != stem {
            let named = sequence.with(|parts| parts.name = stem.clone()).map_err(|e| e.message)?;
            self.edit_state(Some(Arc::new(named)), |_| {});
        }
        let sequence = self.state.sequence.clone().expect("open");
        if let Some(parent) = target.parent() { let _ = std::fs::create_dir_all(parent); }
        let text = document_to_tree(&sequence, &self.state.section).readable();
        atomic_write(&target, text.as_bytes()).map_err(|error| format!("Cannot save {}: {error}", target.display()))?;
        self.path = target.display().to_string();
        self.saved = self.state.clone();
        // v2 save_pulse: said to the operator (a dialog), else left in the summary.
        let said = format!("Saved {}", target.file_name().map_or_else(String::new, |n| n.to_string_lossy().into()));
        if self.operator { self.done(said); } else { self.summary_note = Some(said); }
        Ok(json!({"path": self.path}))
    }

    /// The file dialog a window command without a path asks (v2 ask_open_path /
    /// ask_save_path): its caption, its v2 filter, whether it saves, where it
    /// starts -- a folder, or the file a save proposes -- and the folder its kind
    /// keeps (pulses; value files in config_values), where a start that is gone
    /// opens.  None: no dialog.
    pub fn file_dialog(&self, action: &str, args: &Map<String, Value>) -> Option<(&'static str, &'static str, bool, PathBuf, PathBuf)> {
        const PULSE: &str = "ZLC Pulse (*.json);;All files (*)";
        const VALUES: &str = "Config (*.json)";
        const SUBPULSE: &str = "ZLC Subpulse (*.subpulse.json *.json);;All files (*)";
        if args.get("path").is_some_and(|path| !path.is_null()) { return None; }
        let op = args.get("op").and_then(Value::as_str).unwrap_or("");
        let pulses = self.pulses_directory.clone();
        let values = self.config_values_directory().unwrap_or_else(|| pulses.clone());
        let pulse_folder = self.base_directory().unwrap_or_else(|| pulses.clone());
        let (title, filter, save, start) = match (action, op) {
            ("pulse_open", _) => ("Open Pulse", PULSE, false, pulse_folder),
            ("pulse_save", _) => ("Save Pulse", PULSE, true, if self.path.is_empty() { self.proposed_save_path() } else { PathBuf::from(&self.path).with_extension("json") }),
            ("pulse_config", "load") => ("Open value file", VALUES, false, if self.config_path.is_empty() { values.clone() } else { PathBuf::from(&self.config_path) }),
            ("pulse_config", "save") if flag(args, "save_as") || self.config_path.is_empty() => ("Save value file", VALUES, true,
                if self.config_path.is_empty() { values.join(CURRENT_CONFIG_VALUES) } else { PathBuf::from(&self.config_path) }),
            ("pulse_scan", "load_array") => ("Open scan array", "Scan tables (*.npy *.csv *.txt);;All files (*)", false, pulse_folder),
            ("pulse_scan", "load_program") => ("Open scan program", "Python source (*.py);;All files (*)", false, pulse_folder),
            ("pulse_subpulse", "open") => ("Open Subpulse", SUBPULSE, false, if self.subpulse_path.is_empty() { pulses.clone() } else { PathBuf::from(&self.subpulse_path) }),
            ("pulse_subpulse", "insert_file") => ("Add subpulse", SUBPULSE, false, pulses.clone()),
            ("pulse_subpulse", "save_as" | "export") => {
                let key = optional_text(args, "component_id").unwrap_or_else(|| self.component_context.clone());
                let name = if key.is_empty() { self.subpulse_draft.as_ref().map(|draft| draft.to_sequence().name.clone()) }
                           else { self.state.sequence.as_ref().and_then(|s| s.components.iter().find(|c| c.component_id == key).map(|c| c.name.clone())) };
                ("Save Subpulse", SUBPULSE, true, pulses.join(format!("{}.subpulse.json", name.unwrap_or_else(|| "component".into()))))
            }
            ("pulse_subpulse", "source") if args.get("browse") == Some(&Value::Bool(true)) => {
                let key = optional_text(args, "component_id").unwrap_or_default();
                let current = self.state.sequence.as_ref().and_then(|s| s.components.iter().find(|c| c.component_id == key)).map(|c| c.config_path.clone()).unwrap_or_default();
                ("Choose Subpulse source", SUBPULSE, false, if current.is_empty() { pulses.clone() } else { PathBuf::from(current) })
            }
            _ => return None,
        };
        Some((title, filter, save, start, if action == "pulse_config" { values } else { pulses }))
    }

    /// The path a Save without one proposes (`<pulses>/<name>.json`).
    pub fn proposed_save_path(&self) -> PathBuf {
        if !self.path.is_empty() { return PathBuf::from(&self.path); }
        let name = self.state.sequence.as_ref().map_or("pulse", |s| s.name.as_str());
        self.pulses_directory.join(format!("{}.json", if name.is_empty() { "pulse" } else { name }))
    }

    /// `clear_all`: one safe tick on the same target, the path kept.
    pub fn clear_all(&mut self, discard_changes: bool) -> Answer {
        let Some(sequence) = self.state.sequence.clone() else { return Ok(Value::Null) };
        if self.dirty() && !discard_changes { return Err("Pulse has unsaved changes; pass discard_changes=true to clear it".into()); }
        let blank = cleared(&sequence).map_err(|e| e.message)?;
        let visible = self.state.section.visible_ports.clone();
        self.accept_state(EditorState { sequence: Some(Arc::new(blank)), section: EditorSection { visible_ports: visible, ..Default::default() } });
        Ok(Value::Null)
    }

    // ------------------------------------------------------------- edits
    /// Whether an edit would change file-backed Subpulse content or a value
    /// Config supplies (`_edited_candidate`'s guards while Config is on).
    fn guard_config(&self, action: &str, args: &[Value]) -> Result<()> {
        let Some(sequence) = &self.state.sequence else { return Ok(()) };
        if !sequence.config_enabled { return Ok(()); }
        let refuse = |message: &str| Err(Error::new(Kind::Value, message.to_owned()));
        let order = sequence_item_order(sequence);
        for component in sequence.components.iter().filter(|c| c.from_file()) {
            let mut members: BTreeSet<Item> = component.period_ids.iter().map(|id| (PERIOD.to_owned(), id.clone())).collect();
            for id in &component.bracket_ids {
                members.insert((BRACKET.to_owned(), bracket_post_key(id, "start")));
                members.insert((BRACKET.to_owned(), bracket_post_key(id, "end")));
            }
            let positions = |items: &[Item]| items.iter().enumerate().filter(|(_, item)| members.contains(*item)).map(|(i, _)| i).collect::<Vec<_>>();
            let refused = match action {
                "remove_items" => crate::args::items(&args[0])?.iter().any(|item| members.contains(item)),
                "insert_period" | "insert_spacer" if !args[0].is_null() => {
                    let before = crate::args::item(&args[0])?;
                    let at = positions(&order);
                    order.iter().position(|item| *item == before).is_some_and(|index| at.first().is_some_and(|first| *first < index)
                        && at.last().is_some_and(|last| index <= *last))
                }
                "reorder_items" => {
                    let edited = crate::args::items(&args[0])?;
                    let at = positions(&edited);
                    let inner = |items: &[Item]| items.iter().filter(|item| members.contains(*item)).cloned().collect::<Vec<_>>();
                    inner(&edited) != inner(&order) || (!at.is_empty() && at[at.len() - 1] - at[0] + 1 != at.len())
                }
                "bracket_add" => {
                    let ids = sequence.period_ids();
                    match (args[0].as_str().and_then(|a| ids.iter().position(|id| *id == a)), args[1].as_str().and_then(|b| ids.iter().position(|id| *id == b))) {
                        (Some(a), Some(b)) if a <= b => {
                            let covered: BTreeSet<&str> = ids[a..=b].iter().copied().collect();
                            let inside = component.period_ids.iter().any(|id| covered.contains(id.as_str()));
                            inside && !component.period_ids.iter().all(|id| covered.contains(id.as_str()))
                        }
                        _ => false,
                    }
                }
                _ => false,
            };
            if refused { return refuse("This Subpulse comes from a file. Open its source, or switch to Default to edit its contents."); }
        }
        if matches!(action, "duration" | "analog" | "digital" | "period_name" | "binding" | "bracket") {
            let id = py_str(&args[if action == "binding" { 1 } else { 0 }]);
            let component = if action == "bracket" || (action == "binding" && args[0].as_str() == Some("bracket_count")) {
                sequence.component_for_field(&FieldRef::bracket_count(&id))
            } else { sequence.component_for_period(&id) };
            if component.is_some_and(|c| c.from_file()) {
                return refuse("This Subpulse comes from a file. Open its source, or switch to Default to edit the embedded content.");
            }
        }
        if matches!(action, "duration" | "analog" | "delay" | "bracket") {
            let first = py_str(&args[0]);
            let reference = match action {
                "delay" => FieldRef::delay(&first),
                "duration" => FieldRef::duration(&first),
                "bracket" => FieldRef::bracket_count(&first),
                _ => FieldRef::dac(&first, &py_str(&args[1])),
            };
            let active = self.active_config_values();
            let scanned = self.scanned_fields();
            if sequence.config_bindings().iter().any(|b| b.field_ref == reference && active.iter().any(|(key, _)| *key == b.config_key)
                && !scanned.contains(&b.field_id())) {
                return refuse("This value is supplied by Config. Edit the Config file, or switch to Default.");
            }
        }
        Ok(())
    }

    /// One authoring edit (`_edited_candidate` + `_apply`).
    pub fn edit(&mut self, action: &str, args: &[Value], default_edit: bool) -> Answer {
        let Some(sequence) = self.state.sequence.clone() else { return Err("open a pulse before editing it".into()) };
        if !default_edit { self.guard_config(action, args).map_err(|e| e.message)?; }
        let (candidate, dropped) = edits::edited(&sequence, action, args).map_err(|e| e.message)?;
        if !dropped.is_empty() { self.warn(format!("Unbound {}: the field is no longer set here", dropped.join(", "))); }
        self.edit_state(Some(Arc::new(candidate)), |_| {});
        Ok(Value::Null)
    }

    /// A shape change built from the sequence's parts (`_rebuilt` + `_apply`).
    fn rebuild(&mut self, edit: impl FnOnce(&mut SequenceParts)) -> Answer {
        let Some(sequence) = self.state.sequence.clone() else { return Err("open a pulse before editing it".into()) };
        let candidate = sequence.with(edit).map_err(|e| e.message)?;
        let (pruned, dropped) = prune_orphaned_bindings(&candidate).map_err(|e| e.message)?;
        if !dropped.is_empty() { self.warn(format!("Unbound {}: the field it named is no longer set here", dropped.join(", "))); }
        self.edit_state(Some(Arc::new(pruned)), |_| {});
        Ok(Value::Null)
    }

    /// `pulse_edit {edit, ...}`: the v2 control vocabulary plus the window's own gestures.
    pub fn pulse_edit(&mut self, args: &Map<String, Value>) -> Answer {
        let edit = text_arg(args, "edit")?;
        match edit.as_str() {
            "visible_ports" => {
                let ports = args.get("ports").cloned().unwrap_or(Value::Null);
                let visible = if ports.is_null() { None } else {
                    Some(ports.as_array().ok_or("ports must be a list of port keys or null")?.iter().map(py_str).collect::<Vec<_>>())
                };
                self.edit_state(None, |section| section.visible_ports = visible);
                Ok(Value::Null)
            }
            "fill_port" | "clear_port" => {
                let port = text_arg(args, "port_key")?;
                let Some(sequence) = self.state.sequence.clone() else { return Err("open a pulse before editing it".into()) };
                let active = self.active_config_values();
                if sequence.config_enabled && (sequence.components.iter().any(|c| c.from_file())
                    || sequence.config_bindings().iter().any(|b| b.kind() == FieldKind::Dac && b.field_ref.port.as_deref() == Some(&port)
                        && active.iter().any(|(key, _)| *key == b.config_key))) {
                    return Err("This channel includes Config-supplied content. Switch to Default before editing the whole channel.".into());
                }
                let candidate = port_everywhere(&sequence, &port, edit == "fill_port").map_err(|e| e.message)?;
                let (pruned, dropped) = prune_orphaned_bindings(&candidate).map_err(|e| e.message)?;
                if !dropped.is_empty() { self.warn(format!("Unbound {}: the field it named is no longer set here", dropped.join(", "))); }
                self.edit_state(Some(Arc::new(pruned)), |_| {});
                Ok(Value::Null)
            }
            "port_label" => {
                let (key, label) = (text_arg(args, "port_key")?, text_arg(args, "label")?);
                let Some(target) = self.state.sequence.as_ref().map(|s| s.target.clone()) else { return Err("open a pulse before editing it".into()) };
                match retarget_labels(&target, &[(key, label)]).map_err(|e| e.message)? {
                    Some(renamed) => self.rebuild(|parts| parts.target = renamed),
                    None => Ok(Value::Null),
                }
            }
            "scan_repeats" => {
                let repeats = arg(args, "repeats")?.as_u64().ok_or("repeats must be a non-negative integer")?;
                self.edit_state(None, |section| section.scan_repeats = repeats);
                Ok(Value::Null)
            }
            "run_repeats" if self.state.sequence.as_ref().is_some_and(|s| Some(s.run_repeats) == args.get("repeats").and_then(Value::as_u64)) => Ok(Value::Null),
            _ => {
                let (action, positional) = control_edit_arguments(&edit, args).map_err(|e| e.message)?;
                if (action == "insert_period") && self.state.sequence.is_none() {
                    return if self.start_new_pulse() { Ok(Value::Null) } else { Err(self.last_warning.clone()) };
                }
                if action == "insert_spacer" && self.state.sequence.is_none() { return Err("Add a period first: a spacer is time between periods".into()); }
                let positional = if action == "reorder_items" { vec![self.authored_items(&positional[0])?] } else { positional };
                self.edit(&action, &positional, false)
            }
        }
    }

    /// `_authored_items`: a displayed file-backed block back to its embedded default block.
    fn authored_items(&self, items: &Value) -> std::result::Result<Value, String> {
        let Some(sequence) = &self.state.sequence else { return Ok(items.clone()) };
        if !sequence.config_enabled { return Ok(items.clone()); }
        let Ok(effective) = self.effective_sequence(sequence) else { return Ok(items.clone()) };
        let order = sequence_item_order(sequence);
        let parsed = crate::args::items(items).map_err(|e| e.message)?;
        let mut owners: Vec<(Item, String)> = Vec::new();
        let mut defaults: Vec<(String, Vec<Item>)> = Vec::new();
        for component in sequence.components.iter().filter(|c| c.from_file()) {
            let Some(shown) = effective.components.iter().find(|c| c.component_id == component.component_id) else { continue };
            defaults.push((component.component_id.clone(), order.iter().filter(|(kind, key)| (kind == PERIOD && component.period_ids.contains(key))
                || (kind == BRACKET && component.bracket_ids.iter().any(|id| id == post_bracket(key)))).cloned().collect()));
            owners.extend(shown.period_ids.iter().map(|id| ((PERIOD.to_owned(), id.clone()), component.component_id.clone())));
            for id in &shown.bracket_ids {
                for side in ["start", "end"] { owners.push(((BRACKET.to_owned(), bracket_post_key(id, side)), component.component_id.clone())); }
            }
        }
        let mut result = Vec::new();
        let mut seen = BTreeSet::new();
        for item in parsed {
            match owners.iter().find(|(owned, _)| *owned == item) {
                None => result.push(item),
                Some((_, key)) => if seen.insert(key.clone()) { result.extend(defaults.iter().find(|(id, _)| id == key).map(|(_, items)| items.clone()).unwrap_or_default()); },
            }
        }
        Ok(Value::Array(result.into_iter().map(|(k, v)| json!([k, v])).collect()))
    }

    // ------------------------------------------------------------- Value Config
    pub(crate) fn config_dirty(&self) -> bool { self.config_rows != self.config_saved_rows }

    pub(crate) fn rows_of(values: &ConfigValues) -> Vec<[String; 3]> {
        values.iter().map(|(key, (value, unit))| [key.clone(), number(*value), unit.clone()]).collect()
    }

    pub(crate) fn config_values_directory(&self) -> Option<PathBuf> {
        let base = self.base_directory().unwrap_or_else(|| self.pulses_directory.clone());
        if base.as_os_str().is_empty() { return None; }
        Some(base.parent().map_or_else(|| base.clone(), Path::to_path_buf).join(CONFIG_VALUES_DIRECTORY))
    }

    /// `_config_file_operation`: write and/or activate a value file (inline:
    /// a small local file); the draft follows when `load_draft`.
    fn config_file(&mut self, path: &str, entries: Option<&ConfigValues>, load_draft: bool, activate: bool) -> Answer {
        let result = (|| -> Result<ConfigValues> {
            if let Some(entries) = entries { write_config_values(Path::new(path), entries)?; }
            if activate {
                match &self.link.board {
                    Some(board) => { board.config().load_file(Some(Path::new(path)))?; Ok(board.config().values()) }
                    None => if path.is_empty() { Ok(Vec::new()) } else { read_config_values(Path::new(path)) },
                }
            } else if path.is_empty() { Ok(Vec::new()) } else { read_config_values(Path::new(path)) }
        })();
        let values = match result {
            Ok(values) => values,
            Err(error) => {
                if activate { self.config_read_error = error.message.clone(); }
                return Err(format!("Cannot update Config: {}", error.message));
            }
        };
        if activate {
            self.config_read_error.clear();
            self.config_loaded_path = path.to_owned();
            self.config_loaded_values = values.clone();
        }
        if load_draft {
            self.config_path = path.to_owned();
            self.config_rows = Self::rows_of(&values);
            self.config_saved_rows = self.config_rows.clone();
        }
        self.revision += 1;
        Ok(json!({"path": path}))
    }

    /// `pulse_config {action, ...}`: the Value Config page.
    pub fn pulse_config(&mut self, args: &Map<String, Value>) -> Answer {
        let action = text_arg(args, "op")?;
        let discard = flag(args, "discard_changes");
        match action.as_str() {
            "edit" => {
                let rows = arg(args, "rows")?.as_array().ok_or("rows must be [[name, value, unit], ...]")?;
                let rows: Vec<[String; 3]> = rows.iter().map(|row| match row.as_array().map(Vec::as_slice) {
                    Some([a, b, c]) => Ok([py_str(a), if b.is_null() { String::new() } else { py_str(b) }, py_str(c)]),
                    _ => Err("Config rows must contain name, value and unit".to_owned()),
                }).collect::<std::result::Result<_, _>>()?;
                if rows != self.config_rows { self.config_rows = rows; self.revision += 1; }
                Ok(Value::Null)
            }
            "new" => {
                if self.config_dirty() && !discard { return Err("Config has unsaved changes; save it or explicitly pass discard_changes=true to replace the draft".into()); }
                if !self.config_path.is_empty() || !self.config_rows.is_empty() { self.revision += 1; }
                self.config_path.clear();
                self.config_rows.clear();
                self.config_saved_rows.clear();
                Ok(Value::Null)
            }
            "load" => {
                if self.config_dirty() && !discard { return Err("Config has unsaved changes; save it or explicitly pass discard_changes=true to replace the draft".into()); }
                let path = optional_text(args, "path").ok_or("config.load requires an explicit path")?;
                let path = absolute(Path::new(&path)).display().to_string();
                self.config_file(&path, None, true, true)
            }
            "refresh" => {
                if self.config_path.is_empty() { return Ok(Value::Null); }
                if self.config_dirty() && !discard { return Err("Config has unsaved changes; save it or explicitly pass discard_changes=true to replace the draft".into()); }
                let path = self.config_path.clone();
                let activate = path == self.active_config_path();
                self.config_file(&path, None, true, activate)
            }
            "unload" => self.config_file("", None, false, true),
            "save" => {
                let mut entries: ConfigValues = Vec::new();
                for (index, [name, text, unit]) in self.config_rows.clone().iter().enumerate() {
                    let key = name.trim().to_owned();
                    if key.is_empty() && text.trim().is_empty() && unit.trim().is_empty() { continue; }
                    if key.is_empty() { return Err(format!("Config row {} has no name", index + 1)); }
                    if entries.iter().any(|(existing, _)| *existing == key) { return Err(format!("Duplicate Config name: {key}")); }
                    let value = crate::args::py_float(&Value::String(text.clone())).map_err(|_| format!("Config {key}: value must be a number"))?;
                    entries.push((key, (value, unit.trim().to_owned())));
                }
                let target = optional_text(args, "path").or_else(|| (!self.config_path.is_empty()).then(|| self.config_path.clone()))
                    .ok_or("save_config_values requires an explicit path for this file")?;
                let mut target = PathBuf::from(target);
                target.set_extension("json");
                let target = absolute(&target).display().to_string();
                self.config_file(&target, Some(&entries), true, true)
            }
            "mode" => {
                let enabled = arg(args, "enabled")?.as_bool().ok_or("enabled must be boolean")?;
                let Some(sequence) = self.state.sequence.clone() else { return Err("open a pulse first".into()) };
                let toggled = sequence.with(|parts| parts.config_enabled = enabled).map_err(|e| e.message)?;
                self.edit_state(Some(Arc::new(toggled)), |_| {});
                self.config_error.clear();
                if enabled && !self.active_config_path().is_empty() {
                    let path = self.active_config_path();
                    return self.config_file(&path, None, false, true);
                }
                Ok(Value::Null)
            }
            "binding" => self.edit("config_binding", &[arg(args, "field_id")?.clone(), arg(args, "key")?.clone()], false),
            other => Err(format!("unknown Config action {other:?}")),
        }
    }

    // ------------------------------------------------------------- Target
    /// `apply_target(records)`: attached, only names change; offline the
    /// records are the target and the pulse is carried output by output.
    pub fn apply_target(&mut self, records: &[TargetRecord]) -> Answer {
        let Some(target) = self.current_target() else {
            self.target_feedback_at = self.revision; self.target_feedback = ("there is no target to apply to".into(), "error");
            return Err("there is no target to apply to".into());
        };
        let (changed, feedback) = if self.link.facts.is_some() {
            let wanted: Vec<(String, String)> = records.iter().map(|r| (r.key.clone(), r.signal.trim().to_owned())).collect();
            let keys: BTreeSet<&str> = programmable_ports(&target).into_iter().filter(|p| p.kind != PortKind::Clock).map(|p| p.key.as_str()).collect();
            if wanted.iter().map(|(key, _)| key.as_str()).collect::<BTreeSet<_>>() != keys {
                let message = "this target is the attached board's; ports cannot be added or removed here, only renamed";
                self.target_feedback_at = self.revision; self.target_feedback = (message.into(), "error");
                return Err(message.into());
            }
            match retarget_labels(&target, &wanted).map_err(|e| e.message)? {
                None => { self.target_feedback_at = self.revision; self.target_feedback = ("nothing to rename".into(), "idle"); return Ok(Value::Null); }
                Some(renamed) => (renamed, format!("renamed {}", counted(wanted.iter().filter(|(_, name)| !name.is_empty()).count(), "output"))),
            }
        } else {
            let rewired = match target_from_records(&target, records) {
                Ok(rewired) => rewired,
                Err(error) => { self.target_feedback_at = self.revision; self.target_feedback = (error.message.clone(), "error"); return Err(error.message); }
            };
            if *rewired == *target && rewired.package_pins.iter().collect::<BTreeSet<_>>() == target.package_pins.iter().collect::<BTreeSet<_>>() {
                self.target_feedback_at = self.revision; self.target_feedback = ("nothing to change".into(), "idle");
                return Ok(Value::Null);
            }
            (rewired, format!("applied {}", counted(records.len(), "output")))
        };
        match self.state.sequence.clone() {
            Some(sequence) => {
                let carried = if self.link.facts.is_none() { Some(carried_onto(&sequence, &changed)) } else { None };
                let carried = match carried.transpose() {
                    Ok(carried) => carried,
                    Err(error) => { self.target_feedback_at = self.revision; self.target_feedback = (error.message.clone(), "error"); return Err(error.message); }
                };
                if let Err(error) = self.rebuild(|parts| {
                    parts.target = changed.clone();
                    if let Some((periods, delays)) = carried { parts.periods = periods; parts.delays = delays; }
                }) { self.target_feedback_at = self.revision; self.target_feedback = (error.clone(), "error"); return Err(error); }
            }
            None => { self.link.board_target = Some(changed); self.revision += 1; }
        }
        self.target_feedback_at = self.revision; self.target_feedback = (feedback, "task");
        Ok(Value::Null)
    }

    // ------------------------------------------------------------- Subpulse Config
    fn selected_component_fragment(&self) -> Option<Subpulse> {
        let sequence = self.state.sequence.as_ref()?;
        if !self.component_context.is_empty() { return extract_subpulse(sequence, &self.component_context).ok(); }
        let selected = sequence.components.iter().find(|c| c.component_id == self.selected_subpulse);
        if selected.is_some_and(|c| c.config_path.is_empty() || c.config_path != self.subpulse_path) { return None; }
        self.subpulse_draft.clone()
    }

    fn select_subpulse(&mut self, key: &str, discard: bool) -> Result<()> {
        let component = self.state.sequence.as_ref().and_then(|s| s.components.iter().find(|c| c.component_id == key)).cloned();
        if !key.is_empty() && component.is_none() { return Err(Error::new(Kind::Value, "that Subpulse no longer exists in this Pulse")); }
        match component {
            Some(component) if component.source == Source::Config => {
                if !component.config_path.is_empty() && component.config_path != self.subpulse_path {
                    if self.subpulse_draft != self.subpulse_saved && !discard {
                        return Err(Error::new(Kind::Value, "Subpulse has unsaved changes; save it or explicitly pass discard_changes=true to replace the file draft"));
                    }
                    self.selected_subpulse = key.to_owned();
                    self.component_context.clear();
                    let loaded = read_subpulse(Path::new(&component.config_path))?;
                    self.subpulse_draft = Some(loaded.clone());
                    self.subpulse_saved = Some(loaded);
                    self.subpulse_path = component.config_path.clone();
                    self.subpulse_visible = None;
                }
                self.selected_subpulse = key.to_owned();
                self.component_context.clear();
            }
            _ => { self.selected_subpulse = key.to_owned(); self.component_context = key.to_owned(); }
        }
        Ok(())
    }

    /// `pulse_subpulse {action, ...}`: the Subpulse Config page and grouping on Edit.
    pub fn pulse_subpulse(&mut self, args: &Map<String, Value>) -> Answer {
        let action = text_arg(args, "op")?;
        let before = (self.selected_subpulse.clone(), self.component_context.clone(), self.subpulse_draft.clone(), self.subpulse_path.clone());
        let revision = self.revision;
        let result = self.subpulse_action(&action, args).map_err(|error| format!("Cannot {action} Component: {}", error.message));
        if (self.selected_subpulse.clone(), self.component_context.clone(), self.subpulse_draft.clone(), self.subpulse_path.clone()) != before
            && self.revision == revision { self.revision += 1; }
        result
    }

    fn subpulse_action(&mut self, action: &str, args: &Map<String, Value>) -> Result<Value> {
        let discard = flag(args, "discard_changes");
        let key = optional_text(args, "component_id").unwrap_or_default();
        let path = optional_text(args, "path");
        let value_error = |message: &str| Error::new(Kind::Value, message.to_owned());
        let sequence = self.state.sequence.clone();
        match action {
            "source" => {
                let sequence = sequence.ok_or_else(|| value_error("open a Pulse before assigning a Subpulse source"))?;
                let current = sequence.components.iter().find(|c| c.component_id == key).ok_or_else(|| value_error("no such Subpulse"))?.clone();
                let source = optional_text(args, "source").unwrap_or_else(|| current.source.name().to_owned());
                let path = match args.get("path") {
                    None => current.config_path.clone(),
                    Some(Value::Null) => current.config_path.clone(),
                    Some(value) => {
                        let text = py_str(value);
                        if text.is_empty() { String::new() } else {
                            let location = PathBuf::from(&text);
                            let location = if location.is_absolute() { location } else { self.save_folder().join(location) };
                            absolute(&location).display().to_string()
                        }
                    }
                };
                let editing = self.selected_subpulse == key;
                if editing && source == "config" && !path.is_empty() && path != self.subpulse_path && self.subpulse_draft != self.subpulse_saved && !discard {
                    return Err(value_error("Subpulse has unsaved changes; save it or explicitly pass discard_changes=true to replace the file draft"));
                }
                let candidate = set_subpulse_source(&sequence, &key, &source, Some(&path))?;
                self.edit_state(Some(Arc::new(candidate)), |_| {});
                if editing { self.select_subpulse(&key, true)?; }
            }
            "open_source" => {
                let sequence = sequence.ok_or_else(|| value_error("open a Pulse first"))?;
                let component = sequence.components.iter().find(|c| c.component_id == key).ok_or_else(|| value_error("no such Subpulse"))?;
                if component.config_path.is_empty() { return Err(value_error("choose a Subpulse source file first")); }
                if self.subpulse_draft != self.subpulse_saved && !discard {
                    return Err(value_error("Subpulse has unsaved changes; save it or explicitly pass discard_changes=true to replace the file draft"));
                }
                let loaded = read_subpulse(Path::new(&component.config_path))?;
                self.subpulse_draft = Some(loaded.clone());
                self.subpulse_saved = Some(loaded);
                self.subpulse_path = component.config_path.clone();
                self.selected_subpulse = component.component_id.clone();
                self.component_context.clear();
                self.subpulse_visible = None;
            }
            "select" | "context" => self.select_subpulse(&key, discard)?,
            "new" | "open" => {
                if self.subpulse_draft != self.subpulse_saved && !discard {
                    return Err(value_error("Subpulse has unsaved changes; save it or explicitly pass discard_changes=true to replace the file draft"));
                }
                if action == "open" {
                    let chosen = path.ok_or_else(|| value_error("supply path explicitly when opening a Subpulse"))?;
                    let chosen = absolute(Path::new(&chosen));
                    let loaded = read_subpulse(&chosen)?;
                    self.subpulse_draft = Some(loaded.clone());
                    self.subpulse_saved = Some(loaded);
                    self.subpulse_path = chosen.display().to_string();
                } else {
                    let target = self.current_target().or_else(|| self.deployed.as_ref().map(|d| d.target.clone()))
                        .ok_or_else(|| value_error("no board target is known"))?;
                    let step = sequence.as_ref().map(|s| s.time_step_ns).or(self.link.board_step_ns)
                        .or_else(|| self.compiler_target().ok().map(|(_, clock)| 1e9 / clock)).ok_or_else(|| value_error("no board clock is known"))?;
                    let period = Period::new("period1", zlc_pulse::num::Num::new(NEW_PULSE_PERIOD_NS, "duration")?, "ns", vec![0; target.raw_lanes.len()],
                                             Vec::new(), "", "period")?;
                    self.subpulse_draft = Some(Subpulse::new("component", target, step, vec![period], Vec::new(), Vec::new())?);
                    self.subpulse_saved = None;
                    self.subpulse_path.clear();
                }
                self.component_context.clear();
                self.selected_subpulse.clear();
                self.subpulse_visible = None;
            }
            "save" | "save_as" | "export" => {
                if action == "save" && !self.component_context.is_empty() {
                    let target = path.unwrap_or_else(|| self.path.clone());
                    if target.is_empty() { return Err(value_error("supply path explicitly when saving the Pulse")); }
                    self.save_pulse(&target).map_err(|message| Error::new(Kind::Value, message))?;
                    return Ok(Value::Null);
                }
                let context = if action == "export" { if key.is_empty() { self.component_context.clone() } else { key.clone() } } else { self.component_context.clone() };
                let fragment = if context.is_empty() { self.selected_component_fragment() } else {
                    Some(extract_subpulse(sequence.as_ref().ok_or_else(|| value_error("open a Pulse first"))?, &context)?)
                }.ok_or_else(|| value_error("create or open a Subpulse first"))?;
                let target = path.or_else(|| (context.is_empty() && action == "save" && !self.subpulse_path.is_empty()).then(|| self.subpulse_path.clone()))
                    .ok_or_else(|| value_error("supply path explicitly when saving or exporting a Subpulse"))?;
                let mut target = absolute(Path::new(&target));
                if !target.extension().is_some_and(|e| e.eq_ignore_ascii_case("json")) { target.as_mut_os_string().push(".subpulse.json"); }
                if let Some(parent) = target.parent() { let _ = std::fs::create_dir_all(parent); }
                write_subpulse(&target, &fragment)?;
                if context.is_empty() {
                    self.subpulse_path = target.display().to_string();
                    self.subpulse_saved = Some(fragment);
                    if !self.selected_subpulse.is_empty() {
                        if let Some(sequence) = &self.state.sequence {
                            let selected = sequence.components.iter().find(|c| c.component_id == self.selected_subpulse).cloned();
                            if let Some(selected) = selected {
                                let candidate = set_subpulse_source(sequence, &selected.component_id, selected.source.name(), Some(&self.subpulse_path))?;
                                self.edit_state(Some(Arc::new(candidate)), |_| {});
                            }
                        }
                    }
                }
                let said = format!("Saved Subpulse {}", target.file_name().map_or_else(String::new, |n| n.to_string_lossy().into()));
                if self.operator { self.done(said); } else { self.summary_note = Some(said); }
            }
            "insert" | "insert_file" => {
                let (fragment, chosen) = if action == "insert_file" {
                    let chosen = absolute(Path::new(&path.ok_or_else(|| value_error("supply path explicitly when inserting a Subpulse file"))?));
                    (Some(read_subpulse(&chosen)?), Some(chosen.display().to_string()))
                } else { (self.selected_component_fragment(), None) };
                let fragment = fragment.ok_or_else(|| value_error("create or open a Subpulse first"))?;
                let old: BTreeSet<String> = sequence.as_ref().map(|s| s.components.iter().map(|c| c.component_id.clone()).collect()).unwrap_or_default();
                let order = sequence.as_ref().map(|s| sequence_item_order(s)).unwrap_or_default();
                let before = match args.get("before_item") { Some(value) if !value.is_null() => Some(crate::args::item(value)?), _ => None };
                let position = match &before { None => order.len(), Some(item) => order.iter().position(|o| o == item).ok_or_else(|| value_error("no such timeline item"))? };
                let candidate = insert_subpulse(sequence.as_deref(), &fragment, None, None)?;
                let added = candidate.components.iter().find(|c| !old.contains(&c.component_id)).expect("inserted").clone();
                let block = sequence_item_order(extract_subpulse(&candidate, &added.component_id)?.to_sequence());
                let mut items: Vec<Item> = order[..position].to_vec();
                items.extend(block);
                items.extend(order[position..].iter().cloned());
                let mut candidate = reordered_sequence(&candidate, &items, None)?;
                match &chosen {
                    Some(chosen) => candidate = set_subpulse_source(&candidate, &added.component_id, "config", Some(chosen))?,
                    None => { self.component_context = added.component_id.clone(); self.selected_subpulse = added.component_id.clone(); }
                }
                self.edit_state(Some(Arc::new(candidate)), |_| {});
            }
            "group" => {
                let sequence = sequence.ok_or_else(|| value_error("open a Pulse before grouping its periods"))?;
                let selected: BTreeSet<Item> = crate::args::items(args.get("items").unwrap_or(&Value::Null))?.into_iter().collect();
                if selected.iter().any(|(kind, _)| kind == COMPONENT) { return Err(value_error("ungroup existing components before making a new group")); }
                let ids: Vec<String> = sequence.periods.iter().filter(|p| selected.contains(&(PERIOD.to_owned(), p.period_id.clone()))).map(|p| p.period_id.clone()).collect();
                if ids.is_empty() { return Err(value_error("select the periods to group; Shift-click adds or removes individual items")); }
                let order = sequence_item_order(&sequence);
                let positions: Vec<usize> = order.iter().enumerate().filter(|(_, item)| selected.contains(*item)).map(|(i, _)| i).collect();
                if order[positions[0]..=positions[positions.len() - 1]].iter().cloned().collect::<BTreeSet<_>>() != selected {
                    return Err(value_error("select consecutive complete items; no unselected periods or brackets will be added"));
                }
                let brackets: Vec<String> = sequence.brackets.iter().filter(|b| selected.contains(&(BRACKET.to_owned(), bracket_post_key(&b.bracket_id, "start")))
                    && selected.contains(&(BRACKET.to_owned(), bracket_post_key(&b.bracket_id, "end")))).map(|b| b.bracket_id.clone()).collect();
                if selected.iter().any(|(kind, key)| kind == BRACKET && !brackets.iter().any(|id| id == post_bracket(key))) {
                    return Err(value_error("select both ends and the complete contents of each selected bracket"));
                }
                let name = unique_id(sequence.components.iter().map(|c| c.name.as_str()), "Component ");
                let candidate = group_component(&sequence, &ids, &name, None, &brackets)?;
                self.edit_state(Some(Arc::new(candidate)), |_| {});
            }
            "ungroup" | "remove" => {
                let sequence = sequence.ok_or_else(|| value_error("open a Pulse first"))?;
                let candidate = if action == "ungroup" { ungroup_component(&sequence, &key)? } else { remove_component(&sequence, &key)? };
                self.edit_state(Some(Arc::new(candidate)), |_| {});
            }
            "rename" => {
                let name = optional_text(args, "name").unwrap_or_default();
                let target = if key.is_empty() { self.component_context.clone() } else { key.clone() };
                if !target.is_empty() {
                    let sequence = sequence.ok_or_else(|| value_error("open a Pulse first"))?;
                    let candidate = sequence.with(|parts| for c in &mut parts.components { if c.component_id == target { c.name = name.clone(); } })?;
                    self.edit_state(Some(Arc::new(candidate)), |_| {});
                } else if let Some(draft) = self.subpulse_draft.clone() {
                    if self.selected_component_fragment().is_none() { return Err(value_error("open the selected Subpulse source before editing it")); }
                    let inner = draft.to_sequence();
                    self.subpulse_draft = Some(Subpulse::new(&name, inner.target.clone(), inner.time_step_ns, inner.periods.clone(), inner.bindings.clone(),
                                                             inner.brackets.clone())?);
                }
            }
            "edit" => { self.edit_component(args)?; }
            other => return Err(value_error(&format!("unknown Subpulse action {other:?}"))),
        }
        Ok(Value::Null)
    }

    /// `_edit_component_contents`: one edit of a Subpulse owner -- the one named
    /// (an Add in an expanded Subpulse on the Edit page), else the selected one.
    fn edit_component(&mut self, args: &Map<String, Value>) -> Result<()> {
        let context = optional_text(args, "component_id").unwrap_or_else(|| self.component_context.clone());
        let edit = text_arg(args, "edit").map_err(|m| Error::new(Kind::Value, m))?;
        if matches!(edit.as_str(), "delay" | "run_repeats" | "scan_repeats") {
            return Err(Error::new(Kind::Value, "delays and repeat counts belong to the main Pulse"));
        }
        if !context.is_empty() && self.state.sequence.as_ref().is_some_and(|s| s.config_enabled && s.components.iter().any(|c| c.component_id == context && c.from_file())) {
            return Err(Error::new(Kind::Value, "This Subpulse comes from a file. Open the source file to edit it."));
        }
        if edit == "visible_ports" {
            let ports = args.get("ports").and_then(Value::as_array).map(|items| items.iter().map(py_str).collect::<Vec<_>>());
            if context.is_empty() { self.subpulse_visible = ports.map(|p| p.into_iter().collect()); }
            else { self.edit_state(None, |section| section.visible_ports = ports); }
            return Ok(());
        }
        let (action, positional) = control_edit_arguments(&edit, args)?;
        if !context.is_empty() {
            if action == "document_name" {
                let mut renamed = args.clone();
                renamed.insert("component_id".into(), json!(context));
                renamed.insert("name".into(), positional[0].clone());
                self.subpulse_action("rename", &renamed)?;
                return Ok(());
            }
            if matches!(action.as_str(), "period_name" | "duration" | "digital" | "analog" | "binding" | "config_binding") {
                return self.edit(&action, &positional, true).map(|_| ()).map_err(|m| Error::new(Kind::Value, m));
            }
        }
        let fragment = if context.is_empty() { self.selected_component_fragment() } else {
            Some(extract_subpulse(self.state.sequence.as_ref().expect("a context"), &context)?)
        }.ok_or_else(|| Error::new(Kind::Value, "create or open the selected Subpulse source first"))?;
        let (candidate, dropped) = edits::edited(fragment.to_sequence(), &action, &positional)?;
        if !dropped.is_empty() { self.warn(format!("Unbound {}: the field is no longer set here", dropped.join(", "))); }
        let edited = Subpulse::new(&candidate.name, candidate.target.clone(), candidate.time_step_ns, candidate.periods.clone(), candidate.bindings.clone(),
                                   candidate.brackets.clone())?;
        if context.is_empty() {
            if Some(&edited) != self.subpulse_draft.as_ref() { self.subpulse_draft = Some(edited); self.revision += 1; }
        } else {
            let sequence = self.state.sequence.clone().expect("a context");
            let replaced = replace_component(&sequence, &context, &edited, None, None)?;
            self.edit_state(Some(Arc::new(replaced)), |_| {});
        }
        Ok(())
    }

    // ------------------------------------------------------------- Scan authoring
    pub(crate) fn scan_fields(&self) -> Vec<(String, String)> {
        self.state.sequence.as_ref().and_then(|s| resolve_subpulse_sources(s, self.base_directory().as_deref()).ok())
            .map(|s| s.scan_bindings().iter().map(|b| (b.field_id(), b.unit.clone())).collect()).unwrap_or_default()
    }

    fn has_scan_slots(&self) -> std::result::Result<(), String> {
        let Some(sequence) = &self.state.sequence else { return Err("Bind at least one field to a scan slot first (click a dot in the Edit tab)".into()) };
        match resolve_subpulse_sources(sequence, self.base_directory().as_deref()) {
            Ok(resolved) if !resolved.scan_bindings().is_empty() => Ok(()),
            Ok(_) => Err("Bind at least one field to a scan slot first (click a dot in the Edit tab)".into()),
            Err(error) => Err(error.message),
        }
    }

    pub(crate) fn fields_moved(&self, held: &[(String, String)]) -> bool {
        self.scan_fields().into_iter().filter(|pair| held.contains(pair)).collect::<Vec<_>>() != held
    }

    pub(crate) fn scan_table_stale(&self) -> bool { self.state.section.scan_fields.is_empty() || self.fields_moved(&self.state.section.scan_fields) }

    pub(crate) fn say_scan(&mut self, text: impl Into<String>, word: &'static str) { self.scan_progress = (text.into(), word); }

    fn check_table_fields(&self, fields: &[(String, String)], scanned: &Sequence, named: bool, changed: &str) -> Result<()> {
        if !named {
            if fields != self.scan_fields() { return Err(Error::new(Kind::Value, format!("Scan fields or units changed {changed}"))); }
            return Ok(());
        }
        let held: Vec<(String, String)> = scanned.scan_bindings().iter().map(|b| (b.field_id(), b.unit.clone())).collect();
        if held.iter().any(|pair| !fields.contains(pair)) || self.fields_moved(&held) {
            return Err(Error::new(Kind::Value, format!("A field the table names stopped being Scan or changed unit {changed}")));
        }
        Ok(())
    }

    /// A table a program produced or a file held, kept for the fields it drives.
    pub(crate) fn take_table(&mut self, table: &ScanTable, fields: &[(String, String)], source: Option<&str>, changed: &str) -> Result<usize> {
        let sequence = self.state.sequence.clone().ok_or_else(|| Error::new(Kind::Runtime, "no pulse is open"))?;
        let (scanned, rows) = scan_table_for(&sequence, table)?;
        self.check_table_fields(fields, &scanned, matches!(table, ScanTable::Columns(_)), changed)?;
        let count = rows.len();
        let scan_fields: Vec<(String, String)> = scanned.scan_bindings().iter().map(|b| (b.field_id(), b.unit.clone())).collect();
        let dirty = source.map(|source| self.state.section.scan_source != source);
        let repeats = if count > 1 && sequence.run_repeats == 0 {
            Some(Arc::new(sequence.with(|parts| parts.run_repeats = 1)?))
        } else { None };
        self.edit_state(repeats, |section| {
            section.scan_rows = rows;
            section.scan_fields = scan_fields;
            if let Some(dirty) = dirty { section.scan_source_dirty = dirty; }
        });
        Ok(count)
    }

    /// `pulse_scan {action, ...}`: the Scan page's authoring half (Hold and
    /// Step are board commands).
    pub fn pulse_scan(&mut self, args: &Map<String, Value>) -> Answer {
        let action = text_arg(args, "op")?;
        match action.as_str() {
            "template" => {
                self.has_scan_slots()?;
                let kind = text_arg(args, "kind")?;
                let sequence = self.state.sequence.clone().expect("checked");
                let (geometry, _) = self.compiler_target().map_err(|e| e.message)?;
                let resolved = resolve_subpulse_sources(&sequence, self.base_directory().as_deref()).map_err(|e| e.message)?;
                let source = scan_table_template(&kind, &scan_columns_for(&resolved, &geometry).map_err(|e| e.message)?).map_err(|e| e.message)?;
                self.edit_state(None, |section| { section.scan_source = source; section.scan_source_dirty = true; });
                Ok(Value::Null)
            }
            "edit_source" => {
                let source = text_arg(args, "source")?;
                self.edit_state(None, |section| { section.scan_source = source; section.scan_source_dirty = true; });
                Ok(Value::Null)
            }
            "load_program" => {
                let path = optional_text(args, "path").ok_or("load_scan_program requires an explicit path")?;
                let source = std::fs::read_to_string(&path).map_err(|error| format!("Cannot read {path}: {error}"))?;
                self.edit_state(None, |section| { section.scan_source = source; section.scan_source_dirty = true; });
                self.say_scan(format!("Loaded scan source from {path}"), "task");
                Ok(Value::Null)
            }
            "run" => self.run_scan_program(),
            "load_array" => {
                self.has_scan_slots()?;
                let path = optional_text(args, "path").ok_or("load_scan_array requires an explicit path")?;
                let fields = self.scan_fields();
                let shown = Path::new(&path).file_name().map_or_else(|| path.clone(), |n| n.to_string_lossy().into_owned());
                let outcome = crate::tables::read_table(Path::new(&path), fields.len())
                    .and_then(|table| self.take_table(&table, &fields, None, "while loading the table. Open it again."));
                match outcome {
                    Ok(count) => {
                        self.say_scan(format!("Loaded {} from {path}", counted(count, "scan point")), "task");
                        let progress = self.scan_progress.0.clone();
                        self.done(progress);
                        Ok(json!({"points": count}))
                    }
                    Err(error) => {
                        let message = format!("Cannot read {shown}: {}", error.message);
                        self.say_scan(message.clone(), "error");
                        Err(message)
                    }
                }
            }
            "save_array" => {
                if self.state.section.scan_rows.is_empty() { return Err("There is no scan table to save".into()); }
                let path = match optional_text(args, "path") {
                    Some(path) if !path.to_ascii_lowercase().ends_with(".npy") => return Err("save_scan_array requires a .npy path".into()),
                    Some(path) => PathBuf::from(path),
                    None => {
                        let name = format!("{}-scan", self.state.sequence.as_ref().map_or("scan", |s| s.name.as_str()));
                        crate::tables::unique_path(&self.save_folder(), &name, ".npy")
                    }
                };
                match crate::tables::write_named_npy(&path, &self.state.section.scan_fields, &self.state.section.scan_rows) {
                    Ok(()) => { self.say_scan(format!("Saved {}", path.display()), "task"); Ok(json!({"path": path.display().to_string()})) }
                    Err(error) => { let message = format!("Cannot save the scan table: {}", error.message); self.say_scan(message.clone(), "error"); Err(message) }
                }
            }
            "clear" => {
                let dirty = !self.state.section.scan_source.is_empty();
                self.edit_state(None, |section| { section.scan_rows.clear(); section.scan_fields.clear(); section.scan_source_dirty = dirty; });
                self.say_scan("Scan array cleared; the Python source is unchanged", "task");
                Ok(Value::Null)
            }
            "repeats" => {
                let repeats = arg(args, "repeats")?.as_u64().ok_or("repeats must be a non-negative integer")?;
                self.edit_state(None, |section| section.scan_repeats = repeats);
                Ok(Value::Null)
            }
            other => Err(format!("unknown Scan action {other:?}")),
        }
    }

    /// The program lane runs the operator's Python and answers its `scan_table`.
    fn run_scan_program(&mut self) -> Answer {
        self.has_scan_slots()?;
        if self.scan_running { return Err("The scan program is still running".into()); }
        self.scan_running = true;
        self.say_scan("Running the scan program…", "task");
        let source = self.state.section.scan_source.clone();
        let fields = self.scan_fields();
        self.jobs.push(Job::program(source, fields));
        Ok(json!({"accepted": true}))
    }

    /// The program lane's answer.
    pub(crate) fn scan_program_finished(&mut self, source: &str, fields: &[(String, String)], table: std::result::Result<Option<ScanTable>, String>) {
        self.scan_running = false;
        let outcome = match table {
            Err(error) => Err(format!("Scan program failed: {error}")),
            Ok(None) => Err("The scan program did not assign scan_table".to_owned()),
            Ok(Some(table)) => self.take_table(&table, fields, Some(source), "while generating the table. Generate it again.").map_err(|e| e.message),
        };
        match outcome {
            Ok(count) => self.say_scan(format!("{} ready", counted(count, "scan point")), "task"),
            Err(message) => { self.say_scan(message.clone(), "error"); self.warn(message); }
        }
    }

    // ------------------------------------------------------------- Preview settings
    pub fn pulse_preview(&mut self, args: &Map<String, Value>) -> Answer {
        if let Some(value) = args.get("include_off").and_then(Value::as_bool) { if value != self.include_off { self.include_off = value; self.pinned_size = None; } }
        if let Some(size) = args.get("size") { self.pinned_size = if size.is_null() { None } else { Some(py_str(size)) }; }
        if let Some(value) = args.get("selectors").and_then(Value::as_bool) { self.selectors = value; }
        Ok(Value::Null)
    }

    pub fn show_page(&mut self, page: &str) -> Answer {
        if !PAGES.contains(&page) { return Err(format!("unknown page {page:?}")); }
        self.page = page.to_owned();
        Ok(Value::Null)
    }

    /// What the preview should draw now: the effective pulse, or why not.
    pub fn preview_request(&self) -> Option<std::result::Result<(Sequence, bool, Option<String>), String>> {
        let sequence = self.state.sequence.as_ref()?;
        Some(self.effective_sequence(sequence).map(|effective| (effective, self.include_off, self.pinned_size.clone()))
            .map_err(|error| format!("Config error: {}", error.message)))
    }

    pub fn set_preview_status(&mut self, text: impl Into<String>, word: &'static str) { self.preview_status = (text.into(), word); }

    // ------------------------------------------------------------- projections
    fn document_summary(&self) -> String {
        if let Some(sequence) = &self.state.sequence {
            let shown = if self.path.is_empty() { sequence.name.clone() } else { self.path.clone() };
            let periods = self.effective_sequence(sequence).map_or_else(|_| period_counts(&sequence.periods), |e| period_counts(&e.periods));
            return format!("{shown}{SEPARATOR}{periods}");
        }
        match self.current_target() {
            None => "Open a pulse, or add a period to start one on this board".into(),
            Some(target) => format!("{} on this board{SEPARATOR}Add a period to start a pulse", counted(target.ports.len(), "port")),
        }
    }

    fn schedule(&self) -> Value {
        let config = self.active_config_values();
        let scanned = self.scanned_fields();
        let visible = self.visible();
        let pins = self.link.facts.as_ref().map(|f| f.description.target.package_pins.clone());
        let Some(sequence) = &self.state.sequence else {
            return match (self.current_target(), self.link.board_step_ns) {
                (Some(target), Some(step)) => project_board(&target, step, self.revision, pins.as_deref()),
                _ => empty_schedule(self.revision),
            };
        };
        let (effective, error) = match self.effective_sequence(sequence) {
            Ok(effective) => (effective, String::new()),
            Err(error) => (sequence.with(|parts| parts.config_enabled = false).unwrap_or_else(|_| (**sequence).clone()), error.message),
        };
        let request = ScheduleRequest {
            path: &self.path, generation: 0, revision: self.revision, pins: pins.as_deref(), scan_points: self.state.section.scan_rows.len(),
            config_error: &error, context: Context { visible_ports: visible.as_ref(), config: &config, scanned: Some(&scanned), authored: Some(sequence) },
            cards: Some(&self.cards),
        };
        project_schedule(&effective, &request)
    }

    fn binding_records(&self) -> Vec<Value> {
        let Some(authored) = &self.state.sequence else { return Vec::new() };
        let Ok(sequence) = self.effective_sequence(authored) else { return Vec::new() };
        let grouped = |sequence: &Sequence, reference: &FieldRef| -> String {
            let label = field_label(sequence, reference).unwrap_or_default();
            match sequence.component_for_field(reference) { None => label, Some(c) => label.strip_prefix(&format!("{} / ", c.name)).unwrap_or(&label).to_owned() }
        };
        let record = |sequence: &Sequence, b: &zlc_pulse::model::Binding, note: &str| {
            let component = sequence.component_for_field(&b.field_ref);
            json!({"field_id": b.field_id(), "label": grouped(sequence, &b.field_ref), "scan": b.scan, "source": b.source.name(),
                   "group_id": component.map_or("", |c| c.component_id.as_str()), "group_label": component.map_or("Pulse", |c| c.name.as_str()), "note": note})
        };
        let mut records: Vec<Value> = sequence.bindings.iter().filter(|b| b.scan || b.source == Source::Api).map(|b| record(&sequence, b, "")).collect();
        records.extend(stale_subpulse_bindings(authored, &sequence).iter().map(|b| record(authored, b, "not in file")));
        records
    }

    fn scan_page(&self) -> Value {
        let Some(sequence) = &self.state.sequence else { return json!({"slots_text": "No pulse is open.", "bindings": [], "table_text": "",
            "table_note": "", "source_text": "", "source_dirty": false, "repeats": 0, "busy": false, "progress_text": "", "progress_word": "task",
            "progress_polling": false}) };
        let rows = &self.state.section.scan_rows;
        let columns = self.compiler_target().and_then(|(geometry, _)| {
            let resolved = resolve_subpulse_sources(sequence, self.base_directory().as_deref())?;
            Ok((scan_columns_for(&resolved, &geometry)?, resolved, geometry))
        });
        let (columns, resolved, geometry) = match columns {
            Ok(found) => found,
            Err(error) => return json!({"slots_text": format!("Config error: {}", error.message), "bindings": [], "table_text": "", "table_note": "",
                "source_text": self.state.section.scan_source, "source_dirty": self.state.section.scan_source_dirty,
                "repeats": self.state.section.scan_repeats, "busy": true, "progress_text": self.scan_progress.0, "progress_word": self.scan_progress.1,
                "progress_polling": false}),
        };
        let stale = !rows.is_empty() && self.scan_table_stale();
        let table_text = if stale || rows.is_empty() { String::new() } else {
            self.played_scan(&resolved).and_then(|played| scan_columns_for(&played, &geometry)).map_or_else(|_| String::new(), |columns| crate::tables::table_text(rows, &columns))
        };
        let mut slots = if columns.is_empty() && sequence.api_bindings().is_empty() {
            "No bound fields: open a field's binding control and enable Scan or API.".to_owned()
        } else { "Physical fields available to Scan and API callers:".to_owned() };
        if !rows.is_empty() && !columns.is_empty() && sequence.run_repeats == 0 {
            slots.push_str(" Repeat on Edit is 0, so On Pulse stays at the first scan point until Stop Pulse.");
        }
        // v2 _render_scan_progress: while the board plays a table and the page is shown, the line
        // says where it is (a held row keeps its own line); otherwise it is the page's own words.
        let polling = self.link.state.firing && !rows.is_empty() && self.page == SCAN_PAGE;
        let (progress_text, progress_word) = match (polling, self.scan_progress_line()) {
            (true, Some(line)) => (line, "task"),
            (true, None) if self.link.held_point.is_none() => (String::new(), "task"),
            _ => (self.scan_progress.0.clone(), self.scan_progress.1),
        };
        json!({
            "slots_text": slots, "bindings": self.binding_records(), "table_text": table_text,
            "table_note": if stale { "Scan fields or units changed. Generate or open the scan table again." }
                          else if rows.is_empty() { "No scan table yet: write a program and press Run." } else { "" },
            "source_text": self.state.section.scan_source, "source_dirty": self.state.section.scan_source_dirty || stale,
            "repeats": self.state.section.scan_repeats, "busy": self.scan_running, "progress_text": progress_text,
            "progress_word": progress_word, "progress_polling": polling,
            "points": rows.len(),
        })
    }

    /// "Scan: point i / N  field = value, ...": the board's row of the table it
    /// was loaded with, while it still plays that program (v2 _render_scan_progress).
    fn scan_progress_line(&self) -> Option<String> {
        let state = &self.link.state;
        let (digest, names, rows) = self.link.applied_scan.as_ref()?;
        let cursor = state.cursor?;
        if !state.firing || *digest != state.applied_digest || rows.is_empty() { return None; }
        let point = cursor as usize % rows.len();
        let values = names.iter().zip(&rows[point]).map(|(name, text)| format!("{name} = {text}")).collect::<Vec<_>>().join(", ");
        let count = rows.len().to_string();
        Some(format!("Scan: point {} / {count}{}", crate::text::figure_padded(&(point + 1).to_string(), &count),
                     if values.is_empty() { String::new() } else { format!("  {values}") }))
    }

    fn config_page(&self) -> Value {
        let active = self.active_config_values();
        let scanned = self.scanned_fields();
        let mut bindings = Vec::new();
        let mut groups = Vec::new();
        let mut parked = Vec::new();
        if let Some(sequence) = &self.state.sequence {
            for b in &sequence.bindings {
                if b.source != Source::Config {
                    if b.config_key.is_empty() { continue; }
                    parked.push(b.field_id());
                }
                let default = pulse_field_value(sequence, &b.field_ref, &b.unit).unwrap_or(f64::NAN);
                let effective = (!b.config_key.is_empty() && b.source == Source::Config).then(|| active.iter().find(|(key, _)| *key == b.config_key)).flatten();
                let component = sequence.component_for_field(&b.field_ref);
                let label = field_label(sequence, &b.field_ref).unwrap_or_default();
                let label = component.map_or(label.clone(), |c| label.strip_prefix(&format!("{} / ", c.name)).unwrap_or(&label).to_owned());
                let status = if b.source != Source::Config { format!("{} (Config name kept)", if b.source == Source::Api { "API" } else { "Default" }) }
                    else if !sequence.config_enabled { "Config off".into() }
                    else if !self.config_error.is_empty() { "Not applied — Config error".into() }
                    else if scanned.contains(&b.field_id()) { "Scan table".into() }
                    else if effective.is_none() { "Default".into() } else { "Applied".into() };
                bindings.push(json!([b.field_id(), label, b.config_key, crate::text::bound_text(default, &b.unit, &b.unit),
                                     effective.map_or_else(String::new, |(_, (value, unit))| crate::text::bound_text(*value, unit, &b.unit)), status]));
                groups.push(json!([b.field_id(), component.map_or("", |c| c.component_id.as_str()), component.map_or("Pulse", |c| c.name.as_str())]));
            }
        }
        json!({"file_path": self.config_path, "dirty": self.config_dirty(), "entries": self.config_rows,
               "available_names": active.iter().map(|(key, _)| key.clone()).collect::<Vec<_>>(), "bindings": bindings,
               "active_path": self.active_config_path(), "busy": self.link.busy.is_some() || self.link.stopping, "binding_groups": groups, "parked": parked,
               "enabled": self.state.sequence.as_ref().is_some_and(|s| s.config_enabled), "error": self.config_error,
               "directory": self.config_values_directory().map(|d| d.display().to_string())})
    }

    fn subpulse_page(&self) -> Value {
        let sequence = self.state.sequence.as_ref();
        let enabled = sequence.is_some_and(|s| s.config_enabled);
        let sources: Vec<Value> = sequence.map(|s| s.components.iter().map(|c| {
            let (status, word) = if c.source != Source::Config { ("Default", "idle") } else if c.config_path.is_empty() { ("No file — plays Default", "idle") }
                else if !enabled { ("Config off", "idle") } else if !self.config_error.is_empty() { ("Not applied — Config error", "error") } else { ("Applied", "idle") };
            json!({"component_id": c.component_id, "name": c.name, "source": c.source.name(), "path": c.config_path, "status": status, "word": word})
        }).collect()).unwrap_or_default();
        let contexts: Vec<Value> = sequence.map(|s| s.components.iter().map(|c| json!([c.component_id, c.name])).collect()).unwrap_or_default();
        let fragment = self.selected_component_fragment();
        let context = &self.component_context;
        let document = fragment.as_ref().map(|fragment| {
            let selected = fragment.to_sequence();
            let config = self.active_config_values();
            let visible = if context.is_empty() { self.subpulse_visible.clone() } else { self.visible() };
            let pins = (!context.is_empty()).then(|| self.link.facts.as_ref().map(|f| f.description.target.package_pins.clone())).flatten();
            let request = ScheduleRequest { revision: self.revision, pins: pins.as_deref(),
                context: Context { visible_ports: visible.as_ref(), config: &config, scanned: None, authored: None }, ..Default::default() };
            let mut vm = project_schedule(selected, &request);
            if let (false, Some(parent)) = (context.is_empty(), sequence) {
                // HOLD reads the real preceding Pulse; the ids are the parent's own.
                if let Ok(parent) = parent.with(|parts| parts.config_enabled = false) {
                    let context = Context { visible_ports: visible.as_ref(), config: &config, scanned: None, authored: None };
                    let cards = crate::project::Periods::new(&parent, context);
                    vm["periods"] = Value::Array(selected.periods.iter().filter_map(|p| parent.period(&p.period_id))
                        .map(|p| cards.project(p)).collect());
                }
            }
            vm["item_order"] = Value::Array(sequence_item_order(selected).into_iter().map(|(k, v)| json!([k, v])).collect());
            vm
        });
        let selected_source = sequence.and_then(|s| s.components.iter().find(|c| c.component_id == self.selected_subpulse));
        let path = if !context.is_empty() { String::new() } else if fragment.is_none() { selected_source.map_or_else(String::new, |c| c.config_path.clone()) }
                   else { self.subpulse_path.clone() };
        let bindings: Vec<Value> = fragment.as_ref().map(|f| f.to_sequence().config_bindings().iter()
            .map(|b| json!([b.field_id(), field_label(f.to_sequence(), &b.field_ref).unwrap_or_default(), b.config_key])).collect()).unwrap_or_default();
        let page = json!({"sources": sources, "enabled": enabled, "busy": self.link.busy.is_some() || self.link.stopping, "error": self.config_error,
               "contexts": contexts, "context_id": context, "selected_id": self.selected_subpulse, "path": path,
               "dirty": if context.is_empty() { fragment.is_some() && self.subpulse_draft != self.subpulse_saved } else { self.dirty() },
               "bindings": bindings, "config_names": self.active_config_values().iter().map(|(k, _)| k.clone()).collect::<Vec<_>>(),
               "save_target": if context.is_empty() { "subpulse" } else { "pulse" }});
        crate::project::with_parts(page, [("document", document.unwrap_or(Value::Null))])
    }

    fn target_page(&self) -> Value {
        // v2 set_target_feedback: said until the page is next refreshed (set_target_ports says its caption again).
        let feedback = if self.target_feedback_at == self.revision { self.target_feedback.clone() } else { (String::new(), "idle") };
        let Some(target) = self.current_target() else {
            return json!({"records": [], "editable": false, "caption": "No pulse and no board: nothing is wired yet.", "reserved": [],
                          "feedback": feedback.0, "feedback_word": feedback.1, "width_rules": null});
        };
        let pins = self.link.facts.as_ref().map(|f| f.description.target.package_pins.clone());
        let records: Vec<Value> = project_target(&target, pins.as_deref()).iter().map(TargetRecord::to_json).collect();
        let reserved: Vec<String> = programmable_ports(&target).into_iter().filter(|p| p.kind == PortKind::Clock).map(|p| p.key.clone()).collect();
        let (caption, rules) = match &self.link.facts {
            None => ("Offline: this target came from the pulse file and can be edited.".to_owned(), Value::Null),
            Some(facts) => {
                let width = facts.description.geometry.bus_width;
                let rules = json!({"digital": [1, 1, 1], "dac": [2, width, width]});
                if target.abi_fingerprint() == facts.description.target.abi_fingerprint() {
                    (format!("Wiring read from the attached board; rename freely, the topology is the board's ({} outputs).", records.len()), rules)
                } else {
                    ("The pulse file's own wiring, which the attached board will not load: edit the pulse to fit, save it and open it again.".to_owned(), rules)
                }
            }
        };
        json!({"records": records, "editable": self.link.facts.is_none(), "caption": caption, "reserved": reserved,
               "feedback": feedback.0, "feedback_word": feedback.1, "width_rules": rules,
               "digital_count": records.iter().filter(|r| r["kind"] == "digital").count(), "dac_count": records.iter().filter(|r| r["kind"] == "dac").count()})
    }

    /// Everything the window shows, as one state section (`pulse`).
    pub fn state(&mut self) -> Value {
        let (status, word) = self.link.status_token(self);
        let control = self.link.control(self);
        let config_mode = json!({"enabled": self.state.sequence.as_ref().is_some_and(|s| s.config_enabled),
                                 "busy": self.link.busy.is_some() || self.link.stopping || self.state.sequence.is_none(), "error": self.config_error});
        let mut config_error = String::new();
        if let Some(sequence) = &self.state.sequence { if let Err(error) = self.effective_sequence(sequence) { config_error = error.message; } }
        self.config_error = config_error;
        let summary = if !self.config_error.is_empty() { format!("Config error: {}", self.config_error) }
                      else { self.link.busy.clone().or_else(|| self.summary_note.clone()).unwrap_or_else(|| self.document_summary()) };
        let schedule = self.schedule();
        let order: Vec<Value> = self.state.sequence.as_ref().map(|s| sequence_item_order(s).into_iter().map(|(k, v)| json!([k, v])).collect()).unwrap_or_default();
        let state = json!({
            "revision": self.revision, "path": self.path, "dirty": self.dirty(), "page": self.page,
            "title": format!("PulseGUI{SEPARATOR}{}", self.state.sequence.as_ref().map_or("no pulse", |s| s.name.as_str())),
            "summary": summary, "status": status, "dot": word, "busy": self.link.busy, "stopping": self.link.stopping,
            "message": self.message.as_ref().map(|m| json!({"id": m.id, "kind": m.kind, "text": m.text})),
            "last_warning": self.last_warning,
            "has_pulse": self.state.sequence.is_some(),
            "preview": {"include_off": self.include_off, "size": self.pinned_size, "selectors": self.selectors,
                        "status": self.preview_status.0, "status_word": self.preview_status.1,
                        "sizes": zlc_plot::layout::PANEL_PRESETS.iter().map(|(name, _, _)| *name).collect::<Vec<_>>()},
            "time_units": TIME_UNIT_CHOICES,
            "save_folder": self.save_folder().display().to_string(),
            "proposed_path": self.proposed_save_path().display().to_string(),
        });
        crate::project::with_parts(state, [("schedule", schedule), ("item_order", Value::Array(order)), ("connection", self.link.connection_json()),
            ("control", control), ("config_mode", config_mode), ("scan", self.scan_page()), ("config", self.config_page()),
            ("subpulse", self.subpulse_page()), ("target", self.target_page()), ("board", self.link.state.to_json())])
    }

    /// The schedule cards of the state last made, once it is sent: the next
    /// state takes back each one that is unchanged.
    pub fn recycle(&mut self, state: &mut Value) {
        let cards = state.get_mut("schedule").and_then(|schedule| schedule.get_mut("periods")).map(Value::take);
        if let Some(Value::Array(cards)) = cards { self.cards.get_mut().recycle(cards); }
    }

    /// `_align_sequence_to(board)`: the open pulse on this board, matched by
    /// name and kind; what does not fit is said and dropped, never driven blind.
    pub(crate) fn align_sequence_to(&mut self, board: &BoardDescription) {
        let Some(current) = self.state.sequence.clone() else { return };
        if current.target.abi_fingerprint() == board.target.abi_fingerprint() && current.time_step_ns == board.time_step_ns() { return; }
        let digital: BTreeSet<&str> = board.target.ports.iter().filter(|p| p.kind == PortKind::Digital).map(|p| p.lanes[0].as_str()).collect();
        let was = |lane: &str| current.target.lane_index(lane);
        let mut lost: Vec<String> = current.target.raw_lanes.iter().enumerate()
            .filter(|(index, lane)| !digital.contains(lane.as_str()) && current.periods.iter().any(|p| p.states[*index] != 0)).map(|(_, l)| l.clone()).collect();
        lost.sort();
        let fits = |key: &str| board.target.port(key).zip(current.target.port(key)).is_some_and(|(b, w)| b.kind == w.kind);
        let labels: Vec<(String, String)> = current.target.ports.iter().filter(|p| fits(&p.key)).map(|p| (p.key.clone(), p.label.clone())).collect();
        let target = retarget_labels(&board.target, &labels).ok().flatten().unwrap_or_else(|| board.target.clone());
        let mut dropped: BTreeSet<String> = current.periods.iter().flat_map(|p| p.analog_steps.iter()).filter(|s| !fits(&s.port)).map(|s| s.port.clone()).collect();
        dropped.extend(current.delays.iter().filter(|d| !fits(&d.port)).map(|d| d.port.clone()));
        let periods: Vec<Period> = current.periods.iter().map(|period| {
            let mut copy = period.clone();
            copy.states = board.target.raw_lanes.iter().map(|lane| if digital.contains(lane.as_str()) { was(lane).map_or(0, |i| period.states[i]) } else { 0 }).collect();
            copy.analog_steps.retain(|s| fits(&s.port));
            copy
        }).collect();
        let bindings: Vec<_> = current.bindings.iter().filter(|b| b.field_ref.port.as_deref().is_none_or(|p| fits(p))).cloned().collect();
        let mut dropped_bindings: Vec<String> = current.bindings.iter().filter(|b| !bindings.contains(b))
            .map(|b| field_label(&current, &b.field_ref).unwrap_or_default()).collect();
        dropped_bindings.sort();
        let delays: Vec<_> = current.delays.iter().filter(|d| fits(&d.port)).cloned().collect();
        let step = board.time_step_ns();
        match current.with(|parts| { parts.target = target; parts.time_step_ns = step; parts.periods = periods; parts.bindings = bindings; parts.delays = delays; }) {
            Err(error) => self.warn(format!("This pulse does not fit the attached board ({}); it keeps the wiring it was written for, which this board \
                                             will not load: edit it to fit, save it and open it again", error.message)),
            Ok(candidate) => {
                self.edit_state(Some(Arc::new(candidate)), |_| {});
                let missing: Vec<String> = lost.into_iter().chain(dropped).chain(dropped_bindings).collect();
                if !missing.is_empty() {
                    self.warn(format!("This board has no matching {}; those outputs or bindings were dropped from the pulse rather than driven blind",
                                      missing.join(", ")));
                }
            }
        }
    }

    /// Answers for a remote caller that inspects the editor (v2 `control_state`).
    pub fn control_state(&self) -> Value {
        let sequence = self.state.sequence.as_ref();
        json!({
            "path": self.path, "dirty": self.dirty(), "revision": self.revision, "pending": self.link.busy.is_some() || self.link.stopping,
            "stopping": self.link.stopping, "running": self.link.state.firing, "board": self.link.state.to_json(), "last_warning": self.last_warning,
            "scan": {"rows": self.state.section.scan_rows.len(), "repeats": self.state.section.scan_repeats,
                     "fields": self.state.section.scan_fields.iter().map(|(f, _)| f.clone()).collect::<Vec<_>>(),
                     "source_dirty": self.state.section.scan_source_dirty, "pending": self.scan_running, "progress": self.scan_progress.0},
            "config": {"draft_path": self.config_path, "draft_dirty": self.config_dirty(), "active_path": self.active_config_path(),
                       "enabled": sequence.is_some_and(|s| s.config_enabled), "error": self.config_error},
            "authored": sequence.map(|s| json!({"name": s.name, "time_step_ns": s.time_step_ns, "duration_seconds": s.played_nanoseconds() * 1e-9,
                "total_text": readable(s.played_nanoseconds()),
                "periods": s.periods.iter().map(|p| json!({"period_id": p.period_id, "name": p.name, "kind": p.kind.name(), "duration": p.duration.to_json_value(), "unit": p.unit})).collect::<Vec<_>>(),
                "brackets": s.brackets.iter().map(|b| json!({"bracket_id": b.bracket_id, "start_period_id": b.start_period_id, "end_period_id": b.end_period_id, "count": b.count})).collect::<Vec<_>>(),
                "bindings": s.bindings.iter().map(|b| json!({"field_id": b.field_id(), "label": field_label(s, &b.field_ref).unwrap_or_default(), "unit": b.unit,
                    "scan": b.scan, "source": b.source.name(), "config_key": b.config_key, "field_kind": crate::project::field_kind_word(b.kind())})).collect::<Vec<_>>(),
                "ports": s.target.ports.iter().map(|p| json!({"key": p.key, "label": p.label, "kind": p.kind.name(), "lanes": p.lanes,
                    "signed_range": p.signed_range().map(|(lo, hi)| json!([lo, hi]))})).collect::<Vec<_>>()})),
        })
    }

    /// One `pulse_*` command, whoever sent it. A refusal the operator caused
    /// is said in the window; a remote caller gets it as its answer only.
    pub fn command(&mut self, action: &str, args: &Map<String, Value>, operator: bool) -> Answer {
        self.last_warning.clear();
        self.summary_note = None;
        let shown = self.message.as_ref().map(|m| m.id);
        self.operator = operator;
        let result = self.dispatch(action, args);
        self.operator = false;
        if let Err(message) = &result {
            if operator { self.warn(message.clone()); } else { self.last_warning = message.clone(); }
        }
        if !operator && self.message.as_ref().map(|m| m.id) != shown {
            // What a remote command made the editor say is its answer, not the operator's dialog.
            if let Some(message) = self.message.take() { self.messages = message.id; }
        }
        result
    }

    fn dispatch(&mut self, action: &str, args: &Map<String, Value>) -> Answer {
        match action {
            "pulse_state" => Ok(self.control_state()),
            "pulse_open" => self.open_pulse(&text_arg(args, "path")?, flag(args, "discard_changes")),
            "pulse_save" => {
                let path = optional_text(args, "path").or_else(|| (!self.path.is_empty()).then(|| self.path.clone()))
                    .ok_or("save_pulse requires an explicit path")?;
                self.save_pulse(&path)
            }
            "pulse_new" => {
                if self.dirty() && !flag(args, "discard_changes") {
                    return Err("Pulse has unsaved changes; save it or explicitly pass discard_changes=true to replace the draft".into());
                }
                if self.start_new_pulse() { Ok(Value::Null) } else { Err(self.last_warning.clone()) }
            }
            "pulse_clear" => self.clear_all(flag(args, "discard_changes")),
            "pulse_edit" => self.pulse_edit(args),
            "pulse_scan" => self.pulse_scan(args),
            "pulse_config" => self.pulse_config(args),
            "pulse_subpulse" => self.pulse_subpulse(args),
            "pulse_target" => {
                let records = arg(args, "records")?.as_array().ok_or("records must be a list")?.iter()
                    .map(TargetRecord::from_json).collect::<Result<Vec<_>>>().map_err(|e| e.message)?;
                self.apply_target(&records)
            }
            "pulse_preview" => self.pulse_preview(args),
            "pulse_page" => self.show_page(&text_arg(args, "page")?),
            "pulse_connect" => self.connect(&text_arg(args, "mode")?, &optional_text(args, "endpoint").unwrap_or_default()),
            "pulse_fire" => self.fire(),
            "pulse_hold" => self.hold(None),
            "pulse_step" => self.hold(Some(arg(args, "delta")?.as_i64().ok_or("delta must be an integer")?)),
            "pulse_stop" => self.stop(),
            "pulse_sync" => self.sync(flag(args, "discard_changes")),
            "pulse_validate" => self.validate(),
            other => Err(format!("{other} is not a Pulse Editor command")),
        }
    }

    /// Whether a value is set the way Python reads a flag.
    pub fn is_truthy(value: &Value) -> bool { py_truthy(value) }
}

trait NumJson { fn to_json_value(&self) -> Value; }
impl NumJson for zlc_pulse::num::Num {
    fn to_json_value(&self) -> Value { crate::to_value(&self.to_json()) }
}
