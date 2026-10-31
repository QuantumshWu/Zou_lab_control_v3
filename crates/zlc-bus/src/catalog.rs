//! Every action a client can take, with its parameter schema, its purpose and
//! whether it only reads. The window and remote (MCP/AI) clients submit the same
//! commands through the same queue; nothing either can do is missing from the
//! other's vocabulary.

use std::collections::BTreeMap;

use schemars::{JsonSchema, schema_for};
use serde::Deserialize;
use serde_json::{Map, Value};

/// The plot kinds a live signal can be shown as.
#[derive(Clone, Copy, Debug, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PlotKind { Image, Curve, Histogram, Rolling, FacetGrid }

/// Open a saved Figure archive in a Figure Viewer of its own (v2: one window per Figure).
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Open {
    /// Path of a ZLC Figure archive (.npz). The window may omit it to ask with a file dialog.
    pub path: Option<String>,
    /// An open Figure Viewer (figure-viewer-<n> from get_state windows) to show it in instead, replacing its Figure.
    pub window: Option<String>,
}

/// Save one panel as an image together with its Figure archive. When the image or the
/// archive is already there nothing is written and the save is refused, naming them,
/// unless `overwrite` says to write over them.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Save {
    /// Panel id (from get_state); omitted saves the document's first panel.
    pub panel: Option<String>,
    /// Image path ending in .png, .pdf or .svg; the paired .npz archive is written beside it.
    pub path: String,
    /// Write over the image and archive when either is already there (otherwise refused).
    pub overwrite: Option<bool>,
}

/// Start the camera (the virtual MOT camera on this build).
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct StartCamera {
    /// Exposure time in seconds (> 0).
    pub exposure_seconds: f64,
    /// Acquire exactly this many frames and stop; omitted or null runs until stopped.
    pub frames: Option<u64>,
    /// Frame size [rows, columns] of the virtual camera.
    pub frame_shape_yx: Option<[u64; 2]>,
    /// Sensor region [x, y, width, height].
    pub roi_xywh: Option<[u64; 4]>,
    /// Simulation seed (reproducible frames).
    pub seed: Option<u64>,
    /// Frames the driver may buffer before the instrument reads them.
    pub buffer_frame_count: Option<u64>,
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Empty {}

/// Add a plot panel of a live signal.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AddPanel {
    /// A signal name from list_signals, e.g. "@logic/camera_measurement/frames" or "@logic/panel-1/roi_frame";
    /// omitted adds a blank panel whose signal is chosen later (configure binding).
    pub signal: Option<String>,
    pub kind: PlotKind,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PanelRef {
    /// Panel id from get_state, e.g. "panel-2".
    pub panel: String,
}

/// Save the window's picture.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SaveImage {
    /// A .png file; omitted, the window asks (a remote client's goes to today's data folder).
    pub path: Option<String>,
    /// Which window: devices, task-console (default), figure-viewer or pulse-editor.
    pub window: Option<String>,
}

/// Close one of the instrument's windows, as its × does: a Figure Viewer drops its Figure, a
/// control or the Pulse Editor hides, a log ends; Devices or TaskConsole closes the experiment.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CloseWindow {
    /// A window from get_state windows (e.g. figure-viewer-2, log:camera).
    pub window: Option<String>,
}

/// Show one of the instrument's windows in front.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ShowWindow {
    /// devices (Devices@Zou lab), task-console (TaskConsole@Zou lab), pulse-editor (PulseGUI@Zou lab), figure-viewer-<n>,
    /// control:<device>, log:<device>, slm-editor:<device> (an SLM's Editor, whose draft is the device's
    /// editor_* commands) or review:<node> (a run's open point review); get_state windows lists them.
    pub window: String,
    /// TaskConsole's page to show: monitor or logic.
    pub page: Option<String>,
}

/// One window's picture.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WindowRef {
    /// devices, task-console (default), figure-viewer or pulse-editor.
    pub window: Option<String>,
}

/// Close the experiment, as the window's close button does.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CloseExperiment {
    /// Close although the Pulse Editor holds unsaved changes (otherwise refused).
    pub discard_changes: Option<bool>,
}

/// One topic of the operator guide.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct GuideTopic {
    /// overview (default), setup, devices, acquisition, pulse, calibration, processing, scan, plot, fit, files,
    /// recipes, lifetime, slm, errors; or type:<type_id> for that plugin type's descriptor.
    pub topic: Option<String>,
}

/// One step of a JSON path: an object key or a list index.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum PathStep { Key(String), Index(u64) }

/// Wait until a field of a read command's reply equals a value.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WaitFor {
    /// A command that only reads (get_state, list_signals, get_fit, pulse_state, ...); not wait, wait_for or batch.
    pub tool: String,
    /// Its arguments.
    pub arguments: Option<Map<String, Value>>,
    /// The field: object keys and list indices, e.g. ["devices", 0, "state"].
    pub path: Vec<PathStep>,
    /// The value the field must equal.
    pub equals: Value,
    /// At most this long (default 30, at most 1800).
    pub timeout_seconds: Option<f64>,
}

/// Already-decided commands, run in order.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Batch {
    /// 1-16 calls. An argument {"$result": [i, "key", 0, ...]} is that field of call i's reply (an earlier call).
    pub calls: Vec<BatchCall>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BatchCall {
    /// Any command but batch.
    pub tool: String,
    pub arguments: Option<Map<String, Value>>,
    /// Top-level reply fields that must equal these values for the batch to go on.
    pub expect: Option<Map<String, Value>>,
}

/// A directory of the workspace.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ListFiles {
    /// Workspace-relative (or absolute inside it); default the workspace root.
    pub directory: Option<String>,
    /// First entry (entries are sorted by name).
    pub offset: Option<u64>,
    /// At most this many entries (1-200, default 100).
    pub limit: Option<u64>,
}

/// A file of the workspace.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReadFile {
    /// Workspace-relative (or absolute inside it).
    pub path: String,
    /// First byte of the page (text, or a long Figure summary).
    pub offset: Option<u64>,
    /// At most this many bytes (1-64000, default 16000).
    pub limit: Option<u64>,
    /// A Figure archive's one section, by name from section_names.
    pub section: Option<String>,
}

/// A page of a published signal's numbers.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReadSignal {
    /// A signal from list_signals, e.g. "@logic/occupancy/counts".
    pub signal: String,
    /// Refuse unless the signal's latest publication is of this run generation (from a previous read).
    pub generation: Option<String>,
    /// First value of the page, in the flattened physical storage (C order).
    pub offset: Option<u64>,
    /// At most this many values (1-4096, default 256).
    pub limit: Option<u64>,
    /// The whole dataset: a finite run's dataset so far, or a monitor's retained history; default the latest publication.
    pub dataset: Option<bool>,
}

/// What plots can do, before any panel exists.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PlotCatalog {
    /// Also each fit model's formula and parameters.
    pub details: Option<bool>,
}

/// Open a pulse the opened Figure's run played in the Pulse Editor.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct OpenRecordedPulse {
    /// The played pulse's key, from the Figure's Devices rows (`pulse:<key>` actions).
    pub key: String,
    /// Replace an unsaved Pulse Editor draft.
    pub discard_changes: Option<bool>,
}

/// A fit request: a catalogue model, optionally a custom expression.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FitTarget {
    /// Model key from the panel's fit.models in get_state (e.g. "gaussian_offset"); null or "" removes the fit.
    pub model: Option<String>,
    /// Custom model expression when the model accepts one; "" for none.
    pub expression: Option<String>,
    /// Which selector's data the fit reads, when the panel has several.
    pub selector_kind: Option<String>,
    /// Parameters held fixed, by name: {"offset": 0.0}.
    pub fixed: Option<Value>,
    /// Starting values by parameter name.
    pub initial: Option<Value>,
    /// Bounds by parameter name: {"sigma": [low, high]}.
    pub bounds: Option<Value>,
    /// Solver options as listed by the model.
    pub options: Option<Value>,
    /// Facet grids: fit every cell instead of only the focused one.
    pub fit_all_facets: Option<bool>,
    /// Two-component models: the evidence (BIC gain) required to prefer them.
    pub min_bic_gain: Option<f64>,
}

/// Rebind a panel's input.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Binding {
    /// Signal to read from list_signals.
    pub signal: Option<String>,
    /// How many recent shots the panel reads (>= 1); it also sets the scientific window.
    pub window: Option<u64>,
}

/// One selector as a remote client sets it; the revision is assigned by the instrument.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Selector {
    #[serde(rename = "kind")]
    pub kind: SelectorKind,
    /// x_range: [low, high]; area: {"x": [low, high], "y": [low, high]}; crosshair: [x, y];
    /// threshold: a number. Values are in the panel's data coordinates (its axis units).
    pub value: Value,
    /// Facet grids: the cell the selector belongs to; null otherwise.
    pub facet_index: Option<u64>,
    /// Edit counter of this kind; the instrument assigns the next one when omitted.
    pub revision: Option<u64>,
}

#[derive(Clone, Copy, Debug, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SelectorKind { XRange, Area, Crosshair, Threshold }

impl SelectorKind {
    pub fn wire(self) -> &'static str {
        match self { SelectorKind::XRange => "x_range", SelectorKind::Area => "area",
                     SelectorKind::Crosshair => "crosshair", SelectorKind::Threshold => "threshold" }
    }
}

/// The visible data range of a panel; null axes are automatic.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Viewport {
    pub x: Option<[f64; 2]>,
    pub y: Option<[f64; 2]>,
}

/// Change what a panel shows and how. Every field is optional; only given fields change.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Configure {
    /// Panel id from get_state.
    pub panel: String,
    /// Semantic edits, keyed by the panel's controls with section "semantic" in get_state:
    /// "kind", "cell_kind", "reduction", and one "fate:<domain>:<axis>" row per data axis whose
    /// value is a role ("x", "y", "group", "facet"), "reduce", "pool" or a scope pin.
    pub semantic: Option<Map<String, Value>>,
    /// Display parameters keyed by the controls with section "display" (e.g. "colormap",
    /// "relim_mode", "color_min", "bin_count", "uncertainty"); choices and bounds are listed there.
    pub parameters: Option<Map<String, Value>>,
    pub fit: Option<FitTarget>,
    /// Panel size preset, one of the panel's size_choices (e.g. "2x2").
    pub size: Option<String>,
    pub binding: Option<Binding>,
    /// The complete selector set after the edit (replaces the panel's selectors); [] clears them.
    pub selectors: Option<Vec<Selector>>,
    /// The visible range; null resets to automatic limits.
    pub viewport: Option<Option<Viewport>>,
    /// Facet grids: open one cell by index, or null for the overview.
    pub facet_focus: Option<Option<u64>>,
    /// Histogram classifier thresholds as listed in the panel's target.
    pub classifier_thresholds: Option<Vec<Value>>,
}

/// Open one facet cell (or return to the overview).
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Focus {
    pub panel: String,
    /// Cell index, or null for the overview.
    pub index: Option<u64>,
}

/// Move a scoped (pinned) axis to a neighbouring coordinate.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ScopeStep {
    pub panel: String,
    /// The semantic control name of the scoped fate row.
    pub name: String,
    /// Coordinates to move by (negative goes back).
    pub steps: i64,
}

/// A pointer event on a panel canvas, as the window sends it.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Pointer {
    pub panel: String,
    /// Monotonic browser event time in seconds, unaffected by rendering queues.
    pub time: Option<f64>,
    /// press, move, release, double_click, scroll, leave, key or cancel.
    pub event: String,
    /// Canvas position as a fraction of its width (0 = left).
    pub x: Option<f64>,
    /// Canvas position as a fraction of its height (0 = top).
    pub y: Option<f64>,
    /// The axes entry from the panel's "axes" in get_state that the gesture acts on.
    pub axes: Option<Value>,
    pub button: Option<String>,
    pub buttons: Option<Vec<String>>,
    pub modifiers: Option<Vec<String>>,
    /// Wheel steps for scroll (positive zooms in).
    pub steps: Option<f64>,
    pub key: Option<String>,
    /// The frame sequence the gesture was aimed at (from the panel's frame in get_state).
    pub displayed_sequence: Option<u64>,
}

/// Read the instrument state, optionally for one panel.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct GetState {
    pub panel: Option<String>,
    /// Only this logic node's row in logic.
    pub node: Option<String>,
    /// Only these top-level sections (e.g. ["logic", "devices"]); omitted returns everything.
    pub sections: Option<Vec<String>>,
}

/// Read the accepted fit results of a panel, a page at a time.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct GetFit {
    pub panel: String,
    /// First row (facet grids have one row per cell).
    pub offset: Option<u64>,
    /// At most this many rows (1-4096, default 256).
    pub limit: Option<u64>,
}

/// Wait until something has happened, or until the timeout.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Wait {
    /// Wait until the camera has produced at least this many frames in this run.
    pub produced: Option<u64>,
    /// Wait until this panel shows a frame newer than `after_sequence` (default: its current one).
    pub panel: Option<String>,
    pub after_sequence: Option<u64>,
    /// Wait until this logic node's run reaches `until` (default terminal); an operator request
    /// answers a ready or terminal wait early with reached false.
    pub node: Option<String>,
    pub until: Option<RunMilestone>,
    /// Only this run of the node (e.g. "calibration#2"); refused once another run is the node's.
    pub run: Option<String>,
    /// At most this long (default 30, at most 1800).
    pub timeout_seconds: Option<f64>,
}

#[derive(Clone, Copy, Debug, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RunMilestone { Ready, Terminal, Request }

// ---- Workbench (docs/WORKBENCH_COMMANDS.md §4) ----

/// Plugin types, optionally one.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ListPlugins { pub type_id: Option<String> }

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DeviceLog {
    /// Device instance id from get_state devices.
    pub device: String,
    /// The newest this many lines (default 200, at most 2000).
    pub lines: Option<u64>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NodeRef {
    /// Logic node id from get_state logic.
    pub node: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ListRuns { pub node: Option<String>, pub limit: Option<u64> }

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RunRef {
    /// Run id, e.g. "calibration#2".
    pub run: String,
}

/// A file in the workspace (relative paths are read against its root).
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PathRef { pub path: String }

/// A file; the window may omit it to ask with a file dialog.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct OptionalPath { pub path: Option<String> }

/// Save the apparatus draft. A file this window did not read in full (it could not, or it left
/// entries or settings out) is replaced only when `replace_unread` says so; else save under another name.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SaveApparatus {
    /// A zlc.installation file; the window may omit it to ask with a file dialog.
    pub path: Option<String>,
    /// Replace the apparatus file this window did not read in full (what it left out is lost).
    pub replace_unread: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RecipeName { pub name: Option<String> }

/// Add a device to the apparatus draft (it opens at apply_devices).
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AddDevice {
    /// A device type from list_plugins.
    pub type_id: String,
    /// Any text without surrounding space or `/` (v2); minted from the family when omitted.
    pub instance_id: Option<String>,
    /// Display name; the instance id when omitted.
    pub role: Option<String>,
    /// Init settings by field name; the rest take their defaults.
    pub parameters: Option<Map<String, Value>>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DeviceRef {
    /// Device instance id from get_state devices.
    pub device: String,
}

/// Change a draft device: its name, type (settings reset) or settings.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EditDevice {
    pub device: String,
    pub role: Option<String>,
    pub type_id: Option<String>,
    /// Changed settings only.
    pub parameters: Option<Map<String, Value>>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SimulationValues {
    /// Changed simulation world settings (apparatus.simulation; v2 kept them in the apparatus file).
    pub values: Map<String, Value>,
}

#[derive(Clone, Copy, Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Template { Virtual, Hardware }

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct UseTemplate { pub template: Template }

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Discovered {
    /// An id from apparatus.discovered.
    pub instance_id: String,
}

/// Read an open device's tunable fields.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReadDevice {
    pub device: String,
    /// Read these fields in another spelling of their unit from now on (field -> unit,
    /// one of the field's `units`; "" = its own unit again).
    pub units: Option<BTreeMap<String, String>>,
}

/// Apply one Device Control field to an open device.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TuneDevice {
    pub device: String,
    /// A field name from the device's fields (read_device).
    pub field: String,
    pub value: Value,
    /// The unit the value is written in (the field's own when omitted); the readback is in it too.
    pub unit: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DeviceRisk { pub device: String, pub accepted: bool }

/// Remote on or off for one open device (v2's Device Manager switch).
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PublishDevice {
    /// Device instance id from get_state devices.
    pub device: String,
    /// true: hand it to another bench computer; false: take it back.
    pub published: bool,
}

/// The operator's answer to one browser asking to watch.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ShareDecide {
    /// A browser id from get_state share.clients.
    pub client: String,
    /// true: Allow (it sees the window); false: Reject, or Disconnect a connected one.
    pub allow: bool,
}

/// Add a stopped logic node to the board.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AddNode {
    /// A node type from list_plugins.
    pub type_id: String,
    /// Minted from the type when omitted (occupancy, occupancy2, ...).
    pub node_id: Option<String>,
    pub name: Option<String>,
    pub values: Option<Map<String, Value>>,
    pub source_signal: Option<String>,
    /// Device argument -> device instance id.
    pub device_keys: Option<Map<String, Value>>,
    pub artifact_inputs: Option<Map<String, Value>>,
    pub auto_preview: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RenameNode { pub node: String, pub name: String }

/// Change a node's draft; only given fields change. A running node keeps its configuration.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EditNode {
    pub node: String,
    /// Changed authoring values by field name (the node's controls in get_state).
    pub values: Option<Map<String, Value>>,
    /// The signal it reads, one of its source_choices.
    pub source_signal: Option<String>,
    /// Changed device arguments -> device instance ids (its device_choices).
    pub device_keys: Option<Map<String, Value>>,
    /// Changed artifact inputs: a path, or a list of {path, frames} for frame claims.
    pub artifact_inputs: Option<Map<String, Value>>,
    /// Whether Start opens the node's declared preview panels.
    pub auto_preview: Option<bool>,
    /// Refuse if the draft changed since this revision (get_state draft_revision).
    pub expected_draft_revision: Option<u64>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct StartNode {
    pub node: String,
    /// Refuse if the draft changed since this revision.
    pub expected_draft_revision: Option<u64>,
}

/// Answer a running node's operator request.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AnswerRequest {
    pub node: String,
    /// The request id from its run.request.
    pub request: String,
    /// The request's fields; null cancels the request (the run stops).
    pub response: Option<Map<String, Value>>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct OrderPanels {
    /// Panel ids in their new order; ids left out follow in their current order.
    pub order: Vec<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RenamePanel {
    pub panel: String,
    /// The panel name; null or "" returns to the automatic name.
    pub title: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PanelInterval {
    pub panel: String,
    /// 100, 200, 400 or 800.
    pub interval_ms: u64,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PanelOverlay {
    pub panel: String,
    /// One of the panel's overlay_choices in get_state's board (e.g. "@logic/occupancy/occupied"); null or "" draws no rings.
    pub signal: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PublishOutputs {
    pub panel: String,
    /// ROI/fit output name -> whether the panel publishes it as @logic/<panel>/<name>.
    pub outputs: BTreeMap<String, bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SetBoard {
    /// Pause live display (acquisition continues).
    pub paused: Option<bool>,
    /// Plot gestures (selectors, zoom, pan) on or off.
    pub selectors: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LoadLayout {
    /// A zlc.console-board file; the window may omit it to ask with a file dialog.
    pub path: Option<String>,
    /// Stop running nodes first instead of refusing.
    pub stop_running: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ClearBoard { pub stop_running: Option<bool> }

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RunRecipe {
    /// A recipe from list_recipes.
    pub name: String,
    /// Recipe parameters by name (list_recipes {name} shows them with defaults).
    pub parameters: Option<Map<String, Value>>,
    /// Start its nodes once loaded (default true).
    pub start: Option<bool>,
}

// ---------------------------------------------------------------- pulse
/// A pulse document by path.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PulseDocument {
    /// Path of a zlc.pulse JSON document.
    pub document: String,
}

/// Play a pulse document on the attached board.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PulseRun {
    /// Path of a zlc.pulse JSON document.
    pub document: String,
    /// API field id (pulse_describe) -> [number, unit]; converted exactly into the field's own unit.
    pub api_values: Option<BTreeMap<String, (f64, String)>>,
    /// Complete Pulse runs (0: until pulse_stop); default 1.
    pub run_repeats: Option<u64>,
    /// Scan-table sweeps; 1 without a table.
    pub scan_repeats: Option<u64>,
    /// A table the board plays: these scan slots (no other slot streams), each row run_repeats times.
    pub scan: Option<PulseScanTable>,
    /// A run's two-step play: `load` puts the document on the board without firing (no command_id);
    /// `fire` then fires what that load left there (repeats only). Default: both at once.
    pub stage: Option<String>,
}

/// A scan table for the board: rows in the fields' own units, in this field order.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PulseScanTable {
    /// Field ids (pulse_describe scan_fields) the table drives.
    pub fields: Vec<String>,
    /// One row per point, one value per field.
    pub rows: Vec<Vec<f64>>,
}

/// The Pulse ticks a scan plan schedules, compiled for the attached board.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PulseEstimate {
    /// Path of a zlc.pulse JSON document.
    pub document: String,
    /// One API timing combination per entry (field id -> [number, unit]); at most 256; none: the authored values.
    pub api_points: Option<Vec<BTreeMap<String, (f64, String)>>>,
    /// The plan's board axes: these scan slots, each with its values; the table is their cartesian product.
    pub scan: Option<PulseScanAxes>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PulseScanAxes {
    /// Field ids (pulse_describe scan_fields).
    pub fields: Vec<String>,
    /// Per field, its values in the field's own unit.
    pub columns: Vec<Vec<f64>>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PulseWait {
    /// The FIRE's command_id from pulse_run.
    pub command_id: u64,
    /// At most this long (at most 600).
    pub timeout_seconds: f64,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PulseOpen {
    /// Path of a zlc.pulse JSON document; the window may omit it to ask with a file dialog.
    pub path: Option<String>,
    /// Replace unsaved changes (otherwise refused).
    pub discard_changes: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PulseSave {
    /// Where to save (.json); omitted saves to the open file's path (the window asks when there is none).
    pub path: Option<String>,
    /// The window asks where, whatever the open file's path.
    pub save_as: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PulseDiscard {
    /// Replace unsaved changes (otherwise refused).
    pub discard_changes: Option<bool>,
}

/// One authoring edit; the arguments each edit takes are named in its summary.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PulseEdit {
    /// period_name {period_id,name} | duration {period_id,value,unit} | digital {period_id,port_key,high} |
    /// analog {period_id,port_key,mode: edge|ramp|hold,value} | delay {port_key,value,unit} |
    /// binding {field_kind: duration|analog|delay|bracket_count, period_id|bracket_id, port_key, scan, source: default|api|config} |
    /// config_binding {field_id,key} | bracket {bracket_id,start,end,count} | bracket_add {start,end,count} | bracket_remove {bracket_id} |
    /// period_add / spacer_add {before_item: [period|bracket, id] or null} | period_remove {period_id} | remove_items {items} |
    /// reorder {order: every [kind, id] item} | run_repeats {repeats} | scan_repeats {repeats} | document_name {name} |
    /// visible_ports {ports: [port keys] or null} | fill_port / clear_port {port_key} | port_label {port_key,label}
    pub edit: String,
    pub period_id: Option<String>,
    pub name: Option<String>,
    /// A number or the text typed in the field.
    pub value: Option<Value>,
    pub unit: Option<String>,
    pub port_key: Option<String>,
    pub high: Option<bool>,
    pub mode: Option<String>,
    pub field_kind: Option<String>,
    pub bracket_id: Option<String>,
    pub scan: Option<bool>,
    pub source: Option<String>,
    pub field_id: Option<String>,
    pub key: Option<String>,
    /// Inclusive first period id, or null (an empty bracket at an edge).
    pub start: Option<Value>,
    pub end: Option<Value>,
    pub count: Option<u64>,
    pub before_item: Option<Value>,
    pub items: Option<Value>,
    pub order: Option<Value>,
    pub repeats: Option<u64>,
    pub ports: Option<Value>,
    pub label: Option<String>,
}

/// The Scan page.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PulseScan {
    /// template {kind: columns|grid} | edit_source {source} | load_program {path} | run | load_array {path: .npy/.csv/.txt} |
    /// save_array {path?: .npy} | clear | repeats {repeats: 0 = until Stop}
    pub op: String,
    pub kind: Option<String>,
    pub source: Option<String>,
    pub path: Option<String>,
    pub repeats: Option<u64>,
}

/// The Value Config page.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PulseConfig {
    /// edit {rows: [[name, value, unit], ...]} | new | load {path} | refresh | unload | save {path?} | mode {enabled} |
    /// binding {field_id, key}
    pub op: String,
    pub rows: Option<Value>,
    pub path: Option<String>,
    pub enabled: Option<bool>,
    pub field_id: Option<String>,
    pub key: Option<String>,
    pub discard_changes: Option<bool>,
    /// save: the window asks where, whatever the draft's path.
    pub save_as: Option<bool>,
}

/// The Subpulse Config page and grouping on Edit.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PulseSubpulse {
    /// select {component_id ("" = the independent file draft)} | source {component_id, source: default|config, path} |
    /// open_source {component_id} | new | open {path} | save {path?} | save_as {path} | export {path, component_id?} |
    /// insert {before_item?} | insert_file {path, before_item?} | group {items} | ungroup / remove {component_id} |
    /// rename {name, component_id?} | edit {edit + that edit's arguments, as pulse_edit}
    pub op: String,
    pub component_id: Option<String>,
    pub source: Option<String>,
    pub path: Option<Value>,
    pub before_item: Option<Value>,
    pub items: Option<Value>,
    pub name: Option<String>,
    pub discard_changes: Option<bool>,
    /// source: the window asks for the file.
    pub browse: Option<bool>,
    pub edit: Option<String>,
    pub period_id: Option<String>,
    pub value: Option<Value>,
    pub unit: Option<String>,
    pub port_key: Option<String>,
    pub high: Option<bool>,
    pub mode: Option<String>,
    pub field_kind: Option<String>,
    pub bracket_id: Option<String>,
    pub scan: Option<bool>,
    pub field_id: Option<String>,
    pub key: Option<String>,
    pub start: Option<Value>,
    pub end: Option<Value>,
    pub count: Option<u64>,
    pub order: Option<Value>,
    pub ports: Option<Value>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PulseTarget {
    /// The Target page's records: [{key, kind: digital|dac, signal, endpoints, clock_key, clock_endpoint}].
    pub records: Vec<Value>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PulsePreview {
    /// Draw every output, not only those that switch.
    pub include_off: Option<bool>,
    /// A panel preset ("1x2", ...) or null to let the content choose.
    pub size: Option<Value>,
    /// Allow drag/zoom gestures on the preview.
    pub selectors: Option<bool>,
    /// Save the preview as PNG (beside the pulse when no path is given).
    pub save: Option<bool>,
    pub path: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PulsePage {
    /// Edit | Preview | Scan | Value Config | Subpulse Config | Target
    pub page: String,
}

#[derive(Clone, Copy, Debug, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PulseConnection { Virtual, Remote, Offline }

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PulseConnect {
    pub mode: PulseConnection,
    /// host:port of a pulse server (remote only).
    pub endpoint: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PulseStep {
    /// Scan points to move from the held one (-1 back, 1 forward).
    pub delta: i64,
}

/// What a command is, for every client.
pub struct CommandSpec {
    pub name: &'static str,
    /// One line: what it does and when to use it.
    pub summary: &'static str,
    /// Only reads: offered to every MCP client (v2 `allow_control` off offers only these).
    pub read_only: bool,
    /// Answered from instrument state without entering the work queue.
    pub query: bool,
    /// Composed by the MCP endpoint from other commands (or its own facts); the instrument never receives it.
    pub endpoint: bool,
    schema: fn() -> Value,
    validate: fn(&Value) -> Result<(), String>,
}

impl CommandSpec {
    pub fn input_schema(&self) -> Value { (self.schema)() }
    pub fn validate(&self, params: &Value) -> Result<(), String> { (self.validate)(params) }
}

fn schema<T: JsonSchema>() -> Value {
    let mut value = serde_json::to_value(schema_for!(T)).expect("schema serializes");
    if let Some(object) = value.as_object_mut() { object.remove("$schema"); object.remove("title"); }
    value
}

fn check<T: for<'de> Deserialize<'de>>(params: &Value) -> Result<(), String> {
    T::deserialize(params).map(|_| ()).map_err(|error| error.to_string())
}

macro_rules! command {
    ($name:literal, $ty:ty, read, $query:literal, $summary:literal) => {
        CommandSpec { name: $name, summary: $summary, read_only: true, query: $query, endpoint: false, schema: schema::<$ty>, validate: check::<$ty> }
    };
    ($name:literal, $ty:ty, control, $query:literal, $summary:literal) => {
        CommandSpec { name: $name, summary: $summary, read_only: false, query: $query, endpoint: false, schema: schema::<$ty>, validate: check::<$ty> }
    };
    (endpoint $name:literal, $ty:ty, $read:literal, $summary:literal) => {
        CommandSpec { name: $name, summary: $summary, read_only: $read, query: true, endpoint: true, schema: schema::<$ty>, validate: check::<$ty> }
    };
}

/// The catalogue, in the order a reader learns it.
pub static COMMANDS: &[CommandSpec] = &[
    command!(endpoint "guide", GuideTopic, true,
        "How the instrument's workflows go, by topic (start with overview); type:<type_id> is that plugin type's full descriptor. Changes nothing."),
    command!(endpoint "health", Empty, true,
        "Whether this endpoint and the instrument answer: pid, workspace, seconds since the instrument last answered, calls in flight, control. Answers even while the instrument is busy; not device readiness."),
    command!("get_state", GetState, read, true,
        "Read everything shown: workspace, apparatus, devices, logic rows and runs, board, document, acquisition, and each panel's kind, authored target, editable controls (with choices and current values), axes geometry, frame and computed fit results."),
    command!("get_panel_image", PanelRef, read, true,
        "The panel's current picture as a PNG, exactly the pixels the window shows."),
    command!("get_window_image", WindowRef, read, false,
        "One window as it is on screen (its view, open dialog and every plot) as a PNG, without saving a file."),
    command!("save_image", SaveImage, control, false,
        "Save image…: one window as it is on screen (its view, controls and every plot) as a PNG; the reply also carries the image."),
    command!("close_window", CloseWindow, control, false,
        "Close a window as its × does (a Figure Viewer drops its Figure); get_state windows names them."),
    command!("read_signal", ReadSignal, read, true,
        "A page of a published signal's numbers: values (non-finite as null), validity, sigma, and per value every axis's code, coordinate and label; the latest publication or the whole dataset."),
    command!("list_files", ListFiles, read, true,
        "The workspace's files and folders, workspace-relative, a page at a time."),
    command!("read_file", ReadFile, read, true,
        "A workspace file: text a page of bytes at a time, a PNG/JPEG as an image, a Figure archive (.npz) as a summary of its datasets, sections and members."),
    command!(endpoint "wait_for", WaitFor, true,
        "Re-read a read command until one field of its reply equals a value, at most timeout_seconds; answers matched or timed_out with the last reply."),
    command!(endpoint "batch", Batch, false,
        "Run already-decided commands in order (at most 16, 30 s); stops at the first error, refusal, unmet wait or unmet expect; earlier effects stay."),
    command!("close_experiment", CloseExperiment, control, false,
        "Close the experiment as the window's close button does: stop every run, close the devices, end this endpoint."),
    command!("show_window", ShowWindow, control, false,
        "Show a window in front by the name get_state windows lists it under (devices, task-console, figure-viewer-<n>, control:<device>, review:<node>, ...)."),
    command!("open_recorded_pulse", OpenRecordedPulse, control, false,
        "Open the pulse a run of the opened Figure played (its Devices row) in the Pulse Editor as an unsaved draft."),
    command!("list_signals", Empty, read, true,
        "Live signals that panels can plot: name, shape (axes and sizes), latest revision."),
    command!("get_fit", GetFit, read, false,
        "The panel's accepted fit as a table: parameter names and units, then per fitted cell its values, standard errors, validity, reduced chi-square and any failure message."),
    command!("wait", Wait, read, true,
        "Block until a node's run is ready or terminal (or asks the operator), frames have been produced, or a panel shows a newer frame, at most timeout_seconds."),
    command!("start_camera", StartCamera, control, false,
        "Start acquiring from the camera; a first image panel appears if there is none."),
    command!("stop_camera", Empty, control, false,
        "Stop acquiring; the last data stays for ROI, fits and saving."),
    command!("stop_all", Empty, control, false,
        "Stop every run and drive every registered output (the pulse board, RF outputs) to its safe state. Never refused."),
    command!("add_panel", AddPanel, control, false,
        "Add a plot panel of a live signal (a camera, or a panel-derived signal such as an ROI)."),
    command!("configure", Configure, control, false,
        "Change a panel: what axes play which role (semantic), display parameters, fit model, size, input binding, selectors (ROI/range/threshold/crosshair), viewport or facet focus."),
    command!("focus", Focus, control, false, "Open one facet-grid cell full size, or go back to the overview."),
    command!("reset_view", PanelRef, control, false, "Return a panel to automatic limits."),
    command!("fit", PanelRef, control, false, "Run the panel's configured fit now on the data it shows."),
    command!("scope_step", ScopeStep, control, false, "Step a pinned (scoped) axis to a neighbouring coordinate."),
    command!("pointer", Pointer, control, false,
        "A raw canvas gesture (drag a selector, pan, zoom); prefer configure, which states the result directly."),
    command!("close_panel", PanelRef, control, false, "Close a panel and release its input."),
    command!("edit_panel", PanelRef, control, false,
        "Open a board panel's Edit: the panel held still on the data it shows now (edit-<panel>), its settings shared with the panel; close it with close_panel."),
    command!("refresh_panel_edit", PanelRef, control, false, "Refresh snapshot: the panel's Edit takes the panel's newest data."),
    command!("open", Open, control, false, "Open a saved Figure archive (.npz) in a new Figure Viewer window, or in the one `window` names."),
    command!("save", Save, control, false, "Save a panel as shown as a PNG, PDF or SVG (by the path's suffix) plus its Figure archive (.npz); the reply lists the files written. \
        Either file already there: nothing is written and the save is refused, naming it, unless overwrite is true."),
    // Workbench: plugins, apparatus and devices.
    command!("list_plugins", ListPlugins, read, true,
        "Installed device and logic-node types: what each is for, its settings/fields with choices and units, inputs, outputs, device needs; and plugins that failed to load."),
    command!("plot_catalog", PlotCatalog, read, true,
        "What panels can do before one exists: plot kinds, facet cell kinds, axis roles, reductions and fit models (formulas and parameters with details)."),
    command!("get_device_log", DeviceLog, read, true, "The newest log lines of a device instance's plugin."),
    command!("add_device", AddDevice, control, false, "Add a device to the apparatus draft; it opens at apply_devices."),
    command!("remove_device", DeviceRef, control, false, "Remove a device from the apparatus draft; the open instance closes at apply_devices."),
    command!("edit_device", EditDevice, control, false, "Change a draft device's name, type (settings reset) or Init settings."),
    command!("edit_simulation", SimulationValues, control, false, "Change the simulated world's settings in the draft."),
    command!("use_template", UseTemplate, control, false, "Replace the apparatus draft with every virtual (or every hardware) device type."),
    command!("discover_devices", Empty, control, false, "Scan for connected hardware; results appear in apparatus.discovered."),
    command!("add_discovered", Discovered, control, false, "Copy a discovered device into the apparatus draft."),
    command!("load_apparatus", OptionalPath, control, false, "Replace the apparatus draft from a zlc.installation file."),
    command!("save_apparatus", SaveApparatus, control, false,
        "Save the apparatus draft as a zlc.installation file; replacing one it did not read in full needs replace_unread."),
    command!("apply_devices", Empty, control, false,
        "Init/Apply: open the devices the draft adds or changes, close those it removes (stopping runs that hold them). apparatus.action says which."),
    command!("close_device", DeviceRef, control, false, "Close one open device (its runs stop first); apply_devices reopens it."),
    command!("shut_down_devices", Empty, control, false, "Stop every run and close every device. Never refused."),
    command!("read_device", ReadDevice, read, false, "Read an open device's tunable fields: current readback, limits, whether a running node protects them; units reads fields in another spelling (MHz, mW)."),
    command!("tune_device", TuneDevice, control, false, "Apply one tunable field to an open device; the reply holds the readback."),
    command!("accept_device_risk", DeviceRisk, control, false,
        "Device Control's Accept risk: allow tuning live-safe fields of a device a running node holds."),
    command!("publish_device", PublishDevice, control, false,
        "Remote on hands an open device to another bench computer (its Scan hardware finds it; nothing here uses it meanwhile); off takes it back."),
    command!("share_start", Empty, control, false,
        "Browser sharing on -- the window, read-only, to browsers on the lab network that the operator allows one by one."),
    command!("share_stop", Empty, control, false, "Browser sharing off; every watching browser is told."),
    command!("share_decide", ShareDecide, control, false, "Allow or Reject a browser asking to watch, or Disconnect one."),
    // Workbench: logic nodes and runs.
    command!("add_node", AddNode, control, false, "Add a logic node (measurement, processor or task) as a stopped draft."),
    command!("rename_node", RenameNode, control, false, "Rename a logic node; references use its id, so nothing rewires."),
    command!("edit_node", EditNode, control, false,
        "Edit a node's draft: authoring values, source signal, devices, artifact files, auto preview. issues in get_state say what blocks Start."),
    command!("check_node", NodeRef, control, false, "Re-read a node's files and re-check its draft."),
    command!("revert_node", NodeRef, control, false, "Make a running node's draft its running configuration again."),
    command!("estimate_node", NodeRef, read, false, "The node type's estimate of a run with this draft (duration, shots), when it gives one."),
    command!("start_node", StartNode, control, false,
        "Start (or restart) a node: claims its devices, starts its run; a processor follows its source. Task runs get a run directory."),
    command!("stop_node", NodeRef, control, false, "Stop a node's run (its data is kept and sealed; a processor's signals are withdrawn, as v2). Never refused."),
    command!("remove_node", NodeRef, control, false, "Remove a node; a running one stops first."),
    command!("answer_request", AnswerRequest, control, false, "Answer the operator request a running node is waiting on."),
    command!("list_runs", ListRuns, read, true, "Runs of this session, newest first."),
    command!("get_run", RunRef, read, true, "One run: state, progress, artifacts, and its start.json/run.json records."),
    // Workbench: board and layouts.
    command!("order_panels", OrderPanels, control, false, "Reorder the board's panels."),
    command!("rename_panel", RenamePanel, control, false, "Name a panel (null returns to the automatic name)."),
    command!("set_panel_interval", PanelInterval, control, false, "A panel's update interval: 100, 200, 400 or 800 ms."),
    command!("set_panel_overlay", PanelOverlay, control, false,
        "Draw an image panel with a judgement's site rings (Occupancy's occupied, ...): their status colours from the same shot."),
    command!("publish_panel_outputs", PublishOutputs, control, false, "Which ROI/fit outputs a panel publishes as signals."),
    command!("set_board", SetBoard, control, false, "Pause or resume live display; turn plot gestures on or off."),
    command!("save_layout", OptionalPath, control, false, "Save the board (logic drafts and panels) as a zlc.console-board file."),
    command!("load_layout", LoadLayout, control, false,
        "Replace the board from a zlc.console-board file; refused while nodes run unless stop_running. Notes say what could not be kept."),
    command!("read_layout", PathRef, read, true, "What a layout file would load, without loading it."),
    command!("clear_board", ClearBoard, control, false, "Remove every logic node and panel; refused while nodes run unless stop_running."),
    command!("list_recipes", RecipeName, read, true, "Experiment recipes in the workspace, or one with its parameters and defaults."),
    command!("run_recipe", RunRecipe, control, false,
        "Load a recipe's layout with parameters and start its nodes; the reply names the result panels."),
    // Pulse: the board and its documents (docs/PULSE_V3.md).
    command!("pulse_describe", PulseDocument, read, false,
        "A pulse document's API fields in the order it plays them and its scan slots (field_id, label, kind, unit, is_dac, lo, hi, \
         limit_lo, limit_hi; API fields also value), API fields with no scan plan (unscannable) and its periods."),
    command!("pulse_run", PulseRun, control, false,
        "Play a pulse document on the attached board with these API values (and a scan table): compile, load, fire. The receipt: \
         command_id for pulse_wait_done, the values played, the table as played, timing (period starts, lane rising edges, ticks)."),
    command!("pulse_wait_done", PulseWait, control, false,
        "Wait for one FIRE's report, taken once: {done, status done|error|underflow, cursor, elapsed_seconds, command_id, fault}; \
         {done: false} while it plays. A FIRE that can no longer report (already reported, made safe, replaced) is refused, \
         and so is any client but the run while a run holds the board (taking the report is that run's)."),
    command!("pulse_estimate", PulseEstimate, read, false,
        "The Pulse ticks a scan plan schedules, compiled for the attached board with its Config: per API point the board table's \
         ticks, its first row's and the output-delay bound; sums over the points; clock_hz."),
    command!("pulse_stop", Empty, control, false,
        "Drive the pulse board SAFE now; what the editor plays stops. While a run holds the board it is refused (device is in use by \
         <node>): stop that node, whose own Stop makes the board safe."),
    command!("pulse_state", Empty, read, false,
        "The Pulse Editor's facts: open document, periods, bindings, ports, scan table, Config, board state, last warning."),
    command!("pulse_open", PulseOpen, control, false, "Open a pulse document in the Pulse Editor."),
    command!("pulse_save", PulseSave, control, false, "Save the Pulse Editor's draft as a zlc.pulse document (the file names the pulse)."),
    command!("pulse_new", PulseDiscard, control, false, "Start a new two-period pulse on the attached (or deployed) board."),
    command!("pulse_clear", PulseDiscard, control, false, "Clear the pulse to one safe period, keeping its file path."),
    command!("pulse_edit", PulseEdit, control, false,
        "One edit of the open pulse (durations, TTL levels, DAC steps, delays, bindings, brackets, periods, spacers, ports); the model refuses an illegal one."),
    command!("pulse_scan", PulseScan, control, false, "The Scan page: write, run or load the scan table; its repeats."),
    command!("pulse_config", PulseConfig, control, false, "The Value Config page: edit, load, save the value file; Config on/off; field names."),
    command!("pulse_subpulse", PulseSubpulse, control, false, "Subpulses: group periods, open/save/insert subpulse files, edit one."),
    command!("pulse_target", PulseTarget, control, false, "Apply the Target page (offline: re-wire; attached: rename outputs)."),
    command!("pulse_preview", PulsePreview, control, false, "The Preview page's timeline: off rows, size, gestures, save as PNG."),
    command!("pulse_page", PulsePage, control, false, "Show one page of the Pulse Editor."),
    command!("pulse_connect", PulseConnect, control, false,
        "Attach the Pulse Editor to the virtual board, a remote pulse server, or nothing (refused while the apparatus owns the board)."),
    command!("pulse_validate", Empty, read, false, "Compile what On Pulse would play, with the loaded Config, without touching the board."),
    command!("pulse_fire", Empty, control, false,
        "On Pulse: load what the editor shows and play it (Run and Scan repeats as authored)."),
    command!("pulse_hold", Empty, control, false, "Play the scan point the board is on, held until Stop."),
    command!("pulse_step", PulseStep, control, false, "Hold the scan point delta away from the held one."),
    command!("pulse_sync", PulseDiscard, control, false,
        "Bring the pulse the board holds into the editor (it replaces unsaved changes only with discard_changes)."),
];

pub fn find(name: &str) -> Option<&'static CommandSpec> { COMMANDS.iter().find(|spec| spec.name == name) }

/// What a new client reads first (kept short on purpose; `guide` holds the rest).
pub fn instructions() -> String {
    String::from(
        "ZLC v3 runs a neutral-atom experiment: devices, Pulse timing, camera and waveform acquisition, Calibration, \
         Occupancy, Frame Survival, Derive, Seamless Scan, SLM tasks, live plot/facet/fit panels and saved Figure \
         archives. The window and you send the same commands: what you do shows on screen, the operator's edits show \
         in get_state.\nStart with guide (topic overview) and list_plugins; neither needs devices. The workspace's \
         EXPERIMENT_CONTEXT.md (read_file), when present, holds this apparatus's settings.\nAddress logic nodes by \
         node_id; name is the operator's label and may change. Signals are @logic/<node_id>/<output>.\nThe native \
         chain already computes occupancy and conditional survival; panels pool repeats, facet by site and fit. \
         Configure panels and fits before Start; read results with get_fit and read_signal, not external scripts.\n\
         A reply means the command was applied, not that a run or the hardware finished: wait {node, until} or \
         wait_for a field. Never replay an uncertain Start, Fire or Apply: read get_state first. batch runs \
         already-decided steps and stops at the first refusal. stop_all is never refused.")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn every_command_has_an_object_schema_and_validates_its_examples() {
        for spec in COMMANDS {
            let schema = spec.input_schema();
            assert_eq!(schema["type"], "object", "{}", spec.name);
        }
        let ok = [
            ("start_camera", json!({"exposure_seconds": 0.1})),
            ("add_panel", json!({"signal": "camera/frames", "kind": "image"})),
            ("configure", json!({"panel": "panel-1", "parameters": {"colormap": "viridis"},
                                 "selectors": [{"kind": "area", "value": {"x": [1, 2], "y": [3, 4]}, "facet_index": null}],
                                 "viewport": null, "fit": {"model": "gaussian_offset"}})),
            ("focus", json!({"panel": "panel-3", "index": null})),
            ("wait", json!({"produced": 10, "timeout_seconds": 5})),
            ("stop_all", json!({})),
            ("edit_node", json!({"node": "occupancy", "values": {"model_kind": "default"}, "expected_draft_revision": 3})),
            ("answer_request", json!({"node": "calibration", "request": "calibration#1:1", "response": null})),
            ("wait", json!({"node": "calibration", "until": "terminal"})),
        ];
        for (name, params) in ok { find(name).unwrap().validate(&params).unwrap_or_else(|e| panic!("{name}: {e}")); }
        let bad = [
            ("start_camera", json!({})),
            ("add_panel", json!({"signal": "camera/frames", "kind": "scatter"})),
            ("configure", json!({"panel": "panel-1", "colour": "red"})),
            ("configure", json!({"panel": "panel-1", "selectors": [{"kind": "lasso", "value": 1, "facet_index": null}]})),
            ("stop_all", json!({"now": true})),
            ("use_template", json!({"template": "lab"})),
            ("publish_panel_outputs", json!({"panel": "panel-1", "outputs": {"center": "yes"}})),
        ];
        for (name, params) in bad { assert!(find(name).unwrap().validate(&params).is_err(), "{name} {params}"); }
    }

    #[test]
    fn the_capability_map_stays_short() {
        let text = instructions();
        assert!(text.len() < 2048, "{} bytes", text.len());
    }
}
