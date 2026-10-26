// The workbench sections of the accepted state (docs/WORKBENCH_COMMANDS.md
// §3 on claude/v3-workbench).  The views render these and send §4's commands;
// none of the rules behind them lives in the UI.
import type { Control } from "./protocol.js";
import type { FieldSpec, TreeGroup, Value } from "./fluent/index.js";

/** A workbench form field: the panel Control shape plus §3.7's additions. */
export type WorkbenchControl = FieldSpec & Partial<Pick<Control, "section">> & {
  required?: boolean;
  enabled?: boolean;
  problem?: string;
  blank_label?: string;
  /** The form section the field stands in (AuthoringField.group); the Edit page titles each change. */
  group?: string;
};

export type Workspace = {
  root: string; data: string; today: string; pulses: string;
  layout: { path: string; dirty: boolean } | null;
  notes: string[];
};

export type DeviceType = {
  type_id: string; family: string; label: string; description?: string;
  /** How the family reads as a card title. */
  family_label: string;
  /** The type can scan for its hardware (Scan hardware). */
  discoverable?: boolean;
  capabilities: string[]; physical_outputs: { name: string }[]; addable: boolean;
};

export type Apparatus = {
  path: string; dirty: boolean;
  /** Why the file at `path` was not read in full (v2 `_path_unread`): Save goes through Save as. */
  unread?: string | null;
  simulation?: Record<string, Value>;
  types: DeviceType[];
  unavailable: { module: string; reason: string }[];
  templates: string[];
  discovered: { instance_id: string; type_id: string; role: string; parameters?: Record<string, unknown> }[];
  action: "init" | "apply" | "shut_down" | null;
  busy: string | null;
  /** The Device Manager's last line (v2 _report): text and its status word. */
  report?: { text: string; severity: "idle" | "task" | "warning" | "error" };
};

export type TunableField = WorkbenchControl & {
  current: Value;
  desired: Value;
  live_write: boolean;
  device_limits: [number, number] | null;
  /** The reading and the device's own range in words (v2 format_quantity, in the row's spelling;
   * an inexact conversion rounded after "≈ "), and with every digit (their tooltips). */
  current_text: string;
  limits_text: string;
  current_exact?: string;
  limits_exact?: string;
  editable: boolean;
  reason?: string;
  /** The row at rest: Ready, Protected, Stands outside its bounds. */
  status: string;
  severity: "idle" | "warning";
};

export type DeviceRow = {
  instance_id: string; role: string; type_id: string; family: string;
  parameters?: Record<string, unknown>;
  controls: WorkbenchControl[];
  state: "draft" | "opening" | "open" | "closing" | "failed";
  close_failed?: boolean;
  pending: "add" | "change" | "remove" | null;
  error: string | null;
  claimed_by: { node_id: string; run: string; protected_fields: string[] }[];
  physical_outputs: { name: string }[];
  commands: string[];
  /** The window its Control opens instead of the field form (`slm-editor`), or "". */
  control_window?: string;
  fields: TunableField[] | null;
  /** Device Control's title and policy line, once the device was read. */
  control: { owners: string[]; reason: string; risk_enabled: boolean } | null;
  /** Operator accepted the risk of tuning this claimed device (accept_device_risk). */
  risk_accepted?: boolean;
  /** Remote on: handed to another bench computer (publish_device). */
  published?: boolean;
  /** An open device of this computer, which Remote may hand over. */
  publishable?: boolean;
  /** A device another bench computer publishes: where it is and what it is there. */
  remote?: { host: string; port: number; instance_id: string; publisher: string; role: string; type_id: string; label: string } | null;
};

/** Browser sharing (the `share` section): where browsers watch, and who asks or watches. */
export type Share = {
  active: boolean; urls: string[]; port: number | null; error: string | null;
  clients: { id: string; address: string; browser: string; status: "pending" | "connected" }[];
};

export type RunRow = {
  run: string; node_id: string;
  state: "starting" | "running" | "stopping" | "completed" | "stopped" | "failed";
  ready: boolean; started_at: string; ended_at: string | null;
  progress: { message: string; current: number; total: number } | null;
  request: { request: string; kind: string; title: string; message: string; payload: Record<string, unknown> } | null;
  directory: string | null;
  artifacts: { name: string; path: string; role: string; contract_id: string; size_bytes: number }[];
  generation: string | null; stop_reason: string | null; error: { type: string; message: string } | null;
};

export type LogicRow = {
  node_id: string; name: string; type_id: string;
  node_kind: "measurement" | "processor" | "task" | "panel_publisher" | string;
  description?: string; draft_revision: number;
  values: Record<string, Value>;
  controls: WorkbenchControl[];
  /** The draft fields a selector of its output writes (v2 SelectionMapping.draft_fields). */
  selector_fields?: string[];
  source_signal: string | null;
  /** The source row's words (the input's name, "Signal bundle"); null where the draft uses no dataset input. */
  source_label: string | null;
  source_choices: [string, string][];
  /** Its Signal picker's tree (v2 _source_choices); absent for a bundle, offered flat. */
  source_tree?: TreeGroup[] | null;
  device_keys: Record<string, string>;
  device_choices: Record<string, [string, string][]>;
  /** How each device argument reads as a row label. */
  device_labels: Record<string, string>;
  /** Each file input as the draft holds it: a path, or the files with the frames each claims. */
  artifact_inputs: Record<string, ArtifactValue>;
  /** The node's file inputs as it declares them (label, its codec's dialog filter, frame claims). */
  artifact_specs?: { name: string; label: string; file_filter: string; required: boolean; frame_claims: boolean }[];
  auto_preview: boolean;
  outputs: { signal: string; contract: string; label: string; shape: string | null }[];
  issues: string[];
  can_start: boolean;
  /** Anything to stop: running, following its source, a queued start. */
  can_stop: boolean;
  /** The node declares a preview the Plot switch would open. */
  preview_offered: boolean;
  edited: boolean;
  state: "idle" | "draft" | "running" | "warning" | "error";
  status: string;
  phase: "idle" | "following" | "waiting" | "starting" | "running" | "stopping" | "removing";
  run: RunRow | null;
  /** The node editor's model its plugin answers (`editor_facts`, the latest one
   * while a newer draft is being asked): the editor of that kind draws it in
   * place of the fields it manages.  `editor_facts_error`: why it could not. */
  editor_facts?: EditorFacts | null;
  editor_facts_error?: string | null;
};

export type ArtifactValue = string | null | { path: string; frames?: string[] }[];

export type EditorFacts = { editor: string; manages: string[] } & Record<string, unknown>;

export type NodeType = { type_id: string; label: string; node_kind: string; publishes: string; description?: string };

export type Board = {
  panels: { panel: string; title: string | null; name: string; interval_ms: number; published_outputs: Record<string, boolean>;
    /** The judgement whose site rings the image is drawn with ("": none), and the judgements of its measurement. */
    overlay_signal: string; overlay_choices: [string, string][];
    /** Setting's Signal and Overlay pickers (v2 keyed_choice): the trees they open. */
    signal_tree: TreeGroup[]; overlay_tree: TreeGroup[];
    /** What the panel can publish (ROI frame, fit parameters), each with its signal. */
    /** What the panel can publish (v2 publisher fields); `live`: its signal exists now. */
    outputs: { name: string; label: string; signal: string; shape: string | null; live: boolean; published: boolean }[] }[];
  paused: boolean;
  selectors: boolean;
  /** The Task Console's last line (v2 ConsolePresenter._report). */
  report?: { text: string; severity: "idle" | "task" | "warning" | "error" };
  active_task: string | null;
  frozen: Record<string, string>;
};
