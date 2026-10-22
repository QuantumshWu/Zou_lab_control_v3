// The `pulse` state section (zlc_pulse_editor::Editor::state): v2's
// zlc_ui.pulse view models as JSON.  The view renders these and sends
// pulse_* commands; no pulse rule lives in TypeScript.
import type { Axis } from "../protocol.js";
export type FieldVM = {
  text: string; editable: boolean; scan: boolean; source: "default" | "api" | "config"; can_scan: boolean; can_api: boolean;
  effective_text: string; source_text: string; config_key: string; validator_kind: "none" | "int" | "float";
  validator_lo: number; validator_hi: number; resolution: number; allow_any: boolean; field_id: string; config_names: string[];
  config_active: boolean; config_enabled: boolean; default_text: string; binding_editable: boolean;
};
export type PortRow = { key: string; kind: "digital" | "dac" | "clock"; label: string; endpoint_text: string; endpoint_tooltip: string;
  width: number; lo: number; hi: number; visible: boolean };
export type PeriodVM = { period_id: string; name: string; duration: FieldVM; unit: string; unit_choices: string[];
  digital: [string, boolean][]; analog: [string, string, FieldVM][]; kind: "period" | "spacer"; readonly: boolean };
export type BracketVM = { bracket_id: string; start_period_id: string | null; end_period_id: string | null; count: number; ordinal: number;
  color: string; count_field: FieldVM; readonly: boolean };
export type ComponentVM = { component_id: string; name: string; period_ids: string[]; total_text: string; summary_text: string;
  bracket_ids: string[]; source: "default" | "config"; config_path: string; config_active: boolean; error: string };
export type DelayRow = { port_key: string; value: FieldVM; unit: string; units: string[] };
export type Schedule = {
  revision: number; document_name: string; clock_text: string; total_text: string; total_tooltip: string; period_count: number;
  visible_text: string; summary_text: string; ports: PortRow[]; periods: PeriodVM[]; analog_mode_choices: { label: string; value: string }[];
  brackets: BracketVM[]; run_repeats: number; delay_rows: DelayRow[]; scan_summary_text: string; min_bracket_count: number;
  default_bracket_count: number; components: ComponentVM[]; item_order?: [string, string][];
};
export type Item = [kind: string, key: string];
export type Connection = { choices: { label: string; value: string; endpoint_editable: boolean }[]; selected: string; endpoint: string;
  status: string; status_word: "idle" | "running" | "task" | "warning" | "error"; locked: boolean };
export type Control = { running: boolean; synchronized: boolean; file_dirty: boolean; can_run: boolean; can_stop: boolean; can_sync: boolean;
  can_hold: boolean; can_step: boolean; claimed_by: string | null; held_point: number | null };
export type BindingRecord = { field_id: string; label: string; scan: boolean; source: string; group_id: string; group_label: string; note: string };
export type ScanPage = { slots_text: string; bindings: BindingRecord[]; table_text: string; table_note: string; source_text: string;
  source_dirty: boolean; repeats: number; busy: boolean; progress_text: string; progress_word: "idle" | "running" | "task" | "warning" | "error";
  progress_polling: boolean; points?: number };
export type ConfigPage = { file_path: string; dirty: boolean; entries: [string, string, string][]; available_names: string[];
  bindings: [string, string, string, string, string, string][]; active_path: string; busy: boolean; binding_groups: [string, string, string][];
  parked: string[]; enabled: boolean; error: string; directory: string | null };
export type SubpulsePage = { sources: { component_id: string; name: string; source: string; path: string; status: string; word: string }[];
  enabled: boolean; busy: boolean; error: string; document: Schedule | null; contexts: [string, string][]; context_id: string;
  selected_id: string; path: string; dirty: boolean; bindings: [string, string, string][]; config_names: string[]; save_target: string };
export type TargetRecord = { key: string; kind: "digital" | "dac"; signal: string; endpoints: string[]; clock_key: string | null; clock_endpoint: string | null };
export type TargetPage = { records: TargetRecord[]; editable: boolean; caption: string; reserved: string[]; feedback: string;
  feedback_word: "idle" | "running" | "task" | "warning" | "error"; width_rules: { digital: number[]; dac: number[] } | null };
export type Preview = { include_off: boolean; size: string | null; selectors: boolean; status: string; status_word: "idle" | "running" | "task" | "warning" | "error";
  sizes: string[]; panel: string; frame: { width: number; height: number; sequence: number } | null;
  /** The drawn timeline's axes, hit-tested by the page's pointer. */
  axes: Axis[] };
export type PulseState = {
  revision: number; path: string; dirty: boolean; page: string; title: string; summary: string;
  status: "idle" | "dirty-ready" | "running-synced" | "running-stale" | "unreachable"; dot: "idle" | "running" | "warning" | "error";
  busy: string | null; stopping: boolean; message: { id: number; kind: "info" | "warning"; text: string } | null; last_warning: string;
  has_pulse: boolean; schedule: Schedule; item_order: Item[]; connection: Connection; control: Control;
  config_mode: { enabled: boolean; busy: boolean; error: string }; scan: ScanPage; config: ConfigPage; subpulse: SubpulsePage;
  target: TargetPage; preview: Preview; board: Record<string, unknown>; time_units: string[]; save_folder: string; proposed_path: string;
  apparatus_device: string | null;
};
