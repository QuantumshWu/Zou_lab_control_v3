// A node's own editor (v2 ui_contributions), drawn from the model its plugin
// answers for the draft (`editor_facts`, PLUGIN_PROTOCOL §2).  An editor is
// chosen by the model's `editor` kind, never by the node type; it holds no
// rule of the node's -- what is offered, derived, kept or wrong is in the
// model -- and writes back only the fields (and files) the model says it
// manages.  A `stack` model stacks several (Occupancy: its calibrations,
// then its post-selection).
import type { PathAsk } from "../../fluent/index.js";
import type { EditorFacts, LogicRow } from "../../workbench.js";
import { DeriveVariablesEditor } from "./derive_variables.js";
import { FrameClaimsEditor } from "./frame_claims.js";
import { PairMatrixEditor } from "./pair_matrix.js";
import { PostSelectionEditor } from "./post_selection.js";
import { ScanPlanEditor } from "./scan_plan.js";

/** A draft patch of managed fields (`edit_node {values}`); `what` names it in
 * the status line.  Returns the request id its accepted state will carry. */
export type DraftEdit = (values: Record<string, unknown>, what: string) => number;

/** What an editor can ask of its page: edit managed fields, edit managed file
 * inputs (`edit_node {artifact_inputs}`), and choose a file in the window's dialog. */
export type EditorHost = {
  edit: DraftEdit;
  editArtifacts(inputs: Record<string, unknown>, what: string): number;
  choose(ask: PathAsk): Promise<string | null>;
};

export interface NodeEditor {
  readonly root: HTMLElement;
  /** The row and its model; `locked` is why nothing may be edited now (a
   * running task owns the console), or empty. */
  update(row: LogicRow, facts: EditorFacts, locked: string): void;
  /** The room (CSS px) the editor keeps in the label column for words that
   * come and go; the column is never narrower. */
  labelFloor?(): number;
}

const EDITORS: Record<string, (host: EditorHost) => NodeEditor> = {
  scan_plan: host => new ScanPlanEditor(host.edit),
  pair_matrix: host => new PairMatrixEditor(host.edit),
  derive_variables: host => new DeriveVariablesEditor(host.edit),
  frame_claims: host => new FrameClaimsEditor(host),
  post_selection: host => new PostSelectionEditor(host.edit),
};

/** The editor for a model kind, or null for a kind this UI does not draw
 * (the Edit page then shows the managed fields as plain rows). */
export function nodeEditor(kind: string, host: EditorHost): NodeEditor | null {
  return EDITORS[kind]?.(host) ?? null;
}
