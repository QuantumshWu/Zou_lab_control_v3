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
 * inputs (`edit_node {artifact_inputs}`), and choose a file in the window's dialog.
 * `readOnly`: the page only shows the draft (v2 `read_only_fields`, Panel Edit's
 * Direct producer), so the editor offers, asks and writes nothing. */
export type EditorHost = {
  edit: DraftEdit;
  editArtifacts(inputs: Record<string, unknown>, what: string): number;
  choose(ask: PathAsk): Promise<string | null>;
  readOnly?: boolean;
};

/** The host of a page that only shows a node's draft: nothing it is asked reaches the draft. */
export const READ_ONLY: EditorHost = { edit: () => 0, editArtifacts: () => 0, choose: async () => null, readOnly: true };

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
  scan_plan: host => new ScanPlanEditor(host.edit, Boolean(host.readOnly)),
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

/** A node's editor models: its one, or a stack's, in order. */
export function editorModels(facts: EditorFacts | null | undefined): EditorFacts[] {
  return facts?.editor === "stack" ? (facts.editors as EditorFacts[] | undefined) ?? [] : facts ? [facts] : [];
}

/** The editors a page draws for a node's models: one per model, kept while its
 * kind stays.  The Logic Edit page draws them for editing, the Panel Edit's
 * Direct producer read only (READ_ONLY) -- one implementation of each. */
export class NodeEditors {
  private drawn: { kind: string; view: NodeEditor; facts: EditorFacts; key: string }[] = [];

  constructor(private host: EditorHost) {}

  get length(): number { return this.drawn.length; }
  /** The models an editor draws (a kind this UI does not draw is left to plain rows). */
  get models(): EditorFacts[] { return this.drawn.map(editor => editor.facts); }

  /** Brought to `models`; an editor no model wants leaves the page.  Answers the
   * editors' roots in their models' order, for the page to place. */
  reconcile(models: EditorFacts[]): HTMLElement[] {
    const kept: typeof this.drawn = [];
    models.forEach((model, index) => {
      const editor = this.drawn[index];
      if (editor && editor.kind === model.editor) { editor.facts = model; kept.push(editor); return; }
      const view = nodeEditor(model.editor, this.host);
      if (view) kept.push({ kind: model.editor, view, facts: model, key: "" });
    });
    for (const editor of this.drawn) if (!kept.includes(editor)) editor.view.root.remove();
    this.drawn = kept;
    return kept.map(editor => editor.view.root);
  }

  /** Each editor handed the row.  A state push that changes nothing an editor draws costs it
   * nothing: its model, the lock, the node's name and the fields it manages (an edit of another
   * field leaves it as it is).  True when any editor drew anew. */
  update(row: LogicRow, locked: string): boolean {
    let drew = false;
    for (const editor of this.drawn) {
      const fields = editor.facts.manages.map(name => [row.values[name], row.controls.find(control => control.name === name)]);
      const key = JSON.stringify([editor.facts, locked, row.name, fields]);
      if (key === editor.key) continue;
      editor.key = key;
      editor.view.update(row, editor.facts, locked);
      drew = true;
    }
    return drew;
  }

  /** The room the editors keep in the page's label column (NodeEditor.labelFloor). */
  labelFloor(): number { return Math.max(0, ...this.drawn.map(editor => editor.view.labelFloor?.() ?? 0)); }
}
