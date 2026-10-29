import { DeriveVariablesEditor } from "./derive_variables.js";
import { FrameClaimsEditor } from "./frame_claims.js";
import { PairMatrixEditor } from "./pair_matrix.js";
import { PostSelectionEditor } from "./post_selection.js";
import { ScanPlanEditor } from "./scan_plan.js";
/** The host of a page that only shows a node's draft: nothing it is asked reaches the draft. */
export const READ_ONLY = { edit: () => 0, editArtifacts: () => 0, choose: async () => null, readOnly: true };
const EDITORS = {
    scan_plan: host => new ScanPlanEditor(host.edit, Boolean(host.readOnly)),
    pair_matrix: host => new PairMatrixEditor(host.edit),
    derive_variables: host => new DeriveVariablesEditor(host.edit),
    frame_claims: host => new FrameClaimsEditor(host),
    post_selection: host => new PostSelectionEditor(host.edit),
};
/** The editor for a model kind, or null for a kind this UI does not draw
 * (the Edit page then shows the managed fields as plain rows). */
export function nodeEditor(kind, host) {
    return EDITORS[kind]?.(host) ?? null;
}
/** A node's editor models: its one, or a stack's, in order. */
export function editorModels(facts) {
    return facts?.editor === "stack" ? facts.editors ?? [] : facts ? [facts] : [];
}
/** The editors a page draws for a node's models: one per model, kept while its
 * kind stays.  The Logic Edit page draws them for editing, the Panel Edit's
 * Direct producer read only (READ_ONLY) -- one implementation of each. */
export class NodeEditors {
    host;
    drawn = [];
    constructor(host) {
        this.host = host;
    }
    get length() { return this.drawn.length; }
    /** The models an editor draws (a kind this UI does not draw is left to plain rows). */
    get models() { return this.drawn.map(editor => editor.facts); }
    reset() { for (const editor of this.drawn) {
        editor.view.reset?.();
        editor.key = "";
    } }
    /** Brought to `models`; an editor no model wants leaves the page.  Answers the
     * editors' roots in their models' order, for the page to place. */
    reconcile(models) {
        const kept = [];
        models.forEach((model, index) => {
            const editor = this.drawn[index];
            if (editor && editor.kind === model.editor) {
                editor.facts = model;
                kept.push(editor);
                return;
            }
            const view = nodeEditor(model.editor, this.host);
            if (view)
                kept.push({ kind: model.editor, view, facts: model, key: "" });
        });
        for (const editor of this.drawn)
            if (!kept.includes(editor))
                editor.view.root.remove();
        this.drawn = kept;
        return kept.map(editor => editor.view.root);
    }
    /** Each editor handed the row.  A state push that changes nothing an editor draws costs it
     * nothing: its model, the lock, the node's name and the fields it manages (an edit of another
     * field leaves it as it is).  True when any editor drew anew. */
    update(row, locked) {
        let drew = false;
        for (const editor of this.drawn) {
            const fields = editor.facts.manages.map(name => [row.values[name], row.controls.find(control => control.name === name)]);
            const key = JSON.stringify([editor.facts, locked, row.node_id, row.type_id, row.name, fields, editor.view.contextKey?.(row)]);
            if (key === editor.key)
                continue;
            editor.key = key;
            editor.view.update(row, editor.facts, locked);
            drew = true;
        }
        return drew;
    }
    /** The room the editors keep in the page's label column (NodeEditor.labelFloor). */
    labelFloor() { return Math.max(0, ...this.drawn.map(editor => editor.view.labelFloor?.() ?? 0)); }
}
