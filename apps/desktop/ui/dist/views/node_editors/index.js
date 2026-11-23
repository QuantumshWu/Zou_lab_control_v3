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
/** Each node's last model its plugin answered for the draft as it then stood (by node ID: its
 * type, that draft and its revision), kept across the pages that draw the node (its Edit tab
 * closed and opened again, Panel Edit's producer) while the UI runs; never written anywhere. */
const answered = new Map();
/** The draft a node's model is answered from, as its row says it: values, devices, source, files. */
const draftOf = (row) => JSON.stringify([row.values, row.device_keys, row.source_signal, row.artifact_inputs]);
/** Why a kept model's editors write nothing (nodeModel). */
export const KEPT_LOCK = "Shown from the last editor model of this same draft: read only until the node's plugin answers again";
/** The model a node's editors draw from its row: the row's own; where it has none (its plugin
 * could not answer), the node's last one while the draft is still the one it was answered for
 * (`kept`: drawn locked, KEPT_LOCK, so a model that may be stale never writes); else none, and
 * the fields it would manage are plain rows.  A node of another type under the ID, or one whose
 * draft revision went back (a replacement or a reload under the ID), is another node: its last
 * model goes. */
export function nodeModel(row) {
    // A model of an older draft (the draft's own still asked) is not the one kept for this draft.
    if (row.editor_facts && !row.editor_facts_pending) {
        answered.set(row.node_id, { type: row.type_id, draft: draftOf(row), revision: row.draft_revision, facts: row.editor_facts });
    }
    if (row.editor_facts)
        return { facts: row.editor_facts, kept: false };
    let last = answered.get(row.node_id);
    if (last && (last.type !== row.type_id || row.draft_revision < last.revision)) {
        answered.delete(row.node_id);
        last = undefined;
    }
    const kept = last?.draft === draftOf(row);
    return { facts: kept ? last.facts : null, kept };
}
/** An accepted replacement (Load/New/Clear): any node may be another one, so every kept model goes. */
export function forgetModels() { answered.clear(); }
/** Why the node's editor model could not be had for this draft (`editor_facts_error`), as its pages say it. */
export function editorProblem(row) {
    return row.editor_facts_error ? `The editor cannot be drawn: ${row.editor_facts_error}` : "";
}
/** The editors a page draws for a node's models: one per model, kept while its
 * kind stays.  The Logic Edit page draws them for editing, the Panel Edit's
 * Direct producer read only (READ_ONLY) -- one implementation of each. */
export class NodeEditors {
    host;
    /** Each editor, its model, and the key and the lock it was last drawn with. */
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
                kept.push({ kind: model.editor, view, facts: model, key: "", locked: "" });
        });
        for (const editor of this.drawn)
            if (!kept.includes(editor))
                editor.view.root.remove();
        this.drawn = kept;
        return kept.map(editor => editor.view.root);
    }
    /** Each editor handed the row.  A state push that changes nothing an editor draws costs it
     * nothing: its model, the lock, the node's name and the fields it manages (an edit of another
     * field leaves it as it is).  A model still of an older draft draws nothing where the editor
     * already stands: it would put back what the operator has just changed until the answer comes;
     * standing, it still takes no input under a lock it was not drawn with (a Task taking the console
     * over), and once drawn its own lock takes over.  `kept`: the models are the node's kept ones, of
     * this same draft (nodeModel), and draw at once.  True when any editor drew anew. */
    update(row, locked, kept = false) {
        let drew = false;
        for (const editor of this.drawn) {
            const stands = Boolean(row.editor_facts_pending && !kept && editor.key);
            editor.view.root.inert = stands && Boolean(locked) && locked !== editor.locked;
            if (stands)
                continue;
            const fields = editor.facts.manages.map(name => [row.values[name], row.controls.find(control => control.name === name)]);
            const key = JSON.stringify([editor.facts, locked, row.node_id, row.type_id, row.name, fields, editor.view.contextKey?.(row)]);
            if (key === editor.key)
                continue;
            editor.key = key;
            editor.locked = locked;
            editor.view.update(row, editor.facts, locked);
            drew = true;
        }
        return drew;
    }
    /** The room the editors keep in the page's label column (NodeEditor.labelFloor). */
    labelFloor() { return Math.max(0, ...this.drawn.map(editor => editor.view.labelFloor?.() ?? 0)); }
}
