// The Task Console's Logic rows and Edit pages (v2 LogicRowView,
// LogicEditorView).  Both draw one node's projection and send its intents;
// admission, follow intent, restarts and claims are the workbench's.
import {
  Command, Field, Legend, Readout, StatusLine, assign, command, element, fieldKind, form, row as formRow, sectionTitle, setChecked, setState, setTitle,
  shareLabelColumn, switchButton, type FieldSpec, type PathAsk, type StatusWord, type Value,
} from "../fluent/index.js";
import type { LogicRow } from "../workbench.js";
import { NodeEditors, editorModels } from "./node_editors/index.js";

/** Every word the Start slot can say, on the row and on its Edit page. */
export const START_LABELS = ["Start", "Restart", "Starting…"] as const;

/** Each node state as the status word it is drawn with (v2 STATE_WORDS): a
 * draft is as calm as a stopped node -- what keeps it from starting is
 * something to fill in, not a fault. */
const STATE_WORDS: Record<LogicRow["state"], StatusWord> = { idle: "idle", draft: "idle", running: "running", warning: "warning", error: "error" };
/** The kind column's words, one width on every row. */
const KIND_WORDS: Record<string, string> = { measurement: "(measurement)", processor: "(processor)", task: "(task)", panel_publisher: "(panel publisher)" };
const kindWords = (kind: string): string => KIND_WORDS[kind] ?? `(${kind})`;

/** What a node row needs to be drawn: a workbench row (or the transitional
 * built-in camera, views/task_console.ts) as one shape. */
export type NodeProjection = {
  id: string;
  name: string;
  details: string[];
  kind: string;
  state: LogicRow["state"];
  status: string;
  canStart: boolean;
  canStop: boolean;
  /** A Start waits (behind a device release, its sources). */
  pending: boolean;
  edited: boolean;
  previewOffered: boolean;
  autoPreview: boolean;
  outputs: [name: string, shape: string, description: string][];
  /** A panel's published outputs (v2 panel publisher row): nothing to start
   * or remove; its name, kind, outputs and Edit stand in the node columns. */
  publisher?: boolean;
};

export type NodeIntents = {
  start(id: string): void;
  stop(id: string): void;
  edit(id: string): void;
  remove(id: string): void;
  preview(id: string, on: boolean): void;
};

/** What Start does now, one rule for the row and its Edit page: Restart while
 * there is anything to stop, Starting… while its start waits. */
export function startLabel(projection: NodeProjection): string {
  return projection.pending ? "Starting…" : projection.canStop ? "Restart" : "Start";
}

/** A workbench row as a node projection. */
export function projectRow(row: LogicRow): NodeProjection {
  return {
    id: row.node_id,
    name: row.name,
    details: [row.type_id, `Internal ID: ${row.node_id}`],
    kind: row.node_kind,
    state: row.state,
    status: row.status,
    canStart: row.can_start,
    canStop: row.can_stop,
    pending: row.phase === "waiting",
    edited: row.edited,
    previewOffered: row.preview_offered,
    autoPreview: row.auto_preview,
    // v2 PublishedItemsLegend rows: the output's own name (signal_output_name) and its shape.
    outputs: row.outputs.map(output => [output.signal.split("/").pop() ?? output.label, output.shape || "—", `${output.signal} · ${output.contract}`]),
  };
}

/** One logic node on the Logic page: dot, name, kind and status in fixed
 * columns, then Plot, Start, Stop, Edit and the two-click ×; its outputs as
 * chips under the line.  Start and Stop are two fixed slots that are always
 * there, so nothing moves when a run begins. */
export class LogicRowView {
  readonly root = element("article", "card bordered logic-row");
  private dot = element("span", "dot");
  private name = new Readout("", false, true);
  private kind = element("span", "kind-column");
  private status = element("span", "status-text ink logic-status");
  private plot: HTMLButtonElement;
  private start: Command;
  private stop: Command;
  private remove: Command;
  private legend = new Legend();
  private projection: NodeProjection | null = null;

  constructor(readonly id: string, intents: NodeIntents) {
    this.plot = switchButton(on => intents.preview(this.id, on), "Plot");
    this.plot.classList.add("row-switch");
    this.plot.title = "Open this node's plot panel when it starts";
    this.start = command("Start", () => intents.start(this.id), { words: START_LABELS });
    this.stop = command("Stop", () => intents.stop(this.id), { role: "destructive" });
    const edit = command("Edit", () => intents.edit(this.id));
    this.remove = command("×", () => intents.remove(this.id), { role: "destructive", glyph: true, square: true, confirm: "remove this node" });
    // The kind column is as wide as its widest word on every row.
    for (const words of Object.values(KIND_WORDS)) this.kind.append(element("span", "", words));
    this.kind.append(element("span", "current"));
    this.name.root.classList.add("logic-name");
    const top = element("div", "logic-top");
    top.append(this.dot, this.name.root, this.kind, this.status, this.plot, this.start.root, this.stop.root, edit.root, this.remove.root);
    this.root.append(top, this.legend.root);
    this.root.dataset.node = id;
  }

  update(projection: NodeProjection, takeover: { active: boolean; owns: boolean }): void {
    this.projection = projection;
    this.name.text = projection.name;
    setTitle(this.name.root, [projection.name, ...projection.details].filter(Boolean).join(" · "));
    const words = kindWords(projection.kind);
    const current = this.kind.querySelector<HTMLElement>(".current")!;
    if (current.textContent !== words) current.textContent = words;
    const word = STATE_WORDS[projection.state];
    setState(this.dot, word);
    setState(this.status, word);
    if (this.status.textContent !== projection.status) { this.status.textContent = projection.status; this.status.title = projection.status; }
    assign(this.plot, "hidden", !projection.previewOffered);
    setChecked(this.plot, projection.autoPreview);
    // A running Task owns the console: rows stay readable, and the task's
    // own row keeps its Stop (the strip's Stop task is the same command).
    const free = !takeover.active;
    this.start.text = startLabel(projection);
    this.start.disabled = !free || !projection.canStart || projection.pending;
    this.stop.disabled = !projection.canStop || !(free || takeover.owns);
    assign(this.plot, "disabled", !free);
    this.remove.disabled = !free;
    for (const node of [this.dot, this.status, this.plot, this.start.root, this.stop.root, this.remove.root]) node.classList.toggle("kept-place", Boolean(projection.publisher));
    this.legend.setRows(projection.outputs);
  }

  get state(): StatusWord { return this.projection ? STATE_WORDS[this.projection.state] : "idle"; }
}

export type EditorIntents = NodeIntents & {
  rename(id: string, name: string): number;
  /** One edited value of the draft: `{values}`, `{source_signal}` or `{device_keys}`. */
  editDraft(id: string, change: Record<string, unknown>, label: string): number;
  revert(id: string): void;
  check(id: string): void;
  /** The window's dialog for a path the node reads (v2 FluentPathEdit Browse…): the chosen path, or null. */
  choose(ask: PathAsk): Promise<string | null>;
};

/** One node's Edit page: what it is and how it stands, its Name, its source
 * and devices, its files and its parameters in one label column, and the
 * action bar (Remove … Revert, Plot, Start, Stop).  Only the rows scroll. */
export class LogicEditorView {
  readonly root = element("section", "logic-editor");
  private title = new Readout("", true, true);
  private kind = element("span", "note");
  private statusLine = new StatusLine();
  private body = element("div", "logic-editor-body");
  private footer = element("div", "logic-editor-actions");
  private form = form();
  private fields = new Map<string, Field>();
  private plot: HTMLButtonElement;
  private start: Command;
  private stop: Command;
  private revert: Command;
  private remove: Command;
  private row: LogicRow | null = null;
  private projection: NodeProjection | null = null;
  private groupTitles = new Map<string, HTMLDivElement>();
  /** The node's own editors. */
  private editors: NodeEditors;
  /** Why the node's editor model could not be had for this draft (`editor_facts_error`). */
  private editorError = element("p", "field-error editor-error");
  private artifactFields = new Map<string, Field>();

  constructor(readonly id: string, private intents: EditorIntents) {
    const header = element("div", "logic-editor-header");
    const titleRow = element("div", "logic-editor-title");
    this.title.root.classList.add("heading");
    titleRow.append(this.title.root, this.kind);
    header.append(titleRow, this.statusLine.root);
    this.statusLine.text.classList.add("wrap");
    this.editorError.hidden = true;
    this.body.append(this.form, this.editorError);
    this.editors = new NodeEditors({
      edit: (values, what) => intents.editDraft(id, { values }, what),
      editArtifacts: (inputs, what) => intents.editDraft(id, { artifact_inputs: inputs }, what),
      choose: ask => intents.choose(ask),
    });
    const footer = this.footer;
    this.remove = command("Remove", () => intents.remove(id), { role: "destructive", confirm: "remove this node" });
    this.revert = command("Revert", () => intents.revert(id), { role: "destructive",
      title: "Discard this page's edits and put back the configuration the running node uses; the edits do not come back" });
    this.plot = switchButton(on => intents.preview(id, on), "Plot");
    this.plot.classList.add("row-switch");
    this.plot.title = "Open this node's plot panel when it starts";
    this.start = command("Start", () => intents.start(id), { role: "primary", words: [...START_LABELS, "Restart*"] });
    this.stop = command("Stop", () => intents.stop(id), { role: "destructive" });
    footer.append(this.remove.root, element("span", "spacer"), this.revert.root, this.plot, this.start.root, this.stop.root);
    this.root.append(header, this.body, footer);
    // The page's one label column (v2 LogicEditorView._label_width), shared by
    // its form and its node editor; measured again when the page is laid out
    // anew (shown, resized) -- in that frame, before it is painted.
    new ResizeObserver(() => { if (this.editors.length) this.fitLabels(); }).observe(this.body);
  }

  /** The label column, measured in the frame this update is drawn in (the browser lays the
   * page out for it anyway), never right after this update wrote the DOM. */
  private shareLabels(): void {
    if (this.labelsQueued) return;
    this.labelsQueued = true;
    requestAnimationFrame(() => this.fitLabels());
  }
  private fitLabels(): void {
    this.labelsQueued = false;
    shareLabelColumn(this.body, this.editors.labelFloor());
  }
  private labelsQueued = false;

  /** The node's page from its workbench row; `projection` is the row's
   * drawn words, `takeover` the console-wide Task gate. */
  update(row: LogicRow, projection: NodeProjection, takeover: { active: boolean; owns: boolean }, request?: number, failed = false): void {
    this.row = row;
    this.projection = projection;
    this.title.text = projection.name;
    setTitle(this.title.root, [projection.name, ...projection.details].join(" · "));
    const kind = kindWords(projection.kind);
    if (this.kind.textContent !== kind) this.kind.textContent = kind;
    this.statusLine.show(projection.status, STATE_WORDS[projection.state]);
    const locked = takeover.active ? "A running task owns the console" : "";
    const specs: { key: string; group?: string; spec: FieldSpec; commit(value: Value): number }[] = [];
    // THE NAME: what the node is called on its row, this page and its tab.
    specs.push({ key: "name", spec: { name: "name", label: "Name", kind: "text", value: row.name, required: false, allow_none: true,
      description: `Internal ID: ${row.node_id}`, placeholder: row.node_id }, commit: value => this.intents.rename(row.node_id, String(value ?? "")) });
    // The dataset input the draft uses, under its own words (v2 source_required, source_label).
    const sourceLabel = row.source_label;
    if (sourceLabel) {
      specs.push({ key: "source", spec: { name: "source_signal", label: sourceLabel, kind: "choice", value: row.source_signal, required: true, allow_none: true,
        placeholder: "(not selected)", choices: row.source_choices, ...(row.source_tree ? { tree: row.source_tree } : {}), unavailable_reason: locked },
        commit: value => this.intents.editDraft(row.node_id, { source_signal: value }, sourceLabel) });
    }
    for (const [argument, choices] of Object.entries(row.device_choices)) {
      const label = row.device_labels[argument] ?? argument;
      // v2 _add_selector_row(_device_label(name)): a device row carries no ` *` (only Signal does).
      specs.push({ key: `device:${argument}`, spec: { name: `device-${argument}`, label, kind: "choice", value: row.device_keys[argument] ?? null,
        required: false, allow_none: true, placeholder: "(not selected)", choices, unavailable_reason: locked },
        commit: value => this.intents.editDraft(row.node_id, { device_keys: { [argument]: value } }, label) });
    }
    // The node's own editors (one model, or a stack of them) draw the fields
    // and files their models say they manage.
    const order = this.editors.reconcile(editorModels(row.editor_facts));
    // The editors stand after the form and its error, in their models' order.
    const placed = [...this.body.children].slice(2);
    if (placed.length !== order.length || placed.some((node, index) => node !== order[index])) this.body.append(...order);
    const drawn = this.editors.models;
    const managed = new Set(drawn.flatMap(model => model.manages));
    const managedFiles = new Set(drawn.flatMap(model => (model.manages_artifacts as string[] | undefined) ?? []));
    for (const control of row.controls) {
      if (managed.has(control.name)) continue;
      specs.push({ key: `value:${control.name}`, group: control.group, spec: locked ? { ...control, unavailable_reason: control.unavailable_reason || locked } : control,
        commit: value => this.intents.editDraft(row.node_id, { values: { [control.name]: value } }, control.label) });
    }
    const alive = new Set(specs.map(item => item.key));
    for (const [key, field] of this.fields) if (!alive.has(key)) { field.root.remove(); this.fields.delete(key); }
    let cursor: Element | null = this.form.firstElementChild;
    const place = (node: Element): void => { if (node !== cursor) this.form.insertBefore(node, cursor); else cursor = cursor.nextElementSibling; };
    // A section title where the fields' group changes (AuthoringField.group).
    const titles = new Set<string>();
    let group: string | undefined;
    // The files the node reads stand after its devices, before its fields (v2 artifact_form).
    const artifacts = this.artifactRows(row, managedFiles, locked, request, failed);
    let filed = false;
    for (const item of specs) {
      if (!filed && item.key.startsWith("value:")) { artifacts.forEach(place); filed = true; }
      if (item.group && item.group !== group) {
        titles.add(item.group);
        let title = this.groupTitles.get(item.group);
        if (!title) { title = sectionTitle(item.group); this.groupTitles.set(item.group, title); }
        place(title);
      }
      group = item.group ?? group;
      let field = this.fields.get(item.key);
      if (field && field.kind !== fieldKind(item.spec)) {
        // A row rebuilt as another kind (a text that became a choice of its pulse's names) may be where the next row goes.
        if (field.root === cursor) cursor = cursor.nextElementSibling;
        field.root.remove(); this.fields.delete(item.key); field = undefined;
      }
      if (!field) {
        const commit = item.commit;
        field = new Field(`${row.node_id}-edit`, item.spec, { commit: value => commit(value), browse: ask => this.intents.choose(ask),
          refresh: () => this.intents.check(row.node_id) });
        this.fields.set(item.key, field);
      } else field.update(item.spec, request, failed);
      place(field.root);
    }
    if (!filed) artifacts.forEach(place);
    for (const [name, title] of this.groupTitles) if (!titles.has(name)) { title.remove(); this.groupTitles.delete(name); }
    // Whatever stands after the last placed row is no longer the node's.
    while (cursor) { const next: Element | null = (cursor as Element).nextElementSibling; (cursor as Element).remove(); cursor = next; }
    const problem = row.editor_facts_error ? `The editor cannot be drawn: ${row.editor_facts_error}` : "";
    if (this.editorError.textContent !== problem) this.editorError.textContent = problem;
    assign(this.editorError, "hidden", !problem);
    if (this.editors.length) { if (this.editors.update(row, locked)) this.shareLabels(); }
    else this.body.style.removeProperty("--label-width");
    const free = !takeover.active;
    assign(this.plot, "hidden", !projection.previewOffered);
    setChecked(this.plot, projection.autoPreview);
    assign(this.plot, "disabled", !free);
    this.start.text = startLabel(projection);
    this.start.setDirty(projection.edited && projection.canStop);
    this.start.disabled = !free || !projection.canStart || projection.pending;
    this.stop.disabled = !projection.canStop || !(free || takeover.owns);
    this.revert.disabled = !projection.edited || !free;
    this.remove.disabled = !free;
  }

  complete(request?: number, failed = false): void { for (const field of [...this.fields.values(), ...this.artifactFields.values()]) field.complete(request, failed); }

  /** The files a node reads (v2 artifact_form): each declared input under its
   * own label, a path with Browse... (the window's file dialog) -- or, for one
   * that claims frames and no editor draws, what it claims -- then Refresh
   * files to re-read them after they were written elsewhere (check_node). */
  private artifactRows(row: LogicRow, managed: Set<string>, locked: string, request?: number, failed = false): Element[] {
    const declared = row.artifact_specs ?? Object.keys(row.artifact_inputs).map(name => ({ name, label: name, file_filter: "", required: true, frame_claims: false }));
    const specs = declared.filter(spec => !managed.has(spec.name));
    const alive = new Set(specs.filter(spec => !spec.frame_claims).map(spec => spec.name));
    for (const [name, field] of this.artifactFields) if (!alive.has(name)) { field.root.remove(); this.artifactFields.delete(name); }
    return specs.map(spec => {
      const label = spec.label || spec.name;
      const value = row.artifact_inputs[spec.name];
      if (spec.frame_claims) {
        const claims = Array.isArray(value) ? value.map(file => file.frames?.length ? `${file.path} (${file.frames.join(", ")})` : file.path).join("; ") : "";
        return formRow(label, new Readout(claims || "No files chosen").root);
      }
      const control: FieldSpec = { name: spec.name, label, kind: "path", value: typeof value === "string" ? value : "",
        required: spec.required, filter: spec.file_filter, refreshable: true, unavailable_reason: locked };
      let field = this.artifactFields.get(spec.name);
      if (!field) {
        field = new Field(`${row.node_id}-artifact`, control, {
          commit: value => this.intents.editDraft(row.node_id, { artifact_inputs: { [spec.name]: String(value ?? "") } }, label),
          browse: ask => this.intents.choose(ask), refresh: () => this.intents.check(row.node_id),
        });
        this.artifactFields.set(spec.name, field);
      } else field.update(control, request, failed);
      return field.root;
    });
  }

  get name(): string { return this.projection?.name ?? this.id; }
  get edited(): boolean { return Boolean(this.row?.edited); }
}

/** A panel publisher's page (v2: LogicEditorView without actions): one
 * switch per output the panel can publish, each naming the signal it
 * publishes as; the panel stays where it is, only what it offers changes. */
export class PublisherEditorView {
  readonly root = element("section", "logic-editor");
  private title = new Readout("", true, true);
  private body = element("div", "logic-editor-body");
  private form = form();
  private fields = new Map<string, Field>();
  constructor(readonly id: string, private publish: (outputs: Record<string, boolean>) => number) {
    const header = element("div", "logic-editor-header");
    const titleRow = element("div", "logic-editor-title");
    this.title.root.classList.add("heading");
    titleRow.append(this.title.root, element("span", "note", "(panel publisher)"));
    header.append(titleRow);
    this.body.classList.add("publisher-body");
    this.body.append(this.form);
    this.root.append(header, this.body);
  }
  update(name: string, outputs: { name: string; label: string; signal: string; published: boolean }[], locked: string, request?: number, failed = false): void {
    this.title.text = `${name} outputs`;
    const alive = new Set(outputs.map(output => output.name));
    for (const [key, field] of this.fields) if (!alive.has(key)) { field.root.remove(); this.fields.delete(key); }
    for (const output of outputs) {
      const spec: FieldSpec = { name: output.name, label: output.label, kind: "boolean", value: output.published, required: false,
        description: `publishes as ${output.signal}`, unavailable_reason: locked };
      let field = this.fields.get(output.name);
      if (!field) {
        // One switch is one output (the board merges it, v2 update_panel_published_outputs).
        field = new Field(`publisher-${this.id}`, spec, { commit: value => this.publish({ [output.name]: Boolean(value) }) });
        this.fields.set(output.name, field);
        this.form.append(field.root);
      } else field.update(spec, request, failed);
    }
  }
}
