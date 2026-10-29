// FigureViewer@Zou lab: the saved figure's information pane on the left
// (File, Browse…, Refresh; its record as tabs of names and values; a status
// line) and its panels on the right: the Data chooser with Edit data / New
// data, then the Monitor page with Plot + Add panel over the board, and a
// closable tab per open data working copy or panel Edit.  It is its own
// document (state.viewer, v2's FigureViewer window): its board holds the
// opened Figure's panels and its Data lists that Figure's datasets, while the
// Task Console's live board runs on.
import { ComboBox, InfoPane, TabWidget, command, confirm, element, note, type Graph, type InfoTab } from "../fluent/index.js";
import { confirmPulseDiscard } from "../pulse/widgets.js";
import type { Panel, State } from "../protocol.js";
import { queueStatus, saveImage, type View, type ViewContext } from "../shell.js";
import type { PanelBoard } from "./board.js";
import { DataEditorView, saveDataAs, type DataProjection } from "./data_editor.js";
import { PanelEditView, type EditInfo } from "./panel_edit.js";

const baseName = (path: string): string => path.split(/[\\/]/).pop() ?? path;
/** The record's tabs v2 declares on its InfoPane (they bound its width; until the host sends them, they stand empty). */
const INFO_TABS = ["Plot", "Logic", "Devices", "Flow", "Raw"];

/** One data working copy as the host lists it: its tab's title, whether it holds unsaved work, what it shows. */
type DataTab = { id: string; title: string; dirty: boolean; projection: DataProjection };

export class FigureViewerView implements View {
  readonly key = "figure-viewer";
  readonly title = "Figure Viewer";
  readonly windowTitle = "FigureViewer@Zou lab";
  readonly root = element("div", "window figure-viewer");
  private info: InfoPane;
  // v2: an empty, disabled chooser until a Figure gives it data; it names what Edit data opens.
  private data = new ComboBox(() => {}, "").minimumContents(15);
  private kind = new ComboBox(() => {}, "").minimumContents(12);
  private add = command("Add panel", () => this.addPanel());
  private tabs = new TabWidget({ loneBar: true });
  private monitor = element("div", "monitor-page");
  private empty = note("", true);
  private path = "";
  private flowSignature = "";
  // v2: the button beside the chooser acts on what it holds; the one after it does something else.
  private editData = command("Edit data", () => { if (this.data.value) this.ctx.transport.send({ action: "edit_data", signal: this.data.value }, "Opening the data…"); });
  private newData = command("New data", () => this.ctx.transport.send({ action: "new_data" }, "Making new data…"));
  private editors = new Map<string, DataEditorView>();
  private editorTitles = new Map<string, string>();
  private panelEdits = new Map<string, PanelEditView>();
  private editFocus: string | null = null;
  /** What the viewer last said of its Figure (v2 view.set_status), kept until it says something else:
   * "showing …" once an open is on the board, an Add's next step. */
  private report: { text: string; path: string; word: "idle" | "task" } | null = null;
  /** An open whose Figure is not on the board yet ("opening <file>…"); a later re-render is no open. */
  private opening = false;
  private openRequest = -1;
  private adding: Set<string> | null = null;

  constructor(private ctx: ViewContext, private board: PanelBoard) {
    this.info = new InfoPane("File", {
      // v2 open: "opening <file>…" until the Figure is on the board (the dialog of Browse… says nothing).
      commit: path => { if (path) this.openRequest = ctx.transport.send({ action: "open", path }, `opening ${baseName(path)}…`); },
      browse: () => ctx.transport.send({ action: "open" }),
      refresh: () => { if (this.path) this.openRequest = ctx.transport.send({ action: "open", path: this.path }, `opening ${baseName(this.path)}…`); },
    }, ["schema_fingerprint", "coordinate_frame"], ["Flow"], INFO_TABS);
    this.info.path.input.placeholder = ".npz archive or matching image preview";
    this.info.path.browse.root.title = "Browse NPZ archives and image thumbnails. An image opens the same-named NPZ in its folder.";
    this.info.root.classList.add("figure-info");
    // A Devices row offers the pulse a run played: it opens in the Pulse Editor as an unsaved draft.
    this.info.actionRequested = action => { if (action.startsWith("pulse:")) void this.openRecorded(action.slice("pulse:".length)); };
    // v2: the right half is one white work surface -- the data bar, then the tabs.
    const right = element("div", "card figure-main");
    const dataRow = element("div", "header figure-data");
    dataRow.append(element("label", "row-label", "Data"), this.data.root, this.editData.root, this.newData.root);
    this.data.root.id = "add-signal";
    this.kind.root.id = "add-kind";
    this.add.root.id = "add-panel";
    const addRow = element("nav", "board-row");
    addRow.setAttribute("aria-label", "Add a plot panel");
    // Adding a panel and saving the board's picture act on the board alone, so they head its tab (v2).
    addRow.append(this.kind.root, this.add.root, element("span", "spacer"), command("Save image…", () => saveImage(ctx.transport)).root);
    this.monitor.append(addRow, this.empty);
    this.tabs.addPermanent("monitor", "Monitor", this.monitor);
    right.append(dataRow, this.tabs.root);
    this.root.append(this.info.root, right);
    // v2 FigureViewerView.open_panel_editor: a panel's Edit is a closable tab of this surface.
    board.onEdit = panel => this.editPanel(panel);
  }

  shown(): void {
    this.monitor.append(this.board.root);
    // A saved Figure's panels always take their gestures (v2 panel-only board).
    this.board.interactive = true;
    this.board.shown();
  }
  hidden(): void { this.board.hidden(); }
  /** Shown with its Figure: the viewer's document comes after the window's first state. */
  ready(): boolean { return Boolean(this.ctx.latest.viewer); }

  update(state: State): void {
    const latest = this.ctx.latest;
    const doc = latest.viewer?.document;
    // A Figure opened here (or again: Refresh) is "opening" until its pictures are on the board.
    const reopened = state.request === this.openRequest && !state.error;
    if (reopened) this.openRequest = -1;
    if ((doc?.path ?? "") !== this.path || reopened) { this.report = null; this.opening = Boolean(doc?.path); }
    this.path = doc?.path ?? "";
    this.info.setPath(this.path);
    const tabs = (doc as { info?: InfoTab[] } | undefined)?.info;
    // The record's tabs as the archive gives them; until the host sends them,
    // the five v2 declares, empty.
    this.info.setTabs(tabs?.length ? tabs : INFO_TABS.map(title => [title, []] as InfoTab));
    const flow = (doc as { flow?: Graph } | undefined)?.flow;
    const signature = JSON.stringify(flow ?? null);
    if (signature !== this.flowSignature) { this.flowSignature = signature; this.info.setGraph("Flow", flow ?? { nodes: [], edges: [] }); }
    const signals = latest.viewer?.signals ?? [];
    // v2 set_editable_data_choices: the Figure's datasets and applied data, not what its panels derive.
    const datasets = signals.filter(signal => signal.id.startsWith("@figure/"));
    this.data.setChoices(datasets.map(signal => ({ value: signal.id, label: signal.label, title: signal.reason || "" })));
    if (this.data.value === null || !datasets.some(signal => signal.id === this.data.value)) this.data.value = datasets[0]?.id ?? null;
    this.data.disabled = !datasets.length;
    this.editData.disabled = !datasets.length;
    this.updateEditors();
    this.updatePanelEdits();
    // v2: the viewer offers the console's five kinds, by their own names, "image" first.
    const offered = new Set((latest.viewer?.plot_kinds ?? latest.plot_kinds ?? []).map(kind => kind.id));
    const kinds = ["image", "curve", "rolling", "histogram", "facet_grid"].filter(id => offered.has(id)).map(id => ({ id }));
    this.kind.setChoices(kinds.map(kind => ({ value: kind.id, label: `Plot: ${kind.id}` })));
    if (this.kind.value === null || !kinds.some(kind => kind.id === this.kind.value)) this.kind.value = kinds[0]?.id ?? null;
    // v2 set_panel_kinds: Add panel is there whenever a kind is (a blank panel, its signal chosen in Setting).
    this.add.disabled = !kinds.length;
    // v2 _sync_monitor_empty: a figure to open until there is data, then a panel to add.
    this.empty.hidden = this.board.size > 0;
    const next = datasets.length || tabs?.length ? "No panels yet. Choose a Plot above and click Add panel." : "Browse NPZ files and image previews to open saved data, or make data with New data.";
    if (this.empty.textContent !== next) this.empty.textContent = next;
    // v2 add_selected_panel: "added <panel>; choose a signal in Setting".
    const panels = (latest.viewer?.panels ?? []) as (Panel & { caption_id?: string })[];
    if (this.adding) {
      const made = panels.find(panel => !this.adding!.has(panel.id) && !panel.id.startsWith("edit-"));
      if (made) {
        // v2 add_blank_panel names it by its kind ("image", "image 2", ...): the viewer's board holds that title.
        this.adding = null;
        this.report = { text: `added ${this.panelName(made.id)}; choose a signal in Setting`, path: this.path, word: "task" };
      }
    }
    this.status();
  }

  /** What a panel is called on this viewer's board (v2 _panel_name: its title, else its own name). */
  private panelName(panel: string): string {
    const entry = (this.ctx.latest.viewer as { board?: { panels: { panel: string; title: string | null; name: string }[] } } | undefined)?.board?.panels.find(item => item.panel === panel);
    return entry?.title || entry?.name || panel;
  }

  /** v2 open/update/close_data_editor: one closable tab per open working copy, "Data · name", starred while dirty. */
  private updateEditors(): void {
    const viewer = this.ctx.latest.viewer as { data?: DataTab[]; data_focus?: string | null } | undefined;
    const listed = viewer?.data ?? [];
    for (const [id] of this.editors) {
      if (listed.some(tab => tab.id === id)) continue;
      this.tabs.remove(`data:${id}`);
      this.editors.delete(id);
      this.editorTitles.delete(id);
    }
    for (const tab of listed) {
      let editor = this.editors.get(tab.id);
      const title = `${tab.title}${tab.dirty ? " *" : ""}`;
      if (!editor) {
        editor = new DataEditorView(tab.id, intent => {
          if (intent.op === "save_as") void saveDataAs(this.ctx.transport, tab.id, intent);
          else this.ctx.transport.send({ action: "data_editor", editor: tab.id, intent }, "Editing the data…");
        });
        this.editors.set(tab.id, editor);
        this.tabs.addClosable(`data:${tab.id}`, title, editor.root, () => void this.closeEditor(tab.id), true);
      } else if (this.editorTitles.get(tab.id) !== title) this.tabs.setTitle(`data:${tab.id}`, title);
      this.editorTitles.set(tab.id, title);
      editor.update(tab.projection);
    }
    if (viewer?.data_focus && this.editors.has(viewer.data_focus)) this.tabs.setCurrent(`data:${viewer.data_focus}`);
  }

  /** Closing a copy with unsaved work asks first (v2 confirm_discard); a yes is honoured, a no is said in the copy. */
  private async closeEditor(id: string): Promise<void> {
    const editor = this.editors.get(id);
    const name = (this.editorTitles.get(id) ?? "").replace(/^Data · /, "").replace(/ \*$/, "");
    let discard = false;
    if (editor?.isDirty) {
      discard = await confirm("Unsaved edits", `${name} has unsaved edits. Close it and lose them?`, { confirmText: "Close anyway", cancelText: "Keep editing" });
    }
    this.ctx.transport.send({ action: "close_data_editor", editor: id, discard }, "Closing the data…");
  }

  /** v2 edit_panel: open (holding the panel still on one read) or focus the panel's Edit tab. */
  private editPanel(panel: string): void {
    const edit = `edit-${panel}`;
    if (this.panelEdits.has(edit)) { this.tabs.setCurrent(`edit:${edit}`); return; }
    this.editFocus = edit;
    this.ctx.transport.send({ action: "edit_panel", panel }, "Holding the panel still…");
  }

  /** The Edit tabs follow the Edits the viewer's document carries: one opens with its Edit, closes with it. */
  private updatePanelEdits(): void {
    const panels = (this.ctx.latest.viewer?.panels ?? []) as (Panel & { edit?: EditInfo; caption_id?: string })[];
    const edits = panels.filter(row => row.edit);
    for (const [edit] of this.panelEdits) if (!edits.some(row => row.id === edit)) { this.tabs.remove(`edit:${edit}`); this.panelEdits.delete(edit); }
    for (const row of edits) {
      // v2 "Edit panel · <the panel's title>".
      const name = this.panelName(row.edit!.panel);
      let page = this.panelEdits.get(row.id);
      if (!page) {
        page = new PanelEditView(row.id, row.edit!.panel, this.board, (command, feedback) => this.ctx.transport.send(command, feedback), () => {}, this.ctx.transport);
        this.panelEdits.set(row.id, page);
        this.tabs.addClosable(`edit:${row.id}`, `Edit panel · ${name}`, page.root, () => this.closePanelEdit(row.id), this.editFocus === row.id);
        if (this.editFocus === row.id) this.editFocus = null;
      }
      page.update(row, name, [], String((this.ctx.latest.workspace as { today?: string } | undefined)?.today ?? ""));
      this.tabs.setTitle(`edit:${row.id}`, `Edit panel · ${name}`);
    }
  }

  private closePanelEdit(edit: string): void {
    this.panelEdits.delete(edit);
    this.tabs.remove(`edit:${edit}`);
    this.ctx.transport.send({ action: "close_panel", panel: edit }, "Closing the Edit…");
  }

  private async openRecorded(key: string, discard = false): Promise<void> {
    try {
      await this.ctx.transport.request({ action: "open_recorded_pulse", key, discard_changes: discard }, "Opening the recorded pulse…");
      this.ctx.show("pulse-editor");
    } catch (error) {
      if (discard || !String(error).includes("unsaved changes")) return;
      if (await confirmPulseDiscard()) await this.openRecorded(key, true);
    }
  }

  /** v2 add_selected_panel: a blank panel of the chosen kind; its signal is chosen in its Setting. */
  private addPanel(): void {
    if (this.add.disabled || this.kind.value === null) return;
    this.adding = new Set((this.ctx.latest.viewer?.panels ?? []).map(panel => panel.id));
    this.ctx.transport.send({ action: "add_panel", kind: this.kind.value }, "Adding panel…");
  }

  status(): void {
    // v2 viewer: "opening <file>…" until the opened Figure is on the board, then "showing <its
    // first panel's data>" once, or what the viewer said last; "no figure open" before one is.
    const doc = this.ctx.latest.viewer?.document;
    if (this.opening && !this.board.presenting) {
      this.opening = false;
      const shown = this.ctx.latest.viewer?.panels?.find(panel => panel.binding?.signal)?.binding?.signal;
      this.report = { text: shown ? `showing ${shown}` : "opened; choose a signal in Setting", path: this.path, word: "idle" };
    }
    const said = this.report && this.report.path === (doc?.path ?? "") ? this.report : null;
    const [text, word] = queueStatus(this.ctx.transport, this.opening && doc?.path ? [`opening ${baseName(doc.path)}…`, "task"]
      : said ? [said.text, said.word] : ["no figure open", "idle"]);
    this.info.status.show(text, word);
  }
}

export const figureViewer = (board: PanelBoard) => (ctx: ViewContext): View => new FigureViewerView(ctx, board);
