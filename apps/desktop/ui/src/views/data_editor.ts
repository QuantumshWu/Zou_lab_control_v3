// A data working copy's tab in the Figure Viewer (v2 _DataEditorView): two
// pages chosen at its top -- the Dataset and Axes cards with the selected
// axis's values, and the Data card with the value table -- and the copy's
// own commands under both.  Every Dataset rule is the host's (manual.rs):
// this page shows its projection and sends the operator's intents.
import { ComboBox, Field, GroupBox, Segmented, SpinBox, command, element, fontOf, labelColumnWidth, measurer, note, setText, summary, switchButton, setChecked, token, wholeText, type Value } from "../fluent/index.js";

type Axis = { id: string; name: string; size: number; unit: string; domain: string; domain_label: string };
type AxisRow = { axis_id: string; name: string; size: number; unit: string; mode: "rows" | "columns" | "scope"; index: number; scope_label: string };
type Block = { row: number; column: number; cells: string[][] };
type Table = {
  component: string; component_choices: [string, string, boolean][]; sigma_enabled: boolean; shape: [number, number]; block: Block;
  blank_help: string; row_headers: string[]; column_headers: string[]; structure: [string, number][][]; axes: AxisRow[];
};
export type DataProjection = {
  dataset: { name: string; dtype: string; unit: string; note: string; source: string; dtype_choices: [string, string][] };
  domain_choices: [string, string][]; axes: Axis[]; selected_axis: string;
  axis_values: { values: string[]; blank_hint: string }; table: Table;
  dirty: boolean; modified: boolean; can_apply: boolean; can_save: boolean; save_suggested: string; message: string; message_severity: string;
};
export type Intent = { op: string; [key: string]: unknown };

/** The editor's fixed row labels (v2 _STATIC_LABELS): its one label column is the widest of these and the axis rows'. */
const STATIC_LABELS = ["Name", "Type", "Value unit", "Note", "Editing", "Length", "Unit", "Domain", "Table shows"];
const DOMAIN_INK = ["domain-repeat", "domain-point", "domain-cell"];
let measure: CanvasRenderingContext2D | null = null;

/** A text input that keeps the operator's draft while focused (v2 being_edited). */
function edit(commit: (text: string) => void, placeholder = ""): HTMLInputElement {
  const input = element("input");
  input.type = "text";
  input.spellcheck = false;
  input.placeholder = placeholder;
  input.addEventListener("change", () => commit(input.value));
  input.addEventListener("keydown", event => { if (event.key === "Enter") input.blur(); });
  return input;
}
function project(input: HTMLInputElement, text: string): void { if (document.activeElement !== input && input.value !== text) input.value = text; }

/** FluentTableView over a host-held 2-D slice (v2 _VirtualTextTableModel): only
 * the cells on screen exist; their text comes in blocks the host sends for the
 * rows and columns asked; a typed edit shows ORANGE_TINT until the host answers.
 * The header row, the row numbers and the values are ONE scrolled body, the
 * headers sticky at its top and left, so they move with the values in every
 * frame (a header that followed the scroll by script was drawn a frame late);
 * its bars span the whole table, beside the headers, as QAbstractScrollArea's. */
class DataTable {
  readonly root = element("div", "data-table");
  private corner = element("div", "dt-corner");
  private columnHead = element("div", "dt-columns");
  private columnInner = element("div", "dt-columns-inner");
  private rowHead = element("div", "dt-rows");
  private body = element("div", "dt-body");
  private viewport = element("div", "dt-viewport");
  private sizer = element("div", "dt-sizer");
  private shape: [number, number] = [0, 0];
  private rowHeaders: string[] = [];
  private columnHeaders: string[] = [];
  private block: Block = { row: 0, column: 0, cells: [] };
  private widths: number[] = [];
  private offsets: number[] = [0];
  private rowHeaderWidth = 0;
  private pending = new Map<string, string>();
  /** The cells on screen by "row:column", kept from frame to frame: a cell pressed is still
   * there at its release, so its click and double click arrive. */
  private cells = new Map<string, HTMLElement>();
  /** What the last column takes over its own width (render). */
  private stretch = 0;
  private current: [number, number] | null = null;
  private editor: HTMLInputElement | null = null;
  private frame = 0;
  private asked = "";
  editable = true;
  hint = "";
  edited: (cells: [number, number, string][]) => void = () => {};
  view: (row: number, rows: number, column: number, columns: number) => void = () => {};
  rejected: (message: string) => void = () => {};

  constructor(private stretchLast = true) {
    this.columnHead.append(this.columnInner, this.corner);
    this.body.append(this.rowHead, this.sizer);
    this.viewport.append(this.columnHead, this.body);
    this.root.append(this.viewport);
    this.root.tabIndex = 0;
    this.viewport.addEventListener("scroll", () => this.schedule());
    new ResizeObserver(() => this.schedule()).observe(this.viewport);
    this.sizer.addEventListener("mousedown", event => {
      const cell = (event.target as HTMLElement).closest<HTMLElement>(".dt-cell");
      if (!cell || cell.querySelector("input")) return;
      this.current = [Number(cell.dataset.row), Number(cell.dataset.column)];
      this.root.focus({ preventScroll: true });
      this.schedule();
    });
    this.sizer.addEventListener("dblclick", event => {
      const cell = (event.target as HTMLElement).closest<HTMLElement>(".dt-cell");
      if (cell) this.startEdit(Number(cell.dataset.row), Number(cell.dataset.column), null);
    });
    this.root.addEventListener("keydown", event => this.key(event));
    this.root.addEventListener("paste", event => this.paste(event));
  }

  set(shape: [number, number], rowHeaders: string[], columnHeaders: string[], block: Block): void {
    const reset = shape[0] !== this.shape[0] || shape[1] !== this.shape[1] || columnHeaders.join("\u0001") !== this.columnHeaders.join("\u0001");
    this.shape = shape;
    this.rowHeaders = rowHeaders;
    this.columnHeaders = columnHeaders;
    this.block = block;
    this.pending.clear();
    if (reset) { this.widths = []; this.asked = ""; if (this.current && (this.current[0] >= shape[0] || this.current[1] >= shape[1])) this.current = null; }
    this.measure();
    this.schedule();
  }

  /** v2 ResizeToContents over what has been seen: a column is as wide as its widest header or cell, the last one stretched. */
  private measure(): void {
    measure ??= document.createElement("canvas").getContext("2d");
    if (!measure) return;
    measure.font = getComputedStyle(this.root).font;
    const pad = 2 * token("--table-cell-pad");
    const text = (words: string): number => measure!.measureText(words).width;
    const columns = this.shape[1];
    for (let column = this.widths.length; column < columns; column++) this.widths.push(Math.ceil(text(this.columnHeaders[column] ?? "") + pad + token("--table-header-margin")));
    this.block.cells.forEach(row => row.forEach((value, offset) => {
      const column = this.block.column + offset;
      if (column < columns) this.widths[column] = Math.max(this.widths[column]!, Math.ceil(text(value) + pad));
    }));
    let widest = 0;
    for (const header of this.rowHeaders) widest = Math.max(widest, text(header));
    this.rowHeaderWidth = Math.ceil(widest + pad + token("--table-header-margin"));
    this.offsets = [0];
    for (const width of this.widths) this.offsets.push(this.offsets[this.offsets.length - 1]! + width);
  }

  private schedule(): void { if (!this.frame) this.frame = requestAnimationFrame(() => { this.frame = 0; this.render(); }); }

  private render(): void {
    const [rows, columns] = this.shape;
    const rowHeight = token("--table-row-h"), headerHeight = token("--table-header-h");
    this.root.style.setProperty("--dt-row-head", `${this.rowHeaderWidth}px`);
    this.root.style.setProperty("--dt-height", `${headerHeight + rows * rowHeight + 2 * token("--edge")}px`);
    // The values' view: the scrolled body less the sticky headers.
    const room = this.viewport.clientWidth - this.rowHeaderWidth, high = this.viewport.clientHeight - headerHeight;
    const natural = this.offsets[columns] ?? 0;
    // The last column takes what the view has over the columns' own widths.
    const stretch = this.stretch = this.stretchLast && columns > 0 && natural < room ? room - natural : 0;
    this.sizer.style.width = `${natural + stretch}px`;
    this.sizer.style.height = `${rows * rowHeight}px`;
    // The header row as wide as what it heads, so its face covers the values scrolled under it.
    this.columnHead.style.width = `${this.rowHeaderWidth + natural + stretch}px`;
    // A bar only where the values overflow: a table that fits shows none (no bar from a rounding).
    this.viewport.style.overflowX = natural > room + 0.5 ? "auto" : "hidden";
    this.viewport.style.overflowY = rows * rowHeight > high + 0.5 ? "auto" : "hidden";
    const top = this.viewport.scrollTop, left = this.viewport.scrollLeft;
    const firstRow = Math.max(0, Math.floor(top / rowHeight)), lastRow = Math.min(rows, Math.ceil((top + high) / rowHeight) + 1);
    let firstColumn = 0;
    while (firstColumn < columns - 1 && this.offsets[firstColumn + 1]! <= left) firstColumn++;
    let lastColumn = firstColumn;
    while (lastColumn < columns && this.offsets[lastColumn]! < left + room) lastColumn++;
    // Ask for the block on screen (with a margin) once it leaves what the host sent.
    const { row, column, cells } = this.block;
    const loaded = firstRow >= row && lastRow <= row + cells.length && firstColumn >= column && lastColumn <= column + (cells[0]?.length ?? 0);
    if (!loaded && rows && columns) {
      const ask = [Math.max(0, firstRow - 16), lastRow - firstRow + 32, Math.max(0, firstColumn - 8), lastColumn - firstColumn + 16];
      const key = ask.join(",");
      if (key !== this.asked) { this.asked = key; this.view(ask[0]!, ask[1]!, ask[2]!, ask[3]!); }
    }
    // The cell being typed into is its editor alone, over the cell's whole box.
    const editing = this.editor?.parentElement;
    if (editing) editing.style.cssText = this.place(Number(editing.dataset.row), Number(editing.dataset.column), rowHeight);
    const shown = new Map<string, HTMLElement>();
    for (let r = firstRow; r < lastRow; r++) {
      for (let c = firstColumn; c < lastColumn; c++) {
        const key = `${r}:${c}`;
        if (editing && editing.dataset.row === String(r) && editing.dataset.column === String(c)) continue;
        const text = this.pending.get(key) ?? cells[r - row]?.[c - column] ?? "";
        let node = this.cells.get(key);
        if (!node) {
          node = element("div", "dt-cell");
          node.dataset.row = String(r);
          node.dataset.column = String(c);
          this.sizer.append(node);
        }
        if (node.textContent !== text) node.textContent = text;
        node.style.cssText = this.place(r, c, rowHeight);
        node.classList.toggle("pending", this.pending.has(key));
        node.classList.toggle("current", Boolean(this.current && this.current[0] === r && this.current[1] === c));
        node.title = text || this.hint;
        shown.set(key, node);
      }
    }
    for (const [key, node] of this.cells) if (!shown.has(key)) node.remove();
    this.cells = shown;
    this.columnInner.replaceChildren(...Array.from({ length: lastColumn - firstColumn }, (_, at) => {
      const c = firstColumn + at;
      const node = element("div", "dt-head", this.columnHeaders[c] ?? String(c));
      node.style.cssText = `left:${this.offsets[c]}px;width:${this.width(c)}px`;
      return node;
    }));
    this.rowHead.replaceChildren(...Array.from({ length: lastRow - firstRow }, (_, at) => {
      const r = firstRow + at;
      const node = element("div", "dt-head", this.rowHeaders[r] ?? String(r));
      node.style.cssText = `top:${r * rowHeight}px;height:${rowHeight}px`;
      return node;
    }));
  }

  /** A column's width on screen: the last one with the stretch. */
  private width(c: number): number { return this.widths[c]! + (c === this.shape[1] - 1 ? this.stretch : 0); }
  private place(r: number, c: number, rowHeight: number): string {
    return `top:${r * rowHeight}px;left:${this.offsets[c]}px;width:${this.width(c)}px;height:${rowHeight}px`;
  }

  private startEdit(r: number, c: number, typed: string | null): void {
    if (!this.editable) return;
    this.current = [r, c];
    const holder = element("div", "dt-cell editing");
    holder.dataset.row = String(r);
    holder.dataset.column = String(c);
    holder.style.cssText = this.place(r, c, token("--table-row-h"));
    // The value it replaces is not drawn beside it.
    this.cells.get(`${r}:${c}`)?.remove();
    this.cells.delete(`${r}:${c}`);
    const input = element("input");
    input.type = "text";
    input.spellcheck = false;
    const key = `${r}:${c}`;
    input.value = typed ?? this.pending.get(key) ?? this.block.cells[r - this.block.row]?.[c - this.block.column] ?? "";
    holder.append(input);
    this.editor = input;
    let done = false;
    const finish = (commit: boolean, move: [number, number] | null): void => {
      if (done) return;
      done = true;
      this.editor = null;
      holder.remove();
      const original = this.block.cells[r - this.block.row]?.[c - this.block.column] ?? "";
      if (commit && input.value !== original) { this.pending.set(key, input.value); this.edited([[r, c, input.value]]); }
      if (move) this.current = [Math.min(this.shape[0] - 1, Math.max(0, move[0])), Math.min(this.shape[1] - 1, Math.max(0, move[1]))];
      this.root.focus({ preventScroll: true });
      this.schedule();
    };
    input.addEventListener("keydown", event => {
      event.stopPropagation();
      if (event.key === "Enter") { event.preventDefault(); finish(true, [r + 1, c]); }
      else if (event.key === "Tab") { event.preventDefault(); finish(true, [r, c + (event.shiftKey ? -1 : 1)]); }
      else if (event.key === "Escape") { event.preventDefault(); finish(false, null); }
    });
    input.addEventListener("blur", () => finish(true, null));
    this.sizer.append(holder);
    input.focus();
    if (typed === null) input.select(); else input.setSelectionRange(input.value.length, input.value.length);
  }

  private key(event: KeyboardEvent): void {
    if (!this.current || this.editor) return;
    const [r, c] = this.current;
    const moves: Record<string, [number, number]> = { ArrowUp: [r - 1, c], ArrowDown: [r + 1, c], ArrowLeft: [r, c - 1], ArrowRight: [r, c + 1] };
    const moved = moves[event.key];
    if (moved) {
      event.preventDefault();
      this.current = [Math.min(this.shape[0] - 1, Math.max(0, moved[0])), Math.min(this.shape[1] - 1, Math.max(0, moved[1]))];
      this.reveal();
      this.schedule();
    } else if (event.key === "Enter" || event.key === "F2") { event.preventDefault(); this.startEdit(r, c, null); }
    else if (event.key === "Delete" || event.key === "Backspace") {
      // v2 clear_indices: a cleared cell is a blank edit.
      if (!this.editable) return;
      event.preventDefault();
      this.pending.set(`${r}:${c}`, "");
      this.edited([[r, c, ""]]);
      this.schedule();
    } else if (event.key.length === 1 && !event.ctrlKey && !event.metaKey && !event.altKey) { event.preventDefault(); this.startEdit(r, c, event.key); }
  }

  private reveal(): void {
    if (!this.current) return;
    const rowHeight = token("--table-row-h");
    const [r, c] = this.current;
    const top = r * rowHeight, left = this.offsets[c]!;
    // The values' view, under the sticky header row and right of the row numbers.
    const high = this.viewport.clientHeight - token("--table-header-h"), room = this.viewport.clientWidth - this.rowHeaderWidth;
    if (top < this.viewport.scrollTop) this.viewport.scrollTop = top;
    else if (top + rowHeight > this.viewport.scrollTop + high) this.viewport.scrollTop = top + rowHeight - high;
    if (left < this.viewport.scrollLeft) this.viewport.scrollLeft = left;
    else if (left + this.widths[c]! > this.viewport.scrollLeft + room) this.viewport.scrollLeft = left + this.widths[c]! - room;
  }

  /** v2 set_block_data: a pasted block of tab-separated rows, from the current cell. */
  private paste(event: ClipboardEvent): void {
    if (!this.current || this.editor || !this.editable) return;
    const text = event.clipboardData?.getData("text/plain") ?? "";
    if (!text) return;
    event.preventDefault();
    const rows = text.replace(/\r\n?/g, "\n").replace(/\n$/, "").split("\n").map(line => line.split("\t"));
    const [r, c] = this.current;
    if (r + rows.length > this.shape[0] || rows.some(fields => c + fields.length > this.shape[1])) {
      this.rejected("Paste exceeds the current table; resize axes first");
      return;
    }
    const cells: [number, number, string][] = [];
    rows.forEach((fields, dr) => fields.forEach((value, dc) => { this.pending.set(`${r + dr}:${c + dc}`, value); cells.push([r + dr, c + dc, value]); }));
    this.edited(cells);
    this.schedule();
  }
}

/** One labelled row of the editor's single label column. */
function line(label: string | HTMLElement, ...controls: HTMLElement[]): HTMLDivElement {
  const row = element("div", "data-row");
  const words = typeof label === "string" ? element("span", "data-label", label) : label;
  words.classList.add("data-label");
  row.append(words, ...controls);
  return row;
}

export class DataEditorView {
  readonly root = element("div", "data-editor");
  private page: "dataset" | "data" = "dataset";
  private pages = new Segmented(value => this.showPage(value as "dataset" | "data"));
  private body = element("div", "data-editor-body");
  private datasetCard = new GroupBox("Dataset", [], { summary: summary("") });
  private axesCard: GroupBox;
  private dataCard = new GroupBox("Data");
  private name = edit(text => this.field("name", text));
  private dtype = new ComboBox(value => this.field("dtype", value)).minimumContents(9);
  private unit = edit(text => this.field("unit", text));
  private note = edit(text => this.field("note", text), "Optional reason for this manual edit");
  private axisMode = summary("Edit selected axis");
  private addAxis = command("Add axis", () => this.beginAdd(), { title: "Create a new axis" });
  private axisChoice = new ComboBox(value => { this.adding = false; this.emit("select_axis", { axis_id: value }); }).minimumContents(13);
  private applyAxis = command("Apply axis", () => this.commitAxis(), { words: ["Apply axis", "Create axis"] });
  private removeAxis = command("Remove", () => this.emit("delete_axis", { axis_id: this.selected }), { role: "destructive", title: "Remove the selected axis and keep its current Scope slice" });
  private axisName = edit(() => {});
  private axisLength = element("input");
  private axisLengthBox = element("div", "data-length");
  private axisLengthSpin: SpinBox;
  private axisUnit = edit(() => {});
  private domain = new Segmented(value => { this.domainValue = String(value); this.domain.setCurrent(value); });
  private axisValues = new DataTable();
  private structure = element("div", "data-structure");
  private viewRows = element("div", "data-view-rows");
  private viewFields = new Map<string, Field>();
  private components = new Segmented(value => { this.components.setCurrent(value); this.emit("set_component", { component: value }); });
  private sigma = switchButton(on => this.emit("toggle_sigma", { enabled: on }), "Store sigma");
  private blankHelp = note("");
  private values = new DataTable();
  private message = note("", true);
  private discard = command("Discard", () => this.emit("discard"), { role: "destructive" });
  private apply = command("Apply & preview", () => this.emit("apply_preview", { note: this.note.value }));
  private save = command("Save as…", () => this.emit("save_as", { note: this.note.value }));
  private projection: DataProjection | null = null;
  private listed = new Set<string>();
  private selected = "";
  private adding = false;
  private domainValue = "";
  private component = "values";
  private dirty = false;

  constructor(readonly id: string, private send: (intent: Intent) => void) {
    this.pages.setChoices([["dataset", "Dataset and axes"], ["data", "Data"]]);
    this.axisLength.type = "text";
    this.axisLength.inputMode = "numeric";
    this.axisLengthBox.append(this.axisLength);
    this.axisLengthSpin = new SpinBox(this.axisLengthBox, this.axisLength, { step: () => 1, bounds: () => [1, 2_147_483_647], commit: value => { this.axisLength.value = String(value); } });
    this.axesCard = new GroupBox("Axes", [this.addAxis.root], { summary: this.axisMode });
    const pair = (...rows: HTMLElement[]): HTMLDivElement => { const grid = element("div", "data-pair"); grid.append(...rows); return grid; };
    this.datasetCard.body.append(pair(line("Name", this.name), line("Type", this.dtype.root), line("Value unit", this.unit), line("Note", this.note)));
    const axisValues = element("div", "data-axis-values");
    axisValues.append(element("div", "section-title", "Axis values"), this.axisValues.root);
    this.axesCard.body.append(line("Editing", this.axisChoice.root, this.applyAxis.root, this.removeAxis.root),
      pair(line("Name", this.axisName), line("Length", this.axisLengthBox), line("Unit", this.axisUnit), line("Domain", this.domain.root)), axisValues);
    this.axisValues.edited = cells => this.emit("set_axis_values", { axis_id: this.selected, cells });
    this.axisValues.rejected = text => this.local(text);
    this.axisValues.view = () => {};
    const shows = element("div", "data-shows");
    shows.append(this.components.root, this.sigma);
    this.dataCard.body.append(this.structure, this.viewRows, line("Table shows", shows), this.blankHelp, this.values.root);
    this.values.edited = cells => this.emit("set_cells", { component: this.component, cells });
    this.values.rejected = text => this.local(text);
    this.values.view = (row, rows, column, columns) => this.emit("view", { row, rows, column, columns });
    this.body.append(this.datasetCard.root, this.axesCard.root, this.dataCard.root);
    const actions = element("div", "data-actions");
    actions.append(this.discard.root, element("span", "spacer"), this.apply.root, this.save.root);
    const foot = element("div", "data-foot");
    foot.append(this.message, actions);
    this.message.hidden = true;
    this.root.append(this.pages.root, this.body, foot);
    this.showPage("dataset");
  }

  private emit(op: string, values: Record<string, unknown> = {}): void { this.send({ op, ...values }); }
  private field(field: string, value: Value): void { this.emit("set_dataset_field", { field, value }); }
  private local(text: string): void { this.showMessage(text, "error"); }
  private showMessage(text: string, severity: string): void {
    const words = text && !/[.?!…]$/.test(text) ? `${text}.` : text;
    setText(this.message, words);
    this.message.dataset.state = severity;
    this.message.hidden = !words;
  }

  private showPage(page: "dataset" | "data"): void {
    this.page = page;
    this.pages.setCurrent(page);
    const data = page === "data";
    this.datasetCard.root.hidden = this.axesCard.root.hidden = data;
    this.dataCard.root.hidden = !data;
    this.body.scrollTop = 0;
  }

  private beginAdd(): void {
    this.adding = true;
    this.selected = "";
    this.axisChoice.value = null;
    this.axisName.value = "";
    this.axisLength.value = "1";
    this.axisUnit.value = "";
    this.axisValues.set([0, 1], [], ["Value"], { row: 0, column: 0, cells: [] });
    this.domainValue = this.projection?.domain_choices[0]?.[0] ?? "repeat";
    this.domain.setCurrent(this.domainValue);
    setText(this.axisMode, "New axis");
    this.applyAxis.text = "Create axis";
    this.modeControls(false);
    this.axisName.focus();
  }

  private modeControls(hasAxis: boolean): void {
    const editing = hasAxis || this.adding;
    for (const input of [this.axisName, this.axisLength, this.axisUnit]) input.disabled = !editing;
    this.domain.disabled = !editing;
    this.applyAxis.disabled = !editing;
    this.removeAxis.disabled = !(hasAxis && !this.adding);
    this.axisValues.editable = hasAxis && !this.adding;
    this.axisValues.root.classList.toggle("disabled", !this.axisValues.editable);
  }

  private commitAxis(): void {
    const name = this.axisName.value.trim();
    if (!name) { this.local("Axis name cannot be blank"); return; }
    const values = { name, length: Math.max(1, Math.round(Number(this.axisLength.value) || 1)), unit: this.axisUnit.value.trim(), domain: this.domainValue };
    if (this.adding) this.emit("add_axis", values);
    else if (this.selected) this.emit("edit_axis", { axis_id: this.selected, ...values });
  }

  get isDirty(): boolean { return this.dirty; }

  update(data: DataProjection): void {
    this.projection = data;
    const { dataset, table } = data;
    project(this.name, dataset.name);
    this.dtype.setChoices(dataset.dtype_choices.map(([value, label]) => ({ value, label })));
    this.dtype.value = dataset.dtype;
    project(this.unit, dataset.unit);
    project(this.note, dataset.note);
    setText(this.datasetCard.header.querySelector(".summary-text")!, dataset.source);
    this.domain.setChoices(data.domain_choices);
    const listed = this.listed;
    this.listed = new Set(data.axes.map(axis => axis.id));
    this.axisChoice.setChoices(data.axes.map(axis => ({ value: axis.id, label: `${axis.domain_label} · ${axis.name}` })));
    // A Create landed: the host selects the axis it made, which the list did not hold.
    if (this.adding && data.selected_axis && !listed.has(data.selected_axis)) this.adding = false;
    const selected = data.axes.find(axis => axis.id === data.selected_axis) ?? data.axes[0];
    this.selected = selected?.id ?? "";
    if (this.adding) this.axisChoice.value = null;
    else {
      this.axisChoice.value = this.selected || null;
      this.domainValue = selected?.domain ?? "";
      this.domain.setCurrent(this.domainValue);
      project(this.axisName, selected?.name ?? "");
      if (document.activeElement !== this.axisLength) this.axisLength.value = String(selected?.size ?? 1);
      project(this.axisUnit, selected?.unit ?? "");
      setText(this.axisMode, "Edit selected axis");
      this.applyAxis.text = "Apply axis";
      const values = data.axis_values.values;
      this.axisValues.hint = data.axis_values.blank_hint;
      this.axisValues.set([values.length, 1], values.map((_, index) => String(index)), ["Value"], { row: 0, column: 0, cells: values.map(value => [value]) });
    }
    this.axisLengthSpin.refresh();
    this.modeControls(Boolean(selected));
    this.structureLines(table.structure);
    this.viewControls(table.axes);
    this.components.setChoices(table.component_choices.map(([value, label]) => [value, label] as [string, string]));
    for (const [value, , offered] of table.component_choices) this.components.setWithheld(value, !offered);
    this.component = table.component;
    this.components.setCurrent(table.component);
    setChecked(this.sigma, table.sigma_enabled);
    setText(this.blankHelp, table.blank_help);
    this.values.hint = table.blank_help;
    this.values.set(table.shape, table.row_headers, table.column_headers, table.block);
    this.showMessage(data.message, data.message_severity);
    this.dirty = data.dirty;
    this.apply.setDirty(data.modified);
    this.save.setDirty(data.dirty && !data.modified);
    this.discard.disabled = !data.dirty;
    this.apply.disabled = !data.can_apply;
    this.save.disabled = !data.can_save;
    this.shareLabels();
  }

  /** data_structure_fragments: the shape over the axes' names, a group per domain in its ink. */
  private structureLines(structure: [string, number][][]): void {
    const signature = JSON.stringify(structure);
    if (this.structure.dataset.signature === signature) return;
    this.structure.dataset.signature = signature;
    const lineOf = (names: boolean): HTMLDivElement => {
      const node = element("div", "line");
      structure.forEach((items, domain) => {
        if (domain) node.append(document.createTextNode(" × "));
        const text = items.map(item => String(item[names ? 0 : 1])).join(" × ") || (names ? "—" : "1");
        node.append(element("span", DOMAIN_INK[domain]!, `(${text})`));
      });
      return node;
    };
    this.structure.replaceChildren(lineOf(false), lineOf(true));
  }

  /** One row per axis: Rows ↓, Columns → or its Scope coordinate (v2 FluentCycleComboBox). */
  private viewControls(rows: AxisRow[]): void {
    const wanted = new Set(rows.map(row => row.axis_id));
    for (const [id, field] of this.viewFields) if (!wanted.has(id)) { field.root.remove(); this.viewFields.delete(id); }
    const nodes: HTMLElement[] = [];
    for (const row of rows) {
      const label = `${row.name} (${row.size})${row.unit ? ` [${row.unit}]` : ""}`;
      const spec = { section: "semantic", name: row.axis_id, label, kind: "choice", value: row.mode === "scope" ? "scope" : row.mode,
        choices: [["rows", "Rows ↓"], ["columns", "Columns →"]], minimum_contents: 9,
        cycle: { count: row.size, index: row.mode === "scope" ? row.index : null, label: row.scope_label, value: "scope" } } as const;
      let field = this.viewFields.get(row.axis_id);
      if (!field) {
        field = new Field(`data-${this.id}`, spec as never, {
          commit: value => { if (value === "scope") this.emit("set_table_axis", { axis_id: row.axis_id, mode: "scope" }); else this.emit("set_table_axis", { axis_id: row.axis_id, mode: value }); return 0; },
          scope: steps => {
            const current = this.projection?.table.axes.find(axis => axis.axis_id === row.axis_id);
            const index = Math.max(0, Math.min((current?.size ?? 1) - 1, (current?.index ?? 0) + steps));
            this.emit("set_scope", { axis_id: row.axis_id, index });
            return 0;
          },
        }, false);
        this.viewFields.set(row.axis_id, field);
      } else field.update(spec as never);
      nodes.push(field.root);
    }
    const host = element("div", "form data-view-form");
    if (this.viewRows.firstElementChild?.classList.contains("data-view-form")) {
      const form = this.viewRows.firstElementChild as HTMLElement;
      form.replaceChildren(...nodes);
    } else { host.append(...nodes); this.viewRows.replaceChildren(host); }
    this.viewRows.hidden = rows.length === 0;
  }

  /** v2 _set_label_width: one column for the three cards, wide enough for every fixed label and axis row, capped. */
  private shareLabels(): void {
    const advance = measurer(fontOf(getComputedStyle(this.root)));
    const width = (labels: string[]): number => labelColumnWidth(labels.map(advance), token("--data-label-min"));
    const axisLabels = [...this.viewFields.values()].map(field => wholeText(field.root.querySelector<HTMLElement>(".field-label")!));
    const value = `${Math.min(width([...STATIC_LABELS, ...axisLabels]), Math.max(width(STATIC_LABELS), token("--data-label-cap")))}px`;
    if (this.root.style.getPropertyValue("--label-width") !== value) this.root.style.setProperty("--label-width", value);
  }
}
