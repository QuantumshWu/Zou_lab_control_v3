// Occupancy's calibrations (v2 CalibrationsEditor): one row per file, the
// frames it reads as columns -- First and Last their own, then Rest, then the
// middle frames by number; a marked card is a claim.  The model (occupancy/
// editor.py) says how the source's frames read the files (a frame through
// Rest dashed, one nobody reads empty red, one two files read red, an absent
// middle number amber), what each file is, and what is refused or kept.  A
// file is added with Add calibration…, replaced by its ⋯ (or a click on its
// name) and removed by its ×; the readout model is a segmented choice under
// the files.  The grid writes back the claims and the model.
import { Segmented, SelectionGrid, button, element, form, row as formRow, setText, token, type Cell, type CellState } from "../../fluent/index.js";
import type { EditorFacts, LogicRow } from "../../workbench.js";
import type { EditorHost, NodeEditor } from "./index.js";

type Token = string | number;
type Entry = { path: string; frames: Token[] };
type Facts = EditorFacts & {
  title: string; summary: string;
  grid: { rows: string[]; columns: string[]; cells: [number, number, string][]; rowTips: string[]; mutedRows: number[]; narrowColumns: number[] };
  selected: Cell[]; states: Record<string, CellState>; tips: Record<string, string>;
  columns: (Token | null)[]; entries: Entry[]; new_frames: Token[]; file_filter: string;
  problems: string[]; notes: string[]; hint: string;
};

const key = ([row, column]: readonly [number, number]): string => `${row},${column}`;

export class FrameClaimsEditor implements NodeEditor {
  readonly root = element("div", "node-editor frame-claims");
  private title = element("span", "heading");
  private summary = element("span", "summary-text");
  private add = button("Add calibration…", () => void this.addFile());
  private grid = new SelectionGrid({
    toggle: true, labelsLeft: true, labelColumn: "var(--label-width, var(--label-min))",
    rowActions: [{ value: "replace", glyph: "⋯", title: "Choose another file", role: "quiet" }, { value: "remove", glyph: "×", title: "Remove this calibration" }],
    labelAction: { value: "replace", title: "Click to choose another file" },
  });
  private model = new Segmented(value => this.host.edit({ model_kind: value }, "Readout model"));
  private modelRow: HTMLDivElement;
  private problems = element("div", "frame-claims-lines");
  private notes = element("div", "frame-claims-lines");
  private hint = element("p", "note wrap");
  private facts: Facts | null = null;
  private signature = "";

  constructor(private host: EditorHost) {
    const header = element("div", "title-row editor-title");
    header.append(this.title, this.summary, this.add);
    this.grid.edited = cells => this.marked(cells);
    this.grid.rowAction = (row, action) => { if (action === "remove") this.remove(row); else void this.replace(row); };
    const modelForm = form();
    this.modelRow = formRow("Readout model", this.model.root);
    modelForm.append(this.modelRow);
    this.root.append(header, this.grid.root, modelForm, this.problems, this.notes, this.hint);
  }

  /** A file's name gets the room calibrations are named in (v2 _FILE_NAME_ROOM),
   * fixed, so the column is the same with no file, one or ten. */
  labelFloor(): number { return token("--file-name-room"); }

  update(row: LogicRow, raw: EditorFacts, locked: string): void {
    const facts = raw as Facts;
    this.facts = facts;
    setText(this.title, facts.title);
    setText(this.summary, facts.summary);
    this.add.disabled = Boolean(locked);
    const signature = JSON.stringify(facts.grid);
    if (signature !== this.signature) { this.signature = signature; this.grid.setGrid(facts.grid); }
    this.grid.setStates(facts.states, facts.tips);
    this.grid.setSelected(facts.selected);
    this.grid.disabled = Boolean(locked);
    const shown = facts.entries.length > 0;
    this.grid.root.hidden = this.modelRow.hidden = !shown;
    this.model.setChoices(row.controls.find(control => control.name === "model_kind")?.choices ?? []);
    this.model.setCurrent(row.values.model_kind ?? null);
    this.model.disabled = Boolean(locked);
    // One sentence per block, half a row gap apart: a problem that wraps is
    // still told from the next.  Problems red, notes amber (said, not refused).
    this.lines(this.problems, facts.problems, "field-error");
    this.lines(this.notes, facts.notes, "field-error warning");
    setText(this.hint, facts.hint);
  }

  private lines(host: HTMLElement, texts: string[], className: string): void {
    while (host.children.length > texts.length) host.lastElementChild!.remove();
    while (host.children.length < texts.length) host.append(element("p", className));
    texts.forEach((text, index) => setText(host.children[index]!, text));
    host.hidden = !texts.length;
  }

  private entries(): Entry[] { return (this.facts?.entries ?? []).map(entry => ({ path: entry.path, frames: [...entry.frames] })); }

  private write(entries: Entry[]): void { this.host.editArtifacts({ calibrations: entries }, this.facts?.title ?? "Calibrations"); }

  /** Cells marked and unmarked: a frame (First, Last, Rest or a middle
   * number) belongs to one file, so taking it here takes it from the file that had it. */
  private marked(cells: Cell[]): void {
    const facts = this.facts;
    if (!facts) return;
    const before = new Set(facts.selected.map(key)), after = new Set(cells.map(key));
    const entries = this.entries();
    const tokenOf = (name: string): Token | null => facts.columns[Number(name.split(",")[1])] ?? null;
    for (const name of [...before].filter(name => !after.has(name))) {
      const row = Number(name.split(",")[0]), token = tokenOf(name);
      entries[row]!.frames = entries[row]!.frames.filter(item => item !== token);
    }
    for (const name of [...after].filter(name => !before.has(name)).sort()) {
      const row = Number(name.split(",")[0]), token = tokenOf(name);
      if (token === null) continue;
      entries.forEach((entry, index) => { if (index !== row) entry.frames = entry.frames.filter(item => item !== token); });
      entries[row]!.frames.push(token);
    }
    this.write(entries);
  }

  private remove(row: number): void { this.write(this.entries().filter((_, index) => index !== row)); }

  private async replace(row: number): Promise<void> {
    const entries = this.entries();
    const path = await this.host.choose({ title: "Replace calibration", filter: this.facts?.file_filter, start: entries[row]?.path ?? "" });
    if (!path) return;
    entries[row]!.path = path;
    this.write(entries);
  }

  /** Another calibration: the first one reads every frame, a later one is
   * kept reading none until it is given frames -- the draft still starts. */
  private async addFile(): Promise<void> {
    const path = await this.host.choose({ title: "Add calibration", filter: this.facts?.file_filter, start: "" });
    if (!path) return;
    this.write([...this.entries(), { path, frames: [...(this.facts?.new_frames ?? [])] }]);
  }
}
