// Frame Survival's pairs (v2 SurvivalForm): the selection's closed set on a
// strip beside the section title, a start × end grid of cards, and the lines
// that say what is kept but not published.  The model (frame_survival/
// editor.py) says which cards are selected, which the rule derives, how each
// frame is written and which grid card every listed row names; a gesture is
// written back as the rule's exceptions (`pairs`) or as Selected's own list
// (`custom_pairs`), the rows the model handed over, edited.
import { Segmented, SelectionGrid, element, setText, type Cell, type CellState } from "../../fluent/index.js";
import type { EditorFacts, LogicRow } from "../../workbench.js";
import type { DraftEdit, NodeEditor } from "./index.js";

type PairRow = { start: string; end: string; selected: boolean };
type Facts = EditorFacts & {
  title: string;
  selection: string;
  rule: boolean;
  keeps_list: boolean;
  error: string;
  grid: { rows: string[]; columns: string[]; cells: [number, number, string][] } | null;
  selected: Cell[];
  baseline: Cell[];
  states: Record<string, CellState>;
  tips: Record<string, string>;
  enabled: boolean;
  identity: string[];
  override_rows: PairRow[];
  override_cells: (Cell | null)[];
  list_rows: PairRow[];
  list_cells: (Cell | null)[];
  summary: string;
  note: string;
  parked: string;
};

const key = ([row, column]: readonly [number, number]): string => `${row},${column}`;
const cellOf = (name: string): Cell => name.split(",").map(Number) as unknown as Cell;
const byCell = (a: string, b: string): number => { const [ra, ca] = cellOf(a), [rb, cb] = cellOf(b); return ra - rb || ca - cb; };

export class PairMatrixEditor implements NodeEditor {
  readonly root = element("div", "node-editor pair-matrix");
  private title = element("span", "heading");
  private presets = new Segmented(value => this.preset(String(value)));
  private summary = element("span", "summary-text");
  private grid = new SelectionGrid({ labelColumn: "var(--label-width, var(--label-min))", labelsGrow: true });
  private error = element("p", "field-error");
  private note = element("p", "note wrap warning");
  private parked = element("p", "note wrap");
  private facts: Facts | null = null;
  private gridSignature = "";

  constructor(private edit: DraftEdit) {
    const header = element("div", "title-row editor-title");
    header.append(this.title, this.presets.root, this.summary);
    this.grid.edited = cells => this.marked(cells);
    this.root.append(header, this.grid.root, this.error, this.note, this.parked);
  }

  update(row: LogicRow, raw: EditorFacts, locked: string): void {
    const facts = raw as Facts;
    this.facts = facts;
    setText(this.title, facts.title);
    // The strip is the selection field's own closed set, in its own words.
    this.presets.setChoices(row.controls.find(control => control.name === "selection")?.choices ?? []);
    this.presets.setCurrent(facts.selection);
    this.presets.disabled = Boolean(locked);
    setText(this.summary, facts.summary);
    const grid = facts.grid;
    const signature = JSON.stringify(grid);
    if (signature !== this.gridSignature) {
      this.gridSignature = signature;
      this.grid.setGrid(grid ?? { rows: [], columns: [], cells: [] });
    }
    this.grid.root.hidden = !grid;
    this.grid.setStates(facts.states, facts.tips);
    this.grid.setSelected(facts.selected);
    this.grid.disabled = !facts.enabled || Boolean(locked);
    for (const [line, text] of [[this.error, facts.error], [this.note, facts.note], [this.parked, facts.parked]] as const) {
      setText(line, text);
      line.hidden = !text;
    }
  }

  /** How the matrix writes the pair a card stands for: its ends by identity. */
  private pair([row, column]: Cell, selected: boolean): PairRow {
    const identity = this.facts!.identity;
    return { start: identity[row] ?? "", end: identity[column + 1] ?? "", selected };
  }

  /** A choice on the strip starts afresh: a rule's exceptions go, Selected's
   * list stays.  With no list to come back to, Selected begins with what is
   * selected now (a rule's pairs, or none after None). */
  private preset(choice: string): void {
    const facts = this.facts;
    if (!facts || choice === facts.selection) return;
    const values: Record<string, unknown> = { selection: choice, pairs: [] };
    if (choice === "custom" && !facts.keeps_list && facts.grid) values.custom_pairs = facts.selected.map(cell => this.pair(cell, true));
    this.edit(values, "Published pairs");
  }

  private marked(cells: Cell[]): void {
    const facts = this.facts;
    if (!facts || !facts.enabled) return;
    const current = new Set(cells.map(key));
    const wanted = new Set(facts.selected.map(key));
    if (facts.rule) {
      // The edits are the rule's exceptions: a card that now differs from
      // what the rule selects is one, a card back at the rule is none.
      const baseline = new Set(facts.baseline.map(key));
      const toggled = new Set([...current, ...wanted].filter(name => current.has(name) !== wanted.has(name)));
      const rows: PairRow[] = [], handled = new Set<string>();
      facts.override_rows.forEach((row, index) => {
        const cell = facts.override_cells[index];
        if (!cell) { rows.push(row); return; }   // not in this source: kept as written
        const name = key(cell);
        if (handled.has(name)) return;
        handled.add(name);
        if (!toggled.has(name)) { rows.push(row); return; }
        if (current.has(name) !== baseline.has(name)) rows.push({ ...row, selected: current.has(name) });
      });
      for (const name of [...toggled].sort(byCell)) {
        if (!handled.has(name) && current.has(name) !== baseline.has(name)) rows.push(this.pair(cellOf(name), current.has(name)));
      }
      this.edit({ selection: facts.selection, pairs: rows }, "Pairs");
      return;
    }
    // The edits are Selected's list.  Under None none of it is shown, so
    // nothing of it is unmarked: a marked pair joins it and picks Selected.
    const listed = new Set(facts.list_rows.flatMap((row, index) => {
      const cell = facts.list_cells[index];
      return row.selected && cell ? [key(cell)] : [];
    }));
    if (facts.selection === "none") for (const name of listed) current.add(name);
    const removed = new Set([...listed].filter(name => !current.has(name)));
    const added = [...current].filter(name => !listed.has(name)).sort(byCell);
    const rows: PairRow[] = [];
    facts.list_rows.forEach((row, index) => {
      const cell = facts.list_cells[index], name = cell ? key(cell) : "";
      if (row.selected && removed.has(name)) return;
      const at = row.selected ? -1 : added.indexOf(name);
      if (at >= 0) { added.splice(at, 1); rows.push({ ...row, selected: true }); return; }
      rows.push(row);
    });
    rows.push(...added.map(name => this.pair(cellOf(name), true)));
    this.edit({ selection: "custom", custom_pairs: rows }, "Pairs");
  }
}
