import { element } from "./dom.js";
import { setMiddle } from "./elide.js";
import { setGlyph } from "./ink.js";

export type Cell = readonly [row: number, column: number];
/** A rule reached it, it cannot be honoured, it is kept with nothing to act
 * on this time, or nobody has it (GRID_CELL_STATES). */
export type CellState = "derived" | "error" | "kept" | "missing";
export type RowAction = { value: string; glyph: string; title: string; role?: "destructive" | "quiet" };
export type GridSpec = {
  rows: readonly string[];
  columns: readonly string[];
  /** The label of every selectable cell; a cell without one is no cell. */
  cells: readonly (readonly [row: number, column: number, label: string])[];
  rowTips?: readonly string[];
  /** Rows kept but not in use: their labels read muted. */
  mutedRows?: readonly number[];
  /** Gaps between column groups, holding no cells. */
  narrowColumns?: readonly number[];
};

const key = (row: number, column: number): string => `${row},${column}`;

/** FluentSelectionGrid: a grid of choices whose selection IS the state.
 * Unchosen cells are blank BG cards, chosen ones named SELECTED cards; every
 * state is a fill, a dashed edge only where it says something.  The ordinary
 * table gestures: click one, drag a rectangle, Ctrl adds or removes, a header
 * takes its whole line.  `toggle` marks cells one by one instead (a click
 * flips one, a drag sweeps the run), headers select nothing.  `edited` comes
 * once a gesture is over -- not per cell the pointer crosses, which in v2
 * re-sent the selection on every move of a drag; `setSelected` never emits. */
export class SelectionGrid {
  readonly root = element("div", "selection-grid");
  edited: (cells: Cell[]) => void = () => {};
  rowAction: (row: number, value: string) => void = () => {};
  private cells = new Map<string, HTMLDivElement>();
  private selected = new Set<string>();
  private gesture: { anchor: Cell; base: Set<string>; before: Set<string>; add: boolean; sweep?: boolean } | null = null;

  /** `labelColumn`: the width of the form label column the row labels stand in (set_label_width); their
   * track is that and the form's field gap less a card's inset (_label_offset), so the first card starts
   * on the field edge. `labelsGrow`: a longer label widens the track (else labels are cut to it). */
  constructor(private options: { toggle?: boolean; rowActions?: readonly RowAction[]; labelAction?: { value: string; title: string }; labelsLeft?: boolean; labelColumn?: string; labelsGrow?: boolean } = {}) {
    this.root.role = "grid";
    this.root.addEventListener("pointerup", () => this.endGesture());
    this.root.addEventListener("lostpointercapture", () => this.endGesture());
    // Under capture the pointer's events all come here; the cell under it is
    // looked up, so a drag keeps working past the grid's edge.
    this.root.addEventListener("pointermove", event => {
      if (!this.gesture) return;
      const cell = document.elementFromPoint(event.clientX, event.clientY)?.closest<HTMLElement>(".grid-cell");
      if (cell && this.root.contains(cell)) this.cellEnter(Number(cell.dataset.row), Number(cell.dataset.column));
    });
  }

  setGrid(spec: GridSpec): void {
    this.cells.clear();
    const narrow = new Set(spec.narrowColumns ?? []);
    const actions = this.options.rowActions ?? [];
    const columns: string[] = spec.columns.map((_, index) => narrow.has(index) ? "var(--section-gap)" : "minmax(var(--grid-cell-min), 1fr)");
    if (actions.length) columns.push("var(--section-gap)", ...actions.map(() => "var(--grid-action)"));
    const track = this.options.labelColumn ? `calc(${this.options.labelColumn} + var(--row-gap) - var(--grid-gutter))` : "max-content";
    this.root.style.gridTemplateColumns = `${this.options.labelsGrow ? `minmax(${track}, max-content)` : track} ${columns.join(" ")}`;
    this.root.classList.toggle("labels-left", Boolean(this.options.labelsLeft));
    const nodes: HTMLElement[] = [element("div", "grid-corner")];
    spec.columns.forEach((label, column) => {
      const header = element("div", "grid-header", label);
      if (!this.options.toggle && !narrow.has(column)) header.addEventListener("pointerdown", event => this.headerPress(event, null, column));
      nodes.push(header);
    });
    if (actions.length) nodes.push(element("div", "grid-gap"), ...actions.map(() => element("div", "grid-gap")));
    const labels = new Map(spec.cells.map(([row, column, label]) => [key(row, column), label]));
    const muted = new Set(spec.mutedRows ?? []);
    spec.rows.forEach((label, row) => {
      const header = element("div", "grid-row-label", label);
      // In a form's label column a longer name keeps its start and its end (v2 occupancy _fitted).
      if (this.options.labelColumn) setMiddle(header, label, "none");
      header.classList.toggle("muted", muted.has(row));
      header.title = [spec.rowTips?.[row], this.options.labelAction?.title].filter(Boolean).join("\n");
      if (this.options.labelAction) {
        header.classList.add("action");
        header.addEventListener("click", () => this.rowAction(row, this.options.labelAction!.value));
      } else if (!this.options.toggle) header.addEventListener("pointerdown", event => this.headerPress(event, row, null));
      nodes.push(header);
      spec.columns.forEach((_, column) => {
        const name = labels.get(key(row, column));
        const cell = element("div", name === undefined ? "grid-void" : "grid-cell");
        if (name !== undefined) {
          cell.append(element("span", "card-face"), element("span", "card-name", name));
          cell.title = name;
          cell.dataset.row = String(row);
          cell.dataset.column = String(column);
          cell.addEventListener("pointerdown", event => this.cellPress(event, row, column));
          this.cells.set(key(row, column), cell);
        }
        nodes.push(cell);
      });
      if (actions.length) {
        nodes.push(element("div", "grid-gap"));
        for (const action of actions) {
          const glyph = element("button", `grid-action ${action.role === "quiet" ? "quiet" : "destructive"} glyph square`);
          setGlyph(glyph, action.glyph, "glyph");
          glyph.type = "button";
          glyph.tabIndex = -1;
          glyph.title = action.title;
          glyph.addEventListener("click", () => this.rowAction(row, action.value));
          nodes.push(glyph);
        }
      }
    });
    this.root.replaceChildren(...nodes);
    this.selected = new Set([...this.selected].filter(cell => this.cells.has(cell)));
    this.paint();
  }

  setSelected(cells: readonly Cell[]): void {
    this.selected = new Set(cells.map(([row, column]) => key(row, column)).filter(cell => this.cells.has(cell)));
    this.paint();
  }

  selectedCells(): Cell[] {
    return [...this.selected].map(cell => cell.split(",").map(Number) as unknown as Cell).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  }

  setStates(states: ReadonlyMap<string, CellState> | Record<string, CellState>, tips: Record<string, string> = {}): void {
    const map = states instanceof Map ? states : new Map(Object.entries(states));
    for (const [name, cell] of this.cells) {
      const state = map.get(name);
      if (state) cell.dataset.state = state; else delete cell.dataset.state;
      cell.title = tips[name] ?? cell.querySelector(".card-name")!.textContent ?? "";
    }
  }

  set disabled(disabled: boolean) { this.root.classList.toggle("disabled", disabled); }

  private paint(): void {
    for (const [name, cell] of this.cells) cell.classList.toggle("selected", this.selected.has(name));
  }

  private live(): boolean { return !this.root.classList.contains("disabled"); }

  private cellPress(event: PointerEvent, row: number, column: number): void {
    if (event.button !== 0 || !this.live()) return;
    event.preventDefault();
    this.root.setPointerCapture(event.pointerId);
    const name = key(row, column), before = new Set(this.selected);
    if (this.options.toggle) {
      // The first cell's new state is what the sweep paints.
      const add = !this.selected.has(name);
      this.gesture = { anchor: [row, column], base: new Set(this.selected), before, add, sweep: true };
      if (add) this.selected.add(name); else this.selected.delete(name);
    } else if (event.shiftKey && this.gesture === null && this.lastAnchor) {
      this.gesture = { anchor: this.lastAnchor, base: event.ctrlKey ? new Set(this.selected) : new Set(), before, add: true };
      this.extend(row, column);
    } else {
      const add = !(event.ctrlKey && this.selected.has(name));
      this.gesture = { anchor: [row, column], base: event.ctrlKey ? new Set(this.selected) : new Set(), before, add };
      this.lastAnchor = [row, column];
      this.extend(row, column);
    }
    this.paint();
  }
  private lastAnchor: Cell | null = null;

  private cellEnter(row: number, column: number): void {
    if (!this.gesture) return;
    if (this.gesture.sweep) {
      const name = key(row, column);
      if (this.gesture.add) this.selected.add(name); else this.selected.delete(name);
    } else this.extend(row, column);
    this.paint();
  }

  /** The rectangle from the anchor to (row, column), over the gesture's base. */
  private extend(row: number, column: number): void {
    const gesture = this.gesture!;
    const [r0, c0] = gesture.anchor;
    this.selected = new Set(gesture.base);
    for (let r = Math.min(r0, row); r <= Math.max(r0, row); r++) {
      for (let c = Math.min(c0, column); c <= Math.max(c0, column); c++) {
        const name = key(r, c);
        if (!this.cells.has(name)) continue;
        if (gesture.add) this.selected.add(name); else this.selected.delete(name);
      }
    }
  }

  private headerPress(event: PointerEvent, row: number | null, column: number | null): void {
    if (event.button !== 0 || !this.live()) return;
    event.preventDefault();
    const line = [...this.cells.keys()].filter(name => { const [r, c] = name.split(",").map(Number); return row === null ? c === column : r === row; });
    const before = new Set(this.selected), base = event.ctrlKey ? new Set(this.selected) : new Set<string>();
    for (const name of line) base.add(name);
    this.selected = base;
    this.paint();
    this.gesture = { anchor: [0, 0], base, before, add: true };
    this.endGesture();
  }

  private endGesture(): void {
    if (!this.gesture) return;
    const before = this.gesture.before;
    this.gesture = null;
    const now = this.selectedCells().map(([row, column]) => key(row, column));
    if (now.length !== before.size || now.some(name => !before.has(name))) this.edited(this.selectedCells());
  }
}
