// The Seamless Scan's plan (v2 ScanPlanEditor): one row per axis, outermost
// first -- the knob (a port of the pulse, its API or a device, or a manual
// axis's name), then its values as an independent Range bank (from, to,
// points) or Values list, their unit, which of the two it plays, and its ×.
// Every row is a line of one column set, so from, to, points, the unit,
// Range|Values and × stand at one x on every row.  Under the rows, what the
// plan is or why it cannot be read; then the pulse's API values this run
// sets.  The model (scan/editor.py) offers the knobs, says what the plan
// amounts to and words the API values; this editor writes `plan` (the axes
// as the JSON document) and `api_values` (only what differs from the pulse,
// in the pulse's units whatever spelling a box is read in; re-read against
// the pulse when the model says that changes it).
import { Field, Segmented, SpinBox, TreeCombo, alignUnitPickers, button, choiceTree, element, sectionTitle, setText, token, unitTree, type FieldSpec, type Value } from "../../fluent/index.js";
import type { EditorFacts, LogicRow } from "../../workbench.js";
import type { DraftEdit, NodeEditor } from "./index.js";

/** `units`: the spellings an axis on the port may be authored in, each with
 * its decade against the port's unit, or null where no decimal shift reaches
 * it (a device knob's dBm and mW) -- `limits` are the port's in those
 * (empty: only its own). */
type Port = { port: string; label: string; unit: string; lo: number; hi: number; seed_lo: number; seed_hi: number; group: string; leaf: string;
  units: [symbol: string, decade: number | null][]; limits: Record<string, [number, number]>;
  /** The family each spelling belongs to (its unit picker's trunks). */
  families: Record<string, string> };
/** A device axis's banks in another spelling, as the model converted the draft's. */
type Converted = { values: number[]; explicit: number[] };
type ApiField = { name: string; label: string; kind: "integer" | "number"; unit: string; units: [symbol: string, decade: number][]; families: Record<string, string>;
  description: string; minimum: number | null; maximum: number | null; authored: number; value: number; unavailable_reason: string };
type Facts = EditorFacts & {
  families: { pulse: string; api: string; device: string; manual: string };
  host_advanced: string[];
  ports: Port[];
  withheld: Record<string, string>;
  /** The plan's ports nobody offers now, with their owner and name. */
  unoffered: Record<string, { group: string; leaf: string }>;
  summary: string;
  problem: string;
  /** By port, each device axis's banks in every spelling past the decimal shifts. */
  conversions: Record<string, Record<string, Converted>>;
  api: { fields: ApiField[]; overrides: [string, number][]; parked: { name: string; text: string; problem: boolean }[]; note: string;
    /** The api_values text re-read against this pulse, when that changes it. */
    tidy: string | null } | null;
};
/** One axis as the plan document writes it: both input banks, and which one plays. */
type Axis = { port: string; values: number[]; unit: string; mode: "range" | "values"; value_text: string };

/** The word a manual axis carries in the label column: a hand moves it. */
const BY_HAND = "By hand";

/** numpy.linspace: the authored coordinates in the row's unit. */
function linspace(start: number, stop: number, count: number): number[] {
  if (count <= 1) return [start];
  const step = (stop - start) / (count - 1);
  return Array.from({ length: count }, (_, index) => index === count - 1 ? stop : index * step + start);
}

const ulp = (value: number): number => value === 0 ? Number.MIN_VALUE : 2 ** (Math.floor(Math.log2(Math.abs(value))) - 52);

/** Whether a list is the evenly spaced grid its ends describe, judged before
 * any display rounding (v2 _uniform_values: only the last-bit error two float
 * routes differ by). */
function uniform(values: number[]): boolean {
  const first = values[0], last = values[values.length - 1];
  if (first === undefined || last === undefined || values.length < 2) return true;
  const tolerance = 4 * ulp(Math.max(Math.abs(first), Math.abs(last)));
  return linspace(first, last, values.length).every((value, index) => Math.abs(value - values[index]!) <= tolerance);
}

/** The plan's axes as written; an unreadable plan has no rows (its problem is the model's to say). */
function readPlan(text: Value | unknown): Axis[] {
  try {
    const tree = JSON.parse(String(text ?? "")) as { axes?: Partial<Axis>[] };
    return (tree.axes ?? []).map(axis => ({ port: String(axis.port ?? ""), values: (axis.values ?? []).map(Number), unit: String(axis.unit ?? ""),
      mode: axis.mode === "values" ? "values" : "range", value_text: String(axis.value_text ?? "") }));
  } catch { return []; }
}

/** `value` with its decimal digits shifted by `places` (Decimal.scaleb on its
 * shortest spelling): a prefix change never takes a binary round trip. */
function shifted(value: number, places: number): number {
  const [digits, exponent = "0"] = String(value).split("e");
  return Number(`${digits}e${Number(exponent) + places}`);
}

/** A spin box for one end or the count of a Range bank. */
function rangeSpin(label: string, step: () => number, bounds: () => [number | null, number | null], commit: (value: number) => void, decimal = true): SpinBox {
  const holder = element("div");
  const input = element("input");
  input.type = "text";
  input.spellcheck = false;
  input.inputMode = "decimal";
  input.setAttribute("aria-label", label);
  holder.append(input);
  const spin = new SpinBox(holder, input, { step, bounds, commit, decimal });
  input.addEventListener("change", () => {
    const value = Number(input.value.trim());
    const [low, high] = bounds();
    const valid = input.value.trim() !== "" && Number.isFinite(value) && (low === null || value >= low) && (high === null || value <= high);
    input.classList.toggle("invalid", !valid);
    if (valid) commit(value);
  });
  return spin;
}

class AxisRowView {
  readonly kind = element("span", "field-label label-column axis-kind");
  readonly axis: HTMLElement;
  readonly values = element("div", "axis-values");
  readonly unit = element("div", "axis-unit");
  private unitWord = element("span", "unit-word");
  private unitPicker = new TreeCombo(symbol => this.unitPicked(symbol));
  readonly mode = new Segmented(value => this.pickMode(value === "values" ? "values" : "range"));
  readonly remove = button("×", () => this.owner.removeRow(this), "destructive glyph square");
  readonly notes = element("div", "axis-notes");
  private custom = element("p", "note");
  private error = element("p", "field-error");
  private combo: TreeCombo | null = null;
  private name: HTMLInputElement | null = null;
  private start: SpinBox;
  private stop: SpinBox;
  private points: SpinBox;
  private list = element("input", "axis-list");
  private bank = element("div", "range-bank");
  entry: Axis;
  /** A list nobody's ends describe, kept until a spin is edited. */
  private customValues: number[] | null = null;

  constructor(private owner: ScanPlanEditor, readonly manual: boolean, entry: Axis) {
    this.entry = entry;
    if (manual) {
      this.kind.textContent = BY_HAND;
      const name = element("input", "axis-name");
      name.type = "text";
      name.spellcheck = false;
      name.placeholder = "manual axis name";
      name.addEventListener("change", () => { this.entry = { ...this.entry, port: owner.families.manual + name.value.trim() }; owner.emitPlan(); });
      this.name = name;
      this.axis = name;
    } else {
      this.combo = new TreeCombo(port => this.portChanged(port));
      this.combo.root.classList.add("axis-port");
      this.axis = this.combo.root;
    }
    // The port's limits, read in the unit the axis is authored in.
    const bounds = (): [number | null, number | null] => {
      const port = this.port(), places = port && this.places(port.unit, this.entry.unit);
      if (port && places !== null && places !== undefined) return [shifted(port.lo, places), shifted(port.hi, places)];
      return port?.limits[this.entry.unit] ?? [null, null];
    };
    this.start = rangeSpin("from", () => 1, bounds, () => this.spinsEdited());
    this.stop = rangeSpin("to", () => 1, bounds, () => this.spinsEdited());
    this.points = rangeSpin("points", () => 1, () => [1, 100_000], () => this.spinsEdited(), false);
    this.points.input.inputMode = "numeric";
    this.bank.append(element("span", "bank-word", "from"), this.start.root, element("span", "bank-word", "to"), this.stop.root,
      element("span", "bank-word", "points"), this.points.root);
    this.list.type = "text";
    this.list.spellcheck = false;
    this.list.placeholder = "1, 3, 5, 6, 11";
    this.list.title = "Scan these values in the written order";
    this.list.addEventListener("change", () => { this.entry = { ...this.entry, value_text: this.list.value }; owner.emitPlan(); });
    this.values.append(this.bank, this.list);
    // ONE picker for both ends: from and to are two ends of one sweep.
    this.unitPicker.root.classList.add("unit-picker");
    this.unitPicker.title = "Read this axis in another unit";
    this.unit.append(this.unitWord, this.unitPicker.root);
    this.mode.setChoices([["range", "Range"], ["values", "Values"]]);
    this.mode.root.title = "Scan a from/to/points range, or the values written";
    this.remove.title = "Remove this axis";
    this.error.hidden = this.custom.hidden = true;
    this.notes.append(this.custom, this.error);
  }

  get cells(): HTMLElement[] { return [this.kind, this.axis, this.values, this.unit, this.mode.root, this.remove, this.notes]; }

  private port(): Port | undefined { return this.owner.ports.find(port => port.port === this.entry.port); }

  /** Bring the row to its authored axis; the box under the operator's cursor
   * keeps what they are typing. */
  show(entry: Axis): void {
    this.entry = entry;
    if (this.combo) this.combo.setTree(this.owner.portTree(entry.port), entry.port);
    if (this.name && document.activeElement !== this.name) this.name.value = entry.port.slice(this.owner.families.manual.length);
    const values = entry.values.length ? entry.values : [0];
    this.customValues = uniform(entry.values) ? null : [...entry.values];
    for (const [spin, value] of [[this.start, values[0]], [this.stop, values[values.length - 1]], [this.points, entry.values.length || 1]] as const) {
      if (document.activeElement !== spin.input) { spin.input.value = String(value); spin.input.classList.remove("invalid"); }
      spin.refresh();
    }
    if (document.activeElement !== this.list) this.list.value = entry.value_text;
    // A unit with other spellings is a picker; one without is a word.
    const port = this.port();
    const offered = port?.units.some(([symbol]) => symbol === entry.unit) ? port.units : [];
    this.unitPicker.setTree(unitTree(offered.map(([symbol]) => ({ value: symbol, label: symbol, group: port!.families[symbol] })), entry.unit), entry.unit);
    this.unitPicker.root.hidden = !offered.length;
    setText(this.unitWord, offered.length || entry.unit === "" || entry.unit === "1" ? "" : entry.unit);
    this.unitWord.hidden = !this.unitWord.textContent;
    this.showMode();
  }

  /** Decimal places from `unit` to `other` on this port's ladder (its own unit
   * is decade 0), or null when either is not a rung of it. */
  private places(unit: string, other: string): number | null {
    const ladder = this.port()?.units ?? [];
    const decade = (symbol: string): number | null | undefined => symbol === this.port()?.unit ? 0 : ladder.find(([name]) => name === symbol)?.[1];
    const from = decade(unit), to = decade(other);
    return from == null || to == null ? null : from - to;
  }

  /** Both banks converted to the picked spelling; an empty Values list stays empty. */
  private unitPicked(symbol: string): void {
    const places = this.places(this.entry.unit, symbol);
    if (places === null) { this.converted(symbol); return; }
    const text = this.entry.value_text.trim();
    const listed = text ? text.split(",").map(item => Number(item.trim())) : [];
    if (listed.some(value => !Number.isFinite(value))) {
      this.showError(`cannot convert unit: Values is not a list of numbers`);
      this.unitPicker.value = this.entry.unit;
      return;
    }
    this.entry = { ...this.entry, unit: symbol, values: this.entry.values.map(value => shifted(value, places)),
      value_text: text ? listed.map(value => String(shifted(value, places))).join(", ") : this.entry.value_text };
    this.show(this.entry);
    this.owner.emitPlan();
  }

  /** A spelling no decimal shift reaches: the banks as the model converted the
   * draft's, while this row is still what the draft says. */
  private converted(symbol: string): void {
    const converted = this.owner.conversion(this.entry, symbol);
    if (!converted) {
      this.showError("cannot convert unit: the row changed since it was read; pick the unit again");
      this.unitPicker.value = this.entry.unit;
      return;
    }
    this.entry = { ...this.entry, unit: symbol, values: converted.values,
      value_text: this.entry.value_text.trim() ? converted.explicit.map(String).join(", ") : this.entry.value_text };
    this.show(this.entry);
    this.owner.emitPlan();
  }

  private showError(text: string): void {
    setText(this.error, text);
    this.error.hidden = !text;
  }

  private showMode(): void {
    const values = this.entry.mode === "values";
    this.mode.setCurrent(this.entry.mode);
    this.bank.hidden = values;
    this.list.hidden = !values;
    setText(this.custom, !values && this.customValues ? "custom values" : "");
    this.custom.hidden = !this.custom.textContent;
  }

  set locked(locked: boolean) {
    for (const node of [this.combo?.root, this.name, this.start.input, this.stop.input, this.points.input, this.list, this.unitPicker.root, this.remove]) if (node) node.disabled = locked;
    this.mode.disabled = locked;
    for (const spin of [this.start, this.stop, this.points]) spin.refresh();
  }

  private pickMode(mode: Axis["mode"]): void {
    if (mode === this.entry.mode) return;
    this.entry = { ...this.entry, mode };
    this.showMode();
    this.owner.emitPlan();
  }

  /** Another knob: its own unit, and its initial sweep in the boxes. */
  private portChanged(port: string): void {
    const offered = this.owner.ports.find(item => item.port === port);
    this.customValues = null;
    this.showError("");
    const count = Number(this.points.text) || 5;
    this.entry = { ...this.entry, port, unit: offered?.unit ?? this.entry.unit,
      values: offered ? linspace(offered.seed_lo, offered.seed_hi, count) : this.entry.values };
    this.show(this.entry);
    this.owner.emitPlan();
  }

  /** An edited spin means the uniform grid the spins describe. */
  private spinsEdited(): void {
    const start = Number(this.start.text), stop = Number(this.stop.text), count = Math.round(Number(this.points.text));
    if (![start, stop, count].every(Number.isFinite) || count < 1) return;
    this.customValues = null;
    this.entry = { ...this.entry, values: linspace(start, stop, count) };
    this.showMode();
    this.owner.emitPlan();
  }
}

export class ScanPlanEditor implements NodeEditor {
  readonly root = element("div", "node-editor scan-plan");
  private add = button("Add axis", () => this.addRow(false));
  private addManual = button("Add manual axis", () => this.addRow(true));
  private axes = element("div", "scan-axes");
  private problem = element("p", "field-error");
  private summary = element("p", "note wrap");
  private apiTitle = sectionTitle("API values");
  private apiForm = element("div", "form api-values");
  private parked = element("div", "api-parked");
  private apiNote = element("p", "note wrap");
  private rows: AxisRowView[] = [];
  private apiFields = new Map<string, Field>();
  private planText: string | null = null;
  private parkedSignature = "";
  /** The spelling each API value is read in, when the operator picked another (FluentUnitPicker). */
  private shownUnits = new Map<string, string>();
  private pickers = "";
  private tidied: string | null = null;
  private facts: Facts | null = null;
  private locked = "";
  families: Facts["families"] = { pulse: "pulse:param:", api: "api:", device: "device:", manual: "manual:" };
  ports: Port[] = [];

  constructor(private edit: DraftEdit) {
    // An API value's name stands in the label column in the room the editor
    // keeps for it (labelFloor), never setting it: the column, and every field
    // after it, stays put when a pulse that binds some is chosen.
    this.apiForm.dataset.labelRoom = "";
    this.root.append(sectionTitle("Scan plan", this.add, this.addManual), this.axes, this.problem, this.summary,
      this.apiTitle, this.apiForm, this.parked, this.apiNote);
  }

  labelFloor(): number { return token("--api-label-room"); }

  update(row: LogicRow, raw: EditorFacts, locked: string): void {
    const facts = raw as Facts;
    const portsChanged = JSON.stringify([facts.ports, facts.withheld]) !== JSON.stringify([this.facts?.ports, this.facts?.withheld]);
    this.facts = facts;
    this.families = facts.families;
    this.ports = facts.ports;
    this.locked = locked;
    const text = String(row.values.plan ?? "");
    if (text !== this.planText || portsChanged) {
      this.planText = text;
      this.reconcileRows(this.ordered(readPlan(text)));
    }
    this.add.disabled = !facts.ports.length || Boolean(locked);
    this.addManual.disabled = Boolean(locked);
    for (const view of this.rows) view.locked = Boolean(locked);
    setText(this.problem, facts.problem);
    this.problem.hidden = !facts.problem;
    setText(this.summary, facts.problem ? "" : facts.summary);
    this.summary.hidden = !this.summary.textContent;
    this.updateApi(row, facts);
  }

  /** `axis` in `symbol` as the model converted it, if `axis` is the draft's row on its port. */
  conversion(axis: Axis, symbol: string): Converted | null {
    const drafted = readPlan(this.planText).find(row => row.port === axis.port);
    const same = drafted && drafted.unit === axis.unit && drafted.value_text === axis.value_text && JSON.stringify(drafted.values) === JSON.stringify(axis.values);
    return same ? this.facts?.conversions[axis.port]?.[symbol] ?? null : null;
  }

  /** Every knob under the thing that owns it (v2 _fill_ports); an authored port the bench
   * no longer offers keeps saying what was authored, never silently another. Closed, the box
   * says the knob under its owner ("api · long_after.duration"). */
  portTree(current: string) {
    const entries = this.ports.map(port => ({ key: port.port, group: port.group, label: port.port.startsWith(this.families.device) ? port.leaf : port.label }));
    const lost = this.facts?.unoffered[current];
    if (lost) entries.push({ key: current, group: lost.group, label: `${lost.leaf} (${current in this.facts!.withheld ? "its bracket is empty" : "not in this pulse"})` });
    return choiceTree(entries, current, "ports");
  }

  /** The host's axes (manual, device, API) stand outside the board's, in their order. */
  private ordered(axes: Axis[]): Axis[] {
    const host = (axis: Axis): boolean => this.facts!.host_advanced.some(prefix => axis.port.startsWith(prefix));
    return [...axes.filter(host), ...axes.filter(axis => !host(axis))];
  }

  /** The rows brought to the plan, kept wherever they can be: a row of the
   * right kind at its position is re-pointed, not rebuilt. */
  private reconcileRows(axes: Axis[]): void {
    const kept: AxisRowView[] = [];
    axes.forEach((axis, index) => {
      const manual = axis.port.startsWith(this.families.manual);
      let view = this.rows[index];
      if (!view || view.manual !== manual) view = new AxisRowView(this, manual, axis);
      view.show(axis);
      kept.push(view);
    });
    this.rows = kept;
    this.axes.replaceChildren(...kept.flatMap(view => view.cells));
    this.axes.classList.toggle("has-manual", kept.some(view => view.manual));
  }

  private addRow(manual: boolean): void {
    if (manual) {
      const taken = new Set(this.rows.filter(view => view.manual).map(view => view.entry.port.slice(this.families.manual.length)));
      let ordinal = 1;
      while (taken.has(`manual ${ordinal}`)) ordinal++;
      this.rows.push(new AxisRowView(this, true, { port: `${this.families.manual}manual ${ordinal}`, values: linspace(0, 1, 3), unit: "", mode: "range", value_text: "" }));
    } else {
      // The first knob the plan does not sweep yet (a port may stand on one axis only).
      const used = new Set(this.rows.map(view => view.entry.port));
      const port = this.ports.find(item => !used.has(item.port)) ?? this.ports[0];
      if (!port) return;
      this.rows.push(new AxisRowView(this, false, { port: port.port, values: linspace(port.seed_lo, port.seed_hi, 5), unit: port.unit, mode: "range", value_text: "" }));
    }
    const added = this.rows[this.rows.length - 1]!;
    added.show(added.entry);
    this.emitPlan();
  }

  removeRow(view: AxisRowView): void {
    this.rows = this.rows.filter(item => item !== view);
    this.emitPlan();
  }

  /** The plan the rows hold, host axes first, as the draft's `plan`. */
  emitPlan(): void {
    const order = this.ordered(this.rows.map(view => view.entry));
    this.rows = order.map(entry => this.rows.find(view => view.entry === entry)!);
    this.axes.replaceChildren(...this.rows.flatMap(view => view.cells));
    this.axes.classList.toggle("has-manual", this.rows.some(view => view.manual));
    this.planText = JSON.stringify({ axes: order.map(axis => ({ port: axis.port, unit: axis.unit, values: axis.values, mode: axis.mode, value_text: axis.value_text })) });
    this.edit({ plan: this.planText }, "Scan plan");
  }

  // ---------------------------------------------------------- API values
  private updateApi(row: LogicRow, facts: Facts): void {
    const api = facts.api;
    const fields = api?.fields ?? [];
    const shown = Boolean(fields.length || api?.parked.length);
    this.apiTitle.hidden = this.apiForm.hidden = !shown;
    const alive = new Set(fields.map(field => field.name));
    for (const [name, field] of this.apiFields) if (!alive.has(name)) { field.root.remove(); this.apiFields.delete(name); }
    let cursor: Element | null = this.apiForm.firstElementChild;
    for (const item of fields) {
      // The box holds the pulse's number read in the picked spelling: a shift of its digits.
      const symbol = this.shownSymbol(item), places = this.apiPlaces(item);
      const shown = (value: number | null): number | null => value === null ? null : shifted(value, places);
      const spec: FieldSpec = { name: item.name, label: item.label, kind: item.kind, value: shown(item.value), required: false, allow_none: false,
        unit: symbol, units: item.units.map(([value]) => ({ value, label: value, group: item.families[value] })),
        minimum: shown(item.minimum), maximum: shown(item.maximum), description: item.description,
        unavailable_reason: item.unavailable_reason || this.locked };
      let field = this.apiFields.get(item.name);
      if (!field) {
        field = new Field(`${row.node_id}-api`, spec, {
          commit: value => this.emitValues({ [item.name]: this.inPulseUnit(item.name, Number(value)) }),
          unit: picked => {
            // What the box holds now -- an edit still on its way included -- re-read in the picked spelling.
            const box = this.apiFields.get(item.name)!, held = this.inPulseUnit(item.name, Number(box.value));
            this.shownUnits.set(item.name, picked);
            if (this.facts) this.updateApi(row, this.facts);
            const shownNow = this.facts?.api?.fields.find(field => field.name === item.name);
            if (shownNow) box.respell(shifted(held, this.apiPlaces(shownNow)));
          },
        }, false);
        this.apiFields.set(item.name, field);
      } else field.update(spec);
      if (field.root !== cursor) this.apiForm.insertBefore(field.root, cursor); else cursor = cursor.nextElementSibling;
    }
    // The values the draft keeps but this run does not apply, each with its ×.
    const parked = JSON.stringify([api?.parked ?? [], this.locked]);
    if (parked !== this.parkedSignature) this.parked.replaceChildren(...(api?.parked ?? []).map(line => {
      const holder = element("div", "parked-line");
      const words = element("p", line.problem ? "field-error" : "note wrap", line.text);
      const remove = button("×", () => this.removeKept(line.name), "destructive glyph square");
      remove.title = "Remove this kept value from the draft";
      remove.disabled = Boolean(this.locked);
      holder.append(words, remove);
      return holder;
    }));
    this.parkedSignature = parked;
    this.parked.hidden = !shown || !api?.parked.length;
    setText(this.apiNote, shown ? api?.note ?? "" : "");
    this.apiNote.hidden = !this.apiNote.textContent;
    // The rows' pickers as wide as the widest spelling any offers, measured once they are laid out.
    const pickers = JSON.stringify(fields.map(field => field.units));
    if (pickers !== this.pickers) { this.pickers = pickers; requestAnimationFrame(() => alignUnitPickers(this.apiForm)); }
    // What the model re-read against this pulse goes back to the draft once.
    if (api?.tidy != null && !this.locked && api.tidy !== this.tidied) {
      this.tidied = api.tidy;
      this.edit({ api_values: api.tidy }, "API values");
    }
  }

  /** The spelling an API value is read in: the picked one while its unit offers it, else its own. */
  private shownSymbol(item: ApiField): string {
    const symbol = this.shownUnits.get(item.name);
    return symbol !== undefined && item.units.some(([name]) => name === symbol) ? symbol : item.unit;
  }

  /** Decimal places from the pulse's unit to the spelling a box is read in. */
  private apiPlaces(item: ApiField): number {
    return -(item.units.find(([name]) => name === this.shownSymbol(item))?.[1] ?? 0);
  }

  /** A number read in a box, back in the pulse's unit. */
  private inPulseUnit(name: string, value: number): number {
    const item = this.facts?.api?.fields.find(field => field.name === name);
    return item ? shifted(value, -this.apiPlaces(item)) : value;
  }

  /** Only what this run sets differently, and every value kept: a box left at
   * the pulse's own number is no override (it would outrank the next
   * recalibration); a value no free box holds is carried over as written. */
  private emitValues(changed: Record<string, number>): number {
    const api = this.facts?.api;
    if (!api) return 0;
    const free = new Map(api.fields.filter(field => !field.unavailable_reason).map(field => [field.name, field]));
    const boxed = (name: string): number => {
      const field = this.apiFields.get(name);
      return changed[name] ?? (field ? this.inPulseUnit(name, Number(field.value)) : free.get(name)!.value);
    };
    const written = new Map<string, number>();
    for (const [name, value] of api.overrides) {
      if (!free.has(name)) written.set(name, value);
      else if (boxed(name) !== free.get(name)!.authored) written.set(name, boxed(name));
    }
    for (const [name, field] of free) if (!written.has(name) && boxed(name) !== field.authored) written.set(name, boxed(name));
    return this.writeValues(written);
  }

  private removeKept(name: string): void {
    const written = new Map(this.facts?.api?.overrides ?? []);
    if (written.delete(name)) this.writeValues(written);
  }

  private writeValues(written: Map<string, number>): number {
    return this.edit({ api_values: [...written].map(([name, value]) => `${name} = ${value}`).join("\n") }, "API values");
  }
}
