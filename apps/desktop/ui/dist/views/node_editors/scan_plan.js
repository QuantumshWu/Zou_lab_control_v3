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
// the pulse when the model says that changes it).  An axis added, or its knob
// first selected, starts as a narrow sweep around where the knob stands (firstSweep);
// revisiting a knob in this editor restores its independent input banks.
// it, and an API value first shown, is read in the unit that shows it between
// 1 and 1000 (defaultUnit).  A unit picked converts a
// displayed quantity and keeps a number the operator wrote: an axis's values
// and an API override keep their digits, read in the new unit; an API box
// still at the pulse's own number is that number re-read in it.
import { Field, Segmented, SpinBox, TreeCombo, switchButton, setChecked, alignUnitPickers, button, choiceTree, defaultUnit, element, sectionTitle, setText, shifted, token, unitTree } from "../../fluent/index.js";
const copyAxis = (entry) => ({ ...entry, values: [...entry.values], ...(entry.links ? { links: entry.links.map(link => ({ ...link, values: [...link.values] })) } : {}) });
/** The word a manual axis carries in the label column: a hand moves it. */
const BY_HAND = "By hand";
/** numpy.linspace: the authored coordinates in the row's unit. */
function linspace(start, stop, count) {
    if (count <= 1)
        return [start];
    const step = (stop - start) / (count - 1);
    return Array.from({ length: count }, (_, index) => index === count - 1 ? stop : index * step + start);
}
const ulp = (value) => value === 0 ? Number.MIN_VALUE : 2 ** (Math.floor(Math.log2(Math.abs(value))) - 52);
/** Whether a list is the evenly spaced grid its ends describe, judged before
 * any display rounding (v2 _uniform_values: only the last-bit error two float
 * routes differ by). */
function uniform(values) {
    const first = values[0], last = values[values.length - 1];
    if (first === undefined || last === undefined || values.length < 2)
        return true;
    const tolerance = 4 * ulp(Math.max(Math.abs(first), Math.abs(last)));
    return linspace(first, last, values.length).every((value, index) => Math.abs(value - values[index]) <= tolerance);
}
/** The plan's axes as written; an unreadable plan has no rows (its problem is the model's to say). */
function readPlan(text) {
    try {
        const tree = JSON.parse(String(text ?? ""));
        return (tree.axes ?? []).map(axis => ({ port: String(axis.port ?? ""), values: (axis.values ?? []).map(Number), unit: String(axis.unit ?? ""),
            mode: axis.mode === "values" ? "values" : "range", value_text: String(axis.value_text ?? ""), ...(axis.links || axis.link_enabled ? { link_enabled: axis.link_enabled === true, links: (axis.links ?? []).map(link => ({ ...link, values: [...link.values] })) } : {}) }));
    }
    catch {
        return [];
    }
}
/** `value` + `step` in decimal, each as its shortest spelling: 80 − 0.1 is 79.9, never 79.90000000000001. */
function decimalSum(value, step) {
    const parts = (number) => {
        const [mantissa = "0", exponent = "0"] = String(number).split("e");
        const [whole = "0", fraction = ""] = mantissa.split(".");
        return [BigInt(whole + fraction), Number(exponent) - fraction.length];
    };
    const [a, at] = parts(value), [b, bt] = parts(step), last = Math.min(at, bt);
    return Number(`${a * 10n ** BigInt(at - last) + b * 10n ** BigInt(bt - last)}e${last}`);
}
/** The port's reading and limits in `symbol`: a decimal shift on its ladder, else the device's conversion. */
function inSpelling(port, symbol) {
    const decade = symbol === port.unit ? 0 : port.units.find(([name]) => name === symbol)?.[1];
    if (typeof decade === "number")
        return { current: port.current === null ? null : shifted(port.current, -decade), bounds: [shifted(port.lo, -decade), shifted(port.hi, -decade)] };
    return { current: port.currents[symbol] ?? null, bounds: port.limits[symbol] ?? [null, null] };
}
/** A knob's first sweep, when an axis is added or its knob changed: in the unit `defaultUnit` picks
 * from where it stands (in the family its owner shows it in), from 0.1 below to 0.1 above that
 * in it -- a whole-number knob 1, one value a point -- in five points, inside its limits.  A knob
 * that says nowhere it stands is one point, at its lower limit: never its whole range. */
function firstSweep(port) {
    const family = (symbol) => port.families[symbol] ?? symbol;
    const own = port.display_unit && port.units.some(([symbol]) => symbol === port.display_unit) ? port.display_unit : port.unit;
    const unit = defaultUnit(own, port.units.filter(([symbol]) => family(symbol) === family(own)).map(([symbol]) => {
        const { current, bounds: [low, high] } = inSpelling(port, symbol);
        return { symbol, value: current, low, high };
    }), port.integer);
    const { current, bounds: [low, high] } = inSpelling(port, unit);
    const inside = (value) => Math.min(high ?? Infinity, Math.max(low ?? -Infinity, value));
    if (current === null)
        return { unit, values: [low ?? high ?? 0] };
    const step = port.integer ? 1 : 0.1;
    const from = inside(decimalSum(current, -step)), to = inside(decimalSum(current, step));
    if (!(from < to))
        return { unit, values: [inside(current)] };
    if (!port.integer)
        return { unit, values: linspace(from, to, 5) };
    return { unit, values: Array.from({ length: Math.floor(to) - Math.ceil(from) + 1 }, (_, index) => Math.ceil(from) + index) };
}
/** A spin box for one end or the count of a Range bank. */
function rangeSpin(label, step, bounds, commit, decimal = true) {
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
        if (valid)
            commit(value);
    });
    return spin;
}
/** The Values list's tooltip: what it is. */
const LIST_TIP = "Scan these values in the written order";
class AxisRowView {
    owner;
    manual;
    kind = element("span", "field-label label-column axis-kind");
    axis;
    values = element("div", "axis-values");
    unit = element("div", "axis-unit");
    unitWord = element("span", "unit-word");
    unitPicker = new TreeCombo(symbol => this.unitPicked(symbol));
    mode = new Segmented(value => this.pickMode(value === "values" ? "values" : "range"));
    remove = button("×", () => this.owner.removeRow(this), "destructive glyph square");
    notes = element("div", "axis-notes");
    custom = element("p", "note");
    combo = null;
    name = null;
    start;
    stop;
    points;
    list = element("input", "axis-list");
    bank = element("div", "range-bank");
    entry;
    linkLocked = false;
    linkControls = [];
    /** A list nobody's ends describe, kept until a spin is edited. */
    customValues = null;
    constructor(owner, manual, entry) {
        this.owner = owner;
        this.manual = manual;
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
        }
        else {
            this.combo = new TreeCombo(port => this.portChanged(port));
            this.combo.root.classList.add("axis-port");
            this.axis = this.combo.root;
        }
        const bounds = () => this.bounds();
        this.start = rangeSpin("from", () => 1, bounds, () => this.spinsEdited());
        this.stop = rangeSpin("to", () => 1, bounds, () => this.spinsEdited());
        this.points = rangeSpin("points", () => 1, () => [1, 100_000], () => this.spinsEdited(), false);
        this.points.input.inputMode = "numeric";
        this.bank.append(element("span", "bank-word", "from"), this.start.root, element("span", "bank-word", "to"), this.stop.root, element("span", "bank-word", "points"), this.points.root);
        this.list.type = "text";
        this.list.spellcheck = false;
        this.list.placeholder = "1, 3, 5, 6, 11";
        this.list.title = LIST_TIP;
        this.list.addEventListener("change", () => { this.entry = { ...this.entry, value_text: this.list.value }; owner.emitPlan(); });
        this.values.append(this.bank, this.list);
        // ONE picker for both ends: from and to are two ends of one sweep.
        this.unitPicker.root.classList.add("unit-picker");
        this.unitPicker.title = "Read this axis in another unit";
        this.unit.append(this.unitWord, this.unitPicker.root);
        this.mode.setChoices([["range", "Range"], ["values", "Values"]]);
        this.mode.root.title = "Scan a from/to/points range, or the values written";
        this.remove.title = "Remove this axis";
        this.custom.hidden = true;
        this.notes.append(this.custom);
    }
    get cells() {
        this.linkControls = [];
        const toggle = switchButton(on => { this.entry.link_enabled = on; if (on && !this.entry.links?.length)
            this.addLink();
        else
            this.owner.emitPlan(); }, "Link");
        toggle.classList.add("axis-link");
        toggle.disabled = this.linkLocked;
        setChecked(toggle, this.entry.link_enabled === true);
        const cells = [this.kind, this.axis, this.values, this.unit, this.mode.root, toggle, this.remove, this.notes];
        this.linkControls.push(toggle);
        if (this.entry.link_enabled)
            for (const [index, link] of (this.entry.links ?? []).entries()) {
                const marker = element("span", "linked-marker");
                const picker = new TreeCombo(value => { link.port = value; const port = this.owner.ports.find(p => p.port === value); if (port)
                    link.unit = firstSweep(port).unit; this.owner.emitPlan(); });
                picker.root.classList.add("linked-port");
                picker.setTree(this.owner.portTree(link.port), link.port);
                const holder = element("div", "linked-values");
                if (link.mode === "range") {
                    const bank = element("div", "range-bank");
                    const start = rangeSpin("linked from", () => 1, () => [null, null], value => { link.values[0] = value; this.owner.emitPlan(); });
                    const stop = rangeSpin("linked to", () => 1, () => [null, null], value => { link.values[link.values.length - 1] = value; this.owner.emitPlan(); });
                    start.input.value = String(link.values[0] ?? 0);
                    stop.input.value = String(link.values.at(-1) ?? 1);
                    start.refresh();
                    stop.refresh();
                    bank.append(element("span", "bank-word", "from"), start.root, element("span", "bank-word", "to"), stop.root);
                    holder.append(bank);
                    this.linkControls.push(start.input, stop.input);
                }
                else {
                    const input = element("input");
                    input.type = "text";
                    input.spellcheck = false;
                    input.value = link.mode === "formula" ? link.formula : link.value_text;
                    input.setAttribute("aria-label", link.mode === "formula" ? "Linked formula" : "Linked values");
                    input.placeholder = link.mode === "formula" ? "120 - x" : "1, 2, 3";
                    input.title = link.mode === "formula" ? "x is this axis's number; write numeric constants directly" : LIST_TIP;
                    input.addEventListener("change", () => { if (link.mode === "formula")
                        link.formula = input.value;
                    else
                        link.value_text = input.value; this.owner.emitPlan(); });
                    holder.append(input);
                    this.linkControls.push(input);
                }
                const unit = new TreeCombo(value => { link.unit = value; this.owner.emitPlan(); });
                unit.root.classList.add("linked-unit");
                const port = this.owner.ports.find(p => p.port === link.port);
                unit.setTree(unitTree((port?.units.length ? port.units : [[link.unit, 0]]).map(([symbol]) => ({ value: symbol, label: symbol || "1", group: port?.families[symbol] })), link.unit), link.unit);
                const mode = new Segmented(value => { link.mode = value; this.owner.emitPlan(); });
                mode.root.classList.add("linked-mode");
                mode.setChoices([["range", "Range"], ["values", "Values"], ["formula", "Formula"]]);
                mode.setCurrent(link.mode);
                const add = button("+", () => this.addLink(), "quiet glyph square linked-add");
                add.title = "Add linked parameter";
                add.setAttribute("aria-label", add.title);
                const remove = button("×", () => { this.entry.links.splice(index, 1); if (!this.entry.links.length)
                    this.entry.link_enabled = false; this.owner.emitPlan(); }, "destructive glyph square linked-remove");
                remove.title = "Remove linked parameter";
                remove.setAttribute("aria-label", remove.title);
                const nodes = [marker, picker.root, holder, unit.root, mode.root, add, remove];
                nodes.forEach(node => node.classList.add("linked-row-cell"));
                if (index === (this.entry.links?.length ?? 0) - 1)
                    marker.classList.add("last");
                this.linkControls.push(picker.root, unit.root, ...mode.root.querySelectorAll("button"), add, remove);
                cells.push(...nodes);
            }
        for (const control of this.linkControls)
            if ("disabled" in control)
                control.disabled = this.linkLocked;
        return cells;
    }
    addLink() {
        const port = this.owner.unusedPort();
        if (!port) {
            this.owner.emitPlan();
            return;
        }
        (this.entry.links ??= []).push({ port: port.port, values: [0, 1], unit: firstSweep(port).unit, mode: "formula", formula: "x", value_text: "" });
        this.owner.emitPlan();
    }
    get steps() { return [this.start.chosenStep, this.stop.chosenStep, this.points.chosenStep]; }
    set steps(values) {
        for (const [index, spin] of [this.start, this.stop, this.points].entries())
            spin.chosenStep = values[index];
    }
    port() { return this.owner.ports.find(port => port.port === this.entry.port); }
    /** The port's limits, read in the unit the axis is authored in. */
    bounds() {
        const port = this.port();
        return port ? inSpelling(port, this.entry.unit).bounds : [null, null];
    }
    /** A box holding a number the limits refuse says so (red, its tooltip the limits): a unit
     * picked keeps the digits written, which the new unit's limits may not admit. */
    flag(input, values, tip = "") {
        const [low, high] = this.bounds();
        const outside = values.some(value => (low !== null && value < low) || (high !== null && value > high));
        input.classList.toggle("invalid", outside);
        input.title = outside ? `Outside the limits: ${low ?? "-∞"} to ${high ?? "∞"} ${this.entry.unit}`.trim() : tip;
    }
    /** Bring the row to its authored axis; the box under the operator's cursor
     * keeps what they are typing. */
    show(entry) {
        if (this.entry.port !== entry.port)
            this.steps = this.owner.recall(entry.port)?.steps ?? [null, null, null];
        this.entry = entry;
        if (this.combo)
            this.combo.setTree(this.owner.portTree(entry.port), entry.port);
        if (this.name && document.activeElement !== this.name)
            this.name.value = entry.port.slice(this.owner.families.manual.length);
        const values = entry.values.length ? entry.values : [0];
        this.customValues = uniform(entry.values) ? null : [...entry.values];
        for (const [spin, value] of [[this.start, values[0]], [this.stop, values[values.length - 1]], [this.points, entry.values.length || 1]]) {
            if (document.activeElement !== spin.input) {
                spin.input.value = String(value);
                spin.input.classList.remove("invalid");
            }
            spin.refresh();
        }
        if (entry.values.length) {
            this.flag(this.start.input, [values[0]]);
            this.flag(this.stop.input, [values[values.length - 1]]);
        }
        if (document.activeElement !== this.list)
            this.list.value = entry.value_text;
        this.flag(this.list, entry.value_text.trim() ? entry.value_text.split(",").map(item => Number(item.trim())).filter(Number.isFinite) : [], LIST_TIP);
        // A unit with other spellings is a picker; one without is a word.
        const port = this.port();
        const offered = port?.units.some(([symbol]) => symbol === entry.unit) ? port.units : [];
        this.unitPicker.setTree(unitTree(offered.map(([symbol]) => ({ value: symbol, label: symbol, group: port.families[symbol] })), entry.unit), entry.unit);
        this.unitPicker.root.hidden = !offered.length;
        setText(this.unitWord, offered.length || entry.unit === "" || entry.unit === "1" ? "" : entry.unit);
        this.unitWord.hidden = !this.unitWord.textContent;
        this.showMode();
    }
    /** Another spelling for the axis: its values are the operator's and keep their
     * digits (the default sweep too), now read in it, against the port's limits there. */
    unitPicked(symbol) {
        if (symbol === this.entry.unit)
            return;
        this.entry = { ...this.entry, unit: symbol };
        this.show(this.entry);
        this.owner.emitPlan();
    }
    showMode() {
        const values = this.entry.mode === "values";
        this.mode.setCurrent(this.entry.mode);
        this.bank.hidden = values;
        this.list.hidden = !values;
        setText(this.custom, !values && this.customValues ? "custom values" : "");
        this.custom.hidden = !this.custom.textContent;
    }
    set locked(locked) {
        this.linkLocked = locked;
        for (const node of this.linkControls)
            if ("disabled" in node)
                node.disabled = locked;
        for (const node of [this.combo?.root, this.name, this.start.input, this.stop.input, this.points.input, this.list, this.unitPicker.root, this.remove])
            if (node)
                node.disabled = locked;
        this.mode.disabled = locked;
        for (const spin of [this.start, this.stop, this.points])
            spin.refresh();
    }
    pickMode(mode) {
        if (mode === this.entry.mode)
            return;
        this.entry = { ...this.entry, mode };
        this.showMode();
        this.owner.emitPlan();
    }
    /** A knob owns its banks, mode, unit and steps; row position owns none of them. */
    portChanged(port) {
        if (port === this.entry.port)
            return;
        this.owner.remember(this);
        const offered = this.owner.ports.find(item => item.port === port);
        const kept = this.owner.recall(port);
        this.customValues = null;
        this.entry = kept?.entry ?? { port, ...(offered ? firstSweep(offered) : { unit: "", values: [0] }), mode: "range", value_text: "" };
        this.steps = kept?.steps ?? [null, null, null];
        this.show(this.entry);
        this.owner.emitPlan();
    }
    /** An edited spin means the uniform grid the spins describe. */
    spinsEdited() {
        const start = Number(this.start.text), stop = Number(this.stop.text), count = Math.round(Number(this.points.text));
        if (![start, stop, count].every(Number.isFinite) || count < 1)
            return;
        this.customValues = null;
        this.entry = { ...this.entry, values: linspace(start, stop, count) };
        this.showMode();
        this.owner.emitPlan();
    }
}
export class ScanPlanEditor {
    edit;
    readOnly;
    root = element("div", "node-editor scan-plan");
    add = button("Add axis", () => this.addRow(false));
    addManual = button("Add manual axis", () => this.addRow(true));
    axes = element("div", "scan-axes");
    problem = element("p", "field-error");
    summary = element("p", "note wrap");
    apiTitle = sectionTitle("API values");
    apiForm = element("div", "form api-values");
    parked = element("div", "api-parked");
    apiNote = element("p", "note wrap");
    rows = [];
    apiFields = new Map();
    planText = null;
    parkedSignature = "";
    /** The spelling each API value is read in: chosen when it is first shown (defaultUnit), then the operator's (FluentUnitPicker). */
    shownUnits = new Map();
    pickers = "";
    tidied = null;
    facts = null;
    locked = "";
    context = null;
    lastRevision = null;
    axisMemory = new Map();
    families = { pulse: "pulse:param:", api: "api:", device: "device:", manual: "manual:" };
    ports = [];
    /** `readOnly`: Panel Edit's Direct producer (v2 `read_only_fields`) -- the
     * same axis rows and what the plan is, without Add axis, the API values or
     * anything written back. */
    constructor(edit, readOnly = false) {
        this.edit = edit;
        this.readOnly = readOnly;
        // An API value's name stands in the label column in the room the editor
        // keeps for it (labelFloor), never setting it: the column, and every field
        // after it, stays put when a pulse that binds some is chosen.
        this.apiForm.dataset.labelRoom = "";
        this.root.append(sectionTitle("Scan plan", this.add, this.addManual), this.axes, this.problem, this.summary, this.apiTitle, this.apiForm, this.parked, this.apiNote);
        if (readOnly)
            for (const node of [this.add, this.addManual, this.apiTitle, this.apiForm, this.parked, this.apiNote])
                node.hidden = true;
    }
    labelFloor() { return token("--api-label-room"); }
    /** Editor-local identity, not run identity: a Start or live read changes
     * neither the node's selected pulse nor the devices owning its scan ports. */
    contextKey(row) {
        const sorted = (items) => Object.entries(items ?? {}).sort(([a], [b]) => a.localeCompare(b));
        return [row.node_id, row.type_id, row.source_signal, row.values.pulse_template ?? null, sorted(row.device_keys), sorted(row.artifact_inputs),
            this.lastRevision !== null && row.draft_revision < this.lastRevision ? row.draft_revision : null];
    }
    /** Discard only editor memory; the next accepted draft supplies the rows. */
    reset() {
        this.axisMemory.clear();
        this.rows = [];
        this.axes.replaceChildren();
        this.planText = null;
        this.shownUnits.clear();
        this.apiFields.clear();
        this.apiForm.replaceChildren();
        this.pickers = "";
        this.tidied = null;
    }
    remember(view) {
        const port = this.ports.find(port => port.port === view.entry.port);
        const previous = this.axisMemory.get(view.entry.port);
        this.axisMemory.set(view.entry.port, { entry: copyAxis(view.entry), steps: [...view.steps],
            nativeUnit: port?.unit ?? previous?.nativeUnit ?? null, integer: port?.integer ?? previous?.integer ?? null });
    }
    recall(port) {
        const kept = this.axisMemory.get(port), offered = this.ports.find(item => item.port === port);
        if (!kept || (offered && (kept.nativeUnit !== offered.unit || kept.integer !== offered.integer)))
            return undefined;
        return { ...kept, entry: copyAxis(kept.entry), steps: [...kept.steps] };
    }
    update(row, raw, locked) {
        // A backward draft revision also identifies a replaced node loaded under
        // the same name. Ordinary edits and run-state pushes preserve the memory.
        const backwards = this.lastRevision !== null && row.draft_revision < this.lastRevision;
        const identity = JSON.stringify(this.contextKey({ ...row, draft_revision: this.lastRevision ?? row.draft_revision }));
        if (identity !== this.context || backwards) {
            this.reset();
            this.context = identity;
        }
        else
            for (const view of this.rows)
                this.remember(view);
        this.lastRevision = row.draft_revision ?? null;
        const facts = raw;
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
        for (const view of this.rows)
            this.remember(view);
        this.add.disabled = !facts.ports.length || Boolean(locked);
        this.addManual.disabled = Boolean(locked);
        for (const view of this.rows)
            view.locked = Boolean(locked);
        setText(this.problem, facts.problem);
        this.problem.hidden = !facts.problem;
        setText(this.summary, facts.problem ? "" : facts.summary);
        this.summary.hidden = !this.summary.textContent;
        if (!this.readOnly)
            this.updateApi(row, facts);
    }
    /** Every knob under the thing that owns it (v2 _fill_ports); an authored port the bench
     * no longer offers keeps saying what was authored, never silently another. Closed, the box
     * says the knob under its owner ("api · long_after.duration"). */
    portTree(current) {
        const entries = this.ports.map(port => ({ key: port.port, group: port.group, label: port.port.startsWith(this.families.device) ? port.leaf : port.label }));
        const lost = this.facts?.unoffered[current];
        if (lost)
            entries.push({ key: current, group: lost.group, label: `${lost.leaf} (${current in this.facts.withheld ? "its bracket is empty" : current.startsWith(this.families.device) ? "unavailable" : "not in this pulse"})` });
        return choiceTree(entries, current, "ports");
    }
    /** The host's axes (manual, device, API) stand outside the board's, in their order. */
    ordered(axes) {
        const host = (axis) => [axis.port, ...(axis.link_enabled ? (axis.links ?? []).map(link => link.port) : [])].some(port => this.facts.host_advanced.some(prefix => port.startsWith(prefix)));
        return [...axes.filter(host), ...axes.filter(axis => !host(axis))];
    }
    /** The rows brought to the plan, kept wherever they can be: a row of the
     * right kind at its position is re-pointed, not rebuilt. */
    reconcileRows(axes) {
        const kept = [];
        axes.forEach((axis, index) => {
            const manual = axis.port.startsWith(this.families.manual);
            let view = this.rows[index];
            if (!view || view.manual !== manual)
                view = new AxisRowView(this, manual, axis);
            view.steps = this.recall(axis.port)?.steps ?? [null, null, null];
            view.show(axis);
            kept.push(view);
        });
        this.rows = kept;
        this.axes.replaceChildren(...kept.flatMap(view => view.cells));
        this.axes.classList.toggle("has-manual", kept.some(view => view.manual));
        this.axes.classList.toggle("has-links", kept.some(view => view.entry.link_enabled));
    }
    unusedPort() {
        const used = new Set(this.rows.flatMap(view => [view.entry.port, ...(view.entry.links ?? []).map(link => link.port)]));
        return this.ports.find(port => !used.has(port.port));
    }
    addRow(manual) {
        if (manual) {
            const taken = new Set(this.rows.filter(view => view.manual).map(view => view.entry.port.slice(this.families.manual.length)));
            let ordinal = 1;
            while (taken.has(`manual ${ordinal}`))
                ordinal++;
            this.rows.push(new AxisRowView(this, true, { port: `${this.families.manual}manual ${ordinal}`, values: linspace(0, 1, 3), unit: "", mode: "range", value_text: "" }));
        }
        else {
            // The first knob the plan does not sweep yet (a port may stand on one axis only).
            const used = new Set(this.rows.map(view => view.entry.port));
            const port = this.unusedPort();
            if (!port)
                return;
            const kept = this.recall(port.port);
            const view = new AxisRowView(this, false, kept?.entry ?? { port: port.port, ...firstSweep(port), mode: "range", value_text: "" });
            view.steps = kept?.steps ?? [null, null, null];
            this.rows.push(view);
        }
        const added = this.rows[this.rows.length - 1];
        added.show(added.entry);
        this.emitPlan();
    }
    removeRow(view) {
        this.remember(view);
        this.rows = this.rows.filter(item => item !== view);
        this.emitPlan();
    }
    /** The plan the rows hold, host axes first, as the draft's `plan`. */
    emitPlan() {
        for (const view of this.rows)
            this.remember(view);
        const order = this.ordered(this.rows.map(view => view.entry));
        this.rows = order.map(entry => this.rows.find(view => view.entry === entry));
        this.axes.replaceChildren(...this.rows.flatMap(view => view.cells));
        this.axes.classList.toggle("has-manual", this.rows.some(view => view.manual));
        this.axes.classList.toggle("has-links", this.rows.some(view => view.entry.link_enabled));
        this.planText = JSON.stringify({ axes: order.map(axis => ({ port: axis.port, unit: axis.unit, values: axis.values, mode: axis.mode, value_text: axis.value_text, ...(axis.links || axis.link_enabled ? { link_enabled: axis.link_enabled === true, links: axis.links ?? [] } : {}) })) });
        this.edit({ plan: this.planText }, "Scan plan");
    }
    // ---------------------------------------------------------- API values
    updateApi(row, facts) {
        const api = facts.api;
        const fields = api?.fields ?? [];
        const shown = Boolean(fields.length || api?.parked.length);
        this.apiTitle.hidden = this.apiForm.hidden = !shown;
        const alive = new Set(fields.map(field => field.name));
        for (const [name, field] of this.apiFields)
            if (!alive.has(name)) {
                field.root.remove();
                this.apiFields.delete(name);
            }
        let cursor = this.apiForm.firstElementChild;
        for (const item of fields) {
            // First shown, a value is read in the spelling `defaultUnit` picks from the value this run gives it.
            if (!this.shownUnits.has(item.name)) {
                const at = (value, decade) => value === null ? null : shifted(value, -decade);
                this.shownUnits.set(item.name, defaultUnit(item.unit, item.units.map(([symbol, decade]) => ({ symbol, value: at(item.value, decade),
                    low: at(item.minimum, decade), high: at(item.maximum, decade) })), item.kind === "integer"));
            }
            // The box holds the pulse's number read in the picked spelling: a shift of its digits.
            const symbol = this.shownSymbol(item), places = this.apiPlaces(item);
            const shown = (value) => value === null ? null : shifted(value, places);
            const spec = { name: item.name, label: item.label, kind: item.kind, value: shown(item.value), required: false, allow_none: false,
                unit: symbol, units: item.units.map(([value]) => ({ value, label: value, group: item.families[value] })),
                minimum: shown(item.minimum), maximum: shown(item.maximum), description: item.description,
                unavailable_reason: item.unavailable_reason || this.locked, inexact: true };
            let field = this.apiFields.get(item.name);
            if (!field) {
                field = new Field(`${row.node_id}-api`, spec, {
                    commit: value => this.emitValues({ [item.name]: this.inPulseUnit(item.name, Number(value)) }),
                    unit: picked => {
                        // What the box holds now (an edit still on its way included): the pulse's own number is
                        // re-read in the picked spelling; an override keeps its digits, read in it, so the value
                        // it sets in the pulse's unit is the one the box now says.
                        const box = this.apiFields.get(item.name), written = Number(box.value), held = this.inPulseUnit(item.name, written);
                        const own = held === this.facts?.api?.fields.find(field => field.name === item.name)?.authored;
                        this.shownUnits.set(item.name, picked);
                        if (this.facts)
                            this.updateApi(row, this.facts);
                        const shownNow = this.facts?.api?.fields.find(field => field.name === item.name);
                        if (!shownNow)
                            return;
                        box.respell(own ? shifted(held, this.apiPlaces(shownNow)) : written);
                        if (!own)
                            this.emitValues({ [item.name]: this.inPulseUnit(item.name, written) });
                    },
                }, false);
                this.apiFields.set(item.name, field);
            }
            else
                field.update(spec);
            if (field.root !== cursor)
                this.apiForm.insertBefore(field.root, cursor);
            else
                cursor = cursor.nextElementSibling;
        }
        // The values the draft keeps but this run does not apply, each with its ×.
        const parked = JSON.stringify([api?.parked ?? [], this.locked]);
        if (parked !== this.parkedSignature)
            this.parked.replaceChildren(...(api?.parked ?? []).map(line => {
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
        if (pickers !== this.pickers) {
            this.pickers = pickers;
            requestAnimationFrame(() => alignUnitPickers(this.apiForm));
        }
        // What the model re-read against this pulse goes back to the draft once.
        if (api?.tidy != null && !this.locked && api.tidy !== this.tidied) {
            this.tidied = api.tidy;
            this.edit({ api_values: api.tidy }, "API values");
        }
    }
    /** The spelling an API value is read in: the picked one while its unit offers it, else its own. */
    shownSymbol(item) {
        const symbol = this.shownUnits.get(item.name);
        return symbol !== undefined && item.units.some(([name]) => name === symbol) ? symbol : item.unit;
    }
    /** Decimal places from the pulse's unit to the spelling a box is read in. */
    apiPlaces(item) {
        return -(item.units.find(([name]) => name === this.shownSymbol(item))?.[1] ?? 0);
    }
    /** A number read in a box, back in the pulse's unit. */
    inPulseUnit(name, value) {
        const item = this.facts?.api?.fields.find(field => field.name === name);
        return item ? shifted(value, -this.apiPlaces(item)) : value;
    }
    /** Only what this run sets differently, and every value kept: a box left at
     * the pulse's own number is no override (it would outrank the next
     * recalibration); a value no free box holds is carried over as written. */
    emitValues(changed) {
        const api = this.facts?.api;
        if (!api)
            return 0;
        const free = new Map(api.fields.filter(field => !field.unavailable_reason).map(field => [field.name, field]));
        const boxed = (name) => {
            const field = this.apiFields.get(name);
            return changed[name] ?? (field ? this.inPulseUnit(name, Number(field.value)) : free.get(name).value);
        };
        const written = new Map();
        for (const [name, value] of api.overrides) {
            if (!free.has(name))
                written.set(name, value);
            else if (boxed(name) !== free.get(name).authored)
                written.set(name, boxed(name));
        }
        for (const [name, field] of free)
            if (!written.has(name) && boxed(name) !== field.authored)
                written.set(name, boxed(name));
        return this.writeValues(written);
    }
    removeKept(name) {
        const written = new Map(this.facts?.api?.overrides ?? []);
        if (written.delete(name))
            this.writeValues(written);
    }
    writeValues(written) {
        return this.edit({ api_values: [...written].map(([name, value]) => `${name} = ${value}`).join("\n") }, "API values");
    }
}
