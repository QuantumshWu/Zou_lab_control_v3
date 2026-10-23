// Derive's variables (v2 DeriveEditor): the source's inputs, then one card
// per variable that reads as a vertical formula -- name, "=" and the first
// operand on the head line (Publish and × at its end), one line per later
// term with its operator strip right before its operand, then its
// statistic.  Every line of every card is a line of one column set whose
// operand column is the page's control column.  The model (derive/
// editor.py) types every line against the source's schemas: what each
// operand may be, which axes and labels a pick offers, which statistics
// fit, each line's problem and what the variable comes out as.  The cards
// write back `variables`, the rows they were handed, edited.
import { CheckCombo, ComboBox, Segmented, button, element, form, row as formRow, sectionTitle, setChecked, setText, switchButton, token, } from "../../fluent/index.js";
/** A label set's values, each `[key, text]`: as listed, or its counted rows
 * spelled as the model names them (`spatial-y 0 · spatial-x 1`). */
function labelValues(set) {
    if (set.values)
        return set.values;
    const values = [];
    const { axes, shown } = set.rows ?? { axes: [], shown: 0 };
    for (let row = 0; row < shown; row++) {
        const parts = [];
        let rest = row;
        for (let index = axes.length - 1; index >= 0; index--) {
            const axis = axes[index];
            const label = String(axis.start + (rest % axis.count) * axis.step);
            rest = Math.floor(rest / axis.count);
            parts.unshift(axis.name === null ? label : `${axis.name} ${label}`);
        }
        const key = parts.join(" · ");
        values.push([key, key]);
    }
    return values;
}
/** A variable's name: what a reader's operand says and its signal ends in. */
const NAME = /^[A-Za-z_][A-Za-z0-9_]*$/;
const NUMBER = "#number";
const DOMAIN_INK = { Repeat: "var(--axis-group-0)", Point: "var(--axis-group-1)", Cell: "var(--axis-group-2)" };
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
/** A list row for a grouped combo (a group only when the model gives one). */
const choiceOf = ({ value, label, group }) => group ? { value, label, group } : { value, label };
const copy = (value) => JSON.parse(JSON.stringify(value));
/** One line of a card: its cells on the card's column set, its problem under it. */
class LineView {
    card;
    index;
    kind;
    cells;
    lead = element("div", "derive-lead");
    name = null;
    strip = null;
    operand = new ComboBox(value => this.card.operandChosen(this.index, String(value ?? "")), "choose");
    at = element("div", "derive-at");
    axis = new ComboBox(value => this.card.axisChosen(this.index, String(value ?? "")));
    number = element("input", "derive-number");
    labels = new CheckCombo();
    unit = element("span", "note derive-unit");
    publish = null;
    remove;
    error = element("p", "field-error derive-error");
    /** An axis chosen whose labels are being picked: written with them. */
    pending = null;
    /** What the labels list was last built from (its list's id, the axis, the ticks). */
    labelsKey = "";
    constructor(card, index, kind) {
        this.card = card;
        this.index = index;
        this.kind = kind;
        if (kind === null) {
            const name = element("input", "derive-name");
            name.type = "text";
            name.spellcheck = false;
            name.placeholder = "name";
            // A name is written when it is finished: the variables that read it
            // follow it then, and a published one's signal is renamed once.
            name.addEventListener("change", () => card.rename(name.value.trim()));
            this.name = name;
            this.lead.append(name, element("span", "derive-eq", "="));
            this.publish = switchButton(on => card.edit(row => { row.publish = on; }), "Publish");
            this.publish.classList.add("derive-publish");
        }
        else {
            // The strip stands right before its operand, ending where the name
            // field ends: the card reads down as one formula.
            this.strip = new Segmented(value => card.editTerm(this.index, term => { term.op = String(value); }));
            this.lead.append(this.strip.root, element("span", "derive-eq hold", "="));
        }
        this.operand.root.classList.add("derive-operand");
        this.number.type = "text";
        this.number.spellcheck = false;
        this.number.placeholder = "number";
        this.number.inputMode = "decimal";
        this.number.addEventListener("change", () => {
            const value = Number(this.number.value.trim());
            if (this.number.value.trim() !== "" && Number.isFinite(value))
                card.editTerm(this.index, term => { term.number = value; });
        });
        this.at.append(element("span", "derive-word", "at"), this.axis.root, this.number);
        this.labels.root.classList.add("derive-labels");
        this.labels.edited = values => card.labelsPicked(this.index, values);
        this.labels.closed = changed => { if (!changed && this.pending !== null) {
            this.pending = null;
            card.redraw();
        } };
        this.remove = button("×", () => kind === null ? card.removeCard() : card.edit(row => { row.terms.splice(this.index, 1); }), "destructive glyph square derive-remove");
        this.remove.title = kind === null ? "Remove this variable" : "Remove this line";
        const labelsCell = element("div", "derive-labels-cell");
        labelsCell.append(this.labels.root, this.unit);
        this.cells = [this.lead, this.operand.root, this.at, labelsCell, element("span", "derive-tail"), this.publish ?? element("span", "derive-publish"), this.remove, this.error];
    }
    show(line, row, locked) {
        if (this.name && document.activeElement !== this.name)
            this.name.value = row.name;
        if (this.publish) {
            setChecked(this.publish, row.publish);
            this.publish.title = row.publish ? `Published as ${this.card.signal(row.name)}` : "Internal: other variables read it; nothing publishes it";
        }
        if (this.strip) {
            this.strip.setChoices(line.ops);
            this.strip.setCurrent(line.op);
            this.strip.disabled = locked;
        }
        this.operand.setChoices(line.operands.map(choiceOf));
        this.operand.value = line.operand;
        this.operand.title = line.operand ?? "";
        this.operand.disabled = locked;
        const at = this.pending ?? line.at;
        this.at.hidden = !line.picks && !line.number;
        this.axis.root.hidden = line.number;
        this.at.firstElementChild.toggleAttribute("hidden", line.number);
        this.number.hidden = !line.number;
        if (line.number && document.activeElement !== this.number)
            this.number.value = line.number_text;
        if (line.picks) {
            this.axis.setChoices(line.axes.map(choiceOf));
            this.axis.value = at;
        }
        this.axis.disabled = this.number.disabled = locked;
        const choice = line.axes.find(item => item.value === at);
        const offered = choice?.labels ? this.card.labelSet(choice.labels) : undefined;
        this.labels.root.hidden = !(line.picks && at && !line.number);
        const shown = this.pending !== null ? [] : line.labels;
        const key = JSON.stringify([offered ?? null, at, choice?.group ?? null, shown]);
        if (!this.labels.root.hidden && key !== this.labelsKey) {
            this.labelsKey = key;
            // The list is spelled out when it is opened, not on every revision.
            this.labels.set({ key, items: () => {
                    const items = [];
                    for (const [value, label] of offered?.ends ?? [["", "All"], ["first", "first"], ["last", "last"]])
                        items.push({ value, label });
                    const colour = DOMAIN_INK[choice?.group ?? ""];
                    items.push(colour ? { value: "", label: at, header: true, colour } : { value: "", label: at, header: true });
                    const listed = offered ? labelValues(offered) : [];
                    for (const [value, label] of listed)
                        items.push({ value, label });
                    if (offered?.more)
                        items.push({ value: "", label: offered.more, header: true });
                    const known = new Set(listed.map(([value]) => value));
                    for (const label of shown)
                        if (!known.has(label))
                            items.push({ value: label, label });
                    return items;
                } }, shown);
        }
        this.labels.disabled = locked;
        setText(this.unit, line.number ? line.number_unit : "");
        this.unit.hidden = !line.number;
        if (this.publish)
            this.publish.disabled = locked;
        this.remove.disabled = locked;
        if (this.name)
            this.name.disabled = locked;
        setText(this.error, line.error);
        this.error.hidden = !line.error;
        // A parked helper's problem in the idle ink of a note: it blocks nothing.
        this.error.classList.toggle("muted", line.muted);
    }
}
/** One variable's card. */
class CardView {
    editor;
    index;
    root = element("section", "card bordered derive-card");
    grid = element("div", "derive-lines");
    lines = [];
    statLead = element("div", "derive-lead derive-stat-lead");
    statStrip = new Segmented(value => this.statisticChosen(String(value)));
    overWord = element("span", "derive-word", "over");
    over = new CheckCombo();
    statRemove = button("×", () => this.closeStatistic(), "destructive glyph square derive-remove");
    statError = element("p", "field-error derive-error");
    statCells;
    adds = element("div", "derive-adds");
    addButtons = new Map();
    preview = element("p", "note wrap derive-preview");
    signature = "";
    /** "+ Statistic" opened a statistic line not filled yet. */
    statisticOpen = false;
    card = null;
    locked = false;
    constructor(editor, index) {
        this.editor = editor;
        this.index = index;
        this.statLead.append(element("span", "derive-word", "statistic"), element("span", "derive-eq hold", "="));
        const reduce = element("div", "derive-reduce");
        reduce.append(this.statStrip.root, this.overWord, this.over.root);
        this.over.root.classList.add("derive-over");
        this.over.edited = values => this.edit(row => { row.over = values; });
        this.statRemove.title = "Remove this statistic";
        this.statCells = [this.statLead, reduce, this.statRemove, this.statError];
        this.root.append(this.grid);
    }
    get row() { return this.card.row; }
    signal(name) { return this.editor.signal(name); }
    labelSet(id) { return this.editor.labelSets[id]; }
    show(card, locked, add) {
        this.card = card;
        this.locked = locked;
        const row = card.row;
        const statShown = this.statisticOpen || Boolean(row.statistic) || row.over.length > 0;
        if (row.statistic || row.over.length)
            this.statisticOpen = false;
        const signature = JSON.stringify([card.lines.map(line => line.kind), statShown, add]);
        if (signature !== this.signature) {
            this.signature = signature;
            this.lines = card.lines.map((line, index) => {
                const kept = this.lines[index];
                return kept && kept.kind === line.kind ? kept : new LineView(this, index, line.kind);
            });
            this.lines.forEach((line, index) => { line.index = index; });
            this.addButtons.clear();
            this.adds.replaceChildren(...add.filter(([kind]) => kind !== "statistic" || !statShown).map(([kind, text]) => {
                const command = button(text, () => this.addLine(kind));
                this.addButtons.set(kind, command);
                return command;
            }));
            // What the variable comes out as stands under its lines, the add commands last.
            this.grid.replaceChildren(...this.lines.flatMap(line => line.cells), ...(statShown ? this.statCells : []), this.preview, this.adds);
        }
        card.lines.forEach((line, index) => this.lines[index].show(line, row, locked));
        // The statistic: the strip greys what the value cannot take.
        const statistic = card.statistic;
        this.statStrip.setChoices(statistic.choices.map(([name]) => [name, name]));
        this.statStrip.setCurrent(statistic.current);
        for (const [name, fits] of statistic.choices)
            this.statStrip.setWithheld(name, !fits || locked);
        const overItems = [];
        let group;
        for (const choice of statistic.over_choices) {
            if (choice.group !== group) {
                group = choice.group;
                if (group)
                    overItems.push({ value: "", label: group, header: true, selects: true, colour: DOMAIN_INK[group] });
            }
            overItems.push({ value: choice.value, label: choice.label });
        }
        this.over.set(overItems, statistic.over);
        this.overWord.hidden = this.over.root.hidden = !overItems.length;
        this.over.disabled = locked;
        this.statRemove.disabled = locked;
        setText(this.statError, statistic.error);
        this.statError.hidden = !statistic.error;
        this.statError.classList.toggle("muted", statistic.muted);
        for (const command of this.addButtons.values())
            command.disabled = locked;
        setText(this.preview, card.preview);
        this.preview.hidden = !card.preview;
    }
    /** The card's row edited by `change`, then the draft written. */
    edit(change) {
        const row = copy(this.row);
        change(row);
        this.editor.commit(this.index, row);
    }
    /** The term a line edits; the head line of a variable with no term yet writes its first. */
    editTerm(index, change) {
        this.edit(row => {
            if (!row.terms.length)
                row.terms.push(copy(this.editor.defaults.term));
            change(row.terms[index]);
        });
    }
    rename(name) { this.editor.rename(this.index, name); }
    removeCard() { this.editor.remove(this.index); }
    redraw() { if (this.card)
        this.show(this.card, this.locked, this.editor.addCommands); }
    operandChosen(index, key) {
        const line = this.card.lines[index];
        if (key === NUMBER) {
            this.editTerm(index, term => { if (term.number === null)
                Object.assign(term, { of: "", number: 0, boolean: "", at: "", labels: [] }); });
            requestAnimationFrame(() => { const box = this.lines[index]?.number; box?.focus(); box?.select(); });
            return;
        }
        if (key === "#true" || key === "#false") {
            this.editTerm(index, term => Object.assign(term, { of: "", number: null, boolean: key.slice(1), at: "", labels: [] }));
            return;
        }
        // The pick stays when the new operand has its axis, or while nothing says which it has.
        const keeps = line.operands.find(choice => choice.value === key)?.keeps_pick ?? false;
        this.editTerm(index, term => { Object.assign(term, { of: key, number: null, boolean: "" }); if (!keeps)
            Object.assign(term, { at: "", labels: [] }); });
    }
    /** All clears the pick; an axis (or a domain) opens its labels to pick, and
     * the pick is written once they are picked, so nothing says anything in between. */
    axisChosen(index, axis) {
        const view = this.lines[index];
        if (!axis) {
            view.pending = null;
            this.editTerm(index, term => Object.assign(term, { at: "", labels: [] }));
            return;
        }
        if (axis !== this.card.lines[index].at) {
            view.pending = axis;
            this.redraw();
        }
        requestAnimationFrame(() => view.labels.open());
    }
    labelsPicked(index, values) {
        const view = this.lines[index];
        const at = view.pending;
        view.pending = null;
        if (values.includes(""))
            this.editTerm(index, term => Object.assign(term, { at: "", labels: [] }));
        else if (at !== null)
            this.editTerm(index, term => Object.assign(term, { at, labels: values }));
        else
            this.editTerm(index, term => { term.labels = values; });
    }
    /** The statistic; with nothing to reduce over yet, every Cell axis of the
     * value -- or, with none, the list of its axes to pick. */
    statisticChosen(statistic) {
        const defaults = this.card.statistic.default_over;
        const open = !this.row.over.length && !defaults.length && this.card.statistic.over_choices.length > 0;
        this.edit(row => { row.statistic = statistic; if (!row.over.length)
            row.over = [...defaults]; });
        if (open)
            requestAnimationFrame(() => this.over.open());
    }
    closeStatistic() {
        this.statisticOpen = false;
        if (this.row.statistic || this.row.over.length)
            this.edit(row => { row.statistic = ""; row.over = []; });
        else
            this.redraw();
    }
    /** A line of `kind` after the last (the model says how most formulas want
     * it); a pick it starts asks for its labels at once. */
    addLine(kind) {
        if (kind === "statistic") {
            this.statisticOpen = true;
            this.redraw();
            return;
        }
        const added = this.card.add_terms[kind];
        if (!added)
            return;
        const at = this.row.terms.length || 1;
        this.edit(row => { if (!row.terms.length)
            row.terms.push(copy(this.editor.defaults.term)); row.terms.push({ ...copy(this.editor.defaults.term), ...copy(added.term) }); });
        if (added.pending_at)
            this.editor.pendAfterRedraw(this.index, at, added.pending_at);
    }
}
export class DeriveVariablesEditor {
    edit;
    root = element("div", "node-editor derive-variables");
    inputsForm = form("derive-inputs-form");
    inputs = element("div", "derive-inputs");
    add = button("Add variable", () => this.addVariable());
    problem = element("p", "field-error");
    empty = element("p", "note wrap");
    cardsHost = element("div", "derive-cards");
    footer = element("p", "note wrap");
    cards = [];
    facts = null;
    node = "";
    inputsSignature = "";
    pend = null;
    defaults = { variable: {}, term: { op: "", of: "", number: null, boolean: "", at: "", labels: [] } };
    addCommands = [];
    labelSets = {};
    /** A card line's lead with its "=", and the strips it was measured from. */
    lead = 0;
    leadKey = "";
    constructor(edit) {
        this.edit = edit;
        this.inputsForm.append(formRow("Inputs", this.inputs));
        this.root.append(this.inputsForm, sectionTitle("Variables", this.add), this.problem, this.empty, this.cardsHost, this.footer);
    }
    /** The label column leaves a card line's lead -- a name, or the widest
     * operator strip of every kind, shown or not -- its room, so the cards'
     * operands stand on the page's control column whatever the cards hold
     * (v2 DeriveEditor.label_width_floor). */
    labelFloor() {
        const strips = this.facts?.strips ?? [];
        const key = JSON.stringify(strips);
        if (key !== this.leadKey) {
            const probe = element("div", "derive-lead derive-probe");
            const widths = strips.map(choices => { const strip = new Segmented(() => { }); strip.setChoices(choices); probe.append(strip.root); return strip.root; });
            const eq = element("span", "derive-eq", "=");
            probe.append(eq);
            this.root.append(probe);
            this.lead = Math.max(token("--derive-name-w"), ...widths.map(node => node.scrollWidth)) + eq.offsetWidth;
            probe.remove();
            this.leadKey = key;
        }
        return token("--card-pad") + this.lead + token("--row-gap");
    }
    update(row, raw, locked) {
        const facts = raw;
        this.facts = facts;
        this.node = row.name;
        this.defaults = facts.defaults;
        this.addCommands = facts.add;
        this.labelSets = facts.label_sets;
        const inputs = JSON.stringify(facts.inputs);
        if (inputs !== this.inputsSignature)
            this.inputs.replaceChildren(...facts.inputs.flatMap(input => {
                const name = element("span", "derive-input-name", input.name);
                const kind = element("span", "note", input.kind);
                const axes = element("span", "derive-input-axes");
                input.axes.forEach((axis, index) => {
                    if (index)
                        axes.append(" × ");
                    const word = element("span", "", axis.text);
                    word.style.color = DOMAIN_INK[axis.domain] ?? "";
                    axes.append(word);
                });
                if (!input.axes.length && !input.kind.startsWith("waiting"))
                    axes.append("one value");
                for (const node of [name, kind, axes])
                    node.title = input.tip;
                return [name, kind, axes];
            }));
        this.inputsSignature = inputs;
        this.add.disabled = Boolean(locked);
        setText(this.problem, facts.problem);
        this.problem.hidden = !facts.problem;
        setText(this.empty, facts.empty);
        this.empty.hidden = !facts.empty;
        setText(this.footer, facts.footer);
        while (this.cards.length > facts.cards.length)
            this.cards.pop().root.remove();
        while (this.cards.length < facts.cards.length) {
            const card = new CardView(this, this.cards.length);
            this.cards.push(card);
            this.cardsHost.append(card.root);
        }
        facts.cards.forEach((card, index) => this.cards[index].show(card, Boolean(locked), facts.add));
        const pend = this.pend;
        if (pend && facts.cards[pend.card]?.lines.length && facts.cards[pend.card].lines.length > pend.line) {
            this.pend = null;
            const card = this.cards[pend.card];
            card.axisChosen(pend.line, pend.at);
        }
    }
    signal(variable) { return `${this.node}/${variable}`; }
    /** After the next model, the line `line` of card `card` asks for the labels of `at`. */
    pendAfterRedraw(card, line, at) { this.pend = { card, line, at }; }
    /** Every card's row, one edited, written as v2 compact_variable wrote them. */
    commit(index, edited) {
        const rows = (this.facts?.cards ?? []).map(card => card.row).map((row, position) => position === index ? edited : row)
            .filter((row) => row !== null);
        this.write(rows);
    }
    write(rows) {
        const variable = this.defaults.variable, term = this.defaults.term;
        const compact = rows.map(row => {
            const out = {};
            for (const [key, value] of Object.entries(row)) {
                if (key === "terms")
                    out.terms = row.terms.map(item => Object.fromEntries(Object.entries(item).filter(([column, entry]) => !same(entry, term[column]))));
                else if (key === "name" || !same(value, variable[key]))
                    out[key] = value;
            }
            return out;
        });
        this.edit({ variables: compact }, "Variables");
    }
    /** A card's variable renamed.  What read it by its old name reads it by the
     * new one when that is free -- a name no other variable or input has -- and
     * the card was the variable the old name meant (the first so named). */
    rename(index, name) {
        const rows = copy((this.facts?.cards ?? []).map(card => card.row));
        const old = rows[index].name;
        if (name === old)
            return;
        const others = new Set(rows.filter((_, position) => position !== index).map(row => row.name));
        const first = rows.findIndex(row => row.name === old) === index;
        const follows = Boolean(old) && first && NAME.test(name) && !others.has(name) && !(this.facts?.members ?? []).includes(name);
        rows[index].name = name;
        if (follows)
            for (const row of rows)
                for (const item of row.terms)
                    if (item.of === old)
                        item.of = name;
        this.write(rows);
    }
    remove(index) { this.commit(index, null); }
    /** A published variable reading the first input, its name selected to type over. */
    addVariable() {
        this.write([...(this.facts?.cards ?? []).map(card => card.row), this.facts.new_variable]);
        requestAnimationFrame(() => requestAnimationFrame(() => {
            const name = this.cardsHost.lastElementChild?.querySelector(".derive-name");
            name?.scrollIntoView({ block: "nearest" });
            name?.focus();
            name?.select();
        }));
    }
}
