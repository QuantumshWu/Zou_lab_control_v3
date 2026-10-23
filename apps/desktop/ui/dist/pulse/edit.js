// The Edit page (v2 PulseScheduleView): one strip read left to right -- the
// Port catalog and Delay / Scan columns, then the pulse's items in timeline
// order (period cards, bracket posts, Subpulse cards) -- every row of every
// column on one line per port; under it the Control, Connection and Ports
// cards.  A click picks an item (Shift/Ctrl adds), a click between cards
// picks that gap; dragging a card moves it.  Every change is a command.
import { ComboBox, Readout, Segmented, SpinBox, assign, checkBox, command, element, setChecked, setElided, setMiddle, setState, setText, setTitle, statusDot, switchButton, token } from "../fluent/index.js";
import { RepeatsBox, TextBox, ValueBox, confirmPulseDiscard, drawBadges, editCommand } from "./widgets.js";
const itemKey = (item) => `${item[0]}|${item[1]}`;
const parseKey = (key) => { const at = key.indexOf("|"); return [key.slice(0, at), key.slice(at + 1)]; };
const postBracket = (key) => key.slice(0, key.lastIndexOf(":"));
/** More items than this: the strip draws only the cards near its view (style.css `.long`). */
const LONG_STRIP = 12;
/** A row label (ElidedLabel): capitals centred on the row like a field's
 * text, cut with an ellipsis where its column ends. */
function rowLabel(text = "") {
    const root = element("span", "pulse-label");
    const words = element("span", "elided", text);
    root.append(words);
    return { root, text: words };
}
/** A labelled line of a column: the row every column draws at one height. */
function line(...nodes) {
    const row = element("div", "pulse-row");
    for (const node of nodes)
        row.append(typeof node === "string" ? rowLabel(node).root : node);
    return row;
}
/** One period's card (PeriodCard). Its controls are made when it first nears
 * the timeline's view (Strip.watch): a 500-period pulse opens with the cards
 * on screen made, the rest made a slice at a time after it is drawn. */
class PeriodCard {
    id;
    send;
    owner;
    root = element("section", "card bordered pulse-card pulse-period");
    body = null;
    digital = new Map();
    analog = new Map();
    period = null;
    drawn = null;
    constructor(id, send, owner) {
        this.id = id;
        this.send = send;
        this.owner = owner;
        this.root.dataset.item = itemKey(["period", id]);
    }
    get made() { return this.body !== null; }
    /** Make the card's controls and draw them (once). */
    make() {
        if (this.body)
            return;
        const body = {
            title: element("div", "heading pulse-title"), durationLabel: element("span", "pulse-label", "Duration"),
            duration: new ValueBox(this.send, text => this.edit("duration", { value: text, unit: this.period?.unit }), this.owner),
            unit: new ComboBox(value => this.edit("duration", { value: this.period?.duration.text, unit: value })),
            name: new TextBox(text => this.edit("period_name", { name: text })), rows: element("div", "pulse-ports"),
        };
        this.root.append(line(body.title), line(body.durationLabel), line(body.duration.root), line(body.unit.root), line(body.name.root), body.rows);
        this.body = body;
        if (this.drawn)
            this.draw(...this.drawn);
    }
    edit(edit, args) {
        this.send(editCommand(this.owner, edit, { period_id: this.id, ...args }), "Editing the pulse…");
    }
    update(period, index, total, ports, modes) {
        // What it was drawn from, unchanged (a state keeps the identity of what did
        // not change, transport.ts): nothing to redraw. A 500-card strip redraws one.
        const drawn = this.drawn;
        if (drawn && drawn[0] === period && drawn[1] === index && drawn[2] === total && drawn[3] === ports && drawn[4] === modes)
            return;
        this.drawn = [period, index, total, ports, modes];
        this.period = period;
        // Its width is its kind's, made or not.
        this.root.classList.toggle("spacer", period.kind === "spacer");
        this.root.classList.toggle("readonly", period.readonly);
        if (this.body)
            this.draw(period, index, total, ports, modes);
    }
    draw(period, index, total, ports, modes) {
        const { title, durationLabel, duration, unit, name, rows } = this.body;
        const spacer = period.kind === "spacer";
        setText(title, spacer ? "Spacer" : `Period ${index}/${total}`);
        // A spacer leaves the caption's line empty, so its box stays on the cards' line.
        setText(durationLabel, spacer ? "" : "Duration");
        setTitle(title, `${period.name} (${period.period_id})`);
        duration.tip = spacer ? "Edit value source" : "Edit Scan and value source";
        duration.show(period.duration, !period.readonly);
        unit.setChoices(period.unit_choices.map(choice => [choice, choice]));
        unit.value = period.unit;
        unit.disabled = period.readonly || period.duration.config_active;
        name.show(period.name, !period.readonly && !spacer);
        const digital = new Map(period.digital);
        const analog = new Map(period.analog.map(([key, mode, field]) => [key, { mode, field }]));
        const nodes = [];
        for (const port of ports.filter(port => port.visible)) {
            if (port.kind === "digital" && digital.has(port.key)) {
                let box = this.digital.get(port.key);
                if (!box) {
                    box = checkBox(port.label, high => this.edit("digital", { port_key: port.key, high }));
                    this.digital.set(port.key, box);
                }
                setChecked(box, Boolean(digital.get(port.key)));
                const words = box.querySelector(".words");
                // A spacer's circles carry no labels: its neighbours' rows say which is which (v2: on hover).
                if (words)
                    setText(words, spacer ? "" : port.label);
                setTitle(box, spacer ? port.label : "");
                assign(box, "disabled", period.readonly);
                nodes.push(this.row(box, port.key));
            }
            else if (port.kind === "dac" && analog.has(port.key)) {
                const shown = analog.get(port.key);
                let cell = this.analog.get(port.key);
                if (!cell) {
                    const mode = new ComboBox(value => this.edit("analog", { port_key: port.key, mode: value, value: this.analog.get(port.key)?.value.box.root.value ?? "" }));
                    const value = new ValueBox(this.send, text => this.edit("analog", { port_key: port.key, mode: this.analog.get(port.key)?.mode.value ?? "edge", value: text }), this.owner);
                    const row = element("div", "pulse-row pulse-dac");
                    row.append(mode.root, value.root);
                    cell = { row, mode, value };
                    this.analog.set(port.key, cell);
                }
                // A spacer holds every DAC: its one mode, switched off, the row's width; no value to type.
                cell.mode.setChoices(spacer ? modes.filter(choice => choice.value === shown.mode).map(choice => [choice.value, choice.label])
                    : modes.map(choice => [choice.value, choice.label]));
                cell.mode.value = shown.mode;
                cell.mode.disabled = period.readonly || spacer || shown.field.config_active;
                cell.mode.title = spacer ? `${port.label}: a spacer holds the DAC` : `${port.label}: ${shown.mode}`;
                cell.row.classList.toggle("held", spacer);
                assign(cell.value.root, "hidden", spacer);
                // The port's own code range, said where it is typed (v2 PeriodCard's DAC edit).
                cell.value.tip = `${port.label}: signed integer ${port.lo}..${port.hi} (0 = 0 V)\nEdit Scan and value source`;
                if (!spacer)
                    cell.value.show(shown.field, !period.readonly);
                nodes.push(cell.row);
            }
        }
        if (nodes.length !== rows.children.length || nodes.some((node, at) => rows.children[at] !== node))
            rows.replaceChildren(...nodes);
    }
    row(control, key) {
        let holder = control.parentElement;
        if (!holder || !holder.classList.contains("pulse-row")) {
            holder = element("div", "pulse-row");
            holder.append(control);
            holder.dataset.port = key;
        }
        return holder;
    }
}
/** _reading_ink: a bracket's stroke mixed toward TEXT, a twentieth at a
 * time, until it reads 4.5:1 on ACCENT_TINT (a selected post's face). */
function readingInk(stroke) {
    const rgb = (hex) => [1, 3, 5].map(at => parseInt(hex.slice(at, at + 2), 16));
    const luminance = (channels) => {
        const [r, g, b] = channels.map(value => { const v = value / 255; return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; });
        return 0.2126 * r + 0.7152 * g + 0.0722 * b;
    };
    if (!/^#[0-9a-f]{6}$/i.test(stroke))
        return "";
    const face = luminance(rgb("#E4EFF9")) + 0.05;
    const start = rgb(stroke), end = rgb("#323130");
    let ink = start;
    for (let step = 0; step <= 20; step++) {
        ink = start.map((a, index) => Math.round(a + (end[index] - a) * step / 20));
        if (face / (luminance(ink) + 0.05) >= 4.5)
            break;
    }
    return `#${ink.map(value => value.toString(16).padStart(2, "0")).join("")}`;
}
/** A bracket post (BracketPost): a column of the cards' height and header
 * block, titled "Bracket N" in the bracket's reading ink with its colour on
 * the edge; the end post closes the span with its Repeats count (a count
 * box with its source badge), the start post keeps those lines empty. */
class BracketPost {
    bracketId;
    side;
    send;
    owner;
    root = element("section", "card bordered pulse-card pulse-post");
    label = element("div", "heading pulse-title");
    count = null;
    spin = null;
    constructor(bracketId, side, send, owner) {
        this.bracketId = bracketId;
        this.side = side;
        this.send = send;
        this.owner = owner;
        this.root.dataset.item = itemKey(["bracket", `${bracketId}:${side}`]);
        this.root.append(line(this.label), line(side === "end" ? "Repeats" : ""));
        if (side === "end") {
            const count = this.count = new ValueBox(send, text => this.commitCount(text), owner);
            count.tip = "Edit repeat value source";
            count.root.classList.add("pulse-count");
            this.spin = new SpinBox(count.root, count.box.root, { step: () => 1, bounds: () => [this.bracket?.count_field.validator_lo ?? 1, null],
                commit: value => this.commitCount(String(value)) });
            this.root.append(line(count.root));
        }
    }
    bracket = null;
    commitCount(text) {
        const bracket = this.bracket;
        const count = Number(text);
        if (!bracket || !Number.isInteger(count) || count === bracket.count)
            return;
        this.send(editCommand(this.owner, "bracket", { bracket_id: bracket.bracket_id, start: bracket.start_period_id, end: bracket.end_period_id, count }), "Changing the bracket…");
    }
    update(bracket) {
        this.bracket = bracket;
        this.root.style.setProperty("--bracket-ink", bracket.color);
        this.label.style.color = readingInk(bracket.color);
        setText(this.label, `Bracket ${bracket.ordinal}`);
        setTitle(this.root, `Bracket ${bracket.ordinal}: drag to move this boundary`);
        this.count?.show({ ...bracket.count_field, text: String(bracket.count) }, !bracket.readonly);
        this.spin?.refresh();
    }
}
/** A Subpulse's source popup (ComponentCard._toggle_source): Source,
 * Default or Config, in a popup under the card's badge. */
class SourcePopup {
    send;
    id;
    current;
    root = element("div", "card bordered pulse-binding pulse-source-popup");
    choice;
    constructor(send, id, current) {
        this.send = send;
        this.id = id;
        this.current = current;
        this.root.popover = "auto";
        // v2 _commit_source: a new source is asked of the owner; the popup stays, showing what it holds.
        this.choice = new Segmented(value => {
            if (value !== this.current)
                this.send({ action: "pulse_subpulse", op: "source", component_id: this.id, source: value }, "Choosing the Subpulse source…");
            this.choice.setCurrent(this.current);
        });
        this.choice.setChoices([["default", "Default"], ["config", "Config"]]);
        this.choice.setCurrent(current);
        const row = element("div", "pulse-binding-row");
        // One label column with a value's popup (popup_label_width), so Source's choice stands on one x in both.
        const label = element("span", "pulse-label pulse-popup-label");
        label.append(element("span", "", "Source"), element("span", "sizer", "Config name"));
        row.append(label, this.choice.root);
        this.root.append(row);
        this.root.addEventListener("toggle", event => { if (event.newState === "closed")
            this.root.remove(); });
    }
    get open() { return this.root.isConnected && this.root.matches(":popover-open"); }
    project(current) { this.current = current; this.choice.setCurrent(current); }
    openBelow(anchor) {
        document.body.append(this.root);
        this.root.showPopover();
        const rect = anchor.getBoundingClientRect(), gap = token("--popup-gap"), pad = token("--window-pad");
        this.root.style.left = `${Math.max(pad, Math.min(rect.left, innerWidth - pad - this.root.offsetWidth))}px`;
        const below = rect.bottom + gap;
        this.root.style.top = `${below + this.root.offsetHeight <= innerHeight - pad ? below : Math.max(pad, rect.top - gap - this.root.offsetHeight)}px`;
    }
}
/** A Subpulse on the strip (ComponentCard): closed, one card for the block
 * -- Duration, its total with the source badge, what it holds, its name --
 * then Expand, Edit and Ungroup on the port rows' lines; expanded, a Start
 * and an End card frame its members, read-only names, Collapse held down. */
class ComponentCard {
    id;
    mode;
    send;
    opened;
    root = element("section", "card bordered pulse-card pulse-component");
    title = element("div", "heading pulse-title");
    totalText = element("span", "elided");
    badge = element("button", "pulse-badge");
    summary = element("span", "elided note");
    name;
    nameText = element("span", "elided");
    expand;
    component = null;
    constructor(id, mode, send, opened) {
        this.id = id;
        this.mode = mode;
        this.send = send;
        this.opened = opened;
        this.root.dataset.item = itemKey([mode === "end" ? "component-end" : mode === "start" ? "component-start" : "component", id]);
        this.root.classList.toggle("framing", mode !== "closed");
        this.name = new TextBox(text => send({ action: "pulse_subpulse", op: "rename", component_id: id, name: text }, "Renaming the Subpulse…"));
        this.name.root.placeholder = "Name";
        this.badge.type = "button";
        this.badge.tabIndex = -1;
        let wasOpen = false;
        this.badge.addEventListener("pointerdown", () => { wasOpen = Boolean(this.popup?.open); });
        this.badge.addEventListener("click", () => {
            if (!this.component || wasOpen) {
                wasOpen = false;
                return;
            }
            this.popup = new SourcePopup(send, id, this.component.source);
            this.popup.openBelow(this.badge);
        });
        const value = (text, badge) => {
            const holder = element("div", "pulse-value-line");
            holder.append(text, ...(badge ? [badge] : []));
            return holder;
        };
        this.expand = command(mode === "closed" ? "Expand" : "Collapse", () => this.opened(mode === "closed"));
        this.expand.setActive(mode !== "closed");
        const commands = element("div", "pulse-ports");
        commands.append(this.expand.root, command("Edit", () => send({ action: "pulse_subpulse", op: "select", component_id: id }, "Opening the Subpulse…"), { title: "Edit this Subpulse" }).root, command("Ungroup", () => send({ action: "pulse_subpulse", op: "ungroup", component_id: id }, "Ungrouping…"), { role: "destructive", title: "Dissolve this Subpulse: its embedded (Default) periods become ordinary Pulse items; a Config source link is dropped" }).root);
        this.root.append(line(this.title), line("Duration"), value(this.totalText, this.badge), value(this.summary), mode === "closed" ? line(this.name.root) : value(this.nameText), commands);
    }
    /** v2 focus_component_name: the name field focused, its text selected. */
    focusName() { this.name.root.focus(); this.name.root.select(); }
    popup = null;
    update(component) {
        this.component = component;
        if (this.popup?.open)
            this.popup.project(component.source);
        setText(this.title, this.mode === "closed" ? "Subpulse" : this.mode === "start" ? "Start" : "End");
        this.root.classList.toggle("config", component.config_active);
        setElided(this.totalText, component.total_text);
        drawBadges(this.badge, [component.source === "config" ? ["C", "config"] : ["", "default"]]);
        setTitle(this.badge, component.error || `Subpulse source: ${component.source === "config" ? "Config" : "Default"}`);
        setElided(this.summary, this.mode === "closed" ? component.summary_text : "");
        this.name.show(component.name);
        setTitle(this.name.root, component.name);
        setElided(this.nameText, component.name);
    }
}
/** The left columns: Port catalog and Delay / Scan. */
class CatalogColumn {
    send;
    owner;
    root = element("section", "card bordered pulse-card pulse-catalog");
    name;
    total = element("span", "pulse-value-text");
    periods = element("span", "pulse-value-text");
    visible = element("span", "pulse-value-text");
    rows = element("div", "pulse-ports");
    labels = new Map();
    named = false;
    constructor(send, owner) {
        this.send = send;
        this.owner = owner;
        this.name = new TextBox(text => { if (this.named)
            send(editCommand(owner, "document_name", { name: text }), "Renaming the pulse…"); });
        // v2 set_placeholder_prompt: the field's own name, a faint prompt.
        this.name.root.placeholder = owner === "pulse" ? "Pulse name" : "Subpulse name";
        this.name.root.classList.add("prompt");
        this.root.append(line(element("div", "heading pulse-title", "Port catalog")), line("Name", this.name.root), line("Total", this.total), line("Periods", this.periods), line("Visible", this.visible), this.rows);
    }
    update(schedule, editable, hasPulse) {
        // v2 ChannelNamesPanel: the Name is always a field; with no pulse a rename is ignored.
        this.named = editable && hasPulse;
        this.name.show(schedule.document_name, editable);
        // Readouts (v2 FluentReadoutEdit): a long one gives up its middle.
        setMiddle(this.total, schedule.total_text, "none");
        setTitle(this.total, schedule.total_tooltip);
        setMiddle(this.periods, String(schedule.period_count));
        setMiddle(this.visible, schedule.visible_text);
        const nodes = schedule.ports.filter(port => port.visible).map(port => {
            let cell = this.labels.get(port.key);
            if (!cell) {
                const endpoint = rowLabel();
                // The board's port labels are named on the Pulse's Edit page; a Subpulse reads them (readouts, v2 ports_editable=False).
                const label = this.owner === "pulse"
                    ? new TextBox(text => this.send(editCommand(this.owner, "port_label", { port_key: port.key, label: text }), "Renaming the output…"), "elide-right")
                    : element("span", "pulse-value-text");
                const row = line(endpoint.root, label instanceof TextBox ? label.root : label);
                cell = { row, endpoint, label };
                this.labels.set(port.key, cell);
            }
            setText(cell.endpoint.text, port.endpoint_text);
            setTitle(cell.endpoint.root, port.endpoint_tooltip);
            if (cell.label instanceof TextBox)
                cell.label.show(port.label, this.named);
            else
                setText(cell.label, port.label);
            return cell.row;
        });
        if (nodes.length !== this.rows.children.length || nodes.some((node, index) => this.rows.children[index] !== node))
            this.rows.replaceChildren(...nodes);
    }
}
class DelayColumn {
    send;
    root = element("section", "card bordered pulse-card pulse-delays");
    clock = element("span", "pulse-value-text");
    scan = element("span", "pulse-value-text");
    repeats;
    config = switchButton(on => this.send({ action: "pulse_config", op: "mode", enabled: on }, "Switching Config…"));
    refresh;
    rows = element("div", "pulse-ports");
    cells = new Map();
    constructor(send) {
        this.send = send;
        this.repeats = new RepeatsBox(repeats => send({ action: "pulse_edit", edit: "run_repeats", repeats }, "Changing Run repeats…"));
        this.repeats.root.title = "Complete Pulse runs per scan point (or per On Pulse without a scan); ∞ (0) runs indefinitely";
        this.refresh = command("Refresh", () => send({ action: "pulse_config", op: "refresh" }, "Refreshing Config…"));
        const config = element("div", "pulse-config-mode");
        config.append(this.config, this.refresh.root);
        this.root.append(line(element("div", "heading pulse-title", "Delay / Scan")), line("Clock", this.clock), line("Scan", this.scan), line("Repeats", this.repeats.root), line("Config", config), this.rows);
    }
    update(state) {
        const schedule = state.schedule;
        setMiddle(this.clock, schedule.clock_text);
        setMiddle(this.scan, schedule.scan_summary_text);
        this.repeats.show(schedule.run_repeats, schedule.periods.length > 0);
        setChecked(this.config, state.config_mode.enabled);
        this.config.disabled = state.config_mode.busy;
        this.refresh.disabled = !state.config_mode.enabled || state.config_mode.busy;
        const delays = new Map(schedule.delay_rows.map(row => [row.port_key, row]));
        const nodes = schedule.ports.filter(port => port.visible && delays.has(port.key)).map(port => {
            const delay = delays.get(port.key);
            let cell = this.cells.get(port.key);
            if (!cell) {
                const name = rowLabel();
                const value = new ValueBox(this.send, text => this.send({ action: "pulse_edit", edit: "delay", port_key: port.key, value: text, unit: this.cells.get(port.key)?.unit.value ?? "ns" }, "Changing the delay…"));
                value.tip = "Edit value source";
                const unit = new ComboBox(chosen => this.send({ action: "pulse_edit", edit: "delay", port_key: port.key, value: this.cells.get(port.key)?.value.box.root.value ?? "0", unit: chosen }, "Changing the delay…"));
                const fill = element("button", "pulse-fill");
                fill.type = "button";
                fill.title = "Turn this output on in every period";
                fill.addEventListener("click", () => this.send({ action: "pulse_edit", edit: "fill_port", port_key: port.key }, "Filling the output…"));
                const clearPort = () => { this.send({ action: "pulse_edit", edit: "clear_port", port_key: port.key }, "Clearing the output…"); };
                const clear = element("button", "pulse-clear");
                clear.type = "button";
                clear.title = "Turn this output off in every period";
                clear.addEventListener("click", clearPort);
                // On an analog row there is no "off": the button takes the output's steps away, the row remover every row has.
                const steps = command("×", clearPort, { role: "destructive", glyph: true, square: true, title: "Remove this output's steps from every period" }).root;
                const row = line(name.root, value.root, unit.root, fill, clear, steps);
                cell = { row, name, value, unit, fill, clear, steps };
                this.cells.set(port.key, cell);
            }
            setText(cell.name.text, port.label);
            setTitle(cell.name.root, port.endpoint_tooltip);
            cell.value.show(delay.value, state.has_pulse);
            cell.unit.setChoices(delay.units.map(unit => [unit, unit]));
            cell.unit.value = delay.unit;
            // v2 set_delay_rows: the unit is the value's, switched off with it.
            cell.unit.disabled = !state.has_pulse || delay.value.config_active || !delay.value.editable;
            const digital = port.kind === "digital";
            // Inapplicable actions keep their column in every port row.
            cell.fill.disabled = !state.has_pulse || !digital;
            cell.clear.hidden = !digital;
            cell.steps.hidden = digital;
            cell.clear.disabled = cell.steps.disabled = !state.has_pulse;
            return cell.row;
        });
        if (nodes.length !== this.rows.children.length || nodes.some((node, index) => this.rows.children[index] !== node))
            this.rows.replaceChildren(...nodes);
    }
}
/** v2 bracket_spans_of: each bracket's (start gap, end gap) in a flat order,
 * or null when a post stands before its partner or two brackets overlap
 * without one inside the other (what the document would refuse). */
function bracketSpansOk(order) {
    const positions = new Map();
    const gaps = new Map();
    let before = 0;
    order.forEach((key, index) => {
        const [kind, id] = parseKey(key);
        if (kind === "period") {
            before++;
            return;
        }
        const at = id.lastIndexOf(":");
        const bracket = id.slice(0, at), side = id.slice(at + 1);
        (positions.get(bracket) ?? positions.set(bracket, {}).get(bracket))[side] = index;
        (gaps.get(bracket) ?? gaps.set(bracket, {}).get(bracket))[side] = before;
    });
    const spans = [];
    for (const [bracket, sides] of positions) {
        if (sides.start === undefined || sides.end === undefined || sides.end < sides.start)
            return false;
        spans.push([gaps.get(bracket).start, gaps.get(bracket).end]);
    }
    const contains = (outer, inner) => outer[0] <= inner[0] && inner[1] <= outer[1] && (inner[0] < inner[1] || (outer[0] < inner[0] && inner[0] < outer[1]));
    for (let a = 0; a < spans.length; a++)
        for (let b = a + 1; b < spans.length; b++) {
            const [first, second] = [spans[a], spans[b]];
            if (!(first[1] <= second[0] || second[1] <= first[0] || contains(first, second) || contains(second, first)))
                return false;
        }
    return true;
}
/** The strip of one document's items, with selection and dragging (v2
 * PulseDragContainer + the edit commands of PulseScheduleView): the Edit
 * page's, and the Subpulse page's for the selected owner.  A click picks an
 * item (Shift adds or removes one; the picked item again picks nothing), a
 * click between items picks that gap, Delete or Backspace removes what is
 * picked, and dragging moves it as one proposal the document still owns. */
export class Strip {
    send;
    owner;
    say;
    root = element("div", "pulse-strip");
    catalog;
    delays;
    side = element("div", "pulse-side");
    stub = element("section", "card bordered pulse-side-stub");
    timeline = element("div", "pulse-timeline");
    items = element("div", "pulse-items");
    periods = new Map();
    posts = new Map();
    components = new Map();
    opened = new Set();
    /** The picked items in timeline order, or the picked gap (v2 `_selection`). */
    picked = [];
    gap = null;
    order = [];
    visual = [];
    /** Each Subpulse's members as flat keys, in timeline order (v2 `_item_blocks`). */
    blocks = new Map();
    schedule = null;
    indicator = element("div", "pulse-gap-mark");
    pressed = null;
    dragging = null;
    swallowClick = false;
    /** Set by Add: the next schedule bringing exactly one new card picks it (v2 `_expect_new_card`). */
    expectNew = false;
    last = null;
    /** Unmade cards near the view are made as the timeline scrolls to them. */
    watched = new WeakMap();
    watcher = new IntersectionObserver(entries => {
        for (const entry of entries) {
            const card = this.watched.get(entry.target);
            if (entry.isIntersecting && card)
                this.make(card);
        }
    }, { root: this.timeline, rootMargin: "0px 100% 0px 100%" });
    makingLater = 0;
    sidePanels = true;
    onSidePanels = () => { };
    constructor(send, owner, say) {
        this.send = send;
        this.owner = owner;
        this.say = say;
        this.catalog = new CatalogColumn(send, owner);
        this.delays = owner === "pulse" ? new DelayColumn(send) : null;
        // v2 left_panel_stub: where the hidden side panels stood, what they hold and Show.
        const stubLabel = element("div", "pulse-stub-label", "Name\nDelay");
        this.stub.append(stubLabel, command("Show", () => this.setSidePanels(true)).root);
        this.stub.hidden = true;
        this.side.append(this.catalog.root, ...(this.delays ? [this.delays.root] : []), this.stub);
        // LinkedScrollPanes: ONE scroll offset for the side panels and the cards,
        // so their rows cannot part (v2: one bar drives both panes in one paint).
        // The side panels stand sticky at its left edge; only the cards move
        // sideways under them.  Two scrollers kept in step by a scroll handler
        // drew the follower a frame late while the strip moved.
        this.timeline.append(this.side, this.items);
        this.root.append(this.timeline);
        // The side's width places the strip's first gap and starts the sideways bar's track after it
        // (its layout width, not offsetWidth's whole CSS px: at a fractional DPR that is a device pixel off).
        new ResizeObserver(entries => {
            const width = `${entries.at(-1).borderBoxSize[0].inlineSize}px`;
            if (this.timeline.style.getPropertyValue("--pulse-side-w") !== width)
                this.timeline.style.setProperty("--pulse-side-w", width);
        }).observe(this.side);
        // v2 _clip_scrolled_gutter: the strip's first gap stays clear while the cards scroll under it.
        const gutter = element("div", "pulse-gutter");
        this.items.append(gutter, this.indicator);
        this.indicator.hidden = true;
        // A control the keyboard focuses beneath the side panels (Tab back into a card scrolled under
        // them) is brought out past the first gap, as QScrollArea's focusNextPrevChild shows it.
        let pointing = false;
        this.items.addEventListener("pointerdown", () => { pointing = true; }, true);
        for (const end of ["pointerup", "pointercancel"])
            window.addEventListener(end, () => { pointing = false; }, true);
        this.items.addEventListener("focusin", event => {
            if (pointing || event.target === this.items)
                return;
            const hidden = gutter.getBoundingClientRect().right - event.target.getBoundingClientRect().left;
            if (hidden > 0)
                this.timeline.scrollLeft -= hidden;
        });
        this.items.addEventListener("pointerdown", event => this.press(event));
        this.items.addEventListener("pointermove", event => this.move(event));
        this.items.addEventListener("pointerup", event => this.release(event));
        // The capture lost without a release (a cancel, the window losing the mouse): no drag goes on.
        this.items.addEventListener("lostpointercapture", () => this.cancelPress());
        // A Shift-press over a control picks its card: the control never sees that click.
        this.items.addEventListener("click", event => { if (this.swallowClick) {
            this.swallowClick = false;
            event.preventDefault();
            event.stopPropagation();
        } }, true);
        this.items.addEventListener("keydown", event => {
            if ((event.key === "Delete" || event.key === "Backspace") && event.target === this.items && this.payload().length) {
                event.preventDefault();
                this.remove();
            }
        });
        this.items.tabIndex = 0;
    }
    /** v2 _toggle_left_panels / _show_left_panels: the side panels, or their stub. */
    setSidePanels(shown) {
        this.sidePanels = shown;
        this.catalog.root.hidden = !shown;
        if (this.delays)
            this.delays.root.hidden = !shown;
        this.stub.hidden = shown;
        this.onSidePanels(shown);
    }
    /** The items in timeline order: a closed Subpulse as one card; an open one
     * as its Start card, its members, its End card. */
    visualKeys(schedule, order) {
        const owner = new Map();
        this.blocks.clear();
        for (const component of schedule.components) {
            const block = [];
            for (const item of order) {
                const key = itemKey(item);
                if ((item[0] === "period" && component.period_ids.includes(item[1])) || (item[0] === "bracket" && component.bracket_ids.includes(postBracket(item[1])))) {
                    block.push(key);
                    owner.set(key, component.component_id);
                }
            }
            this.blocks.set(component.component_id, block);
        }
        const keys = [];
        const shown = new Set();
        for (const item of order) {
            const key = itemKey(item);
            const component = owner.get(key);
            if (!component) {
                keys.push(key);
                continue;
            }
            const open = this.opened.has(component);
            if (!shown.has(component)) {
                shown.add(component);
                keys.push(itemKey([open ? "component-start" : "component", component]));
            }
            if (!open)
                continue;
            keys.push(key);
            if (this.blocks.get(component).at(-1) === key)
                keys.push(itemKey(["component-end", component]));
        }
        return keys;
    }
    update(state, schedule, order, editable = true) {
        this.last = state;
        const known = new Set(this.schedule?.periods.map(period => period.period_id) ?? []);
        const periodsChanged = !this.schedule || this.schedule.periods.length !== schedule.periods.length
            || schedule.periods.some(period => !known.has(period.period_id));
        this.schedule = schedule;
        this.order = order;
        this.catalog.update(schedule, editable, state.has_pulse);
        this.delays?.update(state);
        for (const id of [...this.opened])
            if (!schedule.components.some(c => c.component_id === id))
                this.opened.delete(id);
        const previous = this.visual;
        this.visual = this.visualKeys(schedule, order);
        const periods = new Map(schedule.periods.map(period => [period.period_id, period]));
        const brackets = new Map(schedule.brackets.map(bracket => [bracket.bracket_id, bracket]));
        const components = new Map(schedule.components.map(component => [component.component_id, component]));
        const authored = schedule.periods.filter(period => period.kind !== "spacer");
        const ordinal = new Map(authored.map((period, at) => [period, at + 1]));
        const nodes = [];
        const alive = new Set(this.visual);
        const usedCards = new Set();
        for (const key of this.visual) {
            const [kind, id] = parseKey(key);
            if (kind === "period") {
                const period = periods.get(id);
                if (!period)
                    continue;
                let card = this.periods.get(id);
                if (!card) {
                    card = new PeriodCard(id, this.send, this.owner);
                    this.periods.set(id, card);
                }
                card.update(period, ordinal.get(period) ?? 0, authored.length, schedule.ports, schedule.analog_mode_choices);
                nodes.push(card.root);
            }
            else if (kind === "bracket") {
                const bracketId = postBracket(id);
                const bracket = brackets.get(bracketId);
                if (!bracket)
                    continue;
                let post = this.posts.get(id);
                if (!post) {
                    post = new BracketPost(bracketId, id.endsWith(":start") ? "start" : "end", this.send, this.owner);
                    this.posts.set(id, post);
                }
                post.update(bracket);
                nodes.push(post.root);
            }
            else {
                const component = components.get(id);
                if (!component)
                    continue;
                const mode = kind === "component-end" ? "end" : kind === "component-start" ? "start" : "closed";
                const slot = `${mode}|${id}`;
                usedCards.add(slot);
                let card = this.components.get(slot);
                if (!card) {
                    card = new ComponentCard(id, mode, this.send, open => this.expand(id, open));
                    this.components.set(slot, card);
                }
                card.update(component);
                nodes.push(card.root);
            }
        }
        const arrived = [...periods.keys()].filter(id => !known.has(id));
        for (const [id, card] of this.periods)
            if (!alive.has(itemKey(["period", id]))) {
                card.root.remove();
                this.watcher.unobserve(card.root);
                this.periods.delete(id);
            }
        for (const [id, post] of this.posts)
            if (!alive.has(itemKey(["bracket", id]))) {
                post.root.remove();
                this.posts.delete(id);
            }
        for (const [slot, card] of this.components)
            if (!usedCards.has(slot)) {
                card.root.remove();
                this.components.delete(slot);
            }
        let cursor = this.items.firstElementChild?.nextElementSibling ?? null;
        for (const node of nodes) {
            if (cursor === this.indicator)
                cursor = cursor.nextElementSibling;
            if (node !== cursor)
                this.items.insertBefore(node, cursor);
            else
                cursor = cursor.nextElementSibling;
        }
        this.items.classList.toggle("long", nodes.length > LONG_STRIP);
        this.makeNearby();
        // v2 set_items: a changed order forgets a press and keeps what is still there.
        if (this.visual.join() !== previous.join()) {
            this.pressed = null;
            this.picked = this.picked.filter(key => alive.has(key));
            if (this.gap !== null && this.gap > this.visual.length)
                this.gap = null;
        }
        // v2 _reconcile: Add's one new card is picked.
        if (periodsChanged) {
            if (this.expectNew && arrived.length === 1)
                this.show([itemKey(["period", arrived[0]])]);
            this.expectNew = false;
        }
        this.paint();
    }
    // ---------------------------------------------------------- making cards
    /** A card made now: its controls drawn, no longer watched. */
    make(card) { card.make(); this.watcher.unobserve(card.root); }
    /** The cards within a view's width of the timeline's view are made before
     * this draw; the rest when they near the view, or a slice at a time after it. */
    makeNearby() {
        const waiting = [...this.periods.values()].filter(card => !card.made);
        if (!waiting.length)
            return;
        // A short strip is made whole now.
        if (!this.items.classList.contains("long")) {
            for (const card of waiting)
                this.make(card);
            return;
        }
        for (const card of waiting) {
            this.watched.set(card.root, card);
            this.watcher.observe(card.root);
        }
        const view = this.cardsView();
        if (view.width) {
            // One layout of the unmade (empty) cards; making them moves none sideways.
            const boxes = waiting.map(card => card.root.getBoundingClientRect());
            waiting.forEach((card, at) => { if (boxes[at].right >= view.left - view.width && boxes[at].left <= view.right + view.width)
                this.make(card); });
        }
        if (this.makingLater)
            return;
        const slice = () => {
            const started = performance.now();
            for (const card of this.periods.values()) {
                if (card.made)
                    continue;
                this.make(card);
                if (performance.now() - started > 6) {
                    this.makingLater = window.setTimeout(slice);
                    return;
                }
            }
            this.makingLater = 0;
        };
        // After the frame this draw is in.
        this.makingLater = requestAnimationFrame(() => { this.makingLater = window.setTimeout(slice); });
    }
    // ---------------------------------------------------------- selection
    cards() { return [...this.items.querySelectorAll(":scope > [data-item]")]; }
    cardAt(target) {
        const card = target?.closest?.("[data-item]");
        return card && card.parentElement === this.items ? card : null;
    }
    /** v2 _gap_at: one gap per adjacency, both sides of each post. */
    gapAt(x) {
        const cards = this.cards();
        for (let index = 0; index < cards.length; index++) {
            const rect = cards[index].getBoundingClientRect();
            if (x < rect.left + rect.width / 2)
                return index;
        }
        return cards.length;
    }
    /** v2 show_selection: picked items (in timeline order) or a gap, never both. */
    show(items = [], gap = null) {
        const chosen = new Set(items);
        this.gap = gap;
        this.picked = gap === null ? this.visual.filter(key => chosen.has(key)) : [];
        this.paint();
    }
    /** v2 _paint_selection: an open Subpulse's caps stand for its whole block. */
    paint(selected = this.picked) {
        const chosen = new Set(selected);
        for (const key of selected) {
            const [kind, id] = parseKey(key);
            if (kind === "component-start" || kind === "component-end") {
                for (const member of this.blocks.get(id) ?? [])
                    chosen.add(member);
                chosen.add(itemKey(["component-start", id]));
                chosen.add(itemKey(["component-end", id]));
            }
        }
        for (const card of this.cards()) {
            const key = card.dataset.item;
            const [kind, id] = parseKey(key);
            let on = chosen.has(key);
            if (kind === "component-start" || kind === "component-end") {
                const block = this.blocks.get(id) ?? [];
                on ||= block.length > 0 && block.every(member => chosen.has(member));
            }
            card.classList.toggle("selected", on);
        }
        if (this.gap === null || this.dragging) {
            if (!this.dragging)
                this.indicator.hidden = true;
        }
        else
            this.showIndicator(this.gap);
    }
    /** v2 _clicked: Shift adds or removes; the one picked item again picks nothing. */
    clicked(key, append) {
        const current = this.picked;
        if (append)
            this.show(current.includes(key) ? current.filter(item => item !== key) : [...current, key]);
        else
            this.show(current.length === 1 && current[0] === key ? [] : [key]);
    }
    /** v2 selection_payload: a Subpulse as itself, never its contents. */
    payload() {
        const components = new Set(this.picked.filter(key => key.startsWith("component")).map(key => parseKey(key)[1]));
        const covered = new Set([...components].flatMap(id => this.blocks.get(id) ?? []));
        const result = [];
        for (const key of this.picked) {
            const [kind, id] = parseKey(key);
            const item = kind.startsWith("component") ? itemKey(["component", id]) : key;
            if (!covered.has(item) && !result.includes(item))
                result.push(item);
        }
        return result;
    }
    /** v2 flat_order: visual keys back to the document's flat timeline. */
    flatOrder(keys) {
        return keys.flatMap(key => {
            const [kind, id] = parseKey(key);
            if (kind === "component-start" || kind === "component-end")
                return [];
            return kind === "component" ? this.blocks.get(id) ?? [] : [key];
        });
    }
    press(event) {
        if (event.button !== 0)
            return;
        const card = this.cardAt(event.target);
        if (!card)
            return;
        const shift = event.shiftKey;
        // Typing, ticking and opening ARE the click on a control; Shift picks its card.
        if (!shift && event.target.closest("input, select, button, textarea"))
            return;
        const key = card.dataset.item;
        this.pressed = { key, x: event.clientX, y: event.clientY, shift };
        if (shift) {
            this.swallowClick = true;
            event.preventDefault();
        }
        this.items.focus({ preventScroll: true });
        if (!shift && !this.picked.includes(key)) {
            this.paint([key]);
            this.indicator.hidden = true;
        }
        this.items.setPointerCapture(event.pointerId);
    }
    cancelPress() {
        this.pressed = null;
        if (this.dragging) {
            this.dragging.ghost.remove();
            this.dragging = null;
        }
        this.paint();
    }
    /** v2 _drag_items: what one drag moves -- the selection when the grabbed item is in it, whole Subpulses. */
    dragItems(grabbed) {
        const chosen = new Set(this.picked.includes(grabbed) ? this.picked : [grabbed]);
        for (const [id, block] of this.blocks) {
            const caps = [itemKey(["component-start", id]), itemKey(["component-end", id])];
            if (caps.some(cap => chosen.has(cap)) || (block.length && block.every(member => chosen.has(member))))
                for (const key of [...caps, ...block])
                    chosen.add(key);
        }
        return this.visual.filter(key => chosen.has(key));
    }
    /** A read-only card's content (a member of a Subpulse from Config). */
    readonly(key) {
        const [kind, id] = parseKey(key);
        if (kind === "period")
            return Boolean(this.schedule?.periods.find(period => period.period_id === id)?.readonly);
        if (kind === "bracket")
            return Boolean(this.schedule?.brackets.find(bracket => bracket.bracket_id === postBracket(id))?.readonly);
        return false;
    }
    /** v2 _proposal_at: the flat order a drop in `gap` would ask for, or null. */
    proposal(grabbed, gap) {
        const order = this.visual;
        if (!order.includes(grabbed) || gap < 0 || gap > order.length)
            return null;
        const moving = this.dragItems(grabbed);
        const moved = new Set(moving.filter(key => key.startsWith("component")).map(key => parseKey(key)[1]));
        if (moving.some(key => this.readonly(key) && ![...moved].some(id => this.blocks.get(id)?.includes(key))))
            return null;
        const destination = gap - order.slice(0, gap).filter(key => moving.includes(key)).length;
        const remaining = order.filter(key => !moving.includes(key));
        const proposed = [...remaining.slice(0, destination), ...moving, ...remaining.slice(destination)];
        if (proposed.join() === order.join())
            return null;
        const flat = this.flatOrder(proposed);
        return bracketSpansOk(flat) ? flat : null;
    }
    /** Where the cards show: the strip's view right of the side panels, above its sideways bar. */
    cardsView() {
        const view = this.timeline.getBoundingClientRect();
        const left = this.side.getBoundingClientRect().right, right = view.left + this.timeline.clientWidth;
        return { left, right, top: view.top, bottom: view.top + this.timeline.clientHeight, width: Math.max(0, right - left) };
    }
    inTimeline(event) {
        const view = this.cardsView();
        return event.clientX >= view.left && event.clientX < view.right && event.clientY >= view.top && event.clientY < view.bottom;
    }
    move(event) {
        if (this.dragging) {
            this.dragging.ghost.style.left = `${event.clientX - this.dragging.ghost.offsetWidth / 2}px`;
            this.dragging.ghost.style.top = `${event.clientY}px`;
            const gap = this.gapAt(event.clientX);
            if (this.inTimeline(event) && this.proposal(this.dragging.grabbed, gap))
                this.showIndicator(gap);
            else
                this.indicator.hidden = true;
            return;
        }
        const pressed = this.pressed;
        if (!pressed)
            return;
        if (!(event.buttons & 1)) {
            this.pressed = null;
            this.paint();
            return;
        }
        // QApplication.startDragDistance (Windows SM_CXDRAG).
        if (Math.abs(event.clientX - pressed.x) + Math.abs(event.clientY - pressed.y) < 4)
            return;
        this.pressed = null;
        const grabbed = pressed.key;
        if (!this.picked.includes(grabbed))
            this.show(pressed.shift ? [...this.picked, grabbed] : [grabbed]);
        this.show(this.dragItems(grabbed));
        // QDrag's pixmap: the grabbed card, its top centre at the pointer.
        const source = this.cards().find(card => card.dataset.item === grabbed);
        const ghost = source.cloneNode(true);
        ghost.classList.add("pulse-drag-ghost");
        ghost.style.width = `${source.offsetWidth}px`;
        ghost.style.height = `${source.offsetHeight}px`;
        document.body.append(ghost);
        ghost.style.left = `${event.clientX - source.offsetWidth / 2}px`;
        ghost.style.top = `${event.clientY}px`;
        this.dragging = { grabbed, ghost };
        this.indicator.hidden = true;
    }
    release(event) {
        if (event.button !== 0)
            return;
        const dragging = this.dragging;
        if (dragging) {
            this.dragging = null;
            dragging.ghost.remove();
            const flat = this.inTimeline(event) ? this.proposal(dragging.grabbed, this.gapAt(event.clientX)) : null;
            this.paint();
            if (flat)
                this.send(editCommand(this.owner, "reorder", { order: flat.map(parseKey) }), "Moving…");
            return;
        }
        const pressed = this.pressed;
        this.pressed = null;
        if (pressed) {
            const card = this.cards().find(node => { const rect = node.getBoundingClientRect(); return event.clientX >= rect.left && event.clientX < rect.right && event.clientY >= rect.top && event.clientY < rect.bottom; });
            if (card?.dataset.item === pressed.key)
                this.clicked(pressed.key, pressed.shift);
            else
                this.paint();
            return;
        }
        // v2 mouseReleaseEvent: a click on the strip between items picks that gap (again: none).
        if (event.shiftKey || this.cardAt(event.target))
            return;
        const gap = this.gapAt(event.clientX);
        this.show([], this.gap === gap ? null : gap);
    }
    /** v2 _show_indicator_at_gap: a rounded accent caret centred in the gap, a pixel clear of each neighbour, the cards' height. */
    showIndicator(gap) {
        const cards = this.cards();
        if (!cards.length) {
            this.indicator.hidden = true;
            return;
        }
        const at = Math.max(0, Math.min(gap, cards.length));
        const host = this.items.getBoundingClientRect();
        const spacing = parseFloat(getComputedStyle(this.items).columnGap) || 0;
        let left, right;
        if (at === 0) {
            right = cards[0].getBoundingClientRect().left;
            left = right - spacing;
        }
        else if (at === cards.length) {
            left = cards[at - 1].getBoundingClientRect().right;
            right = left + spacing;
        }
        else {
            left = cards[at - 1].getBoundingClientRect().right;
            right = cards[at].getBoundingClientRect().left;
        }
        const clearance = Math.max(1, token("--edge"));
        const width = Math.max(2 * clearance, right - left - 2 * clearance);
        this.indicator.style.left = `${left - host.left + (right - left - width) / 2}px`;
        this.indicator.style.width = `${width}px`;
        this.indicator.hidden = false;
    }
    // ---------------------------------------------------------- the commands
    /** The visual span of an open Subpulse: its Start and End cards' places. */
    openSpan(id) {
        return [this.visual.indexOf(itemKey(["component-start", id])), this.visual.indexOf(itemKey(["component-end", id]))];
    }
    /** v2 _readonly_selection: what is picked comes from a Config Subpulse; said, refused. */
    readonlySelection() {
        const schedule = this.schedule;
        if (!schedule)
            return false;
        const payload = this.payload();
        const inside = this.gap !== null && schedule.components.some(component => {
            if (!component.config_active || !this.opened.has(component.component_id))
                return false;
            const [first, last] = this.openSpan(component.component_id);
            return first < this.gap && this.gap <= last;
        });
        if (payload.some(key => this.readonly(key)) || inside) {
            this.say("This Subpulse comes from Config. Open its source file to edit it.");
            return true;
        }
        return false;
    }
    /** The open Subpulse an Add lands in: the picks all inside it, or the gap between its caps. */
    openOwner() {
        const payload = this.payload();
        for (const component of this.schedule?.components ?? []) {
            if (!this.opened.has(component.component_id))
                continue;
            const block = this.blocks.get(component.component_id) ?? [];
            const [first, last] = this.openSpan(component.component_id);
            if ((payload.length && payload.every(key => block.includes(key))) || (this.gap !== null && first < this.gap && this.gap <= last))
                return component;
        }
        return null;
    }
    /** `_selected_before_item`: add in the picked gap, or right after the selection. */
    beforeItem() {
        let after;
        if (this.gap !== null)
            after = this.gap;
        else {
            const picked = new Set(this.picked);
            for (const key of this.picked) {
                const [kind, id] = parseKey(key);
                if (kind === "component-start" || kind === "component-end")
                    picked.add(itemKey(["component-end", id]));
            }
            if (!picked.size)
                return null;
            after = Math.max(...[...picked].map(key => this.visual.indexOf(key))) + 1;
        }
        const following = after < this.visual.length ? this.flatOrder(this.visual.slice(after)) : [];
        return following.length ? parseKey(following[0]) : null;
    }
    /** v2 _request_insert: Add period / Add spacer where the selection says; the new card is picked once it arrives. */
    insert(kind) {
        if (this.readonlySelection())
            return;
        this.expectNew = true;
        const before = this.beforeItem();
        const edit = kind === "period" ? "period_add" : "spacer_add";
        const owner = this.owner === "pulse" ? this.openOwner() : null;
        if (owner) {
            const inside = before && (this.blocks.get(owner.component_id) ?? []).includes(itemKey(before)) ? before : null;
            this.send({ action: "pulse_subpulse", op: "edit", component_id: owner.component_id, edit, before_item: inside }, kind === "period" ? "Adding a period…" : "Adding a spacer…");
            return;
        }
        this.send(editCommand(this.owner, edit, { before_item: before }), kind === "period" ? "Adding a period…" : "Adding a spacer…");
    }
    /** v2 _request_add_bracket: around the picked periods, around a picked
     * bracket, empty in a picked gap, or around the whole pulse. */
    addBracket() {
        const schedule = this.schedule;
        if (this.readonlySelection() || !schedule || !schedule.periods.length)
            return;
        const ids = schedule.periods.map(period => period.period_id);
        const payload = this.payload();
        const flat = this.flatOrder(payload).map(parseKey);
        const count = schedule.default_bracket_count;
        let start, end;
        const lone = flat.length === 1 && flat[0][0] === "bracket" ? schedule.brackets.find(b => b.bracket_id === postBracket(flat[0][1])) : undefined;
        if (lone)
            [start, end] = [lone.start_period_id, lone.end_period_id];
        else if (payload.length) {
            const chosen = new Set(flat.filter(item => item[0] === "period").map(item => item[1]));
            const positions = ids.map((id, index) => chosen.has(id) ? index : -1).filter(index => index >= 0);
            if (!positions.length || positions.length !== positions.at(-1) - positions[0] + 1) {
                this.say("Select a contiguous set of periods for the bracket; unselected periods are not added.");
                return;
            }
            for (const [kind, key] of flat) {
                if (kind !== "bracket")
                    continue;
                const bracket = schedule.brackets.find(b => b.bracket_id === postBracket(key));
                if (!bracket)
                    continue;
                // bracket_gap_bounds: a missing start is after the last period, a missing end before the first.
                const first = bracket.start_period_id === null ? ids.length : ids.indexOf(bracket.start_period_id);
                const stop = bracket.end_period_id === null ? 0 : ids.indexOf(bracket.end_period_id) + 1;
                if (!ids.slice(first, stop).every(id => chosen.has(id))) {
                    this.say("The selected bracket extends beyond the selected periods.");
                    return;
                }
            }
            [start, end] = [ids[positions[0]], ids[positions.at(-1)]];
        }
        else if (this.gap !== null) {
            const before = this.flatOrder(this.visual.slice(0, this.gap)).filter(key => key.startsWith("period|")).length;
            [start, end] = [before < ids.length ? ids[before] : null, before > 0 ? ids[before - 1] : null];
        }
        else
            [start, end] = [ids[0], ids.at(-1)];
        const owner = this.owner === "pulse" ? this.openOwner() : null;
        if (owner) {
            this.send({ action: "pulse_subpulse", op: "edit", component_id: owner.component_id, edit: "bracket_add",
                start: start !== null && owner.period_ids.includes(start) ? start : null, end: end !== null && owner.period_ids.includes(end) ? end : null, count }, "Adding a bracket…");
            return;
        }
        this.send(editCommand(this.owner, "bracket_add", { start, end, count }), "Adding a bracket…");
    }
    /** v2 _request_remove_items: one atomic removal of what is picked. */
    remove() {
        if (this.readonlySelection())
            return;
        const items = this.payload();
        if (items.length)
            this.send(editCommand(this.owner, "remove_items", { items: items.map(parseKey) }), "Removing…");
    }
    /** v2 _request_group: the picked items become one Subpulse. */
    group() {
        if (this.readonlySelection())
            return false;
        const items = this.payload();
        if (!items.length) {
            this.say("Select the timeline items to group; Shift-click adds or removes one item.");
            return false;
        }
        this.send({ action: "pulse_subpulse", op: "group", items: items.map(parseKey) }, "Grouping…");
        return true;
    }
    /** v2 _request_hide_off_ports: show only the outputs some period drives. */
    hideOff() {
        const schedule = this.schedule;
        if (!schedule)
            return;
        const driven = new Set();
        for (const period of schedule.periods) {
            for (const [key, on] of period.digital)
                if (on)
                    driven.add(key);
            for (const [key, , field] of period.analog)
                if (field.text.trim())
                    driven.add(key);
        }
        if (!driven.size) {
            this.say("Nothing is driven yet, so there is nothing to hide.");
            return;
        }
        this.send(editCommand(this.owner, "visible_ports", { ports: schedule.ports.filter(port => driven.has(port.key)).map(port => port.key) }), "Hiding the off outputs…");
    }
    /** v2 _request_show_all_ports. */
    showAll() {
        if (this.schedule)
            this.send(editCommand(this.owner, "visible_ports", { ports: this.schedule.ports.map(port => port.key) }), "Showing every output…");
    }
    /** v2 _component_action("expand"): open picks the block, closing picks the Subpulse. */
    expand(id, open) {
        if (open)
            this.opened.add(id);
        else
            this.opened.delete(id);
        if (this.last && this.schedule)
            this.update(this.last, this.schedule, this.order);
        this.show(open ? [...(this.blocks.get(id) ?? [])] : [itemKey(["component", id])]);
    }
    /** v2 focus_component_name: the new block's closed card, picked, its name ready to type. */
    focusComponent(id) {
        const card = this.components.get(`closed|${id}`);
        if (!card)
            return;
        card.root.scrollIntoView({ block: "nearest", inline: "nearest" });
        this.show([itemKey(["component", id])]);
        card.focusName();
    }
}
/** The Edit page. */
export class EditPage {
    send;
    say;
    root = element("div", "pulse-edit");
    strip;
    onPulse;
    stopPulse;
    remove;
    addPeriod;
    addSpacer;
    addBracket;
    addSubpulse;
    save;
    open;
    group;
    sync;
    hide;
    connection;
    endpoint = element("input");
    connect;
    connectionStatus = new Readout();
    endpointShown = null;
    hidden;
    addPort;
    hideOff;
    showAll;
    state = null;
    constructor(send, say, focusEdit) {
        this.send = send;
        this.say = say;
        this.strip = new Strip(send, "pulse", say);
        this.onPulse = command("On Pulse", () => send({ action: "pulse_fire" }, "Starting…"), { words: ["On Pulse", "On Pulse*"] });
        this.stopPulse = command("Stop Pulse", () => send({ action: "pulse_stop" }, "Stopping…"), { role: "destructive" });
        this.remove = command("Remove", () => this.strip.remove(), { role: "destructive" });
        this.addPeriod = command("Add period", () => this.strip.insert("period"));
        this.addSpacer = command("Add spacer", () => this.strip.insert("spacer"));
        this.addBracket = command("Add bracket", () => this.strip.addBracket());
        this.addSubpulse = command("Add subpulse", () => { this.expectComponent(); send({ action: "pulse_subpulse", op: "insert_file", before_item: this.strip.beforeItem() }, "Adding a subpulse…"); });
        this.save = command("Save Pulse", () => send({ action: "pulse_save" }, "Saving…"));
        this.open = command("Open…", () => this.openPulse());
        this.group = command("Group Subpulse", () => { if (this.strip.group())
            this.expectComponent(); });
        this.sync = command("Sync", () => send({ action: "pulse_sync" }, "Syncing…"));
        this.hide = command("Hide", () => this.strip.setSidePanels(!this.strip.sidePanels), { words: ["Hide", "Show"], title: "Hide side panels" });
        // v2 _toggle_left_panels: the button says the next press, held down while they are hidden.
        this.strip.onSidePanels = shown => { this.hide.text = shown ? "Hide" : "Show"; this.hide.title = shown ? "Hide side panels" : "Show side panels"; this.hide.setActive(!shown); };
        this.connection = new ComboBox(() => this.showEndpoint());
        this.connection.root.title = "Connection choices are supplied by the host that opened this editor.";
        this.endpoint.type = "text";
        this.endpoint.spellcheck = false;
        this.endpoint.title = "Endpoint for a connection choice that accepts one, as host:port.";
        this.connect = command("Connect", () => send({ action: "pulse_connect", mode: this.connection.value, endpoint: this.endpoint.value }, "Connecting…"));
        this.hidden = new ComboBox(() => { }, "All ports shown");
        this.addPort = command("Add", () => {
            const key = this.hidden.value;
            const ports = this.state?.schedule.ports ?? [];
            if (typeof key === "string")
                send({ action: "pulse_edit", edit: "visible_ports", ports: [...ports.filter(port => port.visible).map(port => port.key), key] }, "Showing the output…");
        });
        this.hideOff = command("Hide off", () => this.strip.hideOff(), { title: "Hide the ports no period drives" });
        this.showAll = command("Show all", () => this.strip.showAll());
        this.focusEdit = focusEdit;
        const control = element("section", "card bordered pulse-control");
        control.append(element("div", "heading", "Control"));
        const grid = element("div", "pulse-control-grid");
        // v2 control grid: On Pulse spans two cells; Sync and Hide share the last.
        const pair = element("div", "pulse-control-pair");
        pair.append(this.sync.root, this.hide.root);
        grid.append(this.onPulse.root, this.stopPulse.root, this.remove.root, this.addPeriod.root, this.addSpacer.root, this.addBracket.root, this.addSubpulse.root, this.save.root, this.open.root, this.group.root, pair);
        control.append(grid);
        const connection = element("section", "card bordered pulse-connection");
        const endpointRow = element("div", "pulse-inline");
        endpointRow.append(this.endpoint, this.connect.root);
        this.connectionStatus.root.classList.add("ink");
        connection.append(element("div", "heading", "Connection"), this.connection.root, endpointRow, this.connectionStatus.root);
        const ports = element("section", "card bordered pulse-port-card");
        const portRow = element("div", "pulse-inline pulse-port-commands");
        portRow.append(this.addPort.root, this.hideOff.root, this.showAll.root);
        // How many ports the combo above holds (v2 hidden_count_label).
        this.hiddenCount.root.classList.add("ink");
        ports.append(element("div", "heading", "Ports"), this.hidden.root, portRow, this.hiddenCount.root);
        const bottom = element("div", "pulse-bottom");
        bottom.append(control, connection, ports);
        this.root.append(this.strip.root, bottom);
    }
    focusEdit;
    hiddenCount = new Readout();
    /** Components before a Group / Add subpulse: the one it makes gets its name focused (v2 focus_component_name). */
    componentsBefore = null;
    expectComponent() {
        this.componentsBefore = { ids: new Set(this.state?.schedule.components.map(c => c.component_id) ?? []), message: this.state?.message?.id ?? 0 };
    }
    openPulse() {
        // Open asks first when the draft would be lost (v2 confirm_pulse_discard).
        const dirty = this.state?.dirty ?? false;
        if (!dirty) {
            this.send({ action: "pulse_open" }, "Opening…");
            return;
        }
        void confirmPulseDiscard().then(discard => { if (discard)
            this.send({ action: "pulse_open", discard_changes: true }, "Opening…"); });
    }
    showEndpoint() {
        const state = this.state;
        if (!state)
            return;
        const choice = state.connection.choices.find(c => c.value === this.connection.value);
        this.endpoint.disabled = !choice?.endpoint_editable || state.connection.locked;
    }
    showCommands() {
        const state = this.state;
        if (!state)
            return;
        const control = state.control;
        this.onPulse.disabled = !control.can_run;
        // set_control_state: the star (and the one primary face) only while On Pulse
        // would put something new on the board and the file is saved.
        const playing = control.running && control.synchronized;
        this.onPulse.setDirty(control.can_run && !state.dirty && !playing);
        this.onPulse.root.title = playing ? "The board is playing this pulse" : "The board is not playing what is on screen; On Pulse puts it there";
        this.stopPulse.disabled = !control.can_stop;
        // v2 gates only On Pulse, Stop, Sync and Add (a port to show); every other
        // command stays available and answers for itself (nothing picked, no pulse, busy).
        this.save.setDirty(state.has_pulse && state.dirty);
        this.sync.disabled = !control.can_sync;
    }
    update(state) {
        this.state = state;
        this.strip.update(state, state.schedule, state.item_order);
        const connection = state.connection;
        this.connection.setChoices(connection.choices.map(choice => [choice.value, choice.label]));
        if (document.activeElement !== this.connection.root)
            this.connection.value = connection.selected;
        this.connection.disabled = connection.locked;
        // The editor projects the endpoint (v2: the window seeds no copy of its
        // own); what the operator typed stands until the projection changes.
        if (connection.endpoint !== this.endpointShown) {
            this.endpointShown = connection.endpoint;
            this.endpoint.value = connection.endpoint;
        }
        // v2 _sync_endpoint_enabled: Connect is the operator's whenever the connection is.
        this.connect.disabled = connection.locked;
        this.showEndpoint();
        this.connectionStatus.text = connection.status;
        setState(this.connectionStatus.root, connection.status_word);
        const hidden = state.schedule.ports.filter(port => !port.visible);
        this.hidden.setChoices(hidden.map(port => [port.key, port.label]));
        if (!hidden.length)
            this.hidden.value = null;
        else if (this.hidden.value === null || !hidden.some(port => port.key === this.hidden.value))
            this.hidden.value = hidden[0].key;
        this.hidden.disabled = !hidden.length;
        this.addPort.disabled = !hidden.length;
        this.hiddenCount.text = hidden.length ? `${hidden.length} ${hidden.length === 1 ? "port" : "ports"} hidden` : "";
        this.showCommands();
        // v2 focus_component_name, after Group / Add subpulse: on Edit, the new block's name ready to type.
        const before = this.componentsBefore;
        if (before) {
            const made = state.schedule.components.filter(c => !before.ids.has(c.component_id));
            if (made.length === 1) {
                this.componentsBefore = null;
                this.focusEdit();
                this.strip.focusComponent(made[0].component_id);
            }
            else if ((state.message?.id ?? 0) > before.message)
                this.componentsBefore = null;
        }
    }
}
export { statusDot };
