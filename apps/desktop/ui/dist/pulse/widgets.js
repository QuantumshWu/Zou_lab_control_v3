// The Pulse Editor's own controls, built from the fluent family: a text box
// that keeps the operator's draft (zlc_ui FluentLineEdit), and a value box
// with its binding badge and popup (FluentScanLineEdit): the badge says
// where the next run's value comes from -- the Pulse default, a Scan slot,
// an API caller or the Config file -- and opens the choice.
import { EditableCombo, Segmented, SpinBox, assign, checked, confirm, element, glyphFont, inkText, metrics, pointPx, setChecked, setText, setTitle, switchButton, token } from "../fluent/index.js";
/** v2 confirm_pulse_discard: the one question Open, Clear all and a recorded pulse ask before the draft goes. */
export const confirmPulseDiscard = () => confirm("Unsaved Pulse", "Discard unsaved Pulse edits?", { confirmText: "Discard" });
/** What a page last drew from: a state keeps the identity of every part that
 * did not change (transport.ts), so a page whose parts are the same objects
 * has nothing to redraw (a page switch or a strip edit redrew all six). */
export class Drawn {
    inputs = [];
    same(...inputs) {
        const same = inputs.length === this.inputs.length && inputs.every((input, at) => input === this.inputs[at]);
        this.inputs = inputs;
        return same;
    }
}
/** One authoring edit, addressed to its owner. */
export function editCommand(owner, edit, args = {}) {
    if (owner === "pulse")
        return { action: "pulse_edit", edit, ...args };
    return { action: "pulse_subpulse", op: "edit", edit, ...args };
}
const TYPED = { int: /^[+-]?\d*$/, float: /^[+-]?\d*\.?\d*(?:[eE][+-]?\d*)?$/ };
/** format_compact_number: `%.12g`, its exponent without "+0". */
function compactNumber(value) {
    if (!Number.isFinite(value) || value === 0)
        return String(value === 0 ? 0 : value);
    const exponent = Math.floor(Math.log10(Math.abs(value)));
    if (exponent < -4 || exponent >= 12) {
        const [mantissa, power] = value.toExponential(11).split("e");
        const digits = mantissa.includes(".") ? mantissa.replace(/0+$/, "").replace(/\.$/, "") : mantissa;
        const sign = power.startsWith("-") ? "-" : "";
        const size = power.replace(/^[+-]/, "");
        return `${digits}e${sign}${sign ? size.padStart(2, "0") : size}`;
    }
    const text = value.toPrecision(12);
    return text.includes(".") ? text.replace(/0+$/, "").replace(/\.$/, "") : text;
}
/** A one-line box showing what the editor says, until the operator types:
 * then their draft stands until they commit it (Enter / leave).  A number
 * box refuses other characters at the keystroke and corrects itself when
 * committed (clamped to its range, snapped to its step). */
export class TextBox {
    commit;
    root = element("input");
    draft = false;
    shown = "";
    numeric = null;
    constructor(commit, className = "") {
        this.commit = commit;
        this.root.type = "text";
        this.root.spellcheck = false;
        if (className)
            this.root.className = className;
        this.root.addEventListener("beforeinput", event => {
            const typed = this.numeric && event.data !== null && /^insert/.test(event.inputType) ? TYPED[this.numeric.kind] : null;
            if (!typed)
                return;
            const { value, selectionStart: start, selectionEnd: end } = this.root;
            if (!typed.test(value.slice(0, start ?? value.length) + event.data + value.slice(end ?? value.length)))
                event.preventDefault();
        });
        this.root.addEventListener("input", () => { this.draft = true; });
        this.root.addEventListener("keydown", event => { if (event.key === "Enter")
            this.finish(); });
        this.root.addEventListener("blur", () => this.finish());
    }
    /** v2 _clamp_to_numeric_bounds, then _snap_to_resolution. */
    correct() {
        const rule = this.numeric;
        const text = this.root.value.trim();
        if (!rule || !text)
            return;
        const value = Number(text);
        if (!Number.isFinite(value))
            return;
        let corrected = text;
        const clamped = Math.min(rule.hi ?? Infinity, Math.max(rule.lo ?? -Infinity, value));
        if (clamped !== value)
            corrected = rule.kind === "int" ? String(Math.round(clamped)) : compactNumber(clamped);
        if (rule.resolution > 0) {
            let snapped = Math.round(Number(corrected) / rule.resolution) * rule.resolution;
            if (!rule.allowAny && snapped <= 0)
                snapped = rule.resolution;
            corrected = compactNumber(snapped);
        }
        if (corrected !== this.root.value)
            this.root.value = corrected;
    }
    finish() {
        if (!this.draft)
            return;
        this.draft = false;
        this.correct();
        if (this.root.value !== this.shown)
            this.commit(this.root.value);
    }
    show(text, editable = true) {
        this.shown = text;
        if (!this.draft && this.root.value !== text)
            this.root.value = text;
        if (this.root.disabled !== !editable)
            this.root.disabled = !editable;
    }
}
/** A repeat count (v2 FluentSpinBox, minimum 0, 0 written ∞): typed and
 * committed like a text box, stepped by its arrows, the wheel once focused,
 * or ↑/↓. */
export class RepeatsBox {
    root = element("div", "pulse-repeats");
    box;
    spin;
    constructor(commit) {
        const read = (text) => {
            const value = text.trim() === "∞" ? 0 : Number(text);
            return Number.isInteger(value) && value >= 0 ? value : null;
        };
        this.box = new TextBox(text => { const value = read(text); if (value !== null)
            commit(value); });
        this.box.root.inputMode = "numeric";
        this.box.numeric = { kind: "int", lo: 0, hi: null, resolution: 0, allowAny: true };
        this.root.append(this.box.root);
        this.spin = new SpinBox(this.root, this.box.root, { step: () => 1, bounds: () => [0, null], commit,
            special: { value: 0, text: "∞" } });
    }
    show(repeats, editable) {
        this.box.show(repeats === 0 ? "∞" : String(repeats), editable);
        this.spin.refresh();
    }
}
/** The badge column (v2 _BindingButton): S scan, then A api or C config,
 * one above the other; a lone placeholder dot for the Pulse default. */
function badges(field) {
    const column = [];
    if (field.scan)
        column.push(["S", "scan"]);
    if (field.source !== "default")
        column.push([field.source === "api" ? "A" : "C", field.source]);
    return column.length ? column : [["", "default"]];
}
const SVG = "http://www.w3.org/2000/svg";
/** v2 _BindingButton.paintEvent, drawn in this window's device pixels (the
 * SVG's user unit is one device px): the column is the frame's inner height
 * and a badge's width; each badge its full diameter where one fits, two
 * sharing the height with a clear logical px above, between and below, the
 * stack in the inner height's parity (a size or the space between one px
 * smaller or larger) so it is centred on whole pixels; every letter Segoe UI
 * Bold max(6, font pt - 5), its INK centred on its badge by a translation
 * snapped to device px (paint_centred_ink). `button` sits on the frame's
 * inner rect (.pulse-badge). */
export function drawBadges(button, column) {
    const dpr = devicePixelRatio;
    const signature = JSON.stringify([column, dpr]);
    if (button.dataset.column === signature)
        return;
    button.dataset.column = signature;
    const { qtDpr, fontPt, scaled, even } = metrics();
    const diameter = even(16, 12);
    // The frame's inner rect (control_height less its edge each side) and the badge's width, in device px.
    const room = (scaled(30, 22) - 2) * qtDpr, across = diameter * qtDpr;
    const count = column.length;
    let between = Math.max(1, Math.round(scaled(1) * qtDpr));
    let size = Math.min(diameter * qtDpr, Math.floor((room - (count + 1) * between) / count));
    if ((room - count * size - (count - 1) * between) & 1) {
        if (count & 1)
            size -= 1;
        else
            between += 1;
    }
    size = Math.max(2, size);
    const stack = count * size + (count - 1) * between;
    const left = Math.floor((across - size) / 2);
    const font = glyphFont(pointPx(Math.max(6, fontPt - 5)), 700);
    const svg = document.createElementNS(SVG, "svg");
    svg.setAttribute("viewBox", `0 0 ${across} ${room}`);
    svg.setAttribute("width", `${across / dpr}`);
    svg.setAttribute("height", `${room / dpr}`);
    column.forEach(([letter, source], index) => {
        const top = Math.floor((room - stack) / 2) + index * (size + between);
        const circle = document.createElementNS(SVG, "circle");
        circle.setAttribute("cx", `${left + size / 2}`);
        circle.setAttribute("cy", `${top + size / 2}`);
        circle.setAttribute("r", `${size / 2}`);
        circle.dataset.source = source;
        svg.append(circle);
        if (!letter)
            return;
        const text = inkText(letter, font, left + size / 2, top + size / 2);
        svg.append(text);
    });
    button.replaceChildren(svg);
}
/** The field's colours (v2 FluentScanLineEdit.set_binding): config green
 * (tinted while the file supplies it), api violet (filled once a caller set
 * the value), scan orange; none for a plain or read-only field. */
function tone(field, editable) {
    if (!editable && !field.config_active)
        return "";
    if (field.config_active)
        return "config-applied";
    if (field.source === "config")
        return "config";
    if (field.source === "api")
        return field.effective_text ? "api-applied" : "api";
    return field.scan ? "scan" : "";
}
/** The edit arguments naming the field a binding is about (from its id). */
export function bindingTarget(fieldId) {
    const [kind, first, second] = fieldId.split(":");
    if (kind === "duration")
        return { field_kind: "duration", period_id: first };
    if (kind === "dac")
        return { field_kind: "analog", period_id: first, port_key: second };
    if (kind === "delay")
        return { field_kind: "delay", port_key: first };
    return { field_kind: "bracket_count", bracket_id: first };
}
/** Where a value comes from, in the popup's order and words (v2 _SOURCES). */
const SOURCES = [["default", "Default"], ["api", "API"], ["config", "Config"]];
/** v2 FluentScanLineEdit's popup: Scan (a switch), Source (Default / API /
 * Config), the Config name (choose or type), the Default and Effective
 * values, and a two-line note.  A choice is asked of the owner and the popup
 * stays open, showing what the owner accepted. */
class BindingPopup {
    send;
    owner;
    field;
    shownText;
    root = element("div", "card bordered pulse-binding");
    scan;
    source;
    name;
    defaultValue = element("span", "pulse-binding-value elided");
    effectiveValue = element("span", "pulse-binding-value elided");
    note = element("p", "pulse-binding-note");
    constructor(send, owner, field, shownText) {
        this.send = send;
        this.owner = owner;
        this.field = field;
        this.shownText = shownText;
        this.root.popover = "auto";
        this.scan = switchButton(on => this.commit(on, this.field.source));
        // A click on a source chooses it; the source already chosen asks nothing.
        this.source = new Segmented(value => { if (value !== this.field.source)
            this.commit(checked(this.scan), String(value)); });
        this.source.setChoices(SOURCES);
        this.name = new EditableCombo(key => {
            if (this.field.source !== "config" || key === this.field.config_key)
                return;
            const named = owner === "pulse" ? { action: "pulse_config", op: "binding", field_id: this.field.field_id, key }
                : editCommand("subpulse", "config_binding", { field_id: this.field.field_id, key });
            send(named, "Naming the Config value…");
        }, "Choose or type a name");
        const label = (text) => element("span", "pulse-label", text);
        this.root.append(label("Scan"), this.scan, label("Source"), this.source.root, label("Config name"), this.name.root, label("Default"), this.defaultValue, label("Effective"), this.effectiveValue, this.note);
        this.root.addEventListener("toggle", event => { if (event.newState === "closed")
            this.root.remove(); });
        this.project(field);
    }
    get open() { return this.root.isConnected && this.root.matches(":popover-open"); }
    commit(scan, source) {
        this.send(editCommand(this.owner, "binding", { ...bindingTarget(this.field.field_id), scan, source }), "Changing the field's source…");
    }
    /** v2 _project_popup: what the owner holds for the field now. */
    project(field) {
        this.field = field;
        setChecked(this.scan, field.scan);
        this.scan.disabled = !(field.can_scan && field.binding_editable);
        this.source.setCurrent(field.source);
        this.source.setWithheld("api", !field.can_api);
        this.source.disabled = !field.binding_editable;
        const named = field.source === "config";
        // Under Default or API a kept name still shows, disabled: it is the name Config switches back to.
        this.name.fill([...new Set(field.config_names)], field.config_key, named);
        this.name.disabled = !(named && field.binding_editable);
        this.name.title = field.config_key;
        let note = field.source_text;
        if (!note) {
            note = named ? (field.effective_text ? "The saved Config supplies this value." : !field.config_key ? "Config name is unassigned; using the Pulse default."
                : "That name is not supplied by Config; using the Pulse default.")
                : field.source === "api" ? "An API caller may override it; otherwise the Pulse default is used." : "Using the Pulse default.";
        }
        const text = this.shownText();
        const fallback = field.default_text || text;
        const effective = field.config_active ? (field.effective_text || text)
            : field.source !== "config" && field.effective_text ? field.effective_text : `${fallback} · default`;
        setText(this.defaultValue, fallback);
        setTitle(this.defaultValue, fallback);
        setText(this.effectiveValue, effective);
        setTitle(this.effectiveValue, effective);
        if (named && !field.config_enabled)
            note = "Default mode; the saved Config reference is not applied.";
        setText(this.note, note);
        setTitle(this.note, note);
    }
    toggleBelow(anchor) {
        if (this.open) {
            this.root.hidePopover();
            return;
        }
        document.body.append(this.root);
        this.root.showPopover();
        const rect = anchor.getBoundingClientRect(), gap = token("--popup-gap"), pad = token("--window-pad");
        const left = Math.max(pad, Math.min(rect.left, innerWidth - pad - this.root.offsetWidth));
        const below = rect.bottom + gap;
        this.root.style.left = `${left}px`;
        this.root.style.top = `${below + this.root.offsetHeight <= innerHeight - pad ? below : Math.max(pad, rect.top - gap - this.root.offsetHeight)}px`;
    }
}
/** A value box with its binding badge inside, on the right (FluentScanLineEdit). */
export class ValueBox {
    send;
    owner;
    root = element("div", "pulse-value");
    box;
    badge = element("button", "pulse-badge");
    field = null;
    popup = null;
    /** What the badge's tooltip says first (v2 FluentScanLineEdit's `tooltip`). */
    tip = "Edit Scan and value source";
    constructor(send, commit, owner = "pulse") {
        this.send = send;
        this.owner = owner;
        this.box = new TextBox(commit);
        this.badge.type = "button";
        this.badge.tabIndex = -1;
        // A press while the popup is open closes it (light dismiss would close it first and the click open it again).
        let wasOpen = false;
        this.badge.addEventListener("pointerdown", () => { wasOpen = Boolean(this.popup?.open); });
        this.badge.addEventListener("click", () => {
            if (!this.field)
                return;
            if (wasOpen) {
                wasOpen = false;
                return;
            }
            this.popup = new BindingPopup(this.send, this.owner, this.field, () => this.box.root.value);
            this.popup.toggleBelow(this.badge);
        });
        this.root.append(this.box.root, this.badge);
    }
    show(field, editable = true) {
        this.field = field;
        const writable = editable && field.editable;
        // v2 _apply_field: the field's own number rule (an int's top too; a float is bounded below).
        const kind = field.validator_kind;
        this.box.numeric = kind === "none" ? null : { kind, lo: Number.isFinite(field.validator_lo) ? field.validator_lo : null,
            hi: kind === "int" && Number.isFinite(field.validator_hi) ? field.validator_hi : null, resolution: field.resolution || 0, allowAny: field.allow_any };
        // v2 set_field_state: read-only while Config supplies it, and still not greyed.
        this.box.show(field.text, writable || field.config_active);
        assign(this.box.root, "readOnly", !writable || field.config_active);
        drawBadges(this.badge, badges(field));
        // A read-only card's binding is not editable either (v2 _read_only).
        const bindable = field.binding_editable && editable;
        assign(this.badge, "disabled", !(bindable && (writable || field.config_active)));
        if (this.popup?.open) {
            if (this.badge.disabled || (!writable && !field.config_active))
                this.popup.root.hidePopover();
            else
                this.popup.project(field);
        }
        const shade = tone(field, writable);
        if (shade)
            this.root.dataset.tone = shade;
        else
            delete this.root.dataset.tone;
        // v2: the tooltip, then the binding said in words, then what the owner says of it.
        const named = field.source === "config" && field.config_key ? `Config '${field.config_key}'` : SOURCES.find(([value]) => value === field.source)[1];
        const summary = [...(field.scan ? ["Scan"] : []), ...(field.source !== "default" ? [named] : [])].join(" + ") || "Default";
        setTitle(this.badge, `${this.tip}\n${summary}\n${field.source_text}`.trim());
    }
}
