import { assign, element, labelColumnWidth, setText, setTitle, token } from "./dom.js";
import { fontOf, measurer, setMiddle, wholeText } from "./elide.js";
import { comboFace, hintWidest } from "./combo.js";
import { SpinBox, approximateText, decimals } from "./spin.js";
import { TreeCombo } from "./tree_combo.js";
import { Segmented, setChecked, switchButton } from "./toggles.js";
import { PathEdit } from "./info_pane.js";
const encode = (value) => JSON.stringify(value);
/** Units set in fixed notches, by spelling: the notch. */
const NATURAL_STEPS = { dB: 0.1, dBm: 0.1, count: 1 };
/** A device number's first step: its field's own step, else its unit's natural
 * notch in the spelling shown, else the last significant digit of its value as
 * shown (812.5 MHz: 0.1 MHz; "≈ 632.5" mVpp: 0.1), else 1. The operator changes
 * it on the step chip. */
export function firstStep(spec) {
    if (spec.step)
        return spec.step;
    if (spec.kind === "integer")
        return 1;
    const natural = NATURAL_STEPS[spec.unit ?? ""];
    if (natural)
        return natural;
    // Twelve significant digits: a reading converted between units carries no float dust into its step;
    // an inexact conversion's carries the digits its ≈ rounding shows, not the ones it hides.
    const shown = typeof spec.value !== "number" ? NaN : spec.inexact ? Number(approximateText(String(spec.value), spec.unit ?? "", null).replace(/^≈ /, ""))
        : Number(spec.value.toPrecision(12));
    return Number.isFinite(shown) && shown !== 0 ? 10 ** -decimals(shown) : 1;
}
/** One shape for both owners: a workbench field says `required` and
 * `blank_label` (v2 FormFieldProps), a panel control `allow_none` and
 * `placeholder`; each is read as the other. */
export function normalField(spec) {
    // A switch has no blank: only a field that says allow_none itself is three-state.
    const allow = spec.allow_none ?? (spec.required === undefined || spec.kind === "boolean" ? undefined : !spec.required);
    // A blank says its blank_label; with none, the box's own words (showValue).
    const placeholder = spec.placeholder ?? spec.blank_label ?? (spec.required === undefined ? undefined : "");
    return allow === spec.allow_none && placeholder === spec.placeholder ? spec : { ...spec, allow_none: allow, placeholder };
}
/** Which editor a spec needs; a field whose kind changes is rebuilt. */
export function fieldKind(raw) {
    const spec = normalField(raw);
    return `${spec.kind}/${Boolean(spec.allow_none)}/${Boolean(spec.segmented)}/${Boolean(spec.automatic)}/${Boolean(spec.units?.length)}/${Boolean(spec.tree)}`;
}
/** One mounted form row (FluentParameterForm row): label, then the control
 * cell (Auto switch, editor, unit picker).  Accepted projections never
 * overwrite a newer local draft; a request completes it. */
export class Field {
    handlers;
    marked;
    root = element("div", "field");
    label = element("label", "field-label");
    cell = element("div", "field-control");
    problem = element("p", "field-error");
    input;
    cycleActive = false;
    cycleWheel = 0;
    automatic = null;
    tree = null;
    unitPicker = null;
    spin = null;
    path = null;
    /** Its step where the field declares none (1, or `firstStep` for a device's number). */
    firstStep = 1;
    spec;
    options = "";
    face = null;
    version = 0;
    dirty = false;
    authoredValue;
    pending = null;
    /** `marked`: a required row says so with ` *` (zlc_ui required_label). */
    constructor(id, spec, handlers, marked = true) {
        this.handlers = handlers;
        this.marked = marked;
        spec = normalField(spec);
        this.spec = spec;
        this.authoredValue = spec.value;
        this.root.dataset.field = spec.name;
        const automatic = Boolean(spec.allow_none && spec.automatic && spec.kind !== "boolean");
        if (automatic) {
            // An Auto row: the switch that says whether Auto holds the value is the
            // first cell of the control column, the editor it governs after it.
            this.automatic = switchButton(on => this.toggleAutomatic(on), ["Auto", "Manual"]);
            this.automatic.setAttribute("aria-label", `${spec.label}: automatic`);
            this.cell.append(this.automatic);
        }
        if (spec.kind === "boolean" && !spec.allow_none) {
            const toggle = switchButton(on => this.submit(on));
            this.input = toggle;
            this.cell.append(toggle);
        }
        else if (spec.tree) {
            this.tree = new TreeCombo(key => this.submit(key === "" ? null : key), spec.allow_none ? "" : spec.placeholder ?? "");
            this.input = this.tree.root;
            this.cell.append(this.input);
        }
        else if (spec.kind === "choice" && spec.segmented && !spec.cycle && !automatic) {
            const segments = new Segmented(value => this.submit(value));
            this.input = segments;
            this.cell.append(segments.root);
        }
        else if (spec.kind === "choice" || spec.kind === "boolean") {
            this.input = this.buildSelect();
            this.cell.append(this.input);
        }
        else if (spec.kind === "path") {
            // FluentPathEdit: a typed path, Browse… (the owner's dialog) and, for a file whose content is read, Refresh.
            const path = this.path = new PathEdit(text => this.submit(text === "" && this.spec.allow_none ? null : text), () => this.browse(), () => this.handlers.refresh?.(this.spec), Boolean(spec.refreshable));
            path.browse.hidden = !handlers.browse;
            this.input = path.input;
            this.cell.append(path.root);
        }
        else {
            const edit = element("input");
            edit.type = "text";
            edit.spellcheck = false;
            if (spec.kind === "integer" || spec.kind === "number")
                edit.inputMode = spec.kind === "integer" ? "numeric" : "decimal";
            // A number (or a pair of them) is a number's width, never the row's (qt_form _CAPPED_KINDS).
            if (spec.kind === "integer" || spec.kind === "number" || spec.kind === "pair") {
                edit.classList.add("number");
                this.cell.classList.add("capped");
            }
            edit.addEventListener("input", event => {
                this.version++;
                this.dirty = true;
                edit.classList.remove("invalid");
                edit.setCustomValidity("");
                this.handlers.edited?.((event instanceof InputEvent && event.isComposing) ? undefined : this.editedValue(edit), this.spec);
            });
            edit.addEventListener("compositionend", () => this.handlers.edited?.(this.editedValue(edit), this.spec));
            edit.addEventListener("change", () => this.commitText(edit));
            edit.addEventListener("keydown", event => {
                if (event.key === "Enter") {
                    this.commitText(edit);
                    edit.blur();
                }
                if (event.key === "Escape") {
                    this.dirty = false;
                    this.pending = null;
                    edit.classList.remove("invalid");
                    edit.setCustomValidity("");
                    this.showValue();
                    this.handlers.cancel?.();
                    edit.blur();
                    event.stopPropagation();
                }
            });
            this.input = edit;
            if ((spec.kind === "integer" || spec.kind === "number") && (!spec.allow_none || spec.spin) && !spec.lossless) {
                // A required number is a spin box; an optional one is a blank-or-number edit, a device's a spin box that may stand blank.
                const holder = element("div");
                holder.append(edit);
                if (spec.spin)
                    this.firstStep = firstStep(spec);
                this.spin = new SpinBox(holder, edit, {
                    step: () => this.spec.step ?? this.firstStep,
                    bounds: () => [this.spec.minimum ?? null, this.spec.maximum ?? null],
                    commit: value => { this.version++; this.submit(value); },
                    decimal: spec.kind === "number",
                    approximate: () => this.spec.inexact ? { unit: this.spec.unit ?? "", step: this.spec.step ?? null } : null,
                });
                this.cell.append(holder);
            }
            else
                this.cell.append(edit);
        }
        if (spec.units?.length) {
            // FluentUnitPicker: which spelling the number is read in, a tree of
            // the unit's families, as wide as its widest spelling, after the box.
            this.unitPicker = new TreeCombo(unit => {
                const id = this.handlers.unit?.(unit, this.spec);
                if (typeof id === "number")
                    this.pending = { id, version: this.version };
            });
            this.unitPicker.root.classList.add("unit-picker");
            this.unitPicker.title = "Read this field in another unit";
            this.cell.append(this.unitPicker.root);
        }
        const control = this.input instanceof Segmented ? this.input.root : this.input;
        control.id = `${id}-${spec.name}`;
        this.label.htmlFor = control.id;
        this.problem.hidden = true;
        this.root.append(this.label, this.cell, this.problem);
        this.update(spec);
    }
    get kind() { return fieldKind(this.spec); }
    get value() { return this.authoredValue; }
    get name() { return this.spec.name; }
    buildSelect() {
        const select = element("select");
        select.addEventListener("change", () => {
            const value = select.value === "@scope" ? this.spec.cycle.value : JSON.parse(select.value);
            this.submit(value);
        });
        // FluentCycleComboBox: with Scope chosen the box says "Scope: <value>";
        // a click on its words (not the arrow) arms the wheel, which then steps
        // the coordinate while the box has the focus; the arrow opens the list.
        select.addEventListener("mousedown", event => {
            if (event.button !== 0 || !this.cycleSelected())
                return;
            const rect = select.getBoundingClientRect();
            if (event.clientX >= rect.right - parseFloat(getComputedStyle(select).paddingRight))
                return;
            event.preventDefault();
            select.focus();
            this.setCycleActive(!this.cycleActive);
        });
        select.addEventListener("click", event => { if (this.cycleSelected() && document.activeElement === select && !select.matches(":open"))
            event.preventDefault(); });
        select.addEventListener("wheel", event => {
            if (!this.cycleActive || document.activeElement !== select || event.deltaY === 0)
                return;
            event.preventDefault();
            const unit = event.deltaMode === WheelEvent.DOM_DELTA_LINE ? 100 / 3 : event.deltaMode === WheelEvent.DOM_DELTA_PAGE ? 300 : 1;
            this.cycleWheel += -event.deltaY * unit;
            const notches = Math.trunc(this.cycleWheel / 100);
            if (notches) {
                this.cycleWheel -= notches * 100;
                this.stepScope(notches);
            }
        }, { passive: false });
        select.addEventListener("keydown", event => {
            if (!this.cycleActive || (event.key !== "ArrowUp" && event.key !== "ArrowDown"))
                return;
            event.preventDefault();
            this.stepScope(event.key === "ArrowUp" ? 1 : -1);
        });
        select.addEventListener("blur", () => this.setCycleActive(false));
        return select;
    }
    complete(response, failed = false) {
        if (this.pending && this.pending.id === response) {
            if (!failed && this.pending.version === this.version)
                this.dirty = false;
            this.pending = null;
            if (this.input instanceof HTMLInputElement)
                this.input.classList.toggle("invalid", failed);
        }
        this.root.classList.toggle("pending", this.pending !== null);
    }
    /** The row brought to `spec`, writing only what changed: a row the edit of another did not
     * touch is not drawn again (its value, choices, focus and words stay as they are). */
    update(spec, response, failed = false) {
        spec = normalField(spec);
        // Read in another spelling, a device's number starts again from its first step in it.
        const respelled = Boolean(spec.spin && this.spin) && spec.unit !== this.spec.unit;
        this.spec = spec;
        if (respelled) {
            this.firstStep = firstStep(spec);
            this.spin.resetStep();
        }
        // A row with no unit picker says its unit in its label; a picker says it itself (_shown_label).
        const label = spec.unit && spec.unit !== "1" && !spec.units?.length ? `${spec.label} (${spec.unit})` : spec.label;
        const words = this.marked && (spec.required ?? !spec.allow_none) ? `${label} *` : label;
        // _fit_label: a label wider than the column the form gives it keeps its start and its end.
        setMiddle(this.label, words, "none");
        setTitle(this.label, spec.description || spec.label);
        const reason = this.reason();
        setText(this.problem, spec.problem || "");
        assign(this.problem, "hidden", !spec.problem);
        if (this.input instanceof Segmented) {
            setTitle(this.input.root, reason);
            this.input.disabled = Boolean(reason);
        }
        else {
            // A path's own tip (the file it names) is its value's (showValue).
            if (reason || !this.path)
                setTitle(this.input, reason || spec.description || "");
            assign(this.input, "disabled", Boolean(reason));
        }
        if (this.path) {
            assign(this.path.browse, "disabled", Boolean(reason));
            assign(this.path.refresh, "disabled", Boolean(reason));
        }
        if (this.automatic)
            assign(this.automatic, "disabled", Boolean(reason));
        if (this.unitPicker) {
            this.unitPicker.disabled = Boolean(reason);
            this.updateUnits();
        }
        this.complete(response, failed);
        if (this.tree)
            this.tree.setTree(spec.tree ?? [], this.tree.value, spec.allow_none ? spec.placeholder || "(none)" : null);
        else if (this.input instanceof HTMLSelectElement)
            this.updateChoices();
        if (this.input instanceof Segmented)
            this.input.setChoices([...(spec.choices || [])]);
        if (!this.dirty && !this.pending)
            this.showValue();
        const cycle = spec.cycle;
        if (this.input instanceof HTMLSelectElement && cycle) {
            const option = [...this.input.options].find(option => option.value === "@scope");
            const text = cycle.index === null ? "Scope" : `Scope: ${cycle.label}`;
            if (option && option.text !== text)
                option.text = text;
            setTitle(this.input, reason || (cycle.index === null ? "" : `Scope: ${cycle.label} · click the words, then scroll or use ↑ / ↓ to change the coordinate (${cycle.count} values)`));
        }
        if (!this.cycleSelected())
            this.setCycleActive(false);
        this.spin?.refresh();
    }
    /** Why the row cannot be edited now (its tip then), or empty. */
    reason() {
        return this.spec.unavailable_reason || (this.spec.enabled === false ? "Not used with the current settings" : "");
    }
    cycleSelected() {
        const cycle = this.spec.cycle;
        return this.input instanceof HTMLSelectElement && !this.input.disabled && Boolean(cycle) && cycle.index !== null && cycle.count > 1;
    }
    setCycleActive(active) {
        this.cycleActive = active;
        this.cycleWheel = 0;
        if (this.input instanceof HTMLSelectElement)
            this.input.classList.toggle("cycle-active", active);
    }
    updateChoices() {
        const choices = this.spec.kind === "boolean" ? [[false, "Off"], [true, "On"]] : [...(this.spec.choices || [])];
        if (this.spec.allow_none && !this.automatic && !choices.some(([value]) => value === null))
            choices.unshift([null, this.spec.placeholder || "(none)"]);
        const held = this.spec.value;
        if (held !== null && held !== "" && !choices.some(([value]) => encode(value) === encode(held))) {
            choices.push([held, `${String(held)} (unavailable)`]);
        }
        const signature = JSON.stringify([choices, Boolean(this.spec.cycle)]);
        if (signature === this.options)
            return;
        this.options = signature;
        const select = this.input;
        const current = select.value;
        const options = choices.map(([value, label]) => new Option(label, encode(value)));
        if (this.spec.cycle)
            options.push(new Option("Scope", "@scope"));
        select.replaceChildren(this.face ??= comboFace(), ...options);
        select.value = current;
        hintWidest(select, options.map(option => option.text), this.spec.minimum_contents ?? 0);
    }
    updateUnits() {
        this.unitPicker.setTree(unitTree(this.spec.units || [], this.spec.unit ?? ""), this.spec.unit ?? "");
    }
    /** The owner reads the row in another spelling (FluentUnitPicker): the box shows what it
     * holds now, re-read; an edit still on its way stays on its way. */
    respell(value) {
        this.authoredValue = value;
        if (this.input instanceof HTMLInputElement)
            this.input.value = value === null ? "" : String(value);
        this.spin?.refresh();
    }
    showValue() {
        const value = this.spec.value;
        this.authoredValue = value;
        const automatic = this.automatic !== null && value === null;
        if (this.automatic) {
            setChecked(this.automatic, automatic);
            // While Auto holds the value its editor shows what Auto resolves to.
            if (!(this.input instanceof Segmented))
                assign(this.input, "disabled", automatic || Boolean(this.spec.unavailable_reason));
        }
        const shown = automatic ? (this.spec.automatic_value ?? null) : value;
        if (this.tree) {
            this.tree.value = shown === null ? "" : String(shown);
        }
        else if (this.input instanceof HTMLSelectElement) {
            const select = this.input;
            // A select told the value it already shows still draws its face again.
            assign(select, "value", this.spec.cycle?.index !== null && this.spec.cycle?.index !== undefined ? "@scope" : encode(shown));
            if (select.selectedIndex < 0 && automatic && select.options.length)
                select.selectedIndex = 0;
            select.classList.toggle("blank", value === null && !this.automatic);
        }
        else if (this.path) {
            const text = shown === null ? "" : String(shown);
            this.path.value = text;
            // v2 _show_tooltip: the path it names (a relative one in its folder), then what the field is.
            const named = text && this.spec.directory && !/^([a-z]:)?[\\/]/i.test(text) ? `${this.spec.directory}\\${text}` : text;
            if (!this.reason())
                setTitle(this.path.input, [named, this.spec.description].filter(Boolean).join("\n"));
        }
        else if (this.input instanceof HTMLInputElement) {
            const rows = this.spec.kind === "rows" ? shown : null;
            // A pair is one fact written "y, x"; the owner reads the text back.
            const text = rows ? `${rows.length} row${rows.length === 1 ? "" : "s"}` : shown === null ? "" : this.spec.text !== undefined && shown === this.spec.value ? this.spec.text
                : Array.isArray(shown) ? shown.join(", ") : String(shown);
            assign(this.input, "readOnly", this.spec.kind === "rows");
            if ((this.spin ? this.spin.text : this.input.value) !== text)
                this.input.value = text;
            // v2 qt_form: a blank-or-number box says its blank label, else "(required)" / "(optional)";
            // a text box its description (cut at 48).
            const numeric = this.spec.kind === "integer" || this.spec.kind === "number";
            const placeholder = numeric ? (this.spec.allow_none ? this.spec.placeholder || (this.spec.required ? "(required)" : "(optional)") : "")
                : this.spec.placeholder || (this.spec.description ?? "").slice(0, 48);
            if (this.input.placeholder !== placeholder)
                this.input.placeholder = placeholder;
        }
        else if (this.input instanceof Segmented) {
            this.input.setCurrent(value);
        }
        else {
            setChecked(this.input, Boolean(value));
            if (this.input.getAttribute("aria-label") !== this.spec.label)
                this.input.setAttribute("aria-label", this.spec.label);
        }
    }
    toggleAutomatic(automatic) {
        if (automatic) {
            this.submit(null);
            return;
        }
        // Manual starts from what the editor shows; an empty editor takes the
        // first choice, or the lowest number the field allows (v2 _automatic_toggled).
        let value = this.spec.automatic_value ?? null;
        if (value === null) {
            if (this.input instanceof HTMLSelectElement)
                value = this.spec.choices?.[0]?.[0] ?? null;
            else if (this.input instanceof HTMLInputElement) {
                const raw = this.input.value.trim();
                value = this.spec.kind === "text" || this.spec.kind === "path" ? raw
                    : raw && Number.isFinite(Number(raw)) ? Number(raw)
                        : this.spec.minimum ?? (this.spec.maximum != null && this.spec.maximum < 0 ? this.spec.maximum : 0);
            }
        }
        this.submit(value);
    }
    /** Browse…: the owner's native dialog, starting at the path's text; a pick is the new value. */
    browse() {
        const spec = this.spec;
        void this.handlers.browse?.({ title: `Choose ${spec.label}`, filter: spec.filter, start: this.path.value, folder: spec.directory, dir: spec.mode === "dir" })
            .then(chosen => { if (chosen !== null) {
            this.path.value = chosen;
            this.submit(chosen);
        } });
    }
    /** The owner has accepted this exact draft, or replaced its unit. */
    acceptEdit() { this.dirty = false; this.pending = null; this.root.classList.remove("pending"); }
    editedValue(input) {
        if (input.disabled || input.readOnly)
            return undefined;
        const raw = input.value.trim();
        if (this.spec.kind !== "integer" && this.spec.kind !== "number")
            return raw;
        // A trailing dot/exponent/sign is still being typed. Never clamp it into
        // a hardware command; the existing commit path owns leave-time clamping.
        const syntax = this.spec.kind === "integer" ? /^[+-]?\d+$/ : /^[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?$/;
        if (!syntax.test(raw))
            return undefined;
        const value = Number(raw);
        if (!Number.isFinite(value) || (this.spec.kind === "integer" && !Number.isInteger(value))
            || (this.spec.minimum != null && value < this.spec.minimum) || (this.spec.maximum != null && value > this.spec.maximum))
            return undefined;
        return value;
    }
    commitText(input) {
        if (!this.dirty || this.pending?.version === this.version)
            return;
        const raw = input.value.trim();
        let value = raw;
        if (raw === "" && this.spec.allow_none && !this.automatic)
            value = null;
        else if (this.spec.kind === "integer" || this.spec.kind === "number") {
            value = Number(raw);
            if (!raw || !Number.isFinite(value) || (this.spec.kind === "integer" && !Number.isInteger(value))) {
                input.classList.add("invalid");
                input.setCustomValidity(this.spec.kind === "integer" ? "Enter a whole number." : "Enter a finite number.");
                input.reportValidity();
                return;
            }
            // v2's leave-time clamp (FluentLineEdit's numeric validator): a number past a bound becomes the bound.
            if (this.spec.minimum != null && value < this.spec.minimum)
                value = this.spec.minimum;
            if (this.spec.maximum != null && value > this.spec.maximum)
                value = this.spec.maximum;
            input.value = String(value);
        }
        input.setCustomValidity("");
        this.submit(value);
    }
    submit(value) {
        if (Object.is(value, this.spec.value) && !this.pending) {
            this.dirty = false;
            this.showValue();
            this.handlers.edited?.(value, this.spec);
            return;
        }
        this.dirty = true;
        this.authoredValue = value;
        if (this.input instanceof HTMLButtonElement)
            setChecked(this.input, Boolean(value));
        if (this.input instanceof Segmented)
            this.input.setCurrent(value);
        if (this.automatic)
            setChecked(this.automatic, value === null);
        const id = this.handlers.commit(value, this.spec);
        this.pending = { id, version: this.version };
        this.root.classList.add("pending");
    }
    stepScope(steps) {
        const id = this.handlers.scope?.(steps, this.spec);
        if (id === undefined)
            return;
        this.pending = { id, version: this.version };
        this.root.classList.add("pending");
    }
}
/** v2 unit_choice_tree: a unit's spellings under the family each belongs to, its own family first. */
export function unitTree(units, unit) {
    const groups = new Map();
    for (const spelling of units) {
        const heading = spelling.group || unit;
        let group = groups.get(heading);
        if (!group)
            groups.set(heading, group = { heading, leaves: [] });
        group.leaves.push({ key: spelling.value, label: spelling.label, text: spelling.label });
    }
    return [...groups.values()];
}
// ------------------------------------------------------------- form layout
/** A form: one label column (setting_label_width) and one control column;
 * rows a row gap apart, section titles a control high, sections a section gap. */
export function form(className = "") { return element("div", `form ${className}`.trim()); }
/** FluentSectionLabel row: the semibold title, its commands at the row's end. */
export function sectionTitle(text, ...trailing) {
    const heading = element("div", "section-title");
    heading.append(element("span", "", text), ...trailing);
    return heading;
}
/** FluentSettingRow: `[label | controls flush-left]`, laid on its form's columns. */
export function row(label, ...controls) {
    const node = element("div", "field");
    const cell = element("div", "field-control");
    cell.append(...controls);
    const words = typeof label === "string" ? element("span", "field-label", label) : label;
    words.classList.add("field-label");
    node.append(words, cell);
    return node;
}
let measure = null;
/** Every unit picker under `host` as wide as the widest spelling any of them
 * offers (FluentUnitPicker is as wide as its widest spelling; DeviceControlView
 * gives its pickers one width so the number boxes share one right edge). */
export function alignUnitPickers(host) {
    const pickers = [...host.querySelectorAll("select.unit-picker")].filter(picker => picker.offsetParent && !picker.hidden);
    if (!pickers.length)
        return;
    measure ??= document.createElement("canvas").getContext("2d");
    if (!measure)
        return;
    let widest = 0;
    for (const picker of pickers) {
        picker.style.minWidth = "";
        measure.font = getComputedStyle(picker).font;
        const text = (words) => measure.measureText(words).width;
        // The picker's own chrome (padding, chevron column, edges) around the words it shows now.
        const chrome = picker.getBoundingClientRect().width - text(picker.selectedOptions[0]?.text ?? "");
        widest = Math.max(widest, chrome + Math.max(...[...picker.options].map(option => text(option.text))));
    }
    const width = `${Math.ceil(widest * devicePixelRatio) / devicePixelRatio}px`;
    for (const picker of pickers)
        picker.style.minWidth = width;
}
/** One label column for several forms (share_label_column): as wide as the
 * widest label any of them has, never below the setting label minimum. */
export function shareLabelColumn(host, floor = 0) {
    // A `.label-column` word stands in the column outside a form ("By hand");
    // words that come and go (a pulse's API values, `[data-label-room]`) get
    // the room their owner keeps for them (`floor`) and never set it.
    const advances = [...host.querySelectorAll(".form .field-label, .label-column")]
        .filter(label => wholeText(label) && label.offsetParent && !label.closest("[data-label-room]"))
        .map(label => measurer(fontOf(getComputedStyle(label)))(wholeText(label)));
    const width = advances.length ? `${labelColumnWidth(advances, Math.max(floor, token("--label-min")))}px` : "";
    if (host.style.getPropertyValue("--label-width") !== width) {
        if (width)
            host.style.setProperty("--label-width", width);
        else
            host.style.removeProperty("--label-width");
    }
}
