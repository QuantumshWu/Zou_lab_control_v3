import { prompt } from "./dialog.js";
import { assign, button, element, token } from "./dom.js";
import { glyphFace, setGlyph } from "./ink.js";
/** The decimal places `value` is written with (1e-05: 5). */
export function decimals(value) {
    const [mantissa = "", exponent] = String(value).split("e");
    const places = mantissa.split(".")[1]?.length ?? 0;
    return Math.max(0, places - Number(exponent ?? 0));
}
/** _grid_step_inside: `steps` notches from `value` on the grid of multiples
 * of `step` inside [low, high]; off the grid the first notch goes to the
 * nearest grid point in its direction.  null when no grid point lies that way. */
export function gridStep(value, step, steps, low, high) {
    if (!(step > 0) || !Number.isFinite(value) || !steps)
        return null;
    const places = Math.min(100, decimals(step));
    const ratio = value / step, nearest = Math.round(ratio);
    const onGrid = Math.abs(ratio - nearest) <= 1e-9 * Math.max(1, Math.abs(ratio));
    const index = onGrid ? nearest + steps : steps > 0 ? Math.floor(ratio) + steps : Math.ceil(ratio) + steps;
    let moved = index * step;
    if (low !== null && moved < low)
        moved = Math.ceil(low / step - 1e-9) * step;
    if (high !== null && moved > high)
        moved = Math.floor(high / step + 1e-9) * step;
    moved = Number(moved.toFixed(places));
    if ((low !== null && moved < low) || (high !== null && moved > high))
        return null;
    if ((steps < 0 && moved >= value) || (steps > 0 && moved <= value))
        return null;
    return moved;
}
/** A canvas font of an element's computed style (its `font` shorthand is empty
 * where the shorthand cannot say everything, as with tabular digits). */
const font = (style) => `${style.fontStyle} ${style.fontWeight} ${style.fontSize} ${style.fontFamily}`;
/** _visible_decimal: `text` with as many of its fractional digits as fit in
 * `room` (rounded, kept inside the bounds), else its whole number; text that
 * is not a plain decimal (an exponent, a word) is shown as written. */
function fittedDecimal(text, room, width, low, high) {
    if (width(text) <= room)
        return text;
    const plain = /^-?\d+\.(\d+)$/.exec(text.trim());
    if (!plain)
        return text;
    const value = Number(text);
    let shown = text;
    for (let places = plain[1].length - 1; places >= 0; places--) {
        let rounded = Number(value.toFixed(places));
        // The grid of this many places never crosses a bound: step inside it.
        const grain = 10 ** -places;
        if (high !== null && rounded > high)
            rounded = Number((rounded - grain).toFixed(places));
        if (low !== null && rounded < low)
            rounded = Number((rounded + grain).toFixed(places));
        shown = String(rounded);
        if (width(shown) <= room)
            return shown;
    }
    return shown;
}
/** _step_chip_text: the step as a decimal, or %g past a millionth or a million. */
function stepText(step) {
    const exponent = Math.floor(Math.log10(step));
    return `±${exponent >= -4 && exponent <= 6 ? String(Number(step.toPrecision(15))) : step.toExponential().replace(/\.?0+e/, "e")}`;
}
/** FluentDoubleSpinBox: a field with a two-cell stepper painted from its frame.
 * The wheel changes the value only once the box has the focus (otherwise the
 * page scrolls), one notch per step on the step's grid.  A decimal box shows
 * its step left of the stepper (_StepChip) -- an operator who cannot see the
 * step cannot know what the arrows do -- and a click on it asks for another;
 * the chip yields to digits that would not fit beside it.  A decimal longer
 * than its box is shown rounded to the box's width (v2 _visible_value); the
 * value stays the one the owner wrote (`text`), and focusing the box puts
 * it back whole for editing. */
export class SpinBox {
    root;
    input;
    options;
    up = button("", () => this.stepBy(1), "step up");
    down = button("", () => this.stepBy(-1), "step down");
    wheel = 0;
    chip = null;
    /** The step the operator chose on the chip, over the owner's (setSingleStep from Edit step). */
    chosen = null;
    /** The room the chip takes from the digits now (0 while it yields), and what it was fitted to. */
    chipRoom = 0;
    fitted = "";
    /** The owner's text while the box shows it rounded, and the rounded text it shows. */
    full = null;
    shown = "";
    constructor(root, input, options) {
        this.root = root;
        this.input = input;
        this.options = options;
        root.classList.add("spin");
        this.up.tabIndex = this.down.tabIndex = -1;
        this.up.setAttribute("aria-label", "Increase");
        this.down.setAttribute("aria-label", "Decrease");
        for (const cell of [this.up, this.down])
            cell.addEventListener("mousedown", event => { event.preventDefault(); input.focus(); });
        root.append(this.up, this.down);
        if (options.decimal) {
            // Hidden until it is fitted: the digits never start under it.
            root.classList.add("decimal", "chip-yields");
            const chip = this.chip = element("button", "step-chip");
            chip.type = "button";
            chip.tabIndex = -1;
            chip.append(element("span"));
            chip.addEventListener("mousedown", event => event.preventDefault());
            chip.addEventListener("click", () => void this.editStep());
            root.append(chip);
            SpinBox.owners.set(root, this);
            SpinBox.resized.observe(root);
        }
        input.addEventListener("wheel", event => {
            if (document.activeElement !== input || input.disabled || event.deltaY === 0)
                return;
            event.preventDefault();
            const unit = event.deltaMode === WheelEvent.DOM_DELTA_LINE ? 100 / 3 : event.deltaMode === WheelEvent.DOM_DELTA_PAGE ? 100 * 3 : 1;
            this.wheel += -event.deltaY * unit;
            const notches = Math.trunc(this.wheel / 100);
            if (notches) {
                this.wheel -= notches * 100;
                this.stepBy(notches);
            }
        }, { passive: false });
        input.addEventListener("keydown", event => {
            const steps = { ArrowUp: 1, ArrowDown: -1, PageUp: 10, PageDown: -10 }[event.key];
            if (!steps)
                return;
            event.preventDefault();
            this.stepBy(steps);
        });
        // What is typed takes the chip's room the moment it needs it (refresh fits it).
        input.addEventListener("input", () => this.refresh());
        input.addEventListener("focus", () => {
            // Edited whole: the digits the box rounded away come back, selected
            // only if the focus selected the box's text (Tab), never by a click.
            if (this.full === null || input.value !== this.shown)
                return;
            const all = input.selectionStart === 0 && input.selectionEnd === input.value.length;
            input.value = this.full;
            this.full = null;
            if (all)
                input.select();
        });
        input.addEventListener("blur", () => { this.wheel = 0; if (this.options.decimal)
            SpinBox.queue(this); });
    }
    /** What the box holds: the owner's text, whatever rounding it is shown with. */
    get text() { return this.full !== null && this.input.value === this.shown ? this.full : this.input.value; }
    step() { return this.chosen ?? this.options.step(); }
    current() {
        const raw = this.text.trim();
        if (this.options.special && raw === this.options.special.text)
            return this.options.special.value;
        const value = raw === "" ? NaN : Number(raw);
        return Number.isFinite(value) ? value : null;
    }
    target(steps) {
        const value = this.current();
        if (value === null)
            return null;
        const [low, high] = this.options.bounds();
        return gridStep(value, this.step(), steps, low, high);
    }
    /** A step changes the value and selects nothing: writing the new text puts
     * a focused box's caret at its end (v2/Qt selected all on every step, a
     * selection nobody made, that typing would have replaced). */
    stepBy(steps) {
        if (this.input.disabled || this.input.readOnly)
            return;
        const moved = this.target(steps);
        if (moved === null)
            return;
        this.input.value = this.options.special?.value === moved ? this.options.special.text : String(moved);
        this.refresh();
        this.options.commit(moved);
    }
    /** stepEnabled: a direction the box cannot move in is dimmed. */
    refresh() {
        const live = !this.input.disabled && !this.input.readOnly;
        assign(this.up, "disabled", !live || this.target(1) === null);
        assign(this.down, "disabled", !live || this.target(-1) === null);
        if (!this.chip)
            return;
        const text = stepText(this.step());
        setGlyph(this.chip.firstElementChild, text, "chip");
        assign(this.chip, "disabled", this.input.disabled);
        const tip = live ? `Step ${text.slice(1)} · click to change` : `Step ${text.slice(1)}`;
        if (this.chip.title !== tip)
            this.chip.title = tip;
        if (this.fitted !== `${text} ${this.input.value}`)
            SpinBox.queue(this);
    }
    /** The owner's step again (its number is read in another unit, where the chosen one means something else). */
    resetStep() { this.chosen = null; this.refresh(); }
    async editStep() {
        if (this.input.readOnly)
            return;
        const answer = await prompt("Edit step", "Step", String(this.step()));
        const step = Math.abs(Number(answer?.trim() || NaN));
        // A zero or unreadable step is no step (setSingleStep ignores it).
        if (!Number.isFinite(step) || step === 0)
            return;
        this.chosen = step;
        this.refresh();
    }
    // The chips and digits are fitted together, a frame after what changed
    // them: every box read first, then every box written, so a form of number
    // boxes costs one layout however many it holds. A box resized is fitted in
    // the frame that resized it (the observer runs after that frame's layout,
    // which then lays the chips out again), not a frame later.
    static owners = new WeakMap();
    static resized = new ResizeObserver(entries => {
        for (const entry of entries) {
            const spin = SpinBox.owners.get(entry.target);
            if (spin)
                SpinBox.pending.add(spin);
        }
        SpinBox.fit();
    });
    static pending = new Set();
    static measure = null;
    static queue(spin) {
        if (!SpinBox.pending.size)
            requestAnimationFrame(() => SpinBox.fit());
        SpinBox.pending.add(spin);
    }
    static fit() {
        const spins = [...SpinBox.pending].filter(spin => spin.root.isConnected);
        SpinBox.pending.clear();
        const context = SpinBox.measure ??= document.createElement("canvas").getContext("2d");
        if (!context || !spins.length)
            return;
        const chrome = token("--chip-chrome");
        const fits = spins.map(spin => {
            const input = spin.input, chip = spin.chip;
            if (!input.clientWidth)
                return null;
            const style = getComputedStyle(input);
            // The digits' whole room: the chip yields to them, never the other way round.
            const digits = input.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight) + spin.chipRoom;
            const label = chip.firstElementChild.textContent ?? "";
            // The chip's text is its ink, rounded up to whole Qt px (_step_chip_metrics).
            const width = glyphFace("chip", label).ink / devicePixelRatio;
            const room = Math.max(0, width + chrome);
            context.font = font(style);
            const measured = (text) => context.measureText(text).width + 2;
            // The owner's value -- what it wrote last, or what is being typed -- fitted
            // to the room unless it is being edited.
            const editing = document.activeElement === input;
            const full = editing || input.value !== spin.shown ? input.value : spin.full ?? input.value;
            const [low, high] = spin.options.bounds();
            const text = editing ? full : fittedDecimal(full, digits, measured, low, high);
            return { width, room, text, full, editing, shown: measured(text) <= digits - room, key: `${label} ${text}` };
        });
        spins.forEach((spin, index) => {
            const fit = fits[index];
            if (!fit)
                return;
            if (!fit.editing) {
                if (spin.input.value !== fit.text)
                    spin.input.value = fit.text;
                spin.shown = fit.text;
                spin.full = fit.text === fit.full ? null : fit.full;
            }
            spin.root.style.setProperty("--chip-text-w", `${fit.width}px`);
            spin.root.classList.toggle("chip-yields", !fit.shown);
            spin.chipRoom = fit.shown ? fit.room : 0;
            spin.fitted = fit.key;
        });
    }
}
/** A standalone number box: an input in a spin holder. */
export function spinBox(options) {
    const holder = document.createElement("div");
    const input = document.createElement("input");
    input.type = "text";
    input.inputMode = options.decimal ? "decimal" : "numeric";
    input.spellcheck = false;
    if (options.value !== undefined)
        input.value = String(options.value);
    if (options.label)
        input.setAttribute("aria-label", options.label);
    holder.append(input);
    const spin = new SpinBox(holder, input, options);
    spin.refresh();
    return spin;
}
