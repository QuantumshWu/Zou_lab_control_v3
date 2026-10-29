import { prompt } from "./dialog.js";
import { assign, button, element, token } from "./dom.js";
import { glyphFace, setGlyph } from "./ink.js";

/** The decimal places `value` is written with (1e-05: 5). */
export function decimals(value: number): number {
  const [mantissa = "", exponent] = String(value).split("e");
  const places = mantissa.split(".")[1]?.length ?? 0;
  return Math.max(0, places - Number(exponent ?? 0));
}

/** _grid_step_inside: `steps` notches from `value` on the grid of multiples
 * of `step` inside [low, high]; off the grid the first notch goes to the
 * nearest grid point in its direction.  null when no grid point lies that way. */
export function gridStep(value: number, step: number, steps: number, low: number | null, high: number | null): number | null {
  if (!(step > 0) || !Number.isFinite(step) || !Number.isFinite(value) || !Number.isSafeInteger(steps) || !steps
      || (low !== null && !Number.isFinite(low)) || (high !== null && !Number.isFinite(high))) return null;
  // The step's decimal spelling defines the grid. Compute its integer index
  // and product exactly, then round once to the wire's Number. Binary division
  // and multiplication followed by toFixed both lose tiny decimal steps and
  // can create long tails, especially after an SI unit change.
  const decimal = (number: number): [bigint, number] => {
    const [mantissa = "0", exponent = "0"] = String(number).split("e");
    return [BigInt(mantissa.replace(".", "")), Number(exponent) - (mantissa.split(".")[1]?.length ?? 0)];
  };
  const parts = [value, step, ...(low === null ? [] : [low]), ...(high === null ? [] : [high])].map(decimal);
  const exponent = Math.min(...parts.map(([, exponent]) => exponent));
  const integer = (number: number): bigint => { const [digits, power] = decimal(number); return digits * 10n ** BigInt(power - exponent); };
  const [at, grain] = [integer(value), integer(step)];
  const floor = (number: bigint): bigint => number / grain - (number % grain < 0n ? 1n : 0n);
  const ceil = (number: bigint): bigint => -floor(-number);
  const asNumber = (number: bigint): number => Number(`${number}e${exponent}`);
  const below = floor(at), remainder = at - below * grain;
  const nearest = below + (remainder * 2n >= grain ? 1n : 0n);
  // Only representation-sized dust is on-grid. The former 1e-9 * index
  // tolerance classified genuine hardware offsets as whole grid positions.
  const distance = asNumber(at - nearest * grain);
  const onGrid = remainder === 0n || Math.abs(distance) <= Number.EPSILON * Math.abs(value);
  let index = (onGrid ? nearest : steps > 0 ? below : ceil(at)) + BigInt(steps);
  if (low !== null && index * grain < integer(low)) index = ceil(integer(low));
  if (high !== null && index * grain > integer(high)) index = floor(integer(high));
  const moved = asNumber(index * grain);
  if (!Number.isFinite(moved)) return null;
  if ((low !== null && moved < low) || (high !== null && moved > high)) return null;
  if ((steps < 0 && moved >= value) || (steps > 0 && moved <= value)) return null;
  return moved;
}

/** A canvas font of an element's computed style (its `font` shorthand is empty
 * where the shorthand cannot say everything, as with tabular digits). */
const font = (style: CSSStyleDeclaration): string => `${style.fontStyle} ${style.fontWeight} ${style.fontSize} ${style.fontFamily}`;

/** _visible_decimal: `text` with as many of its fractional digits as fit in
 * `room` (rounded, kept inside the bounds), else its whole number; text that
 * is not a plain decimal (an exponent, a word) is shown as written. */
function fittedDecimal(text: string, room: number, width: (text: string) => number, low: number | null, high: number | null): string {
  if (width(text) <= room) return text;
  const plain = /^-?\d+\.(\d+)$/.exec(text.trim());
  if (!plain) return text;
  const value = Number(text);
  let shown = text;
  for (let places = plain[1]!.length - 1; places >= 0; places--) {
    let rounded = Number(value.toFixed(places));
    // The grid of this many places never crosses a bound: step inside it.
    const grain = 10 ** -places;
    if (high !== null && rounded > high) rounded = Number((rounded - grain).toFixed(places));
    if (low !== null && rounded < low) rounded = Number((rounded + grain).toFixed(places));
    shown = String(rounded);
    if (width(shown) <= room) return shown;
  }
  return shown;
}

/** The ≈ rule (twin of zlc_core::units::approximate, by which Device Control's
 * reading and limits are written): a number whose shortest spelling has at most
 * 12 significant digits is shown as written; a longer one that reads the same at
 * 12 and 14 significant digits is float dust (0.30000000000000004), shown at 12;
 * any other is an inexact conversion's result (a logarithm, √2, a load), shown
 * "≈ " and rounded -- to 0.01 in a logarithmic unit (dB, dBm), else to 4
 * significant digits, or to the last digit of the field's own `step` where that
 * is coarser.  Only the text is rounded: the box still holds the number whole. */
export function approximateText(text: string, unit: string, step: number | null): string {
  const value = Number(text);
  if (!/^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i.test(text.trim()) || !Number.isFinite(value)) return text;
  // The shortest round-trip digits' count and the leading digit's exponent.
  const digits = (number: number): [number, number] => {
    const [mantissa = "", exponent = "0"] = Math.abs(number).toExponential().split("e");
    return [mantissa.replace(".", "").length, Number(exponent)];
  };
  const [significant, exponent] = digits(value);
  if (significant <= 12) return text;
  const twelve = Number(value.toPrecision(12));
  if (twelve === Number(value.toPrecision(14))) return String(twelve);
  let last = /^dB/.test(unit) ? -2 : exponent - 3;
  if (step !== null && step > 0 && Number.isFinite(step)) { const [count, top] = digits(step); last = Math.max(last, top - count + 1); }
  const shown = last <= 0 ? Number(value.toFixed(Math.min(100, -last))) : Math.round(value / 10 ** last) * 10 ** last;
  // String(-0) is "0": a level rounded to nothing is never -0.
  return `≈ ${String(shown)}`;
}

/** _step_chip_text: the step as a decimal, or %g past a millionth or a million. */
function stepText(step: number): string {
  const exponent = Math.floor(Math.log10(step));
  return `±${exponent >= -4 && exponent <= 6 ? String(Number(step.toPrecision(15))) : step.toExponential().replace(/\.?0+e/, "e")}`;
}

export type SpinOptions = { step(): number; bounds(): [number | null, number | null]; commit(value: number): void;
  /** QSpinBox specialValueText: this value is shown as these words (Repeats 0 = ∞). */
  special?: { value: number; text: string };
  /** A decimal box permits editing its step; every box displays the step. */
  decimal?: boolean;
  /** A number that may be an inexact conversion's result is shown by the ≈ rule
   * (`approximateText`) in this unit and declared step; null: as written. */
  approximate?(): { unit: string; step: number | null } | null };

/** FluentDoubleSpinBox: a field with a two-cell stepper painted from its frame.
 * The wheel changes the value only once the box has the focus (otherwise the
 * page scrolls), one notch per step on the step's grid. Every box shows
 * its step left of the stepper (_StepChip) -- an operator who cannot see the
 * step cannot know what the arrows do. Decimal steps can be edited;
 * the chip yields to digits that would not fit beside it.  A decimal longer
 * than its box is shown rounded to the box's width (v2 _visible_value); the
 * value stays the one the owner wrote (`text`), and focusing the box puts
 * it back whole for editing; so does a number an inexact conversion gave,
 * shown "≈ -23.98" by the ≈ rule (`approximate`). */
export class SpinBox {
  readonly up = button("", () => this.stepBy(1), "step up");
  readonly down = button("", () => this.stepBy(-1), "step down");
  private wheel = 0;
  private chip: HTMLElement | null = null;
  /** The step the operator chose on the chip, over the owner's (setSingleStep from Edit step). */
  private chosen: number | null = null;
  /** The room the chip takes from the digits now (0 while it yields), and what it was fitted to. */
  private chipRoom = 0;
  private fitted = "";
  /** The owner's text while the box shows it rounded, and the rounded text it shows. */
  private full: string | null = null;
  private shown = "";

  constructor(readonly root: HTMLElement, readonly input: HTMLInputElement, private options: SpinOptions) {
    root.classList.add("spin");
    this.up.tabIndex = this.down.tabIndex = -1;
    this.up.setAttribute("aria-label", "Increase");
    this.down.setAttribute("aria-label", "Decrease");
    for (const cell of [this.up, this.down]) cell.addEventListener("mousedown", event => { event.preventDefault(); input.focus(); });
    root.append(this.up, this.down);
    // Every stepper explains its arrows, including fixed integer steps.
    root.classList.add("step-hint", "chip-yields");
    if (options.decimal) root.classList.add("decimal");
    const chip = this.chip = options.decimal ? element("button", "step-chip") : element("span", "step-chip fixed-step");
    chip.append(element("span"));
    if (chip instanceof HTMLButtonElement) {
      chip.type = "button";
      chip.tabIndex = -1;
      chip.addEventListener("mousedown", event => event.preventDefault());
      chip.addEventListener("click", () => void this.editStep());
    }
    root.append(chip);
    SpinBox.owners.set(root, this);
    SpinBox.resized.observe(root);
    input.addEventListener("wheel", event => {
      if (document.activeElement !== input || input.disabled || event.deltaY === 0) return;
      event.preventDefault();
      const unit = event.deltaMode === WheelEvent.DOM_DELTA_LINE ? 100 / 3 : event.deltaMode === WheelEvent.DOM_DELTA_PAGE ? 100 * 3 : 1;
      this.wheel += -event.deltaY * unit;
      const notches = Math.trunc(this.wheel / 100);
      if (notches) { this.wheel -= notches * 100; this.stepBy(notches); }
    }, { passive: false });
    input.addEventListener("keydown", event => {
      const steps = { ArrowUp: 1, ArrowDown: -1, PageUp: 10, PageDown: -10 }[event.key];
      if (!steps) return;
      event.preventDefault();
      this.stepBy(steps);
    });
    // What is typed takes the chip's room the moment it needs it (refresh fits it).
    input.addEventListener("input", () => this.refresh());
    input.addEventListener("focus", () => {
      // Edited whole: the digits the box rounded away come back, selected
      // only if the focus selected the box's text (Tab), never by a click.
      if (this.full === null || input.value !== this.shown) return;
      const all = input.selectionStart === 0 && input.selectionEnd === input.value.length;
      input.value = this.full;
      this.full = null;
      if (all) input.select();
    });
    input.addEventListener("blur", () => { this.wheel = 0; SpinBox.queue(this); });
  }
  /** What the box holds: the owner's text, whatever rounding it is shown with. */
  get text(): string { return this.full !== null && this.input.value === this.shown ? this.full : this.input.value; }
  private step(): number { return this.chosen ?? this.options.step(); }
  /** The operator's explicit step, for an editor that restores each axis' draft. */
  get chosenStep(): number | null { return this.chosen; }
  set chosenStep(value: number | null) {
    if (value !== null && (!(value > 0) || !Number.isFinite(value))) throw new RangeError("Step must be positive and finite");
    this.chosen = value;
    this.refresh();
  }
  private current(): number | null {
    const raw = this.text.trim();
    if (this.options.special && raw === this.options.special.text) return this.options.special.value;
    const value = raw === "" ? NaN : Number(raw);
    return Number.isFinite(value) ? value : null;
  }
  private target(steps: number): number | null {
    const value = this.current();
    if (value === null) return null;
    const [low, high] = this.options.bounds();
    return gridStep(value, this.step(), steps, low, high);
  }
  /** A step changes the value and selects nothing: writing the new text puts
   * a focused box's caret at its end (v2/Qt selected all on every step, a
   * selection nobody made, that typing would have replaced). */
  stepBy(steps: number): void {
    if (this.input.disabled || this.input.readOnly) return;
    const moved = this.target(steps);
    if (moved === null) return;
    this.input.value = this.options.special?.value === moved ? this.options.special.text : String(moved);
    this.refresh();
    this.options.commit(moved);
  }
  /** stepEnabled: a direction the box cannot move in is dimmed. */
  refresh(): void {
    const live = !this.input.disabled && !this.input.readOnly;
    assign(this.up, "disabled", !live || this.target(1) === null);
    assign(this.down, "disabled", !live || this.target(-1) === null);
    if (!this.chip) return;
    const text = stepText(this.step());
    setGlyph(this.chip.firstElementChild as HTMLElement, text, "chip");
    if (this.chip instanceof HTMLButtonElement) assign(this.chip, "disabled", !live);
    this.chip.classList.toggle("disabled", this.input.disabled);
    const tip = live && this.options.decimal ? `Step ${text.slice(1)} · click to change` : `Step ${text.slice(1)}`;
    if (this.chip.title !== tip) this.chip.title = tip;
    if (this.fitted !== `${text} ${this.input.value}`) SpinBox.queue(this);
  }

  /** The owner's step again (its number is read in another unit, where the chosen one means something else). */
  resetStep(): void { this.chosenStep = null; }

  private async editStep(): Promise<void> {
    if (this.input.readOnly) return;
    const answer = await prompt("Edit step", "Step", String(this.step()));
    const step = Math.abs(Number(answer?.trim() || NaN));
    // A zero or unreadable step is no step (setSingleStep ignores it).
    if (!Number.isFinite(step) || step === 0) return;
    this.chosen = step;
    this.refresh();
  }

  // The chips and digits are fitted together, a frame after what changed
  // them: every box read first, then every box written, so a form of number
  // boxes costs one layout however many it holds. A box resized is fitted in
  // the frame that resized it (the observer runs after that frame's layout,
  // which then lays the chips out again), not a frame later.
  private static owners = new WeakMap<Element, SpinBox>();
  private static resized = new ResizeObserver(entries => {
    for (const entry of entries) { const spin = SpinBox.owners.get(entry.target); if (spin) SpinBox.pending.add(spin); }
    SpinBox.fit();
  });
  private static pending = new Set<SpinBox>();
  private static measure: CanvasRenderingContext2D | null = null;
  private static queue(spin: SpinBox): void {
    if (!SpinBox.pending.size) requestAnimationFrame(() => SpinBox.fit());
    SpinBox.pending.add(spin);
  }
  private static fit(): void {
    const spins = [...SpinBox.pending].filter(spin => spin.root.isConnected);
    SpinBox.pending.clear();
    const context = SpinBox.measure ??= document.createElement("canvas").getContext("2d");
    if (!context || !spins.length) return;
    const chrome = token("--chip-chrome");
    const fits = spins.map(spin => {
      const input = spin.input, chip = spin.chip!;
      if (!input.clientWidth) return null;
      const style = getComputedStyle(input);
      // The digits' whole room: the chip yields to them, never the other way round.
      const digits = input.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight) + spin.chipRoom;
      const label = chip.firstElementChild!.textContent ?? "";
      // The chip's text is its ink, rounded up to whole Qt px (_step_chip_metrics).
      const width = glyphFace("chip", label).ink / devicePixelRatio;
      const room = Math.max(0, width + chrome);
      context.font = font(style);
      const measured = (text: string): number => context.measureText(text).width + 2;
      // The owner's value -- what it wrote last, or what is being typed -- fitted
      // to the room unless it is being edited.
      const editing = document.activeElement === input;
      const full = editing || input.value !== spin.shown ? input.value : spin.full ?? input.value;
      const [low, high] = spin.options.bounds();
      // An inexact conversion's number shows its ≈ rounding (never fitted further), float dust its 12 digits.
      const approximate = editing ? null : spin.options.approximate?.();
      const plain = approximate ? approximateText(full, approximate.unit, approximate.step) : full;
      const text = editing || !spin.options.decimal || plain.startsWith("≈") ? plain : fittedDecimal(plain, digits, measured, low, high);
      return { width, room, text, full, editing, shown: measured(text) <= digits - room, key: `${label} ${text}` };
    });
    spins.forEach((spin, index) => {
      const fit = fits[index];
      if (!fit) return;
      if (!fit.editing) {
        if (spin.input.value !== fit.text) spin.input.value = fit.text;
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
export function spinBox(options: SpinOptions & { value?: number; label?: string }): SpinBox {
  const holder = document.createElement("div");
  const input = document.createElement("input");
  input.type = "text";
  input.inputMode = options.decimal ? "decimal" : "numeric";
  input.spellcheck = false;
  if (options.value !== undefined) input.value = String(options.value);
  if (options.label) input.setAttribute("aria-label", options.label);
  holder.append(input);
  const spin = new SpinBox(holder, input, options);
  spin.refresh();
  return spin;
}
