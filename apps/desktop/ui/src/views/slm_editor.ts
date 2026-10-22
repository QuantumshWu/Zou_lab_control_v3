// "<role> SLM Editor" (v2 SlmEditorControl in its own window, opened by an
// SLM's Control): the Pattern page -- Paint (Toggle, Brush, Erase), Intensity,
// Selectors, Size, Preset…, Clear; the Input pupil and Zernike layers with
// their Edit…; the target and the science phase side by side; the file
// commands, Adopt device command and Send to SLM; the status and device
// lines -- and the Wavefront page (steering, Noll Z4-Z11, Reset wavefront,
// the wavefront phase).  The draft, its solve and its rules are the SLM's
// (its editor_* commands); the pictures are the host's panels
// (slm-<device>-<plot>).  This page draws what the editor states and sends
// what the operator does.
import {
  ComboBox, Popup, Segmented, TabWidget, command, element, frame, row, setChecked, setState, spinBox, switchButton,
  type SpinBox, type StatusWord, type Value, type Command as Button,
} from "../fluent/index.js";
import type { State } from "../protocol.js";
import type { View, ViewContext } from "../shell.js";
import type { DeviceRow } from "../workbench.js";
import type { PanelBoard } from "./board.js";

/** One number the editor offers, with its bounds and what it is called under each preset. */
type NumberField = {
  name: string; label: string; type: "int" | "float"; value: number; minimum: number; maximum: number; decimals: number; step: number;
  description: string; kinds?: string[]; labels?: Record<string, string>; descriptions?: Record<string, string>;
};
type FileAction = { save: boolean; name: string; filter: string; title: string };
type Catalog = {
  shape_yx: [number, number]; presets: string[]; paint_modes: [string, string][]; intensity: NumberField;
  fields: { preset: NumberField[]; pupil: NumberField[]; operator: NumberField[]; text: { name: string; label: string; placeholder: string; kinds: string[] } };
  note: string; plot_size: string; plots: [string, string][]; files: Record<string, FileAction>;
};
type EditorState = {
  open: boolean; device: string; status: string; status_word: StatusWord; error: string | null;
  device_status: string; device_status_word: StatusWord; send_enabled: boolean; adopt_enabled: boolean; command_pending: boolean;
  pupil: { enabled: boolean; center_xy: [number, number]; diameter_xy: [number, number] }; pupil_text: string;
  operator_values: Record<string, number | boolean>; operator_text: string;
  preset: { kind: string; values: Record<string, number>; text: string }; plot_size: string;
};

/** QDoubleSpinBox's reading: the value at its decimals, no trailing zeros. */
const shown = (value: number, decimals: number): string => String(Number(value.toFixed(decimals)));

/** A number box of one field: its bounds, its step, its decimals; typing commits on Enter or leaving it. */
class NumberBox {
  readonly spin: SpinBox;
  constructor(readonly field: NumberField, commit: (value: number) => void) {
    const decimal = field.type === "float";
    this.spin = spinBox({ step: () => field.step, bounds: () => [field.minimum, field.maximum], decimal, label: field.label,
      commit: value => commit(this.clamp(value)) });
    const input = this.spin.input;
    input.value = shown(field.value, field.decimals);
    if (field.description) input.title = field.description;
    const typed = (): void => {
      const value = Number(this.spin.text.trim());
      if (this.spin.text.trim() === "" || !Number.isFinite(value)) { input.value = shown(this.value, field.decimals); return; }
      const clamped = this.clamp(value);
      input.value = shown(clamped, field.decimals);
      this.spin.refresh();
      commit(clamped);
    };
    input.addEventListener("change", typed);
    input.addEventListener("keydown", event => { if (event.key === "Enter") input.blur(); });
  }
  private clamp(value: number): number {
    const rounded = this.field.type === "int" ? Math.round(value) : Number(value.toFixed(this.field.decimals));
    return Math.min(this.field.maximum, Math.max(this.field.minimum, rounded));
  }
  get value(): number { const value = Number(this.spin.text); return Number.isFinite(value) ? value : this.field.value; }
  set value(value: number) {
    if (document.activeElement === this.spin.input) return;
    const text = shown(value, this.field.decimals);
    if (this.spin.text !== text) { this.spin.input.value = text; this.spin.refresh(); }
  }
  set disabled(disabled: boolean) { this.spin.input.disabled = disabled; this.spin.refresh(); }
}

class SlmEditorWindow implements View {
  readonly key = "slm-editor";
  readonly title = "SLM Editor";
  readonly windowTitle: string;
  readonly root = element("div", "window slm-editor");
  private tabs = new TabWidget();
  private catalog: Catalog | null = null;
  private state: EditorState | null = null;
  private built = false;
  private named = "";
  /** The latest of the editor's own status and a refusal of this window's command (v2 _report_error). */
  private refusal = "";
  // Pattern page
  private mode = new Segmented(value => this.pickMode(String(value)));
  private paintMode = "toggle";
  private intensity: NumberBox | null = null;
  private selectors = switchButton(on => this.setSelectors(on), "Selectors");
  private size = new ComboBox(value => { if (value !== null) this.send("size", { size: value }, "Resizing the plots…"); });
  private presetButton: Button = command("Preset…", () => this.presetPopup?.toggle());
  private presetPopup: Popup | null = null;
  private presetKind = new ComboBox(value => this.showPresetKind(String(value)));
  private presetBoxes = new Map<string, { box: NumberBox; row: HTMLElement; label: HTMLElement }>();
  private presetText = element("input");
  private presetTextRow: HTMLElement | null = null;
  private pupilSwitch = switchButton(on => this.send("pupil", { enabled: on }, "Solving with the new pupil…"), "Input pupil");
  private pupilText = element("span", "ink layer-text");
  private pupilButton: Button = command("Edit…", () => this.pupilPopup?.toggle());
  private pupilPopup: Popup | null = null;
  private pupilBoxes = new Map<string, NumberBox>();
  /** The editor's pupil and preset as last written into the popups' boxes. */
  private pupilShown = "";
  private presetShown = "";
  private zernikeSwitch = switchButton(on => this.send("operator", { enabled: on }), "Zernike");
  private zernikeText = element("span", "ink layer-text");
  private plots = element("div", "slm-plots");
  private adopt: Button = command("Adopt device command", () => this.send("adopt", {}, "Adopting the device command…"));
  private sendButton: Button = command("Send to SLM", () => this.send("send", {}, "Sending to the SLM…"), { role: "primary" });
  private statusLine = element("p", "ink slm-status");
  private deviceStatus = element("p", "ink slm-status");
  // Wavefront page
  private operatorBoxes = new Map<string, NumberBox>();
  private wavefront = element("div", "slm-plots slm-wavefront-plot");
  private paint: { pointer: number; points: [number, number][]; frame: number } | null = null;

  constructor(private ctx: ViewContext, private board: PanelBoard, private device: string) {
    this.windowTitle = `${device} SLM Editor`;
    this.root.append(this.tabs.root);
    this.tabs.root.classList.add("slm-tabs");
    // v2: the editor is opened (or raised) by its window; its draft is the device's.
    ctx.transport.send({ action: `${device}__editor_open` }, "Opening the SLM Editor…");
    board.interactive = false;
  }

  private send(name: string, params: Record<string, unknown>, feedback?: string): number {
    this.refusal = "";
    return this.ctx.transport.send({ action: `${this.device}__editor_${name}`, ...params }, feedback);
  }

  // ------------------------------------------------------------- building
  private build(catalog: Catalog): void {
    this.built = true;
    const pattern = element("div", "slm-page");
    // Paint · Intensity · Selectors · Size · Preset… · Clear
    const edit = frame(true, "slm-row");
    this.mode.setChoices(catalog.paint_modes.map(([value, label]) => [value, label] as [Value, string]));
    this.mode.setCurrent(this.paintMode);
    this.intensity = new NumberBox(catalog.intensity, () => {});
    this.intensity.spin.root.classList.add("slm-intensity");
    this.selectors.classList.add("row-switch");
    this.selectors.title = "Enable plot selectors, wheel zoom, and pan; disable to paint the target";
    this.size.root.classList.add("slm-size");
    const clear = command("Clear", () => this.send("spots", { spots: [] }, "Clearing the target…"), { role: "destructive",
      title: "Empty the target; the SLM keeps its current phase" });
    edit.append(element("span", "row-label", "Paint"), this.mode.root, element("span", "row-label", "Intensity"), this.intensity.spin.root,
      this.selectors, element("span", "row-label", "Size"), this.size.root, this.presetButton.root, clear.root, element("span", "spacer"));
    this.presetPopup = new Popup(this.presetButton, this.presetForm(catalog), ["--slm-preset-min-w", "--slm-preset-min-h"]);
    // Input pupil · Zernike
    const layers = frame(true, "slm-layers");
    this.pupilSwitch.title = "On uses the configured Gaussian for solver illumination. Off uses uniform full-raster illumination; center and diameter still define the Zernike unit disk.";
    this.pupilPopup = new Popup(this.pupilButton, this.pupilForm(catalog), ["--slm-pupil-min-w", "--slm-pupil-min-h"]);
    const zernikeEdit = command("Edit…", () => this.tabs.setCurrent("wavefront"));
    layers.append(this.pupilSwitch, this.pupilText, this.pupilButton.root, this.zernikeSwitch, this.zernikeText, zernikeEdit.root);
    // The target and the science phase, side by side, scrolling when bigger than the page.
    const scroll = element("div", "slm-plot-scroll");
    scroll.append(this.plots);
    // Files · Adopt device command · Send to SLM
    const files = frame(true, "slm-row");
    const file = (label: string, action: string): HTMLButtonElement => command(label, () => this.file(action)).root;
    files.append(file("Load target", "load_target"), file("Import target", "import_target"), file("Save target", "save_target"),
      file("Load science context", "load_context"), file("Save science context", "save_context"), element("span", "spacer"),
      this.adopt.root, this.sendButton.root);
    pattern.append(edit, layers, scroll, files, this.statusLine, this.deviceStatus);
    this.tabs.addPermanent("pattern", "Pattern", pattern);
    this.tabs.addPermanent("wavefront", "Wavefront", this.wavefrontPage(catalog));
    this.tabs.setCurrent("pattern");
  }

  private presetForm(catalog: Catalog): HTMLElement {
    const body = element("div", "form slm-popup-form");
    this.presetKind.setChoices(catalog.presets.map(kind => [kind, kind] as [Value, string]));
    body.append(row("Pattern", this.presetKind.root));
    for (const field of catalog.fields.preset) {
      const box = new NumberBox(field, () => {});
      const node = row(field.label, box.spin.root);
      this.presetBoxes.set(field.name, { box, row: node, label: node.querySelector(".field-label")! });
      body.append(node);
    }
    const text = catalog.fields.text;
    this.presetText.type = "text";
    this.presetText.spellcheck = false;
    this.presetText.placeholder = text.placeholder;
    this.presetTextRow = row(text.label, this.presetText);
    body.append(this.presetTextRow);
    const actions = element("div", "form-actions slm-popup-actions");
    actions.append(command("Apply", () => this.applyPreset(), { role: "primary" }).root);
    body.append(actions);
    return body;
  }

  private pupilForm(catalog: Catalog): HTMLElement {
    const body = element("div", "form slm-popup-form");
    for (const field of catalog.fields.pupil) {
      const box = new NumberBox(field, () => {});
      this.pupilBoxes.set(field.name, box);
      body.append(row(field.label, box.spin.root));
    }
    const actions = element("div", "form-actions slm-popup-actions");
    actions.append(command("Apply pupil", () => {
      const values: Record<string, number> = {};
      for (const [name, box] of this.pupilBoxes) values[name] = box.value;
      this.pupilPopup?.close();
      this.send("pupil", values, "Solving with the new pupil…");
    }, { role: "primary" }).root);
    body.append(actions);
    return body;
  }

  private wavefrontPage(catalog: Catalog): HTMLElement {
    const page = element("div", "slm-wavefront");
    const parameters = element("div", "slm-wavefront-parameters");
    parameters.append(element("p", "wrap slm-note", catalog.note));
    const grid = element("div", "slm-operator-grid");
    for (const field of catalog.fields.operator) {
      const box = new NumberBox(field, value => this.send("operator", { [field.name]: value }));
      this.operatorBoxes.set(field.name, box);
      const label = element("span", "row-label", field.label);
      label.title = field.description;
      grid.append(label, box.spin.root);
    }
    const reset = command("Reset wavefront", () => this.send("operator", { reset: true }), { role: "destructive",
      title: "Set steering and every Zernike coefficient to zero; the SLM keeps its current phase" });
    const resetRow = element("div", "slm-reset");
    resetRow.append(reset.root);
    parameters.append(grid, resetRow);
    const scroll = element("div", "slm-plot-scroll");
    scroll.append(this.wavefront);
    page.append(parameters, scroll);
    return page;
  }

  // ------------------------------------------------------------- intents
  private pickMode(mode: string): void { this.paintMode = mode; this.mode.setCurrent(mode); }

  private setSelectors(on: boolean): void {
    setChecked(this.selectors, on);
    // v2: Selectors on gives the plots their gestures, and no painting.
    this.board.interactive = on;
    this.mode.disabled = on;
    if (this.intensity) this.intensity.disabled = on;
  }

  private showPresetKind(kind: string): void {
    this.presetKind.value = kind;
    for (const entry of this.presetBoxes.values()) {
      const field = entry.box.field;
      entry.row.hidden = !(field.kinds ?? []).includes(kind);
      entry.label.textContent = field.labels?.[kind] ?? field.label;
      entry.label.title = field.descriptions?.[kind] ?? "";
    }
    if (this.presetTextRow && this.catalog) this.presetTextRow.hidden = !this.catalog.fields.text.kinds.includes(kind);
  }

  private applyPreset(): void {
    const kind = String(this.presetKind.value);
    const values: Record<string, unknown> = { kind, text: this.presetText.value };
    for (const [name, entry] of this.presetBoxes) values[name] = entry.box.value;
    this.refusal = "";
    // v2 hides the popup once the preset is accepted; a refused one stays open with its error.
    this.ctx.transport.request({ action: `${this.device}__editor_preset`, ...values }, "Applying the preset…")
      .then(() => this.presetPopup?.close(), () => {});
  }

  private file(action: string): void {
    this.refusal = "";
    this.ctx.transport.send({ action: "slm_file", device: this.device, file: action }, "Choosing a file…");
  }

  /** v2's paint on the target: a press, then (Brush, Erase) the drag's positions once per frame. */
  private bindPaint(canvas: HTMLCanvasElement): void {
    if (canvas.dataset.paint) return;
    canvas.dataset.paint = "1";
    const at = (event: PointerEvent): [number, number] => {
      const rect = canvas.getBoundingClientRect();
      return [(event.clientX - rect.left) / rect.width, (event.clientY - rect.top) / rect.height];
    };
    const flush = (): void => {
      const stroke = this.paint;
      if (!stroke) return;
      stroke.frame = 0;
      if (stroke.points.length) this.stroke("drag", stroke.points.splice(0));
    };
    canvas.addEventListener("pointerdown", event => {
      if (this.board.interactive || event.button !== 0 || !this.state?.open) return;
      event.preventDefault();
      canvas.setPointerCapture(event.pointerId);
      this.paint = { pointer: event.pointerId, points: [], frame: 0 };
      this.stroke("press", [at(event)]);
    });
    canvas.addEventListener("pointermove", event => {
      const stroke = this.paint;
      if (!stroke || stroke.pointer !== event.pointerId || !this.drags()) return;
      stroke.points.push(at(event));
      if (!stroke.frame) stroke.frame = requestAnimationFrame(flush);
    });
    const end = (event: PointerEvent): void => {
      const stroke = this.paint;
      if (!stroke || stroke.pointer !== event.pointerId) return;
      if (stroke.frame) cancelAnimationFrame(stroke.frame);
      flush();
      this.paint = null;
      if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
    };
    canvas.addEventListener("pointerup", end);
    canvas.addEventListener("lostpointercapture", end);
  }

  /** Whether the paint mode goes on painting while the press drags (the catalogue says). */
  private drags(): boolean { return this.paintMode !== (this.catalog?.paint_modes[0]?.[0] ?? "toggle"); }

  private stroke(stroke: "press" | "drag", points: [number, number][]): void {
    this.refusal = "";
    this.ctx.transport.send({ action: "slm_paint", device: this.device, mode: this.paintMode, stroke, points,
      intensity: this.intensity?.value ?? 1 });
  }

  // ------------------------------------------------------------- state
  update(state: State): void {
    const device = (this.ctx.latest.devices as DeviceRow[] | undefined)?.find(item => item.instance_id === this.device);
    const title = `${device?.role ?? this.device} SLM Editor`;
    if (title !== this.named) { this.named = title; document.title = title; }
    // A refusal of this window's own command is the newest word (v2 _report_error), until the editor speaks again;
    // one the editor reported itself is already its status line, in v2's words (str(error), no type name).
    if (state.error && state.request && !(this.state?.error && state.error.endsWith(this.state.error))) this.refusal = state.error;
    const slm = state.slm as { device: string; state: EditorState; catalog: Catalog | null } | undefined;
    if (slm && slm.device === this.device) {
      if (slm.catalog) this.catalog = slm.catalog;
      if (slm.state && "status" in slm.state) { this.state = slm.state; this.refusal = ""; }
      if (this.catalog && !this.built) this.build(this.catalog);
    }
    this.mount();
    this.show();
  }

  /** Shown once its pages and their plots, in their pictures' room (or saying why they have none), are in;
   * or once the editor refused to open. */
  ready(): boolean {
    const panels = this.ctx.latest.panels as { id: string; error?: string | null }[] | undefined;
    return Boolean(this.refusal) || (this.built && (this.catalog?.plots ?? []).map(([plot]) => `slm-${this.device}-${plot}`)
      .every(id => this.board.views.get(id)?.placed || Boolean(panels?.find(panel => panel.id === id)?.error)));
  }

  /** The host's panels in their places: the target and the phase on Pattern, the wavefront on its page. */
  private mount(): void {
    const plots = this.catalog?.plots ?? [];
    for (const host of [this.plots, this.wavefront]) {
      const wanted = plots.filter(([plot]) => (plot === "wavefront") === (host === this.wavefront))
        .map(([plot]) => this.board.views.get(`slm-${this.device}-${plot}`)?.root).filter((node): node is HTMLElement => Boolean(node));
      if (wanted.length !== host.children.length || wanted.some((node, index) => host.children[index] !== node)) host.replaceChildren(...wanted);
    }
    const painted = this.board.views.get(`slm-${this.device}-target`);
    if (painted) this.bindPaint(painted.canvas);
    // The wavefront takes no gesture (v2 kept its host's interaction off).
    this.board.quiet.add(`slm-${this.device}-wavefront`);
    const target = this.board.views.get(`slm-${this.device}-target`);
    const choices = (this.ctx.latest.panels as { id: string; size_choices?: string[] }[] | undefined)?.find(row => row.id === target?.id)?.size_choices;
    if (choices?.length) this.size.setChoices(choices.map(choice => [choice, choice] as [Value, string]));
  }

  private show(): void {
    const state = this.state;
    if (!state || !this.built) return;
    const [text, word] = this.refusal ? [this.refusal, "error" as StatusWord] : [state.status, state.status_word];
    if (this.statusLine.textContent !== text) this.statusLine.textContent = text;
    setState(this.statusLine, word);
    if (this.deviceStatus.textContent !== state.device_status) this.deviceStatus.textContent = state.device_status;
    setState(this.deviceStatus, state.device_status_word);
    this.sendButton.disabled = !state.send_enabled;
    this.adopt.disabled = !state.adopt_enabled;
    this.size.value = state.plot_size;
    setChecked(this.pupilSwitch, state.pupil.enabled);
    if (this.pupilText.textContent !== state.pupil_text) this.pupilText.textContent = state.pupil_text;
    setState(this.pupilText, "idle");
    setChecked(this.zernikeSwitch, Boolean(state.operator_values.enabled));
    if (this.zernikeText.textContent !== state.operator_text) this.zernikeText.textContent = state.operator_text;
    setState(this.zernikeText, "idle");
    const pupil: Record<string, number> = { center_x: state.pupil.center_xy[0], center_y: state.pupil.center_xy[1],
      diameter_x: state.pupil.diameter_xy[0], diameter_y: state.pupil.diameter_xy[1] };
    // v2: a popup's boxes keep what the operator typed until the editor's own values
    // change (an Apply, a loaded context); a reopened popup shows the unapplied edits.
    const pupilWords = JSON.stringify(pupil);
    if (pupilWords !== this.pupilShown && !this.pupilPopup?.open) {
      this.pupilShown = pupilWords;
      for (const [name, box] of this.pupilBoxes) box.value = pupil[name] ?? box.value;
    }
    for (const [name, box] of this.operatorBoxes) box.value = Number(state.operator_values[name] ?? 0);
    const presetWords = JSON.stringify(state.preset);
    if (presetWords !== this.presetShown && !this.presetPopup?.open) {
      this.presetShown = presetWords;
      this.showPresetKind(state.preset.kind);
      for (const [name, entry] of this.presetBoxes) entry.box.value = state.preset.values[name] ?? entry.box.value;
      if (document.activeElement !== this.presetText) this.presetText.value = state.preset.text;
    }
  }

  status(): void { this.show(); }
}

export const slmEditor = (board: PanelBoard, device: string) => (ctx: ViewContext): View => new SlmEditorWindow(ctx, board, device);
