import { assign, element, type Value } from "./dom.js";
import { hidePopup, isShown, showPopup } from "./popover.js";

export type Choice = {
  value: Value;
  label: string;
  /** Tooltip of the row (what a logic kind publishes, why a family is unavailable). */
  title?: string;
  /** A choice the owner withholds: listed, greyed, not pickable. */
  disabled?: boolean;
  /** Rows under one heading of the popup (a unit family, a device domain). */
  group?: string;
};
export type ChoiceInput = Choice | readonly [Value, string];

const encode = (value: Value): string => JSON.stringify(value);
const normal = (choice: ChoiceInput): Choice => Array.isArray(choice) ? { value: choice[0], label: choice[1] } : choice as Choice;

/** FluentComboBox: a field like an edit (ACCENT edge under the pointer or
 * while open) with the chevron in the stepper column; its popup a white card
 * with the picked row SELECTED and the row under the pointer ACCENT_TINT.
 * For data and long lists; a closed set the owner declares is Segmented.
 * `changed` is the operator's pick; `value =` projects one and never emits. */
let measure: CanvasRenderingContext2D | null | undefined;
const REFERENCE_PX = 100;

/** FluentComboBox.sizeHint: the box is at least as wide as the widest thing it
 * may say (its placeholder or a pickable row, without a grouped row's indent)
 * and as `minimum` "M"s (setMinimumContentsLength), with its chrome. Kept in em
 * of the box's own font (fluent.css reads --combo-min), so it holds at any scale;
 * a fixed or ignored size is the owner's CSS min-width. */
export function hintWidest(select: HTMLSelectElement, labels: Iterable<string>, minimum = 0): void {
  if (measure === undefined) {
    measure = document.createElement("canvas").getContext("2d");
    const style = getComputedStyle(document.body);
    if (measure) measure.font = `${style.fontWeight} ${REFERENCE_PX}px ${style.fontFamily}`;
  }
  if (!measure) return;
  let widest = minimum > 0 ? measure.measureText("M".repeat(minimum)).width : 0;
  for (const label of labels) widest = Math.max(widest, measure.measureText(label.trimStart()).width);
  const value = `calc(${Math.ceil(widest * 100 / REFERENCE_PX) / 100}em + var(--combo-chrome))`;
  if (select.style.getPropertyValue("--combo-min") !== value) select.style.setProperty("--combo-min", value);
}

/** Where a combo box's list opens (QComboBox::showPopup): below the box when
 * the whole list fits there; otherwise on the side with more room -- above when
 * the list fits there or shows more of itself there -- as tall as that side
 * allows, the rest scrolling.  `natural` is the list's own height, `below` and
 * `above` the room the window leaves it on each side; a tie opens below. */
export function dropPlacement(natural: number, below: number, above: number): { up: boolean; height: number } {
  const up = natural > below && above > below;
  return { up, height: Math.max(0, Math.min(natural, up ? above : below)) };
}

/** The window a list opens in, without a scroll bar (a fixed popup's containing block). */
const windowHeight = (): number => document.documentElement.clientHeight;
const px = (value: string): number => parseFloat(value) || 0;

/** Which side of `box` an open `list` (a .drop popup bound to its box: fluent.css places it,
 * its CSS margins its popup gaps) drops to, as dropPlacement says from the room the window
 * leaves it when it opens: `.up` above, capped at `--drop-room` where the side is shorter
 * than the list.  Given `up`, that side (a list that grows while open stays on the side it
 * opened on), capped to it.  Returns the side. */
export function placeDrop(list: HTMLElement, box: Element, up?: boolean): boolean {
  list.classList.remove("up");
  list.style.removeProperty("--drop-room");
  const css = getComputedStyle(list);
  const margins = px(css.marginTop) + px(css.marginBottom);
  const natural = list.getBoundingClientRect().height;
  const rect = box.getBoundingClientRect(), below = windowHeight() - rect.bottom - margins, above = rect.top - margins;
  const drop = up === undefined ? dropPlacement(natural, below, above) : { up, height: Math.max(0, Math.min(natural, up ? above : below)) };
  list.classList.toggle("up", drop.up);
  if (drop.height < natural) list.style.setProperty("--drop-room", `${drop.height}px`);
  return drop.up;
}

/** The closed box's words: the picked row's, ending in an ellipsis where the
 * box is narrower (v2 paints elidedText); first child of `select`, kept there.
 * The platform copies the row only when the pick changes: a picked row whose own
 * words change (Scope's coordinate, a blank's words) is copied again here.
 * `changed` hears every change of the select's rows. */
export function selectedFace(select: HTMLSelectElement, changed: () => void = () => {}): HTMLButtonElement {
  const face = element("button", "combo-face");
  face.type = "button";
  const shown = document.createElement("selectedcontent");
  face.append(shown);
  new MutationObserver(() => {
    const picked = select.selectedOptions[0];
    if (shown.textContent !== (picked?.textContent ?? "")) shown.replaceChildren(...[...picked?.childNodes ?? []].map(node => node.cloneNode(true)));
    changed();
  }).observe(select, { subtree: true, childList: true, characterData: true });
  return face;
}

/** A select's face and its drop-down: the closed box says the picked row
 * (selectedFace); its list is a .drop popup bound to the box (popover.ts),
 * never the browser's own ::picker, which a drag of the frame around it would
 * close.  The list shows the select's rows, a group's under its muted heading,
 * the picked one SELECTED, the cursor's (the keys' or the pointer's)
 * ACCENT_TINT.  It opens on a press of the box (a press while open closes it)
 * or ↑ ↓ Home End Enter Space F4, and picks on a release over a row, Enter or
 * Space.  A pick sets the select and says `change` when it changed it,
 * `repick` when it picked the row already picked. */
export function comboFace(select: HTMLSelectElement): HTMLButtonElement {
  const list = element("div", "drop select-list");
  list.setAttribute("role", "listbox");
  let rows: HTMLOptionElement[] = [];
  let cursor = -1;
  let typed = "", typedAt = 0;
  const open = (): boolean => isShown(list);
  const nodes = (): HTMLElement[] => [...list.querySelectorAll<HTMLElement>(".select-row")];
  const paint = (): void => {
    nodes().forEach((node, index) => {
      node.classList.toggle("cursor", index === cursor);
      node.classList.toggle("picked", rows[index]!.selected);
    });
  };
  const render = (): void => {
    rows = [];
    const children: HTMLElement[] = [];
    const add = (option: HTMLOptionElement): void => {
      if (option.hidden) return;
      const node = element("div", "select-row");
      node.append(...[...option.childNodes].map(child => child.cloneNode(true)));
      node.dataset.index = String(rows.length);
      if (option.title) node.title = option.title;
      node.classList.toggle("disabled", option.disabled);
      rows.push(option);
      children.push(node);
    };
    for (const child of select.children) {
      if (child instanceof HTMLOptionElement) add(child);
      else if (child instanceof HTMLOptGroupElement) {
        children.push(element("div", "select-heading", child.label));
        for (const option of child.children) if (option instanceof HTMLOptionElement) add(option);
      }
    }
    list.replaceChildren(...children);
    cursor = Math.min(cursor, rows.length - 1);
    paint();
  };
  const move = (index: number): void => {
    if (index < 0 || index >= rows.length) return;
    cursor = index;
    paint();
    nodes()[index]?.scrollIntoView({ block: "nearest" });
  };
  /** The next pickable row after `from` going `step` (±1); `from` itself when there is none. */
  const pickable = (from: number, step: number): number => {
    for (let at = from + step; at >= 0 && at < rows.length; at += step) if (!rows[at]!.disabled) return at;
    return from;
  };
  const pick = (index: number): void => {
    const option = rows[index];
    if (!option || option.disabled) return;
    hidePopup(list);
    if (option.selected) { select.dispatchEvent(new Event("repick")); return; }
    option.selected = true;
    select.dispatchEvent(new Event("input", { bubbles: true }));
    select.dispatchEvent(new Event("change", { bubbles: true }));
  };
  const show = (): void => {
    if (select.disabled || open()) return;
    cursor = -1;
    render();
    showPopup(list, select);
    placeDrop(list, select);
    move(rows.findIndex(option => option.selected));
  };
  const face = selectedFace(select, () => { if (open()) render(); });
  list.addEventListener("toggle", event => select.classList.toggle("open", (event as ToggleEvent).newState === "open"));
  // The list leaves the focus on the box; a row is picked where the press is let go.
  list.addEventListener("pointerdown", event => event.preventDefault());
  list.addEventListener("pointerup", event => {
    const row = (event.target as Element).closest<HTMLElement>(".select-row");
    if (row && event.button === 0) pick(Number(row.dataset.index));
  });
  list.addEventListener("pointermove", event => {
    const row = (event.target as Element).closest<HTMLElement>(".select-row");
    const index = Number(row?.dataset.index ?? -1);
    if (row && index !== cursor && !rows[index]?.disabled) { cursor = index; paint(); }
  });
  // An owner's own press that has already said no (a cycle box arming its wheel) opens nothing.
  select.addEventListener("mousedown", event => {
    if (event.button !== 0 || event.defaultPrevented) return;
    event.preventDefault();
    select.focus();
    if (open()) hidePopup(list); else show();
  });
  select.addEventListener("keydown", event => {
    if (event.defaultPrevented) return;
    if (!open()) {
      if (["ArrowUp", "ArrowDown", "Home", "End", "Enter", " ", "F4"].includes(event.key)) { event.preventDefault(); show(); }
      // The closed box answers no letter (the browser's would pick behind the operator's back).
      else if (event.key.length === 1 && !event.ctrlKey && !event.metaKey) event.preventDefault();
      return;
    }
    switch (event.key) {
      case "ArrowDown": move(pickable(cursor, 1)); break;
      case "ArrowUp": move(pickable(cursor, -1)); break;
      case "Home": move(pickable(-1, 1)); break;
      case "End": move(pickable(rows.length, -1)); break;
      case "Enter": case " ": pick(cursor); break;
      case "F4": hidePopup(list); break;
      case "Tab": hidePopup(list); return;
      default: {
        if (event.key.length !== 1 || event.ctrlKey || event.metaKey) return;
        // QAbstractItemView.keyboardSearch: the next row starting with what was typed.
        const now = performance.now();
        typed = now - typedAt < 400 ? typed + event.key : event.key;
        typedAt = now;
        const wanted = typed.toLowerCase(), start = typed.length > 1 ? cursor : cursor + 1;
        for (let step = 0; step < rows.length; step++) {
          const index = (Math.max(0, start) + step) % rows.length;
          if (!rows[index]!.disabled && rows[index]!.text.toLowerCase().startsWith(wanted)) { move(index); break; }
        }
      }
    }
    // The open list's key, not its owner's.
    event.preventDefault();
    event.stopPropagation();
  });
  return face;
}

export class ComboBox {
  readonly root = element("select");
  private face = comboFace(this.root);
  private signature = "";
  private current: Value = null;
  private minimum = 0;

  /** `activated`: QComboBox.activated -- a pick of the row already shown is a pick too
   * (v2 PulsePreviewView: re-picking the drawn size pins it); the select reports only a change. */
  constructor(private changed: (value: Value) => void, private placeholder = "", activated = false) {
    this.root.addEventListener("change", () => {
      const raw = this.root.value;
      const value = raw === "" ? null : JSON.parse(raw) as Value;
      this.current = value;
      this.root.classList.toggle("blank", value === null);
      this.changed(value);
    });
    if (activated) this.root.addEventListener("repick", () => { if (this.root.value !== "") this.changed(this.current); });
  }

  setChoices(choices: readonly ChoiceInput[]): void {
    const list = choices.map(normal);
    const signature = encode(list as unknown as Value) + this.placeholder;
    if (signature === this.signature) return;
    this.signature = signature;
    const nodes: HTMLElement[] = [];
    let group: HTMLOptGroupElement | null = null;
    for (const choice of list) {
      const option = new Option(choice.label, encode(choice.value));
      if (choice.title) option.title = choice.title;
      option.disabled = Boolean(choice.disabled);
      if (choice.group) {
        if (!group || group.label !== choice.group) { group = element("optgroup"); group.label = choice.group; nodes.push(group); }
        group.append(option);
      } else { group = null; nodes.push(option); }
    }
    // The empty choice: the placeholder in MUTED_INK, never offered in the list.
    const blank = new Option(this.placeholder, "");
    blank.hidden = true;
    this.root.replaceChildren(this.face, blank, ...nodes);
    this.show();
    this.hint(list);
  }

  /** setMinimumContentsLength: never narrower than `length` "M"s. */
  minimumContents(length: number): this { this.minimum = length; this.hint(); return this; }

  private hint(list: readonly { label: string; disabled?: boolean }[] = [...this.root.options].filter(option => !option.hidden).map(option => ({ label: option.text, disabled: option.disabled }))): void {
    hintWidest(this.root, [this.placeholder, ...list.filter(choice => !choice.disabled).map(choice => choice.label)], this.minimum);
  }

  get value(): Value { return this.current; }
  set value(value: Value) { this.current = value; this.show(); }
  set disabled(disabled: boolean) { assign(this.root, "disabled", disabled); }
  set title(text: string) { if (this.root.title !== text) this.root.title = text; }

  private show(): void {
    const key = this.current === null ? "" : encode(this.current);
    if (this.root.value !== key) this.root.value = key;
    if (this.root.selectedIndex < 0) this.root.value = "";
    this.root.classList.toggle("blank", this.root.value === "");
  }
}

/** An editable FluentComboBox (QComboBox.setEditable): a name typed into its
 * field or picked from its list.  `commit` hears the text when it is picked,
 * on Enter or when the field is left; `fill` (fill_editable_combo) shows the
 * accepted name and never takes a draft out from under the cursor. */
export class EditableCombo {
  readonly root = element("div", "edit-combo");
  readonly input = element("input");
  private list = element("div", "drop edit-combo-list");
  private names: string[] = [];
  private accepted = "";
  private draft = false;

  constructor(private commit: (text: string) => void, placeholder = "") {
    this.input.type = "text";
    this.input.spellcheck = false;
    this.input.placeholder = placeholder;
    const open = element("button", "edit-combo-open");
    open.type = "button";
    open.tabIndex = -1;
    this.root.append(this.input, open);
    // The chevron opens the list and, while it is open, closes it.
    open.addEventListener("pointerdown", event => event.preventDefault());
    open.addEventListener("click", () => { if (isShown(this.list)) hidePopup(this.list); else this.showList(); });
    this.list.addEventListener("toggle", event => this.root.classList.toggle("open", (event as ToggleEvent).newState === "open"));
    this.input.addEventListener("input", () => { this.draft = true; });
    this.input.addEventListener("keydown", event => { if (event.key === "Enter") this.finish(); });
    this.input.addEventListener("blur", () => this.finish());
  }

  private finish(): void {
    if (!this.draft) return;
    this.draft = false;
    const text = this.input.value.trim();
    if (text !== this.accepted) this.commit(text);
  }

  private showList(): void {
    if (this.input.disabled || !this.names.length) return;
    this.list.replaceChildren(...this.names.map(name => {
      const row = element("div", "edit-combo-row", name);
      row.classList.toggle("picked", name === this.input.value);
      row.addEventListener("pointerup", () => {
        hidePopup(this.list);
        this.input.value = name;
        this.draft = false;
        if (name !== this.accepted) this.commit(name);
      });
      return row;
    }));
    // Below the field, or above it where the list fits better (placeDrop).
    showPopup(this.list, this.root);
    placeDrop(this.list, this.root);
  }

  /** v2 fill_editable_combo: offer `names`, show `key` (the accepted one) unless a draft is being typed. */
  fill(names: readonly string[], key: string, keepDraft = true): void {
    this.names = [...names];
    this.accepted = key;
    const typing = keepDraft && this.draft && document.activeElement === this.input;
    if (!typing) { this.draft = false; if (this.input.value !== key) this.input.value = key; }
  }

  set disabled(disabled: boolean) { assign(this.input, "disabled", disabled); this.root.classList.toggle("disabled", disabled); if (disabled) hidePopup(this.list); }
  set title(text: string) { if (this.input.title !== text) this.input.title = text; }
}
