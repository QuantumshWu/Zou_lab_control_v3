import { element, setText, setTitle, token } from "./dom.js";
import { checkBox, setChecked } from "./toggles.js";

/** One row of a checkable list; a `header` row titles the rows under it
 * (and, when `selects`, ticks or clears all of them). */
export type CheckItem = { value: string; label: string; header?: boolean; selects?: boolean; colour?: string; disabled?: boolean };
/** A list made only when the popup opens (a long pick list), named by `key`. */
export type LazyItems = { key: string; items(): CheckItem[] };

/** FluentComboBox set_checkable: a field like a combo whose popup ticks
 * several of a list.  The closed box reads the ticked values; `edited`
 * comes once, when the popup closes having changed them (one edit per
 * gesture, never per tick). */
export class CheckCombo {
  readonly root = element("button", "check-combo");
  private words = element("span", "check-combo-text");
  private list = element("div", "menu check-menu");
  private items: CheckItem[] = [];
  private lazy: (() => CheckItem[]) | null = null;
  private checked: string[] = [];
  private opened: string[] | null = null;
  private signature = "";
  edited: (values: string[]) => void = () => {};
  /** The popup closed; `changed` whether it wrote anything. */
  closed: (changed: boolean) => void = () => {};

  constructor(private placeholder = "choose") {
    this.root.type = "button";
    this.root.append(this.words);
    // A press while the list is open closes it (light dismiss): its click does not open it again.
    let wasOpen = false;
    this.root.addEventListener("pointerdown", () => { wasOpen = this.list.matches(":popover-open"); });
    this.root.addEventListener("click", () => { if (!wasOpen) this.open(); wasOpen = false; });
    this.list.popover = "auto";
    this.list.role = "listbox";
    this.list.addEventListener("toggle", event => {
      const open = (event as ToggleEvent).newState === "open";
      // The field says it is open while its list is (v2: the ACCENT edge while its popup shows).
      this.root.classList.toggle("open", open);
      if (open || !this.opened) return;
      const before = this.opened;
      this.opened = null;
      const changed = before.length !== this.checked.length || before.some((value, index) => value !== this.checked[index]);
      if (changed) this.edited([...this.checked]);
      this.closed(changed);
    });
  }

  set disabled(disabled: boolean) { this.root.disabled = disabled; }

  /** The list and what is ticked; a tick the list lacks is kept and shown.
   * A lazy list is made when the popup next opens. */
  set(items: readonly CheckItem[] | LazyItems, checked: readonly string[]): void {
    const lazy = "key" in items ? items : null;
    const signature = lazy ? `lazy:${lazy.key}` : JSON.stringify(items);
    if (signature !== this.signature) {
      this.signature = signature;
      if (lazy) { this.lazy = lazy.items; this.items = []; this.list.replaceChildren(); }
      else { this.lazy = null; this.items = [...items as readonly CheckItem[]]; this.build(); }
    }
    if (!this.opened) this.checked = [...checked];
    this.show();
  }

  open(): void {
    if (this.lazy) { this.items = this.lazy(); this.lazy = null; this.build(); }
    if (this.root.disabled || !this.items.length) return;
    this.opened = [...this.checked];
    document.body.append(this.list);
    this.list.showPopover();
    const rect = this.root.getBoundingClientRect(), gap = token("--popup-gap"), pad = token("--window-pad");
    this.list.style.minWidth = `${rect.width}px`;
    const height = this.list.offsetHeight, width = this.list.offsetWidth;
    const below = rect.bottom + gap;
    this.list.style.left = `${Math.max(pad, Math.min(rect.left, innerWidth - pad - width))}px`;
    this.list.style.top = `${below + height <= innerHeight - pad ? below : Math.max(pad, rect.top - gap - height)}px`;
  }

  private build(): void {
    const rows: HTMLElement[] = [];
    let group: string[] = [];
    let groupHead: HTMLButtonElement | null = null;
    const close = (): void => { if (groupHead) groupHead.dataset.values = JSON.stringify(group); };
    for (const item of this.items) {
      if (item.header) {
        close();
        group = [];
        const head = element("button", `check-menu-header${item.selects ? " selects" : ""}`, item.label);
        head.type = "button";
        if (item.colour) head.style.color = item.colour;
        head.disabled = !item.selects;
        if (item.selects) head.addEventListener("click", () => this.tickAll(JSON.parse(head.dataset.values ?? "[]") as string[]));
        groupHead = head;
        rows.push(head);
        continue;
      }
      group.push(item.value);
      const box = checkBox(item.label, on => this.tick(item.value, on));
      box.classList.add("check-menu-item");
      box.dataset.value = item.value;
      box.disabled = Boolean(item.disabled);
      rows.push(box);
    }
    close();
    this.list.replaceChildren(...rows);
  }

  private tick(value: string, on: boolean): void {
    const rest = this.checked.filter(item => item !== value);
    this.checked = on ? this.ordered([...rest, value]) : rest;
    this.show();
  }

  /** A selectable header ticks its whole group, or clears it when all are ticked. */
  private tickAll(values: string[]): void {
    const all = values.every(value => this.checked.includes(value));
    const rest = this.checked.filter(item => !values.includes(item));
    this.checked = all ? rest : this.ordered([...rest, ...values]);
    this.show();
  }

  /** Ticked values in the list's order, a kept one the list lacks at the end. */
  private ordered(values: string[]): string[] {
    const order = this.items.filter(item => !item.header).map(item => item.value);
    return [...values].sort((a, b) => (order.indexOf(a) + 1 || Infinity) - (order.indexOf(b) + 1 || Infinity));
  }

  private show(): void {
    for (const box of this.list.querySelectorAll<HTMLButtonElement>(".check-menu-item")) setChecked(box, this.checked.includes(box.dataset.value ?? ""));
    const text = this.checked.join(", ");
    setText(this.words, text || this.placeholder);
    this.root.classList.toggle("blank", !text);
    setTitle(this.root, text);
  }
}
