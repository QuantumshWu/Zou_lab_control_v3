import { comboFace, hintWidest } from "./combo.js";
import { element } from "./dom.js";

/** One keyed choice: its opaque key, its row's words, and what the closed box
 * says when it is picked (v2 choice_tree_groups' `full_label`). */
export type TreeLeaf = { key: string; label: string; text: string };
/** One heading of the tree and the choices under it (a producer's signals, a device's knobs, a unit's family). */
export type TreeGroup = { heading: string; leaves: TreeLeaf[] };
/** A choice as its owner offers it: `group` is the heading it is filed under. */
export type TreeEntry = { key: string; label?: string; group?: string; text?: string };

/** Python's str order (code points), as v2 sorts its choices. */
const codepoints = (left: string, right: string): number => left < right ? -1 : left > right ? 1 : 0;

/** v2 choice_tree_groups: every choice once, under the first heading that files
 * it (`unfiled` when none does), the headings in case-blind order with `unfiled`
 * last and each one's choices by key; a current key nobody offers still stands,
 * under `unfiled`.  A leaf reads its label (else its key); the closed box its
 * `text`, else "<heading> · <label>". */
export function choiceTree(entries: readonly TreeEntry[], current: string, unfiled: string): TreeGroup[] {
  const offered = new Map<string, TreeEntry>();
  for (const entry of entries) if (entry.key && !offered.has(entry.key)) offered.set(entry.key, entry);
  if (current && !offered.has(current)) offered.set(current, { key: current });
  const groups = new Map<string, TreeLeaf[]>();
  for (const key of [...offered.keys()].sort(codepoints)) {
    const entry = offered.get(key)!;
    const heading = entry.group || unfiled;
    const label = entry.label || key;
    let leaves = groups.get(heading);
    if (!leaves) groups.set(heading, leaves = []);
    leaves.push({ key, label, text: entry.text || `${heading} · ${label}` });
  }
  return [...groups.keys()].sort((left, right) => (Number(left === unfiled) - Number(right === unfiled)) || codepoints(left.toLowerCase(), right.toLowerCase()))
    .map(heading => ({ heading, leaves: groups.get(heading)! }));
}

/** One row the open tree shows: a heading (its index in the groups), a leaf, or the none row. */
type Row = { kind: "heading"; group: number } | { kind: "leaf"; group: number; leaf: TreeLeaf } | { kind: "none" };

/** FluentTreeComboBox: a keyed-choice box whose drop-down is a collapsible
 * tree -- a heading per owner, its choices on its branch.  Closed, it says the
 * picked choice's `text` (or the none row's words, or its placeholder, in
 * MUTED_INK).  Open: a click anywhere on a heading opens or closes it and the
 * popup grows to show what is open; a click on a choice picks it.  The picked
 * choice's heading is open when the popup opens, and what is open stays open.
 * Keys: ↑/↓/Home/End open it; in it ↑/↓/Home/End move, → opens a heading (or
 * goes into it), ← closes it (or goes up to it), Enter picks, Esc closes, and
 * typing finds the next shown row starting with what was typed.  `changed` is
 * the operator's pick; `value =` / `setTree` project one and never emit. */
export class TreeCombo {
  readonly root = element("select", "tree-combo");
  private face = comboFace();
  /** The box's empty value: the none row's words, else the placeholder. */
  private blank = new Option("", "");
  private list = element("div", "tree-combo-list");
  private groups: TreeGroup[] = [];
  private none: string | null = null;
  private current = "";
  private signature = "";
  private minimum = 0;
  /** The headings open in the popup: kept across openings and refills. */
  private expanded = new Set<string>();
  private rows: Row[] = [];
  private cursor = -1;
  private typed = "";
  private typedAt = 0;

  constructor(private changed: (key: string) => void, private placeholder = "") {
    this.blank.hidden = true;
    this.root.replaceChildren(this.face, this.blank);
    this.list.popover = "auto";
    this.list.setAttribute("role", "tree");
    // The box's own list never opens: the tree is its drop-down, opened by the
    // click (after the press's light dismiss), closed by a click while open.
    let wasOpen = false;
    this.root.addEventListener("pointerdown", event => {
      if (event.button !== 0) return;
      event.preventDefault();
      wasOpen = this.open;
      if (!this.root.disabled) this.root.focus();
    });
    this.root.addEventListener("mousedown", event => event.preventDefault());
    this.root.addEventListener("click", event => {
      event.preventDefault();
      if (wasOpen) { if (this.open) this.list.hidePopover(); } else this.show();
      wasOpen = false;
    });
    this.root.addEventListener("keydown", event => this.key(event));
    this.list.addEventListener("toggle", event => this.root.classList.toggle("open", (event as ToggleEvent).newState === "open"));
    this.list.addEventListener("pointerdown", event => event.preventDefault());
    this.list.addEventListener("pointerup", event => {
      const row = (event.target as Element).closest<HTMLElement>(".tree-row");
      if (row && event.button === 0) this.activate(Number(row.dataset.index), true);
    });
  }

  private get open(): boolean { return this.list.matches(":popover-open"); }

  /** Reconcile the tree (v2 set_choice_tree): `none` is the blank row's words
   * ("Off", "(not selected)"), absent when a choice is required. */
  setTree(groups: readonly TreeGroup[], current: string, none: string | null = null): void {
    const signature = JSON.stringify([groups, none, this.placeholder]);
    if (signature !== this.signature) {
      this.signature = signature;
      this.groups = groups.map(group => ({ heading: group.heading, leaves: [...group.leaves] }));
      this.none = none;
      // Every choice is an option of the box, never listed by it: the closed box says
      // the picked one's text and is as wide as the widest it may say (v2 _collapsed_text_candidates).
      const options = this.groups.flatMap(group => group.leaves.map(leaf => { const option = new Option(leaf.text, leaf.key); option.hidden = true; return option; }));
      this.root.replaceChildren(this.face, this.blank, ...options);
      this.blank.text = none ?? this.placeholder;
      hintWidest(this.root, [this.placeholder, ...(none === null ? [] : [none]), ...options.map(option => option.text)], this.minimum);
      if (this.open) this.render();
    }
    this.value = current;
  }

  /** setMinimumContentsLength: never narrower than `length` "M"s. */
  minimumContents(length: number): this { this.minimum = length; this.signature = ""; return this; }

  get value(): string { return this.current; }
  set value(key: string) {
    this.current = key;
    // The none row picked, nothing picked, or a key nobody offers: the box's empty value.
    const leaf = key !== "" && this.leaf(key) !== undefined;
    if (this.root.value !== (leaf ? key : "")) this.root.value = leaf ? key : "";
    this.root.classList.toggle("blank", !leaf);
    if (this.open) this.render();
  }
  set disabled(disabled: boolean) { this.root.disabled = disabled; if (disabled && this.open) this.list.hidePopover(); }
  set title(text: string) { if (this.root.title !== text) this.root.title = text; }

  private leaf(key: string): TreeLeaf | undefined {
    for (const group of this.groups) for (const leaf of group.leaves) if (leaf.key === key) return leaf;
    return undefined;
  }

  /** v2 showPopup: the picked choice's heading opened, the popup below the box. */
  private show(): void {
    if (this.root.disabled) return;
    const at = this.groups.findIndex(group => group.leaves.some(leaf => leaf.key === this.current));
    if (at >= 0) this.expanded.add(this.groups[at]!.heading);
    if (!this.list.isConnected) document.body.append(this.list);
    this.list.showPopover();
    this.cursor = -1;
    this.render();
    // The picked row is the tree's selection (SELECTED) when it opens.
    this.move(this.rows.findIndex(row => row.kind === "leaf" ? row.leaf.key === this.current : row.kind === "none" && this.current === ""));
  }

  /** What a row is, to find it again among the rows a toggle reveals. */
  private identity(row: Row | undefined): string {
    return row === undefined ? "" : row.kind === "none" ? "none" : row.kind === "heading" ? `h:${row.group}` : `l:${row.leaf.key}`;
  }

  /** The rows the tree's open headings reveal, then the popup's size and place (v2 _resize_popup_to_contents). */
  private render(): void {
    const rows: Row[] = [];
    if (this.none !== null) rows.push({ kind: "none" });
    this.groups.forEach((group, index) => {
      rows.push({ kind: "heading", group: index });
      if (this.expanded.has(group.heading)) for (const leaf of group.leaves) rows.push({ kind: "leaf", group: index, leaf });
    });
    this.rows = rows;
    this.list.replaceChildren(...rows.map((row, index) => {
      const node = element("div", `tree-row ${row.kind}`);
      node.dataset.index = String(index);
      const item = element("span", "tree-item");
      if (row.kind === "heading") {
        const open = this.expanded.has(this.groups[row.group]!.heading);
        node.classList.toggle("open", open);
        node.setAttribute("role", "treeitem");
        node.setAttribute("aria-expanded", String(open));
        node.append(element("span", "tree-branch"));
        item.textContent = this.groups[row.group]!.heading;
      } else {
        node.setAttribute("role", "treeitem");
        item.textContent = row.kind === "leaf" ? row.leaf.label : this.none!;
      }
      node.append(item);
      return node;
    }));
    this.paintCursor();
    this.place();
  }

  private place(): void {
    const field = this.root.getBoundingClientRect();
    const style = this.list.style;
    style.minWidth = `${field.width}px`;
    style.maxWidth = `${innerWidth}px`;
    const top = field.bottom;
    style.top = `${top}px`;
    // Every row it reveals, as far down as the window goes; the rest scrolls (never above the box).
    style.maxHeight = `${Math.max(0, innerHeight - top)}px`;
    const width = this.list.offsetWidth;
    const left = width <= field.width ? field.left : field.right - width;
    style.left = `${Math.max(0, Math.min(left, innerWidth - width))}px`;
  }

  private paintCursor(): void {
    [...this.list.children].forEach((node, index) => node.classList.toggle("current", index === this.cursor));
  }

  private move(index: number, scroll = true): void {
    if (index < 0 || index >= this.rows.length) return;
    this.cursor = index;
    this.paintCursor();
    if (scroll) this.list.children[index]?.scrollIntoView({ block: "nearest" });
  }

  /** A heading opens or closes (and the popup grows to fit); a choice is picked. */
  private activate(index: number, pointer: boolean): void {
    const row = this.rows[index];
    if (!row) return;
    if (row.kind === "heading") {
      if (!pointer) return;
      this.toggle(row.group);
      return;
    }
    const key = row.kind === "leaf" ? row.leaf.key : "";
    this.list.hidePopover();
    const changed = key !== this.current;
    this.value = key;
    if (changed) this.changed(key);
  }

  private toggle(group: number, open = !this.expanded.has(this.groups[group]!.heading)): void {
    const heading = this.groups[group]!.heading;
    if (open === this.expanded.has(heading)) return;
    if (open) this.expanded.add(heading); else this.expanded.delete(heading);
    const at = this.identity(this.rows[this.cursor]);
    this.render();
    this.cursor = this.rows.findIndex(row => this.identity(row) === at);
    this.paintCursor();
  }

  private key(event: KeyboardEvent): void {
    if (!this.open) {
      if (["ArrowUp", "ArrowDown", "Home", "End", "Enter", " ", "F4"].includes(event.key)) { event.preventDefault(); this.show(); }
      // The box's own list never answers a letter.
      else if (event.key.length === 1 && !event.ctrlKey && !event.metaKey) event.preventDefault();
      return;
    }
    const row = this.rows[this.cursor];
    const last = this.rows.length - 1;
    switch (event.key) {
      case "ArrowDown": this.move(Math.min(last, this.cursor + 1)); break;
      case "ArrowUp": this.move(Math.max(0, this.cursor - 1)); break;
      case "Home": this.move(0); break;
      case "End": this.move(last); break;
      case "ArrowRight":
        if (row?.kind === "heading") {
          if (!this.expanded.has(this.groups[row.group]!.heading)) this.toggle(row.group, true);
          else if (this.rows[this.cursor + 1]?.kind === "leaf") this.move(this.cursor + 1);
        }
        break;
      case "ArrowLeft":
        if (row?.kind === "heading") this.toggle(row.group, false);
        else if (row?.kind === "leaf") this.move(this.rows.findIndex(item => item.kind === "heading" && item.group === row.group));
        break;
      case "Enter": if (row) this.activate(this.cursor, false); break;
      case "Escape": this.list.hidePopover(); break;
      case "Tab": this.list.hidePopover(); return;
      default:
        if (event.key.length !== 1 || event.ctrlKey || event.metaKey) return;
        this.search(event.key);
    }
    // The open tree's key, not its owner's (an Esc closes the tree, not the Setting around it).
    event.preventDefault();
    event.stopPropagation();
  }

  /** QAbstractItemView.keyboardSearch: the next shown row starting with what was typed. */
  private search(character: string): void {
    const now = performance.now();
    this.typed = now - this.typedAt < 400 ? this.typed + character : character;
    this.typedAt = now;
    const text = (row: Row): string => (row.kind === "heading" ? this.groups[row.group]!.heading : row.kind === "leaf" ? row.leaf.label : this.none ?? "").toLowerCase();
    const wanted = this.typed.toLowerCase();
    const start = this.typed.length > 1 ? this.cursor : this.cursor + 1;
    for (let step = 0; step < this.rows.length; step++) {
      const index = (Math.max(0, start) + step) % this.rows.length;
      if (text(this.rows[index]!).startsWith(wanted)) { this.move(index); return; }
    }
  }
}
