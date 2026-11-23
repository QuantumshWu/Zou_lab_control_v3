import { hintWidest, placeDrop, selectedFace } from "./combo.js";
import { assign, element } from "./dom.js";
import { hidePopup, isShown, showPopup } from "./popover.js";
import { flattenTree, type FlatTreeGroup, type TreeGroup, type TreeLeaf } from "./tree_model.js";
export type { TreeGroup, TreeLeaf } from "./tree_model.js";

/** One keyed choice: its opaque key, its row's words, and what the closed box
 * says when it is picked (v2 choice_tree_groups' `full_label`). */
/** One heading of the tree and the choices under it (a producer's signals, a device's knobs, a unit's family). */
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

/** One row the open tree shows: a heading (its index in the groups), a leaf, or the none row;
 * `id` says which, to find it again among the rows a change reveals. */
type Row = ({ kind: "heading"; group: number } | { kind: "leaf"; group: number; leaf: TreeLeaf } | { kind: "none" }) & { id: string };
type Heading = Extract<Row, { kind: "heading" }>;

/** Where a row (`id`; null: the rows' own top, unscrolled) stood on the screen before the rows changed. */
type Mark = { id: string | null; top: number };

/** FluentTreeComboBox: a keyed-choice box whose drop-down is a collapsible
 * tree -- a heading per owner, its choices on its branch.  Closed, it says the
 * picked choice's `text` (or the none row's words, or its placeholder, in
 * MUTED_INK).  Open: a click anywhere on a heading opens or closes it, the
 * heading standing still (hold); a click on a choice picks it.  The picked
 * choice's heading is open when the popup opens, and what is open stays open.
 * Keys: ↑/↓/Home/End open it; in it ↑/↓/Home/End move, → opens a heading (or
 * goes into it), ← closes it (or goes up to it), Enter picks, Esc closes, and
 * typing finds the next shown row starting with what was typed.  `changed` is
 * the operator's pick; `value =` / `setTree` project one and never emit. */
export class TreeCombo {
  readonly root = element("select", "tree-combo");
  private face = selectedFace(this.root);
  /** The box's empty value: the none row's words, else the placeholder. */
  private blank = new Option("", "");
  private list = element("div", "drop tree-combo-list");
  private groups: FlatTreeGroup[] = [];
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
  /** The side this opening chose and the room it leaves the list (placeDrop): kept while it is open. */
  private drop = { up: false, room: 0 };

  constructor(private changed: (key: string) => void, private placeholder = "") {
    this.blank.hidden = true;
    this.root.replaceChildren(this.face, this.blank);
    this.list.setAttribute("role", "tree");
    // The box's own list never opens: the tree is its drop-down, opened by a
    // press of the box and closed by a press while open.
    this.root.addEventListener("mousedown", event => {
      if (event.button !== 0) return;
      event.preventDefault();
      if (this.root.disabled) return;
      this.root.focus();
      if (this.open) hidePopup(this.list); else this.show();
    });
    this.root.addEventListener("keydown", event => this.key(event));
    this.list.addEventListener("toggle", event => this.root.classList.toggle("open", (event as ToggleEvent).newState === "open"));
    this.list.addEventListener("pointerdown", event => event.preventDefault());
    this.list.addEventListener("pointerup", event => {
      const row = (event.target as Element).closest<HTMLElement>(".tree-row");
      if (row && event.button === 0) this.activate(Number(row.dataset.index), true);
    });
  }

  private get open(): boolean { return isShown(this.list); }

  /** Reconcile the tree (v2 set_choice_tree): `none` is the blank row's words
   * ("Off", "(not selected)"), absent when a choice is required. */
  setTree(groups: readonly TreeGroup[], current: string, none: string | null = null): void {
    const signature = JSON.stringify([groups, none, this.placeholder]);
    if (signature !== this.signature) {
      // Open, the cursor's row stands still (else the first row in sight).
      const marks = this.open ? this.marks([this.rows[this.cursor]?.id, ...this.inSight()]) : [];
      this.signature = signature;
      this.groups = flattenTree(groups);
      this.none = none;
      // Every choice is an option of the box, never listed by it: the closed box says
      // the picked one's text and is as wide as the widest it may say (v2 _collapsed_text_candidates).
      const options = this.groups.flatMap(group => group.leaves.map(leaf => { const option = new Option(leaf.text, leaf.key); option.hidden = true; return option; }));
      this.root.replaceChildren(this.face, this.blank, ...options);
      this.blank.text = none ?? this.placeholder;
      hintWidest(this.root, [this.placeholder, ...(none === null ? [] : [none]), ...options.map(option => option.text)], this.minimum);
      if (this.open) { this.render(); this.hold(marks); }
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
    // The open tree keeps its rows (a press on one is let go on the same one): only which is picked changes.
    if (this.open) this.paintPicked();
  }
  set disabled(disabled: boolean) { assign(this.root, "disabled", disabled); if (disabled) hidePopup(this.list); }
  set title(text: string) { if (this.root.title !== text) this.root.title = text; }
  dispose(): void { hidePopup(this.list); this.list.remove(); }
  private openPath(index: number): void {
    for (let at: number | null = index; at !== null; at = this.groups[at]!.parent) this.expanded.add(this.groups[at]!.path);
  }

  private leaf(key: string): TreeLeaf | undefined {
    for (const group of this.groups) for (const leaf of group.leaves) if (leaf.key === key) return leaf;
    return undefined;
  }

  /** v2 showPopup: the picked choice's heading opened, the popup below the box
   * (above it where it fits better: placeDrop). */
  private show(): void {
    if (this.root.disabled) return;
    const at = this.groups.findIndex(group => group.leaves.some(leaf => leaf.key === this.current));
    if (at >= 0) this.openPath(at);
    showPopup(this.list, this.root);
    this.cursor = -1;
    this.render();
    this.drop = placeDrop(this.list, this.root);
    // The picked row is the tree's selection (SELECTED) when it opens.
    this.move(this.rows.findIndex(row => this.picked(row)));
  }

  /** The rows the tree's open headings reveal, the cursor kept on its row. */
  private render(): void {
    const cursor = this.rows[this.cursor]?.id;
    const rows: Row[] = [];
    if (this.none !== null) rows.push({ kind: "none", id: "none" });
    this.groups.forEach((group, index) => {
      for (let parent = group.parent; parent !== null; parent = this.groups[parent]!.parent) if (!this.expanded.has(this.groups[parent]!.path)) return;
      rows.push({ kind: "heading", group: index, id: `h:${group.path}` });
      if (this.expanded.has(group.path)) for (const leaf of group.leaves) rows.push({ kind: "leaf", group: index, leaf, id: `l:${leaf.key}` });
    });
    this.rows = rows;
    this.cursor = rows.findIndex(row => row.id === cursor);
    this.list.replaceChildren(...rows.map((row, index) => {
      const node = element("div", `tree-row ${row.kind}`);
      node.dataset.index = String(index);
      const depth = row.kind === "none" ? 0 : this.groups[row.group]!.depth + (row.kind === "leaf" ? 1 : 0);
      node.style.setProperty("--tree-depth", String(depth)); node.setAttribute("aria-level", String(depth + 1));
      const item = element("span", "tree-item");
      if (row.kind === "heading") {
        const open = this.expanded.has(this.groups[row.group]!.path);
        node.classList.toggle("open", open);
        node.setAttribute("role", "treeitem");
        node.setAttribute("aria-expanded", String(open));
        node.append(element("span", "tree-branch"));
        item.textContent = this.groups[row.group]!.heading;
      } else {
        node.setAttribute("role", "treeitem");
        item.textContent = row.kind === "leaf" ? row.leaf.label : this.none!;
        node.setAttribute("aria-selected", String(this.picked(row)));
      }
      node.append(item);
      return node;
    }));
    this.paintCursor();
  }

  /** The shown row `id` says, if it is shown. */
  private node(id: string | null | undefined): Element | undefined {
    return this.list.children[this.rows.findIndex(row => row.id === id)];
  }

  /** The rows in sight in the open list, top first. */
  private inSight(): string[] {
    const frame = this.list.getBoundingClientRect();
    return this.rows.filter((_, index) => { const rect = this.list.children[index]!.getBoundingClientRect(); return rect.bottom > frame.top && rect.top < frame.bottom; })
      .map(row => row.id);
  }

  /** Where `ids`' rows stand on the screen (an id no row has, none), then the rows' own top. */
  private marks(ids: readonly (string | undefined)[]): Mark[] {
    const marks: Mark[] = [];
    for (const id of ids) { const node = this.node(id); if (node) marks.push({ id: id!, top: node.getBoundingClientRect().top }); }
    marks.push({ id: null, top: this.list.getBoundingClientRect().top - this.list.scrollTop });
    return marks;
  }

  /** The rows changed while it is open (a heading opened or closed, a new tree): the side and
   * its room stay the opening's, the cap stays on, and the first of `marks` still shown stands
   * where it stood (A), o down the rows.  Below the box the list keeps its top, as tall as its
   * rows up to the room, and scrolls the mark back to A; above it the list keeps its bottom (B)
   * and grows up no further than keeps the mark at A (B + o - A), the rest scrolling.  Its
   * height is the layout's (its rows up to the cap), never measured with the cap lifted. */
  private hold(marks: readonly Mark[]): void {
    const list = this.list, frame = list.getBoundingClientRect();
    const mark = marks.find(item => item.id === null || this.node(item.id))!;
    const node = this.node(mark.id);
    const offset = node ? node.getBoundingClientRect().top - frame.top + list.scrollTop : 0;
    const room = this.drop.up ? Math.min(this.drop.room, frame.bottom + offset - mark.top) : this.drop.room;
    // A height is never below nothing (a negative one would lift the cap).
    list.style.setProperty("--drop-room", `${Math.max(0, room)}px`);
    list.scrollTop = list.getBoundingClientRect().top + offset - mark.top;
  }

  private picked(row: Row): boolean { return row.kind === "leaf" ? row.leaf.key === this.current : row.kind === "none" && this.current === ""; }

  private paintPicked(): void {
    this.rows.forEach((row, index) => { if (row.kind !== "heading") this.list.children[index]?.setAttribute("aria-selected", String(this.picked(row))); });
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

  /** A heading opens or closes (standing still: hold); a choice is picked. */
  private activate(index: number, pointer: boolean): void {
    const row = this.rows[index];
    if (!row) return;
    if (row.kind === "heading") {
      this.toggle(row);
      return;
    }
    const key = row.kind === "leaf" ? row.leaf.key : "";
    hidePopup(this.list);
    const changed = key !== this.current;
    this.value = key;
    if (changed) this.changed(key);
  }

  private toggle(row: Heading, open = !this.expanded.has(this.groups[row.group]!.path)): void {
    const heading = this.groups[row.group]!.path;
    if (open === this.expanded.has(heading)) return;
    const marks = this.marks([row.id]);
    if (open) this.expanded.add(heading); else this.expanded.delete(heading);
    this.render();
    this.hold(marks);
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
          if (!this.expanded.has(this.groups[row.group]!.path)) this.toggle(row, true);
          else if (this.rows[this.cursor + 1]) this.move(this.cursor + 1);
        }
        break;
      case "ArrowLeft":
        if (row?.kind === "heading") {
          if (this.expanded.has(this.groups[row.group]!.path)) this.toggle(row, false);
          else { const parent = this.groups[row.group]!.parent; if (parent !== null) this.move(this.rows.findIndex(item => item.kind === "heading" && item.group === parent)); }
        }
        else if (row?.kind === "leaf") this.move(this.rows.findIndex(item => item.kind === "heading" && item.group === row.group));
        break;
      case "Enter": if (row) this.activate(this.cursor, false); break;
      case "Tab": hidePopup(this.list); return;
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
