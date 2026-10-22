import { command } from "./button.js";
import { FlowGraph, type Graph } from "./flow_graph.js";
import { element, labelColumnWidth, setText, token } from "./dom.js";
import { wordsWidth } from "./ink.js";
import { elideMiddle, fontOf, measurer } from "./elide.js";
import { Menu } from "./menu.js";
import { StatusStrip } from "./status.js";
import { TabWidget } from "./tabs.js";

/** A row's value: a scalar, a list, a record (`{"$fields": [[name, value], ...]}` when
 * its fields read in their written order, not by name), or an action the row offers. */
export type InfoValue = string | number | boolean | null | InfoValue[] | { [name: string]: InfoValue };
export type InfoRow = readonly [label: string, value: InfoValue];
export type InfoTab = readonly [title: string, rows: readonly InfoRow[]];

/** A flat list longer than this reads as its count and range. */
const LIST_PREVIEW = 8;

const isScalar = (value: InfoValue): boolean => value === null || typeof value !== "object";
const isFlatList = (value: InfoValue): value is InfoValue[] => Array.isArray(value) && value.every(isScalar);
const isAction = (value: InfoValue): value is { text: string; action: string } =>
  value !== null && typeof value === "object" && !Array.isArray(value)
  && Object.keys(value).length === 2 && typeof value.text === "string" && typeof value.action === "string";
const isComposite = (value: InfoValue): boolean => !isAction(value) && value !== null && typeof value === "object" && !isFlatList(value);
const isOrdered = (value: InfoValue): value is { $fields: [string, InfoValue][] } =>
  value !== null && typeof value === "object" && !Array.isArray(value) && Object.keys(value).length === 1 && Array.isArray((value as { $fields?: unknown }).$fields);
/** A composite's (name, value) entries: a list by index, an ordered record as written, a record by name. */
const entriesOf = (value: InfoValue): (readonly [string, InfoValue])[] => Array.isArray(value) ? value.map((item, index) => [`[${index}]`, item] as const)
  : isOrdered(value) ? value.$fields.map(([name, item]) => [name, item] as const) : Object.entries(value as Record<string, InfoValue>);

/** One value as it reads beside its name (info_pane.value_text). */
export function valueText(value: InfoValue): string {
  if (value === null) return "none";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : String(Number(value.toPrecision(6)));
  if (typeof value === "string") return value;
  if (isFlatList(value)) {
    if (!value.length) return "(none)";
    const numbers = value.filter((item): item is number => typeof item === "number");
    if (value.length > LIST_PREVIEW && numbers.length === value.length)
      return `${value.length} numbers, ${valueText(Math.min(...numbers))} to ${valueText(Math.max(...numbers))}`;
    const shown = value.slice(0, LIST_PREVIEW).map(valueText).join(", ");
    return value.length > LIST_PREVIEW ? `${shown}, … (${value.length} in all)` : shown;
  }
  if (Array.isArray(value)) return `${value.length} items`;
  return `${entriesOf(value).length} fields`;
}

/** The whole value for the clipboard, nested by indentation. */
export function copyText(value: InfoValue, indent = ""): string {
  if (isFlatList(value)) return value.map(valueText).join(", ") || "(none)";
  if (value !== null && typeof value === "object") {
    return entriesOf(value).map(([name, item]) => isComposite(item)
      ? `${indent}${name}:\n${copyText(item, `${indent}  `)}` : `${indent}${name}: ${copyText(item)}`).join("\n");
  }
  return valueText(value);
}

type Node = { row: HTMLDivElement; label: HTMLSpanElement; name: string; path: string; value: InfoValue; depth: number; children: Node[]; open: boolean };

/** InfoTree: names and values as a tree, every field on its own row, open
 * from the start; guides down every level a row sits under and a chevron on
 * a row that opens; a filter finds a name or value anywhere and tints what
 * it found (ORANGE_TINT); one rounded wash per row under the pointer
 * (ACCENT_TINT) or selected (SELECTED); Copy value, Copy name, Expand all and
 * Collapse all on any row.  A value is never wrapped or cut: the tree scrolls
 * sideways; a name wider than its column shows its two ends (Qt ElideMiddle). */
export class InfoTree {
  readonly root = element("div", "info-tree");
  private nodes: Node[] = [];
  private selected: Node | null = null;
  actionRequested: (action: string) => void = () => {};

  constructor() { new ResizeObserver(() => this.fitNames()).observe(this.root); }

  /** v2 _fit_name_column: the name column as wide as its widest name at its depth (every name,
   * open or not), up to six tenths of the view (the values must start on screen), never below the
   * declared label column; a name past it shows its two ends. Measured by the font, not the layout. */
  private fitNames(): void {
    const first = this.nodes[0]?.label;
    if (!this.root.clientWidth || !first) return;
    const style = getComputedStyle(first);
    const measure = measurer(fontOf(style));
    // A name's cell: its words after its depth's indent, and the end padding.
    const inset = new Map<number, number>();
    const all: Node[] = [];
    const walk = (node: Node): void => { all.push(node); node.children.forEach(walk); };
    this.nodes.forEach(walk);
    const end = parseFloat(style.paddingRight);
    const indent = (node: Node): number => {
      let left = inset.get(node.depth);
      if (left === undefined) { left = parseFloat(getComputedStyle(node.label).paddingLeft); inset.set(node.depth, left); }
      return left + end;
    };
    // A row stands a gutter in from the tree's edge: its name column is that much narrower.
    const gutter = parseFloat(getComputedStyle(this.nodes[0]!.row).marginLeft);
    const widest = Math.ceil(gutter + all.reduce((most, node) => Math.max(most, measure(node.name) + indent(node)), 0));
    const width = `${Math.min(widest, Math.floor(this.root.clientWidth * 0.6))}px`;
    if (this.root.style.getPropertyValue("--info-name-w") !== width) this.root.style.setProperty("--info-name-w", width);
    // The column: the wider of the declared label column and the names', read off a shown row.
    const shown = all.find(node => !node.row.hidden)?.label;
    if (!shown) return;
    const column = shown.getBoundingClientRect().width;
    for (const node of all) setText(node.label, elideMiddle(node.name, column - indent(node), measure));
  }

  setRows(rows: readonly InfoRow[]): void {
    this.selected = null;
    const build = (name: string, value: InfoValue, depth: number, above: string): Node => {
      const row = element("div", "info-row");
      row.style.setProperty("--depth", String(depth));
      for (let level = 0; level < depth; level++) {
        const guide = element("span", "info-guide");
        guide.style.setProperty("--level", String(level));
        row.append(guide);
      }
      const chevron = element("span", "info-chevron");
      const label = element("span", "info-name", name);
      const shown = element("span", "info-value");
      if (isAction(value)) {
        const action = command(value.text, () => this.actionRequested(value.action));
        shown.append(action.root);
      } else shown.textContent = valueText(value);
      row.append(chevron, label, shown);
      const path = above ? `${above}.${name}` : name;
      const node: Node = { row, label, name, path, value, depth, children: [], open: true };
      if (isComposite(value)) {
        row.classList.add("branch", "open");
        node.children = entriesOf(value).map(([child, item]) => build(child, item, depth + 1, path));
        chevron.addEventListener("click", event => { event.stopPropagation(); this.setOpen(node, !node.open); });
        row.addEventListener("dblclick", () => this.setOpen(node, !node.open));
      }
      row.addEventListener("mousedown", () => this.select(node));
      row.addEventListener("contextmenu", event => {
        event.preventDefault();
        this.select(node);
        // v2 InfoTree's menu: the value, the name from the top (devices.camera.exposure_seconds), the tree open or shut.
        new Menu([
          { label: "Copy value", action: () => void navigator.clipboard?.writeText(isAction(node.value) ? node.value.text : copyText(node.value)) },
          { label: "Copy name", action: () => void navigator.clipboard?.writeText(node.path) },
          "separator",
          { label: "Expand all", action: () => this.setAll(true) },
          { label: "Collapse all", action: () => this.setAll(false) },
        ]).openAt(event.clientX, event.clientY);
      });
      return node;
    };
    this.nodes = rows.map(([name, value]) => build(name, value, 0, ""));
    // The top level keeps a chevron column only when some top-level row opens.
    this.root.classList.toggle("decorated", this.nodes.some(node => node.children.length > 0));
    const flat: HTMLElement[] = [];
    const walk = (node: Node): void => { flat.push(node.row); node.children.forEach(walk); };
    this.nodes.forEach(walk);
    this.root.replaceChildren(...flat);
    for (const node of this.nodes) this.setOpen(node, true);
    requestAnimationFrame(() => this.fitNames());
  }

  private setAll(open: boolean): void {
    const walk = (node: Node): void => { if (node.children.length) { node.children.forEach(walk); this.setOpen(node, open); } };
    this.nodes.forEach(walk);
  }

  private setOpen(node: Node, open: boolean): void {
    node.open = open;
    node.row.classList.toggle("open", open);
    const show = (child: Node, visible: boolean): void => {
      child.row.hidden = !visible;
      for (const grand of child.children) show(grand, visible && child.open);
    };
    for (const child of node.children) show(child, open && !node.row.hidden);
  }

  private select(node: Node): void {
    this.selected?.row.classList.remove("selected");
    this.selected = node;
    node.row.classList.add("selected");
  }

  /** Keep the rows whose name or value holds `text`, with the rows above them;
   * the cells that matched are tinted. */
  applyFilter(text: string): void {
    const needle = text.trim().toLowerCase();
    const visit = (node: Node, parentOpen: boolean): boolean => {
      const nameHit = Boolean(needle) && node.name.toLowerCase().includes(needle);
      const valueHit = Boolean(needle) && !isComposite(node.value) && valueText(node.value).toLowerCase().includes(needle);
      node.row.querySelector(".info-name")!.classList.toggle("match", nameHit);
      node.row.querySelector(".info-value")!.classList.toggle("match", valueHit);
      let below = false;
      for (const child of node.children) below = visit(child, parentOpen && node.open || Boolean(needle)) || below;
      const keep = !needle || nameHit || valueHit || below;
      node.row.hidden = !keep || (!needle && !parentOpen);
      return nameHit || valueHit || below;
    };
    for (const node of this.nodes) visit(node, true);
  }

  /** Open on the top-level row `label`; false when there is none. */
  showRow(label: string): boolean {
    const node = this.nodes.find(node => node.name === label);
    if (!node) return false;
    this.select(node);
    node.row.scrollIntoView({ block: "nearest" });
    return true;
  }
}

/** FluentPathEdit: a typed path, Browse… and (for a file whose content is
 * read) Refresh.  The UI never opens a file itself: Browse asks the owner,
 * which shows the native dialog; Enter or leaving the box commits the text.
 * While not being edited a path too long for the box is elided in the
 * middle (v2 set_display_elide(ElideMiddle)); focus shows all of it. */
export class PathEdit {
  readonly root = element("div", "path-edit");
  readonly input = element("input");
  readonly browse = command("Browse…", () => this.browsed());
  readonly refresh = command("Refresh", () => this.refreshed(), { title: "Re-read this file from disk" });
  private full = "";
  constructor(private committed: (path: string) => void, private browsed: () => void, private refreshed: () => void = () => {}, refreshable = false) {
    this.input.type = "text";
    this.input.spellcheck = false;
    this.input.addEventListener("change", () => this.committed(this.input.value.trim()));
    this.input.addEventListener("keydown", event => { if (event.key === "Enter") this.input.blur(); });
    this.input.addEventListener("focus", () => { this.input.value = this.full; });
    this.input.addEventListener("blur", () => { this.full = this.input.value; this.show(); });
    new ResizeObserver(() => this.show()).observe(this.input);
    this.refresh.hidden = !refreshable;
    this.root.append(this.input, this.browse.root, this.refresh.root);
  }
  set value(path: string) {
    if (document.activeElement === this.input || path === this.full) return;
    this.full = path;
    this.input.title = path;
    this.show();
  }
  get value(): string { return document.activeElement === this.input ? this.input.value : this.full; }
  /** The whole path if it fits the text area, else its two ends (Qt ElideMiddle). */
  private show(): void {
    if (document.activeElement === this.input) return;
    const style = getComputedStyle(this.input);
    const room = this.input.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
    const text = room > 0 ? elideMiddle(this.full, room, measurer(fontOf(style))) : this.full;
    if (this.input.value !== text) this.input.value = text;
  }
}

/** The pane's label column (setting_label_width) for the names its rows may carry; 0 when there are none. */
function labelColumn(labels: readonly string[]): number {
  if (!labels.length) return 0;
  const advance = measurer(fontOf(getComputedStyle(document.body)));
  return labelColumnWidth(labels.map(advance));
}

/** InfoPane._apply_pane_width: v2's pane is as wide as its label column and
 * scaled 280, or as its tab bar wants -- the tabs' natural widths
 * (QTabBar.tabSizeHint: the title's QFontMetrics.size(), padding, margins, the
 * first tab's page inset) and the corner the ⋯ takes, hidden or not --
 * whichever is wider, both counting the pane's window-pad margins.  Here those
 * margins lie outside the pane, so it is that less the two of them. */
function paneWidth(labels: number, titles: readonly string[]): number {
  const q = token("--q"), pad = token("--tab-pad-h"), margin = token("--tab-margin-h");
  const tabs = titles.reduce((sum, title) => sum + wordsWidth(title) * q + 2 * pad + 2 * margin, 0)
    + (titles.length ? token("--card-pad") - margin : 0) + token("--tab-overflow") + token("--tab-overflow-margin");
  return Math.max((labels || token("--label-min")) + token("--info-pane-extra") - 2 * token("--window-pad"), tabs);
}

/** InfoPane: a path header (label, path, Browse…, Refresh), read-only tabs
 * of rows (each with its filter above its tree) and a status strip.  The
 * pane does not know what the rows are; which tab the reader is on is kept
 * across a refresh. */
export class InfoPane {
  readonly root = element("div", "info-pane");
  readonly header = element("div", "card info-header");
  readonly path: PathEdit;
  readonly tabs = new TabWidget();
  readonly status = new StatusStrip();
  private trees = new Map<string, { tree: InfoTree; filter: HTMLInputElement }>();
  /** Tabs that show a graph of the rows (v2 graph_tabs: the Figure's Flow). */
  private graphs = new Map<string, FlowGraph>();
  private signature = "";
  actionRequested: (action: string) => void = () => {};

  /** `labelNames`: the names the owner declares its rows may carry; they,
   * not the rows of whatever file is open, set the pane's one stable split
   * (setting_label_width), so opening another file never moves it; the
   * declared tab titles (`tabTitles`) bound it too (_apply_pane_width). */
  constructor(label: string, path: { commit(path: string): void; browse(): void; refresh(): void }, labelNames: readonly string[] = [], private graphTabs: readonly string[] = [], tabTitles: readonly string[] = []) {
    this.path = new PathEdit(path.commit, path.browse, path.refresh, true);
    const width = labelColumn(labelNames);
    if (width) this.root.style.setProperty("--info-label-w", `${width}px`);
    this.root.style.setProperty("--info-pane-w", `${paneWidth(width, tabTitles)}px`);
    this.header.append(element("span", "row-label", label), this.path.root);
    this.tabs.root.classList.add("info-tabs");
    this.root.append(this.header, this.tabs.root, this.status.root);
  }

  setTabs(tabs: readonly InfoTab[]): void {
    const signature = JSON.stringify(tabs);
    if (signature === this.signature) return;
    this.signature = signature;
    const showing = this.tabs.currentKey;
    const titles = new Set(tabs.map(([title]) => title));
    for (const title of [...this.trees.keys(), ...this.graphs.keys()]) if (!titles.has(title)) { this.tabs.remove(title); this.trees.delete(title); this.graphs.delete(title); }
    for (const [title, rows] of tabs) {
      if (this.graphTabs.includes(title)) {
        if (!this.graphs.has(title)) {
          const graph = new FlowGraph();
          graph.rowRequested = (tab, label) => this.showRow(tab, label);
          this.tabs.addPermanent(title, title, graph.root);
          this.graphs.set(title, graph);
        }
        continue;
      }
      let entry = this.trees.get(title);
      if (!entry) {
        const page = element("div", "info-tab");
        const filter = element("input");
        filter.type = "text";
        filter.placeholder = "filter names and values";
        filter.spellcheck = false;
        const tree = new InfoTree();
        tree.actionRequested = action => this.actionRequested(action);
        filter.addEventListener("input", () => tree.applyFilter(filter.value));
        page.append(filter, tree.root);
        this.tabs.addPermanent(title, title, page);
        entry = { tree, filter };
        this.trees.set(title, entry);
      }
      entry.tree.setRows(rows);
      entry.tree.applyFilter(entry.filter.value);
    }
    if (showing && titles.has(showing)) this.tabs.setCurrent(showing);
  }

  /** Replace a graph tab's picture (its nodes may name the rows they stand for). */
  setGraph(title: string, graph: Graph): void { this.graphs.get(title)?.set(graph); }

  showRow(title: string, label: string): boolean {
    const entry = this.trees.get(title);
    if (!entry) return false;
    if (entry.filter.value) { entry.filter.value = ""; entry.tree.applyFilter(""); }
    if (!entry.tree.showRow(label)) return false;
    this.tabs.setCurrent(title);
    return true;
  }

  setStatus(text: string): void { this.status.show(text, "idle"); }
  setPath(path: string): void { this.path.value = path; }
  set label(text: string) { setText(this.header.querySelector(".row-label")!, text); }
}
