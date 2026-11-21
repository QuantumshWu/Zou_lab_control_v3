import { command } from "./button.js";
import { FlowGraph } from "./flow_graph.js";
import { contentWidth, element, labelColumnWidth, setText, token } from "./dom.js";
import { wordsWidth } from "./ink.js";
import { elideMiddle, fontOf, measurer } from "./elide.js";
import { Menu } from "./menu.js";
import { StatusStrip } from "./status.js";
import { TabWidget } from "./tabs.js";
/** A flat list longer than this reads as its count and range. */
const LIST_PREVIEW = 8;
const isScalar = (value) => value === null || typeof value !== "object";
const isFlatList = (value) => Array.isArray(value) && value.every(isScalar);
const isAction = (value) => value !== null && typeof value === "object" && !Array.isArray(value)
    && Object.keys(value).length === 2 && typeof value.text === "string" && typeof value.action === "string";
const isComposite = (value) => !isAction(value) && value !== null && typeof value === "object" && !isFlatList(value);
const isOrdered = (value) => value !== null && typeof value === "object" && !Array.isArray(value) && Object.keys(value).length === 1 && Array.isArray(value.$fields);
/** A composite's (name, value) entries: a list by index, an ordered record as written, a record by name. */
const entriesOf = (value) => Array.isArray(value) ? value.map((item, index) => [`[${index}]`, item])
    : isOrdered(value) ? value.$fields.map(([name, item]) => [name, item]) : Object.entries(value);
/** One value as it reads beside its name (info_pane.value_text). */
export function valueText(value) {
    if (isAction(value))
        return value.text;
    if (value === null)
        return "none";
    if (typeof value === "boolean")
        return value ? "true" : "false";
    if (typeof value === "number")
        return Number.isInteger(value) ? String(value) : String(Number(value.toPrecision(6)));
    if (typeof value === "string")
        return value;
    if (isFlatList(value)) {
        if (!value.length)
            return "(none)";
        const numbers = value.filter((item) => typeof item === "number");
        if (value.length > LIST_PREVIEW && numbers.length === value.length)
            return `${value.length} numbers, ${valueText(Math.min(...numbers))} to ${valueText(Math.max(...numbers))}`;
        const shown = value.slice(0, LIST_PREVIEW).map(valueText).join(", ");
        return value.length > LIST_PREVIEW ? `${shown}, … (${value.length} in all)` : shown;
    }
    if (Array.isArray(value))
        return `${value.length} items`;
    return `${entriesOf(value).length} fields`;
}
/** The whole value for the clipboard, nested by indentation. */
export function copyText(value, indent = "") {
    if (isFlatList(value))
        return value.map(valueText).join(", ") || "(none)";
    if (value !== null && typeof value === "object") {
        return entriesOf(value).map(([name, item]) => isComposite(item)
            ? `${indent}${name}:\n${copyText(item, `${indent}  `)}` : `${indent}${name}: ${copyText(item)}`).join("\n");
    }
    return valueText(value);
}
/** Rows drawn past each edge of the view. */
const OVERSCAN = 8;
/** InfoTree: names and values as a tree, every field on its own row, open
 * from the start; guides down every level a row sits under and a chevron on
 * a row that opens; a filter finds a name or value anywhere and tints what
 * it found (ORANGE_TINT); one rounded wash per row under the pointer
 * (ACCENT_TINT) or selected (SELECTED); Copy value, Copy name, Expand all and
 * Collapse all on any row.  A value is never wrapped or cut: the tree scrolls
 * sideways; a name wider than its column shows its two ends (Qt ElideMiddle).
 * As QTreeView, only the rows in view are drawn: the fields are a model, the
 * shown rows its depth-first list, each drawn at its offset in a spacer as
 * tall as all of them. */
export class InfoTree {
    root = element("div", "info-tree");
    rows = element("div", "info-rows");
    nodes = [];
    all = [];
    /** The shown rows in order, and where each starts (one more: where the last ends). */
    flat = [];
    offsets = new Float64Array(1);
    drawn = new Map();
    owner = new WeakMap();
    selected = null;
    reveal = null;
    needle = "";
    /** While a needle is set: the branches closed since (the filter opens every other). */
    shut = null;
    metrics = null;
    measureName = () => 0;
    stale = { metrics: true, flat: true };
    /** The view width the names were fitted to, and the name cell it gave. */
    fitted = 0;
    column = 0;
    actionRequested = () => { };
    constructor() {
        this.root.append(this.rows);
        new ResizeObserver(() => this.update()).observe(this.root);
        // The scale (scale.js and --fluent on the root) or a font changed: measure the kinds again.
        const restyled = () => { this.stale.metrics = true; this.update(); };
        new MutationObserver(restyled).observe(document.documentElement, { attributes: true, attributeFilter: ["style"] });
        document.fonts.addEventListener("loadingdone", restyled);
        this.root.addEventListener("scroll", () => this.paint(), { passive: true });
        const nodeAt = (event) => {
            const row = event.target.closest(".info-row");
            return row ? this.owner.get(row) : undefined;
        };
        this.root.addEventListener("mousedown", event => { const node = nodeAt(event); if (node)
            this.select(node); });
        this.root.addEventListener("click", event => {
            const node = nodeAt(event);
            if (node?.branch && event.target.closest(".info-chevron"))
                this.setOpen(node, !this.isOpen(node));
        });
        this.root.addEventListener("dblclick", event => { const node = nodeAt(event); if (node?.branch)
            this.setOpen(node, !this.isOpen(node)); });
        this.root.addEventListener("contextmenu", event => {
            const node = nodeAt(event);
            if (!node)
                return;
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
    }
    setRows(rows) {
        const all = [];
        const build = (name, value, depth, above) => {
            const path = above ? `${above}.${name}` : name;
            const node = {
                name, path, value, text: valueText(value), depth, children: [], branch: isComposite(value), action: isAction(value), open: true,
                width: 0, nameHit: false, valueHit: false, found: false,
            };
            all.push(node);
            if (node.branch)
                node.children = entriesOf(value).map(([child, item]) => build(child, item, depth + 1, path));
            return node;
        };
        this.nodes = rows.map(([name, value]) => build(name, value, 0, ""));
        this.all = all;
        // The top level keeps a chevron column only when some top-level row opens.
        this.root.classList.toggle("decorated", this.nodes.some(node => node.children.length > 0));
        this.selected = null;
        this.reveal = null;
        this.metrics = null;
        this.mark();
        this.clear();
        this.stale = { metrics: true, flat: true };
        this.update();
    }
    /** Keep the rows whose name or value holds `text`, with the rows above them;
     * the cells that matched are tinted. */
    applyFilter(text) {
        const needle = text.trim().toLowerCase();
        if (needle === this.needle)
            return;
        this.needle = needle;
        this.mark();
        this.stale.flat = true;
        this.update();
    }
    /** Open on the top-level row `label`; false when there is none. */
    showRow(label) {
        const node = this.nodes.find(node => node.name === label);
        if (!node)
            return false;
        this.select(node);
        this.reveal = node;
        this.update();
        return true;
    }
    /** What the needle finds: each node's hits, and whether it or a row below it holds one. */
    mark() {
        const needle = this.needle;
        this.shut = needle ? new Set() : null;
        const visit = (node) => {
            node.nameHit = Boolean(needle) && node.name.toLowerCase().includes(needle);
            node.valueHit = Boolean(needle) && !node.branch && node.text.toLowerCase().includes(needle);
            let below = false;
            for (const child of node.children)
                below = visit(child) || below;
            return node.found = node.nameHit || node.valueHit || below;
        };
        this.nodes.forEach(visit);
    }
    isOpen(node) { return this.shut ? !this.shut.has(node) : node.open; }
    setOpen(node, open) {
        if (!this.shut)
            node.open = open;
        else if (open)
            this.shut.delete(node);
        else
            this.shut.add(node);
        this.stale.flat = true;
        this.update();
    }
    setAll(open) {
        const branches = this.all.filter(node => node.children.length);
        for (const node of branches)
            node.open = open;
        if (this.shut)
            this.shut = new Set(open ? [] : branches);
        this.stale.flat = true;
        this.update();
    }
    select(node) {
        if (this.selected)
            this.drawn.get(this.selected)?.classList.remove("selected");
        this.selected = node;
        this.drawn.get(node)?.classList.add("selected");
    }
    /** Measure, list, fit and draw what changed; a hidden tree waits until it is shown. */
    update() {
        const width = this.root.clientWidth;
        if (!width)
            return;
        if (!this.all.length) {
            this.clear();
            this.flat = [];
            this.rows.style.height = "0px";
            return;
        }
        let redraw = false;
        if (this.stale.metrics) {
            this.stale.metrics = false;
            if (this.measure())
                this.stale.flat = true;
        }
        if (this.stale.flat) {
            this.stale.flat = false;
            this.layout();
            redraw = true;
        }
        if (width !== this.fitted) {
            this.fitNames(width);
            redraw = true;
        }
        if (redraw)
            for (const [node, row] of this.drawn)
                this.refresh(node, row);
        if (this.reveal) {
            // scrollIntoView({ block: "nearest" }) at the row's offset.
            const at = this.flat.indexOf(this.reveal);
            this.reveal = null;
            if (at >= 0) {
                const top = this.offsets[at], bottom = this.offsets[at + 1];
                if (top < this.root.scrollTop)
                    this.root.scrollTop = top;
                else if (bottom > this.root.scrollTop + this.root.clientHeight)
                    this.root.scrollTop = bottom - this.root.clientHeight;
            }
        }
        this.paint();
    }
    /** A probe row of each kind, read in one layout (Metrics); when it reads
     * anything new, every name and value measured by the font.  True then. */
    measure() {
        const plain = this.all.find(node => !node.action), action = this.all.find(node => node.action);
        const probes = [plain, action].map(node => node && this.draw(node));
        for (const probe of probes)
            if (probe) {
                probe.style.visibility = "hidden";
                this.rows.append(probe);
            }
        // The declared label column: the name cell with no name fitted.
        this.root.style.removeProperty("--info-name-w");
        this.fitted = 0;
        const probe = (probes[0] ?? probes[1]), name = probe.querySelector(".info-name");
        const style = getComputedStyle(name);
        const [plainHeight, actionHeight] = probes.map(row => row ? row.getBoundingClientRect().height : 0);
        const label = name.getBoundingClientRect().width, gutter = parseFloat(getComputedStyle(probe).marginLeft);
        const font = fontOf(style), end = parseFloat(style.paddingRight);
        const indents = [];
        const deepest = this.all.reduce((most, node) => Math.max(most, node.depth), 0);
        for (let depth = 0; depth <= deepest; depth++) {
            probe.style.setProperty("--depth", String(depth));
            indents.push(parseFloat(getComputedStyle(name).paddingLeft));
        }
        const padding = (cell) => { const box = getComputedStyle(cell); return parseFloat(box.paddingLeft) + parseFloat(box.paddingRight); };
        const valuePad = probes[0] ? padding(probes[0].querySelector(".info-value")) : 0;
        // A FluentButton: its words (fitWords' Qt advance, or the font's where wider) and its chrome.
        let buttonFont = "", chrome = 0, q = 0;
        const words = (text) => Math.max(wordsWidth(text) * q, measurer(buttonFont)(text));
        if (probes[1]) {
            const holder = probes[1].querySelector(".info-action"), face = holder.querySelector("button");
            buttonFont = fontOf(getComputedStyle(face));
            q = token("--q");
            chrome = padding(holder) + face.getBoundingClientRect().width - words(action.text);
        }
        for (const probe of probes)
            probe?.remove();
        const read = { plain: plainHeight, action: actionHeight, gutter, label, end, indents, font, valuePad, buttonFont, chrome, q, widest: 0 };
        if (this.metrics && JSON.stringify({ ...this.metrics, widest: 0 }) === JSON.stringify(read))
            return false;
        // A name's cell: its words after its depth's indent and the end padding, a gutter in from the tree's edge.
        const measure = this.measureName = measurer(font);
        let widest = 0;
        for (const node of this.all) {
            widest = Math.max(widest, measure(node.name) + indents[node.depth]);
            node.width = node.action ? chrome + words(node.text) : valuePad + measure(node.text);
        }
        this.metrics = { ...read, widest: Math.ceil(gutter + widest + end) };
        return true;
    }
    /** The shown rows: depth first through open branches (while a needle is
     * set, the found rows only); their offsets; the widest value among them. */
    layout() {
        const metrics = this.metrics, flat = [];
        const visit = (node) => {
            if (this.shut && !node.found)
                return;
            flat.push(node);
            if (this.isOpen(node))
                node.children.forEach(visit);
        };
        this.nodes.forEach(visit);
        this.flat = flat;
        this.offsets = new Float64Array(flat.length + 1);
        let widest = 0;
        flat.forEach((node, at) => {
            this.offsets[at + 1] = this.offsets[at] + (node.action ? metrics.action : metrics.plain);
            widest = Math.max(widest, node.width);
        });
        this.rows.style.height = `${this.offsets[flat.length]}px`;
        this.root.style.setProperty("--info-value-w", `${widest}px`);
    }
    /** v2 _fit_name_column: the name column as wide as its widest name at its depth (every name,
     * open or not), up to six tenths of the view (the values must start on screen), never below the
     * declared label column; a name past it shows its two ends. Measured by the font, not the layout. */
    fitNames(width) {
        const metrics = this.metrics, names = Math.min(metrics.widest, Math.floor(width * 0.6));
        if (this.root.style.getPropertyValue("--info-name-w") !== `${names}px`)
            this.root.style.setProperty("--info-name-w", `${names}px`);
        this.column = Math.max(metrics.label, names - metrics.gutter);
        this.fitted = width;
    }
    /** The shown row at `y` down the spacer. */
    at(y) {
        let low = 0, high = this.flat.length - 1;
        while (low < high) {
            const middle = (low + high + 1) >> 1;
            if (this.offsets[middle] <= y)
                low = middle;
            else
                high = middle - 1;
        }
        return low;
    }
    /** Draw the rows in view and a few past it, in order; drop the rest. */
    paint() {
        const metrics = this.metrics;
        if (!metrics || this.stale.flat || !this.flat.length)
            return;
        const top = this.root.scrollTop;
        const first = Math.max(0, this.at(top) - OVERSCAN), last = Math.min(this.flat.length, this.at(top + this.root.clientHeight) + 1 + OVERSCAN);
        const wanted = new Set(this.flat.slice(first, last));
        for (const [node, row] of this.drawn)
            if (!wanted.has(node)) {
                row.remove();
                this.drawn.delete(node);
            }
        let next = this.rows.firstChild;
        for (let at = first; at < last; at++) {
            const node = this.flat[at];
            let row = this.drawn.get(node);
            if (!row) {
                row = this.draw(node);
                this.refresh(node, row);
                this.drawn.set(node, row);
            }
            const offset = `${this.offsets[at]}px`;
            if (row.style.top !== offset)
                row.style.top = offset;
            if (row === next)
                next = next.nextSibling;
            else
                this.rows.insertBefore(row, next);
        }
    }
    clear() {
        for (const row of this.drawn.values())
            row.remove();
        this.drawn.clear();
    }
    /** One row: a guide per level it sits under, the chevron, the name and the value. */
    draw(node) {
        const row = element("div", "info-row");
        row.style.setProperty("--depth", String(node.depth));
        for (let level = 0; level < node.depth; level++) {
            const guide = element("span", "info-guide");
            guide.style.setProperty("--level", String(level));
            row.append(guide);
        }
        const label = element("span", "info-name", node.name);
        const shown = element("span", "info-value");
        if (isAction(node.value)) {
            // v2's embedded FluentButton consumes its left press; opening the
            // action preserves the tree's selection and its separate hover wash.
            shown.classList.add("info-action");
            const action = node.value.action;
            const face = command(node.value.text, () => this.actionRequested(action));
            face.root.addEventListener("mousedown", event => { if (event.button === 0)
                event.stopPropagation(); });
            shown.append(face.root);
        }
        else
            shown.textContent = node.text;
        row.append(element("span", "info-chevron"), label, shown);
        if (node.branch)
            row.classList.add("branch");
        this.owner.set(row, node);
        return row;
    }
    /** A drawn row's state: its kind's height (a text in a taller fallback font keeps
     * the row's slot), its name in the column, the filter's tint, open, selected. */
    refresh(node, row) {
        const metrics = this.metrics, label = row.querySelector(".info-name");
        const height = `${node.action ? metrics.action : metrics.plain}px`;
        if (row.style.height !== height)
            row.style.height = height;
        setText(label, elideMiddle(node.name, this.column - metrics.indents[node.depth] - metrics.end, this.measureName));
        label.classList.toggle("match", node.nameHit);
        row.querySelector(".info-value").classList.toggle("match", node.valueHit);
        row.classList.toggle("open", node.branch && this.isOpen(node));
        row.classList.toggle("selected", node === this.selected);
    }
}
/** FluentPathEdit: a typed path, Browse… and (for a file whose content is
 * read) Refresh.  The UI never opens a file itself: Browse asks the owner,
 * which shows the native dialog; Enter or leaving the box commits the text.
 * While not being edited a path too long for the box is elided in the
 * middle (v2 set_display_elide(ElideMiddle)); focus shows all of it. */
export class PathEdit {
    committed;
    browsed;
    refreshed;
    root = element("div", "path-edit");
    input = element("input");
    browse = command("Browse…", () => this.browsed());
    refresh = command("Refresh", () => this.refreshed(), { title: "Re-read this file from disk" });
    full = "";
    constructor(committed, browsed, refreshed = () => { }, refreshable = false) {
        this.committed = committed;
        this.browsed = browsed;
        this.refreshed = refreshed;
        this.input.type = "text";
        this.input.spellcheck = false;
        this.input.addEventListener("change", () => this.committed(this.input.value.trim()));
        this.input.addEventListener("keydown", event => { if (event.key === "Enter")
            this.input.blur(); });
        this.input.addEventListener("focus", () => { this.input.value = this.full; });
        this.input.addEventListener("blur", () => { this.full = this.input.value; this.show(); });
        new ResizeObserver(() => this.show()).observe(this.input);
        this.refresh.hidden = !refreshable;
        this.root.append(this.input, this.browse.root, this.refresh.root);
    }
    set value(path) {
        if (document.activeElement === this.input || path === this.full)
            return;
        this.full = path;
        this.input.title = path;
        this.show();
    }
    get value() { return document.activeElement === this.input ? this.input.value : this.full; }
    /** The whole path if it fits the text area, else its two ends (Qt ElideMiddle). */
    show() {
        if (document.activeElement === this.input)
            return;
        const style = getComputedStyle(this.input);
        const room = this.input.clientWidth ? contentWidth(this.input, style) : 0;
        const text = room > 0 ? elideMiddle(this.full, room, measurer(fontOf(style))) : this.full;
        if (this.input.value !== text)
            this.input.value = text;
    }
}
/** The pane's label column (setting_label_width) for the names its rows may carry; 0 when there are none. */
function labelColumn(labels) {
    if (!labels.length)
        return 0;
    const advance = measurer(fontOf(getComputedStyle(document.body)));
    return labelColumnWidth(labels.map(advance));
}
/** InfoPane._apply_pane_width: v2's pane is as wide as its label column and
 * scaled 280, or as its tab bar wants -- the tabs' natural widths
 * (QTabBar.tabSizeHint: the title's QFontMetrics.size(), padding, margins, the
 * first tab's page inset) and the corner the ⋯ takes, hidden or not --
 * whichever is wider, both counting the pane's window-pad margins.  Here those
 * margins lie outside the pane, so it is that less the two of them. */
function paneWidth(labels, titles) {
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
    graphTabs;
    root = element("div", "info-pane");
    header = element("div", "card info-header");
    path;
    tabs = new TabWidget();
    status = new StatusStrip();
    trees = new Map();
    /** Tabs that show a graph of the rows (v2 graph_tabs: the Figure's Flow). */
    graphs = new Map();
    signature = "";
    actionRequested = () => { };
    /** `labelNames`: the names the owner declares its rows may carry; they,
     * not the rows of whatever file is open, set the pane's one stable split
     * (setting_label_width), so opening another file never moves it; the
     * declared tab titles (`tabTitles`) bound it too (_apply_pane_width). */
    constructor(label, path, labelNames = [], graphTabs = [], tabTitles = []) {
        this.graphTabs = graphTabs;
        this.path = new PathEdit(path.commit, path.browse, path.refresh, true);
        const width = labelColumn(labelNames);
        if (width)
            this.root.style.setProperty("--info-label-w", `${width}px`);
        this.root.style.setProperty("--info-pane-w", `${paneWidth(width, tabTitles)}px`);
        this.header.append(element("span", "row-label", label), this.path.root);
        this.tabs.root.classList.add("info-tabs");
        this.root.append(this.header, this.tabs.root, this.status.root);
    }
    setTabs(tabs) {
        const signature = JSON.stringify(tabs);
        if (signature === this.signature)
            return;
        this.signature = signature;
        const showing = this.tabs.currentKey;
        const titles = new Set(tabs.map(([title]) => title));
        for (const title of [...this.trees.keys(), ...this.graphs.keys()])
            if (!titles.has(title)) {
                this.tabs.remove(title);
                this.trees.delete(title);
                this.graphs.delete(title);
            }
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
        if (showing && titles.has(showing))
            this.tabs.setCurrent(showing);
    }
    /** Replace a graph tab's picture (its nodes may name the rows they stand for). */
    setGraph(title, graph) { this.graphs.get(title)?.set(graph); }
    showRow(title, label) {
        const entry = this.trees.get(title);
        if (!entry)
            return false;
        if (entry.filter.value) {
            entry.filter.value = "";
            entry.tree.applyFilter("");
        }
        if (!entry.tree.showRow(label))
            return false;
        this.tabs.setCurrent(title);
        return true;
    }
    setStatus(text) { this.status.show(text, "idle"); }
    setPath(path) { this.path.value = path; }
    set label(text) { setText(this.header.querySelector(".row-label"), text); }
}
