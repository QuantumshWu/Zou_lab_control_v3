import { comboFace, hintWidest, placeDrop } from "./combo.js";
import { assign, element } from "./dom.js";
import { flattenTree } from "./tree_model.js";
/** Python's str order (code points), as v2 sorts its choices. */
const codepoints = (left, right) => left < right ? -1 : left > right ? 1 : 0;
/** v2 choice_tree_groups: every choice once, under the first heading that files
 * it (`unfiled` when none does), the headings in case-blind order with `unfiled`
 * last and each one's choices by key; a current key nobody offers still stands,
 * under `unfiled`.  A leaf reads its label (else its key); the closed box its
 * `text`, else "<heading> · <label>". */
export function choiceTree(entries, current, unfiled) {
    const offered = new Map();
    for (const entry of entries)
        if (entry.key && !offered.has(entry.key))
            offered.set(entry.key, entry);
    if (current && !offered.has(current))
        offered.set(current, { key: current });
    const groups = new Map();
    for (const key of [...offered.keys()].sort(codepoints)) {
        const entry = offered.get(key);
        const heading = entry.group || unfiled;
        const label = entry.label || key;
        let leaves = groups.get(heading);
        if (!leaves)
            groups.set(heading, leaves = []);
        leaves.push({ key, label, text: entry.text || `${heading} · ${label}` });
    }
    return [...groups.keys()].sort((left, right) => (Number(left === unfiled) - Number(right === unfiled)) || codepoints(left.toLowerCase(), right.toLowerCase()))
        .map(heading => ({ heading, leaves: groups.get(heading) }));
}
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
    changed;
    placeholder;
    root = element("select", "tree-combo");
    face = comboFace(this.root);
    /** The box's empty value: the none row's words, else the placeholder. */
    blank = new Option("", "");
    list = element("div", "tree-combo-list");
    groups = [];
    none = null;
    current = "";
    signature = "";
    minimum = 0;
    /** The headings open in the popup: kept across openings and refills. */
    expanded = new Set();
    rows = [];
    cursor = -1;
    typed = "";
    typedAt = 0;
    /** The side this opening chose (placeDrop): kept while a heading opens or closes. */
    up;
    /** A resized window chooses the side again. */
    resized = () => { this.up = undefined; this.place(); };
    constructor(changed, placeholder = "") {
        this.changed = changed;
        this.placeholder = placeholder;
        this.blank.hidden = true;
        this.root.replaceChildren(this.face, this.blank);
        this.list.popover = "auto";
        this.list.setAttribute("role", "tree");
        // The box's own list never opens: the tree is its drop-down, opened by the
        // click (after the press's light dismiss), closed by a click while open.
        let wasOpen = false;
        this.root.addEventListener("pointerdown", event => {
            if (event.button !== 0)
                return;
            event.preventDefault();
            wasOpen = this.open;
            if (!this.root.disabled)
                this.root.focus();
        });
        this.root.addEventListener("mousedown", event => event.preventDefault());
        this.root.addEventListener("click", event => {
            event.preventDefault();
            if (wasOpen) {
                if (this.open)
                    this.list.hidePopover();
            }
            else
                this.show();
            wasOpen = false;
        });
        this.root.addEventListener("keydown", event => this.key(event));
        this.list.addEventListener("toggle", event => {
            const open = event.newState === "open";
            this.root.classList.toggle("open", open);
            if (open)
                addEventListener("resize", this.resized);
            else
                removeEventListener("resize", this.resized);
        });
        this.list.addEventListener("pointerdown", event => event.preventDefault());
        this.list.addEventListener("pointerup", event => {
            const row = event.target.closest(".tree-row");
            if (row && event.button === 0)
                this.activate(Number(row.dataset.index), true);
        });
    }
    get open() { return this.list.matches(":popover-open"); }
    /** Reconcile the tree (v2 set_choice_tree): `none` is the blank row's words
     * ("Off", "(not selected)"), absent when a choice is required. */
    setTree(groups, current, none = null) {
        const signature = JSON.stringify([groups, none, this.placeholder]);
        if (signature !== this.signature) {
            this.signature = signature;
            this.groups = flattenTree(groups);
            this.none = none;
            // Every choice is an option of the box, never listed by it: the closed box says
            // the picked one's text and is as wide as the widest it may say (v2 _collapsed_text_candidates).
            const options = this.groups.flatMap(group => group.leaves.map(leaf => { const option = new Option(leaf.text, leaf.key); option.hidden = true; return option; }));
            this.root.replaceChildren(this.face, this.blank, ...options);
            this.blank.text = none ?? this.placeholder;
            hintWidest(this.root, [this.placeholder, ...(none === null ? [] : [none]), ...options.map(option => option.text)], this.minimum);
            if (this.open)
                this.render();
        }
        this.value = current;
    }
    /** setMinimumContentsLength: never narrower than `length` "M"s. */
    minimumContents(length) { this.minimum = length; this.signature = ""; return this; }
    get value() { return this.current; }
    set value(key) {
        this.current = key;
        // The none row picked, nothing picked, or a key nobody offers: the box's empty value.
        const leaf = key !== "" && this.leaf(key) !== undefined;
        if (this.root.value !== (leaf ? key : ""))
            this.root.value = leaf ? key : "";
        this.root.classList.toggle("blank", !leaf);
        if (this.open)
            this.render();
    }
    set disabled(disabled) { assign(this.root, "disabled", disabled); if (disabled && this.open)
        this.list.hidePopover(); }
    set title(text) { if (this.root.title !== text)
        this.root.title = text; }
    dispose() { if (this.open)
        this.list.hidePopover(); this.list.remove(); }
    openPath(index) {
        for (let at = index; at !== null; at = this.groups[at].parent)
            this.expanded.add(this.groups[at].path);
    }
    leaf(key) {
        for (const group of this.groups)
            for (const leaf of group.leaves)
                if (leaf.key === key)
                    return leaf;
        return undefined;
    }
    /** v2 showPopup: the picked choice's heading opened, the popup below the box
     * (above it where it fits better: placeDrop). */
    show() {
        if (this.root.disabled)
            return;
        const at = this.groups.findIndex(group => group.leaves.some(leaf => leaf.key === this.current));
        if (at >= 0)
            this.openPath(at);
        if (!this.list.isConnected)
            document.body.append(this.list);
        this.list.showPopover();
        this.cursor = -1;
        this.up = undefined;
        this.render();
        // The picked row is the tree's selection (SELECTED) when it opens.
        this.move(this.rows.findIndex(row => row.kind === "leaf" ? row.leaf.key === this.current : row.kind === "none" && this.current === ""));
    }
    /** What a row is, to find it again among the rows a toggle reveals. */
    identity(row) {
        return row === undefined ? "" : row.kind === "none" ? "none" : row.kind === "heading" ? `h:${row.group}` : `l:${row.leaf.key}`;
    }
    /** The rows the tree's open headings reveal, then the popup's size and place (v2 _resize_popup_to_contents). */
    render() {
        const rows = [];
        if (this.none !== null)
            rows.push({ kind: "none" });
        this.groups.forEach((group, index) => {
            for (let parent = group.parent; parent !== null; parent = this.groups[parent].parent)
                if (!this.expanded.has(this.groups[parent].path))
                    return;
            rows.push({ kind: "heading", group: index });
            if (this.expanded.has(group.path))
                for (const leaf of group.leaves)
                    rows.push({ kind: "leaf", group: index, leaf });
        });
        this.rows = rows;
        this.list.replaceChildren(...rows.map((row, index) => {
            const node = element("div", `tree-row ${row.kind}`);
            node.dataset.index = String(index);
            const depth = row.kind === "none" ? 0 : this.groups[row.group].depth + (row.kind === "leaf" ? 1 : 0);
            node.style.setProperty("--tree-depth", String(depth));
            node.setAttribute("aria-level", String(depth + 1));
            const item = element("span", "tree-item");
            if (row.kind === "heading") {
                const open = this.expanded.has(this.groups[row.group].path);
                node.classList.toggle("open", open);
                node.setAttribute("role", "treeitem");
                node.setAttribute("aria-expanded", String(open));
                node.append(element("span", "tree-branch"));
                item.textContent = this.groups[row.group].heading;
            }
            else {
                node.setAttribute("role", "treeitem");
                item.textContent = row.kind === "leaf" ? row.leaf.label : this.none;
                node.setAttribute("aria-selected", String(row.kind === "leaf" ? row.leaf.key === this.current : this.current === ""));
            }
            node.append(item);
            return node;
        }));
        this.paintCursor();
        this.place();
    }
    place() {
        const field = this.root.getBoundingClientRect();
        const style = this.list.style;
        style.minWidth = `${field.width}px`;
        style.maxWidth = `${innerWidth}px`;
        // Every row it reveals, below the box or above it, as far as the window goes on that side; the rest scrolls.
        this.up = placeDrop(this.list, field, this.up);
        const width = this.list.offsetWidth;
        const left = width <= field.width ? field.left : field.right - width;
        style.left = `${Math.max(0, Math.min(left, innerWidth - width))}px`;
    }
    paintCursor() {
        [...this.list.children].forEach((node, index) => node.classList.toggle("current", index === this.cursor));
    }
    move(index, scroll = true) {
        if (index < 0 || index >= this.rows.length)
            return;
        this.cursor = index;
        this.paintCursor();
        if (scroll)
            this.list.children[index]?.scrollIntoView({ block: "nearest" });
    }
    /** A heading opens or closes (and the popup grows to fit); a choice is picked. */
    activate(index, pointer) {
        const row = this.rows[index];
        if (!row)
            return;
        if (row.kind === "heading") {
            this.toggle(row.group);
            return;
        }
        const key = row.kind === "leaf" ? row.leaf.key : "";
        this.list.hidePopover();
        const changed = key !== this.current;
        this.value = key;
        if (changed)
            this.changed(key);
    }
    toggle(group, open = !this.expanded.has(this.groups[group].path)) {
        const heading = this.groups[group].path;
        if (open === this.expanded.has(heading))
            return;
        if (open)
            this.expanded.add(heading);
        else
            this.expanded.delete(heading);
        const at = this.identity(this.rows[this.cursor]);
        this.render();
        this.cursor = this.rows.findIndex(row => this.identity(row) === at);
        this.paintCursor();
    }
    key(event) {
        if (!this.open) {
            if (["ArrowUp", "ArrowDown", "Home", "End", "Enter", " ", "F4"].includes(event.key)) {
                event.preventDefault();
                this.show();
            }
            // The box's own list never answers a letter.
            else if (event.key.length === 1 && !event.ctrlKey && !event.metaKey)
                event.preventDefault();
            return;
        }
        const row = this.rows[this.cursor];
        const last = this.rows.length - 1;
        switch (event.key) {
            case "ArrowDown":
                this.move(Math.min(last, this.cursor + 1));
                break;
            case "ArrowUp":
                this.move(Math.max(0, this.cursor - 1));
                break;
            case "Home":
                this.move(0);
                break;
            case "End":
                this.move(last);
                break;
            case "ArrowRight":
                if (row?.kind === "heading") {
                    if (!this.expanded.has(this.groups[row.group].path))
                        this.toggle(row.group, true);
                    else if (this.rows[this.cursor + 1])
                        this.move(this.cursor + 1);
                }
                break;
            case "ArrowLeft":
                if (row?.kind === "heading") {
                    if (this.expanded.has(this.groups[row.group].path))
                        this.toggle(row.group, false);
                    else {
                        const parent = this.groups[row.group].parent;
                        if (parent !== null)
                            this.move(this.rows.findIndex(item => item.kind === "heading" && item.group === parent));
                    }
                }
                else if (row?.kind === "leaf")
                    this.move(this.rows.findIndex(item => item.kind === "heading" && item.group === row.group));
                break;
            case "Enter":
                if (row)
                    this.activate(this.cursor, false);
                break;
            case "Escape":
                this.list.hidePopover();
                break;
            case "Tab":
                this.list.hidePopover();
                return;
            default:
                if (event.key.length !== 1 || event.ctrlKey || event.metaKey)
                    return;
                this.search(event.key);
        }
        // The open tree's key, not its owner's (an Esc closes the tree, not the Setting around it).
        event.preventDefault();
        event.stopPropagation();
    }
    /** QAbstractItemView.keyboardSearch: the next shown row starting with what was typed. */
    search(character) {
        const now = performance.now();
        this.typed = now - this.typedAt < 400 ? this.typed + character : character;
        this.typedAt = now;
        const text = (row) => (row.kind === "heading" ? this.groups[row.group].heading : row.kind === "leaf" ? row.leaf.label : this.none ?? "").toLowerCase();
        const wanted = this.typed.toLowerCase();
        const start = this.typed.length > 1 ? this.cursor : this.cursor + 1;
        for (let step = 0; step < this.rows.length; step++) {
            const index = (Math.max(0, start) + step) % this.rows.length;
            if (text(this.rows[index]).startsWith(wanted)) {
                this.move(index);
                return;
            }
        }
    }
}
