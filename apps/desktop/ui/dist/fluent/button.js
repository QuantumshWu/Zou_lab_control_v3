import { assign, element } from "./dom.js";
import { fitWords, setGlyph } from "./ink.js";
const ROLE_CLASSES = ["primary", "quiet", "destructive", "destructive-primary"];
/** How long a two-click remover waits for its second click (arm_interval_ms:
 * the system double-click interval, never under a quarter second). */
export const ARM_INTERVAL_MS = 500;
/** FluentButton(confirm_twice=...): the first click arms the white-on-red
 * face, a second within the interval acts; left alone it returns to rest.
 * Only removing a plot panel or a logic node is armed. */
export function armable(node, phrase, confirmed) {
    let timer = 0;
    const rest = () => { window.clearTimeout(timer); timer = 0; node.classList.remove("armed"); node.title = `Click twice to ${phrase}`; };
    node.title = `Click twice to ${phrase}`;
    node.addEventListener("click", () => {
        if (node.classList.contains("armed")) {
            rest();
            confirmed();
            return;
        }
        node.classList.add("armed");
        node.title = `Click again to ${phrase}`;
        timer = window.setTimeout(rest, ARM_INTERVAL_MS);
    });
    // Hidden, disabled or left: back to rest (v2 disarms on hide/disable).
    node.addEventListener("blur", rest);
    return node;
}
/** FluentButton: a command drawn by its role.  The label is never elided;
 * a toggle command shows that it is on with `active`, set by its owner from
 * accepted state, never by its own click. */
export class Command {
    root;
    label;
    role;
    dirty = false;
    words = null;
    constructor(text, action, options = {}) {
        this.root = element("button");
        this.root.type = "button";
        this.label = text;
        this.role = options.role ?? "standard";
        if (options.glyph)
            this.root.classList.add("glyph");
        if (options.square)
            this.root.classList.add("square");
        if (options.title)
            this.root.title = options.title;
        if (options.words?.length) {
            // Every word in one grid cell: the slot is the widest of them, only
            // the current one is visible (v2 measures sizeHint for each word).
            this.words = element("span", "words");
            this.root.append(this.words);
            for (const word of options.words)
                this.words.append(element("span", "", word));
        }
        if (options.confirm)
            armable(this.root, options.confirm, action);
        else
            this.root.addEventListener("click", action);
        this.show();
    }
    get text() { return this.label; }
    set text(text) { if (text !== this.label) {
        this.label = text;
        this.show();
    } }
    set disabled(disabled) { assign(this.root, "disabled", disabled); if (disabled && this.root.classList.contains("armed"))
        this.root.classList.remove("armed"); }
    get disabled() { return this.root.disabled; }
    set hidden(hidden) { this.root.hidden = hidden; }
    set title(text) { if (this.root.title !== text)
        this.root.title = text; }
    setRole(role) { if (role !== this.role) {
        this.role = role;
        this.show();
    } }
    /** set_active: the SELECTED face a toggle command keeps while it is on. */
    setActive(active) { this.root.classList.toggle("active", active); }
    /** set_dirty: pressing this would save or apply something new -- drawn
     * primary with a trailing `*`, the one thing on its surface to press. */
    setDirty(dirty) { if (dirty !== this.dirty) {
        this.dirty = dirty;
        this.show();
    } }
    show() {
        const shown = this.dirty && !this.label.endsWith("*") ? `${this.label}*` : this.label;
        const role = this.dirty ? "primary" : this.role;
        for (const name of ROLE_CLASSES)
            this.root.classList.toggle(name, name === role);
        if (this.root.classList.contains("glyph")) {
            setGlyph(this.root, shown, "glyph");
            return;
        }
        if (!this.words) {
            if (this.root.textContent !== shown)
                this.root.textContent = shown;
            fitWords(this.root, [shown]);
            return;
        }
        let current = [...this.words.children].find(node => node.textContent === shown);
        if (!current) {
            current = element("span", "", shown);
            this.words.append(current);
        }
        for (const node of this.words.children)
            node.classList.toggle("current", node === current);
        fitWords(this.root, [...this.words.children].map(node => node.textContent ?? ""));
    }
}
/** A command button: `command("Save…", save, { role: "primary" }).root`. */
export function command(text, action, options = {}) {
    return new Command(text, action, options);
}
