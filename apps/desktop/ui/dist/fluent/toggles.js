import { assign, button, element } from "./dom.js";
const encode = (value) => JSON.stringify(value);
export function checked(node) { return node.getAttribute("aria-checked") === "true"; }
export function setChecked(node, on) {
    const word = String(on);
    if (node.getAttribute("aria-checked") !== word)
        node.setAttribute("aria-checked", word);
}
// ------------------------------------------------------------------ switch
/** FluentSwitch: a solid track with a white thumb and no outline.  Its
 * label is a fixed word beside it ("Selectors", "Plot"), or a pair of words
 * that say its state (["Auto", "Manual"]), as wide as the widest of them.
 * A click asks; the owner shows the answer with `setChecked`. */
export function switchButton(action, label) {
    const toggle = button("", () => action(!checked(toggle)), "switch");
    toggle.role = "switch";
    toggle.setAttribute("aria-checked", "false");
    toggle.append(element("span", "track"));
    if (typeof label === "string")
        toggle.append(element("span", "switch-label", label));
    else if (label) {
        const holder = element("span", "switch-words");
        holder.append(element("span", "on", label[0]), element("span", "off", label[1]));
        toggle.append(holder);
    }
    return toggle;
}
// ---------------------------------------------------------------- check box
/** FluentCheckBox: a square solid box whose edge is its fill (NEUTRAL_FILL,
 * ACCENT ticked) with a white tick, its words beside it.  For picking several
 * of a list; an on/off setting is a switch.  Words wider than the room its
 * row gives the box end in an ellipsis, whole in the tooltip while cut. */
export function checkBox(text, changed) {
    const box = button("", () => {
        const next = !checked(box);
        setChecked(box, next);
        changed(next);
    }, "check");
    box.role = "checkbox";
    box.setAttribute("aria-checked", "false");
    box.dataset.tipCut = "";
    box.append(element("span", "box"), element("span", "words", text));
    return box;
}
// --------------------------------------------------------------- segmented
/** FluentSegmentedButtons: one of a closed set the OWNER declares closed,
 * side by side.  A DIVIDER track; the current choice is the SELECTED pill,
 * the pointer the ACCENT_TINT wash; no outline, no focus face.  A choice
 * among data (frame labels, a growing list) is a combo instead. */
export class Segmented {
    activated;
    root = element("div", "segmented");
    buttons = new Map();
    signature = "";
    constructor(activated) {
        this.activated = activated;
        this.root.role = "radiogroup";
    }
    setChoices(choices) {
        const signature = encode(choices);
        if (signature === this.signature)
            return;
        this.signature = signature;
        this.buttons.clear();
        this.root.replaceChildren(...choices.map(([value, label]) => {
            const choice = button(label, () => this.activated(value));
            choice.role = "radio";
            choice.tabIndex = -1;
            choice.setAttribute("aria-checked", "false");
            this.buttons.set(encode(value), choice);
            return choice;
        }));
    }
    setCurrent(value) {
        const key = encode(value);
        for (const [choice, node] of this.buttons)
            setChecked(node, choice === key);
    }
    /** A choice the owner withholds keeps its place in DISABLED_INK. */
    setWithheld(value, withheld) {
        const node = this.buttons.get(encode(value));
        if (node)
            node.disabled = withheld;
    }
    set disabled(disabled) {
        this.root.classList.toggle("disabled", disabled);
        for (const node of this.buttons.values())
            assign(node, "disabled", disabled);
    }
}
