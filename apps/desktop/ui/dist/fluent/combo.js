import { assign, element } from "./dom.js";
const encode = (value) => JSON.stringify(value);
const normal = (choice) => Array.isArray(choice) ? { value: choice[0], label: choice[1] } : choice;
/** FluentComboBox: a field like an edit (ACCENT edge under the pointer or
 * while open) with the chevron in the stepper column; its popup a white card
 * with the picked row SELECTED and the row under the pointer ACCENT_TINT.
 * For data and long lists; a closed set the owner declares is Segmented.
 * `changed` is the operator's pick; `value =` projects one and never emits. */
let measure;
const REFERENCE_PX = 100;
/** FluentComboBox.sizeHint: the box is at least as wide as the widest thing it
 * may say (its placeholder or a pickable row, without a grouped row's indent)
 * and as `minimum` "M"s (setMinimumContentsLength), with its chrome. Kept in em
 * of the box's own font (fluent.css reads --combo-min), so it holds at any scale;
 * a fixed or ignored size is the owner's CSS min-width. */
export function hintWidest(select, labels, minimum = 0) {
    if (measure === undefined) {
        measure = document.createElement("canvas").getContext("2d");
        const style = getComputedStyle(document.body);
        if (measure)
            measure.font = `${style.fontWeight} ${REFERENCE_PX}px ${style.fontFamily}`;
    }
    if (!measure)
        return;
    let widest = minimum > 0 ? measure.measureText("M".repeat(minimum)).width : 0;
    for (const label of labels)
        widest = Math.max(widest, measure.measureText(label.trimStart()).width);
    const value = `calc(${Math.ceil(widest * 100 / REFERENCE_PX) / 100}em + var(--combo-chrome))`;
    if (select.style.getPropertyValue("--combo-min") !== value)
        select.style.setProperty("--combo-min", value);
}
/** Where a combo box's list opens (QComboBox::showPopup): below the box when
 * the whole list fits there; otherwise on the side with more room -- above when
 * the list fits there or shows more of itself there -- as tall as that side
 * allows, the rest scrolling.  `natural` is the list's own height, `below` and
 * `above` the room the window leaves it on each side; a tie opens below. */
export function dropPlacement(natural, below, above) {
    const up = natural > below && above > below;
    return { up, height: Math.max(0, Math.min(natural, up ? above : below)) };
}
/** The window a list opens in, without a scroll bar (a fixed popup's containing block). */
const windowHeight = () => document.documentElement.clientHeight;
const px = (value) => parseFloat(value) || 0;
/** Place an open `list` (a fixed popover whose CSS margins are its popup gaps)
 * under `box` or over it, as dropPlacement says; given `up`, on that side (a
 * list that grows while open stays on the side it opened on), capped to it.
 * Returns the side. */
export function placeDrop(list, box, up) {
    const style = list.style;
    style.maxHeight = "";
    const css = getComputedStyle(list);
    const margins = px(css.marginTop) + px(css.marginBottom);
    const natural = list.getBoundingClientRect().height;
    const view = windowHeight(), below = view - box.bottom - margins, above = box.top - margins;
    const drop = up === undefined ? dropPlacement(natural, below, above) : { up, height: Math.max(0, Math.min(natural, up ? above : below)) };
    style.top = drop.up ? "" : `${box.bottom}px`;
    style.bottom = drop.up ? `${view - box.top}px` : "";
    if (drop.height < natural)
        style.maxHeight = `${drop.height}px`;
    return drop.up;
}
/** The select whose list was placed last: placed again when the window is resized while it is open. */
let placedPicker = null;
/** A select's list is the browser's ::picker(select): fluent.css opens it
 * above the box for `drop-up` and caps it at `--drop-room`; this sets them as
 * dropPlacement says, from the list's height before either applies. */
function placePicker(select) {
    if (!select.matches(":open"))
        return;
    select.classList.remove("drop-up");
    select.style.removeProperty("--drop-room");
    const picker = getComputedStyle(select, "::picker(select)");
    const natural = px(picker.height) + (picker.boxSizing === "border-box" ? 0
        : px(picker.paddingTop) + px(picker.paddingBottom) + px(picker.borderTopWidth) + px(picker.borderBottomWidth));
    const margins = px(picker.marginTop) + px(picker.marginBottom);
    const box = select.getBoundingClientRect();
    const drop = dropPlacement(natural, windowHeight() - box.bottom - margins, box.top - margins);
    select.classList.toggle("drop-up", drop.up);
    if (drop.height < natural)
        select.style.setProperty("--drop-room", `${drop.height}px`);
    // The row the list opened on stays in sight in the shorter list.
    const focused = document.activeElement;
    if (focused instanceof HTMLOptionElement && select.contains(focused))
        focused.scrollIntoView({ block: "nearest" });
    if (!placedPicker)
        addEventListener("resize", () => { if (placedPicker?.matches(":open"))
            placePicker(placedPicker); });
    placedPicker = select;
}
/** The closed box's words: the picked row's, ending in an ellipsis where the
 * box is narrower (v2 paints elidedText); first child of `select`, kept there.
 * The platform copies the row only when the pick changes: a picked row whose own
 * words change (Scope's coordinate, a blank's words) is copied again here.
 * Its list opens below the box, or above it where it fits better (placePicker):
 * placed when the list opens and focuses a row (a move between rows is not an
 * opening), or, when no row can take the focus, after the press or key that opened it. */
export function comboFace(select) {
    const face = element("button", "combo-face");
    face.type = "button";
    const shown = document.createElement("selectedcontent");
    face.append(shown);
    new MutationObserver(() => {
        const picked = select.selectedOptions[0];
        if (shown.textContent !== (picked?.textContent ?? ""))
            shown.replaceChildren(...[...picked?.childNodes ?? []].map(node => node.cloneNode(true)));
    }).observe(select, { subtree: true, childList: true, characterData: true });
    select.addEventListener("focusin", event => {
        const from = event.relatedTarget;
        if (event.target instanceof HTMLOptionElement && !(from instanceof HTMLOptionElement && select.contains(from)))
            placePicker(select);
    });
    const opened = () => { requestAnimationFrame(() => { if (document.activeElement === select)
        placePicker(select); }); };
    select.addEventListener("pointerdown", opened);
    select.addEventListener("keydown", opened);
    return face;
}
export class ComboBox {
    changed;
    placeholder;
    root = element("select");
    face = comboFace(this.root);
    signature = "";
    current = null;
    minimum = 0;
    /** `activated`: QComboBox.activated -- a pick of the row already shown is a pick too
     * (v2 PulsePreviewView: re-picking the drawn size pins it); the select reports only a change. */
    constructor(changed, placeholder = "", activated = false) {
        this.changed = changed;
        this.placeholder = placeholder;
        let picked = false;
        this.root.addEventListener("change", () => {
            picked = true;
            const raw = this.root.value;
            const value = raw === "" ? null : JSON.parse(raw);
            this.current = value;
            this.root.classList.toggle("blank", value === null);
            this.changed(value);
        });
        if (!activated)
            return;
        // A press on a row starts a pick; its click ends it, after the change a new row reports.
        this.root.addEventListener("pointerdown", event => { if (event.target.closest("option"))
            picked = false; });
        this.root.addEventListener("click", event => {
            const option = event.target.closest("option");
            if (!option || picked || option.value === "" || option.value !== this.root.value)
                return;
            this.changed(this.current);
        });
    }
    setChoices(choices) {
        const list = choices.map(normal);
        const signature = encode(list) + this.placeholder;
        if (signature === this.signature)
            return;
        this.signature = signature;
        const nodes = [];
        let group = null;
        for (const choice of list) {
            const option = new Option(choice.label, encode(choice.value));
            if (choice.title)
                option.title = choice.title;
            option.disabled = Boolean(choice.disabled);
            if (choice.group) {
                if (!group || group.label !== choice.group) {
                    group = element("optgroup");
                    group.label = choice.group;
                    nodes.push(group);
                }
                group.append(option);
            }
            else {
                group = null;
                nodes.push(option);
            }
        }
        // The empty choice: the placeholder in MUTED_INK, never offered in the list.
        const blank = new Option(this.placeholder, "");
        blank.hidden = true;
        this.root.replaceChildren(this.face, blank, ...nodes);
        this.show();
        this.hint(list);
    }
    /** setMinimumContentsLength: never narrower than `length` "M"s. */
    minimumContents(length) { this.minimum = length; this.hint(); return this; }
    hint(list = [...this.root.options].filter(option => !option.hidden).map(option => ({ label: option.text, disabled: option.disabled }))) {
        hintWidest(this.root, [this.placeholder, ...list.filter(choice => !choice.disabled).map(choice => choice.label)], this.minimum);
    }
    get value() { return this.current; }
    set value(value) { this.current = value; this.show(); }
    set disabled(disabled) { assign(this.root, "disabled", disabled); }
    set title(text) { if (this.root.title !== text)
        this.root.title = text; }
    show() {
        const key = this.current === null ? "" : encode(this.current);
        if (this.root.value !== key)
            this.root.value = key;
        if (this.root.selectedIndex < 0)
            this.root.value = "";
        this.root.classList.toggle("blank", this.root.value === "");
    }
}
/** An editable FluentComboBox (QComboBox.setEditable): a name typed into its
 * field or picked from its list.  `commit` hears the text when it is picked,
 * on Enter or when the field is left; `fill` (fill_editable_combo) shows the
 * accepted name and never takes a draft out from under the cursor. */
export class EditableCombo {
    commit;
    root = element("div", "edit-combo");
    input = element("input");
    list = element("div", "edit-combo-list");
    names = [];
    accepted = "";
    draft = false;
    constructor(commit, placeholder = "") {
        this.commit = commit;
        this.input.type = "text";
        this.input.spellcheck = false;
        this.input.placeholder = placeholder;
        const open = element("button", "edit-combo-open");
        open.type = "button";
        open.tabIndex = -1;
        this.list.popover = "auto";
        this.root.append(this.input, open);
        // The list opens on the click, after the press's light dismiss (opened by the press, its
        // release closed it again); a press while it is open closes it (the light dismiss).
        let wasOpen = false;
        open.addEventListener("pointerdown", event => { event.preventDefault(); wasOpen = this.listOpen; });
        open.addEventListener("click", () => { if (!wasOpen)
            this.showList(); wasOpen = false; });
        // Placed again when the window is resized while it is open.
        this.list.addEventListener("toggle", event => {
            if (event.newState === "open")
                addEventListener("resize", this.place);
            else
                removeEventListener("resize", this.place);
        });
        this.input.addEventListener("input", () => { this.draft = true; });
        this.input.addEventListener("keydown", event => { if (event.key === "Enter")
            this.finish(); });
        this.input.addEventListener("blur", () => this.finish());
    }
    get listOpen() { return this.list.matches(":popover-open"); }
    finish() {
        if (!this.draft)
            return;
        this.draft = false;
        const text = this.input.value.trim();
        if (text !== this.accepted)
            this.commit(text);
    }
    showList() {
        if (this.input.disabled || !this.names.length)
            return;
        this.list.replaceChildren(...this.names.map(name => {
            const row = element("div", "edit-combo-row", name);
            row.classList.toggle("picked", name === this.input.value);
            row.addEventListener("pointerup", () => {
                this.list.hidePopover();
                this.input.value = name;
                this.draft = false;
                if (name !== this.accepted)
                    this.commit(name);
            });
            return row;
        }));
        // Inside the field: a popup holding the field stays open under its list (popover ancestry).
        if (!this.list.isConnected)
            this.root.append(this.list);
        this.list.showPopover();
        this.place();
    }
    /** Below the field, or above it where the list fits better (placeDrop). */
    place = () => {
        const rect = this.root.getBoundingClientRect();
        this.list.style.minWidth = `${rect.width}px`;
        this.list.style.left = `${rect.left}px`;
        placeDrop(this.list, rect);
    };
    /** v2 fill_editable_combo: offer `names`, show `key` (the accepted one) unless a draft is being typed. */
    fill(names, key, keepDraft = true) {
        this.names = [...names];
        this.accepted = key;
        const typing = keepDraft && this.draft && document.activeElement === this.input;
        if (!typing) {
            this.draft = false;
            if (this.input.value !== key)
                this.input.value = key;
        }
    }
    set disabled(disabled) { assign(this.input, "disabled", disabled); this.root.classList.toggle("disabled", disabled); }
    set title(text) { if (this.input.title !== text)
        this.input.title = text; }
}
