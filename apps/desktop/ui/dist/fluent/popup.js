import { element, token } from "./dom.js";
import { hidePopup, isShown, showPopup } from "./popover.js";
import { shareLabelColumn } from "./form.js";
/** FluentPopup + FluentSettingsPopupAnchor: a white rounded card of settings
 * that opens beside the command that asks for it (below it, else above it,
 * kept inside the window: a .beside popup bound to the command, popover.ts),
 * at its owner's width and at least its minimum height (two tokens), taller
 * when its content needs it.  The command is a toggle: it wears the active
 * face while the popup is open, and a click on it while open closes it.  A
 * click elsewhere or Escape closes it too.  Its content and its draft are the
 * owner's; a form in it stands on the one label column. */
export class Popup {
    anchor;
    body;
    minimum;
    root = element("div", "popup card bordered beside");
    constructor(anchor, body, minimum) {
        this.anchor = anchor;
        this.body = body;
        this.minimum = minimum;
        this.root.append(body);
        this.root.addEventListener("toggle", event => anchor.setActive(event.newState === "open"));
    }
    get open() { return isShown(this.root); }
    toggle() {
        if (this.open) {
            this.close();
            return;
        }
        // As wide as its minimum (v2: the popup opens at it; its form fills it), taller when its rows need it.
        this.root.style.width = `${token(this.minimum[0])}px`;
        this.root.style.minHeight = `${token(this.minimum[1])}px`;
        showPopup(this.root, this.anchor.root);
        shareLabelColumn(this.body);
    }
    close() { hidePopup(this.root); }
}
