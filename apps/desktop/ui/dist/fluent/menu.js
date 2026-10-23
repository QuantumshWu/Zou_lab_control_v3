import { element, token } from "./dom.js";
/** _FluentRoundedMenu: a white rounded card with the DIVIDER edge, its rows
 * washed ACCENT_TINT under the pointer, no tick gutter.  Built for one
 * click and gone after it, so it lists what is true now (New…'s templates,
 * the tab overflow).  Light dismiss: a click outside or Escape closes it. */
export class Menu {
    root = element("div", "menu");
    constructor(items) {
        this.root.popover = "auto";
        this.root.role = "menu";
        for (const item of items) {
            if (item === "separator") {
                this.root.append(element("div", "menu-separator"));
                continue;
            }
            const row = element("button", "menu-item", item.label);
            row.type = "button";
            row.role = "menuitem";
            row.disabled = Boolean(item.disabled);
            if (item.title)
                row.title = item.title;
            row.addEventListener("click", () => { this.close(); item.action(); });
            this.root.append(row);
        }
        this.root.addEventListener("toggle", event => { if (event.newState === "closed")
            this.root.remove(); });
    }
    /** Below `anchor`, a popup gap under it, its left edge on the anchor's,
     * kept inside the window. */
    openBelow(anchor) {
        document.body.append(this.root);
        this.root.showPopover();
        const rect = anchor.getBoundingClientRect(), gap = token("--popup-gap"), pad = token("--window-pad");
        const width = this.root.offsetWidth, height = this.root.offsetHeight;
        const left = Math.max(pad, Math.min(rect.left, innerWidth - pad - width));
        const below = rect.bottom + gap;
        const top = below + height <= innerHeight - pad ? below : Math.max(pad, rect.top - gap - height);
        this.root.style.left = `${left}px`;
        this.root.style.top = `${top}px`;
    }
    /** A context menu (QMenu.exec at the pointer): its corner at the point, flipped to stay in the window. */
    openAt(x, y) {
        document.body.append(this.root);
        this.root.showPopover();
        const pad = token("--window-pad"), width = this.root.offsetWidth, height = this.root.offsetHeight;
        const left = x + width <= innerWidth - pad ? x : Math.max(pad, x - width);
        const top = y + height <= innerHeight - pad ? y : Math.max(pad, y - height);
        this.root.style.left = `${left}px`;
        this.root.style.top = `${top}px`;
    }
    close() { if (this.root.matches(":popover-open"))
        this.root.hidePopover(); }
}
