import type { Command } from "./button.js";
import { element, token } from "./dom.js";
import { shareLabelColumn } from "./form.js";

/** FluentPopup + FluentSettingsPopupAnchor: a white rounded card of settings
 * that opens beside the command that asks for it (below it, else above it,
 * kept inside the window), at its owner's width and at least its minimum
 * height (two tokens), taller when its content needs it.  The command is a toggle: it wears the active
 * face while the popup is open, and a press on it while open closes it
 * rather than opening it again.  Light dismiss: a click outside or Escape
 * closes it.  Its content and its draft are the owner's; a form in it stands on the one label column. */
export class Popup {
  readonly root = element("div", "popup card bordered");
  /** Whether the popup was open when the anchor was pressed (light dismiss closes it before the click). */
  private pressedOpen = false;

  constructor(readonly anchor: Command, readonly body: HTMLElement, private minimum: [width: string, height: string]) {
    this.root.popover = "auto";
    this.root.append(body);
    this.root.addEventListener("toggle", event => anchor.setActive((event as ToggleEvent).newState === "open"));
    anchor.root.addEventListener("pointerdown", () => { this.pressedOpen = this.open; });
  }

  get open(): boolean { return this.root.matches(":popover-open"); }

  toggle(): void {
    const close = this.open || this.pressedOpen;
    this.pressedOpen = false;
    if (close) { this.close(); return; }
    if (!this.root.isConnected) document.body.append(this.root);
    // As wide as its minimum (v2: the popup opens at it; its form fills it), taller when its rows need it.
    this.root.style.width = `${token(this.minimum[0])}px`;
    this.root.style.minHeight = `${token(this.minimum[1])}px`;
    this.root.showPopover();
    shareLabelColumn(this.body);
    const rect = this.anchor.root.getBoundingClientRect(), gap = token("--popup-gap"), pad = token("--window-pad");
    const width = this.root.offsetWidth, height = this.root.offsetHeight;
    const left = Math.max(pad, Math.min(rect.left, innerWidth - pad - width));
    const below = rect.bottom + gap;
    const top = below + height <= innerHeight - pad ? below : Math.max(pad, rect.top - gap - height);
    this.root.style.left = `${left}px`;
    this.root.style.top = `${top}px`;
  }

  close(): void { if (this.open) this.root.hidePopover(); }
}
