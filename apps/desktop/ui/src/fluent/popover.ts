import { token } from "./dom.js";

/** Every popup of the interface -- a combo box's list, a menu, a settings
 * popup -- stands beside the box that opened it and is bound to it: the box is
 * the popup's anchor (showPopover's source), and fluent.css places the popup
 * with `position-area` against it.  Its place is part of the layout that
 * places the box, so whatever moves the box -- a dragged Setting frame or
 * dialog, a carried card, a scrolled form, a board relayout, a resized window
 * -- moves the popup in the same frame; scrolled out of sight, the box takes
 * its popup with it (`position-visibility`).  While a popup is open the wheel
 * is the popup's, as Qt gives it to the active popup: turned outside it,
 * nothing behind it scrolls (a scroll the compositor makes on its own would
 * carry the box a frame ahead of its popup).
 *
 * It closes only when the operator closes it (Qt's popup, which a drag
 * elsewhere does not close): a pick, Esc, a press on its own box (the box's
 * toggle), or a click elsewhere -- a press and its release without a drag
 * between them.  A drag that starts outside it (a Setting caption, a title
 * band, a dialog, a plot) leaves it open.  Its box hidden or taken away, it
 * closes with it.  Opening one closes every other except the popups it stands
 * in (a combo's list over the popup that holds the combo). */
type Open = { popup: HTMLElement; box: Element | null };

// showPopover's options (Chromium 133+), which TypeScript's DOM library does not know yet.
declare global { interface HTMLElement { showPopover(options?: { source?: HTMLElement }): void } }

const shown: Open[] = [];
/** A press while a popup is open, until its release: where, the popups it was in (or on whose
 * box), and how far it has gone since. */
let press: { target: Node | null; within: HTMLElement[]; x: number; y: number; travel: number } | null = null;
let watching = false;
/** A box hidden or taken away closes its popup (its size goes to nothing). */
const boxes = new ResizeObserver(entries => {
  for (const entry of entries) {
    const box = entry.target;
    if (box.isConnected && box.checkVisibility()) continue;
    for (const open of opened()) if (open.box === box) hidePopup(open.popup);
  }
});

const grabWheel = (event: WheelEvent): void => {
  const open = opened();
  if (open.length && !open.some(({ popup }) => popup.contains(event.target as Node))) event.preventDefault();
};

/** The popups open now, bottom first (one taken out of the page with its host closed silently). */
function opened(): Open[] {
  for (let at = shown.length - 1; at >= 0; at--) if (!isShown(shown[at]!.popup)) forget(shown[at]!.popup);
  return [...shown];
}

/** Open `popup` beside `box` (null: where its owner places it, a context menu at the pointer). */
export function showPopup(popup: HTMLElement, box: Element | null): void {
  if (isShown(popup)) return;
  watch();
  for (const open of opened().reverse()) if (!(box && open.popup.contains(box))) hidePopup(open.popup);
  popup.popover = "manual";
  // A modal dialog's controls are the only live ones: a popup of one stands in it.
  const host = box?.closest("dialog") ?? document.body;
  if (popup.parentElement !== host) host.append(popup);
  if (!popup.dataset.popup) {
    popup.dataset.popup = "";
    popup.addEventListener("beforetoggle", event => { if ((event as ToggleEvent).newState === "closed") forget(popup); });
  }
  if (!shown.length) addEventListener("wheel", grabWheel, { capture: true, passive: false });
  shown.push({ popup, box });
  if (box) boxes.observe(box);
  if (box instanceof HTMLElement) popup.showPopover({ source: box }); else popup.showPopover();
}

/** Close `popup`, and first the popups that stand in it. */
export function hidePopup(popup: HTMLElement): void {
  if (!isShown(popup)) return;
  for (const open of opened().reverse()) if (open.box && popup.contains(open.box)) hidePopup(open.popup);
  popup.hidePopover();
}

export function isShown(popup: HTMLElement): boolean {
  return popup.matches(":popover-open");
}

/** The popup open beside `box`, if one is. */
export function popupOf(box: Element): HTMLElement | undefined {
  return opened().find(open => open.box === box)?.popup;
}

function forget(popup: HTMLElement): void {
  const at = shown.findIndex(open => open.popup === popup);
  if (at < 0) return;
  const box = shown.splice(at, 1)[0]!.box;
  if (box && !shown.some(open => open.box === box)) boxes.unobserve(box);
  if (!shown.length) removeEventListener("wheel", grabWheel, { capture: true });
}

/** Whether a popup or its box holds `target`. */
const holds = (open: Open, target: Node | null): boolean => Boolean(target && (open.popup.contains(target) || open.box?.contains(target)));

/** The popups from the top down to the one the press was in close. Where the press was is what
 * it was in when it was made (`within`): a list that redraws its rows before the release has
 * taken the pressed row out of the page, and the press was in that list all the same. A popup
 * the press itself opened is found by its box. */
function dismiss(target: Node | null, within: readonly HTMLElement[]): void {
  for (const open of opened().reverse()) {
    if (within.includes(open.popup) || holds(open, target)) return;
    hidePopup(open.popup);
  }
}

function watch(): void {
  if (watching) return;
  watching = true;
  // Capture on the window: heard before any control under the pointer takes the press.
  addEventListener("pointerdown", event => {
    const [target, open] = [event.target as Node | null, opened()];
    press = open.length ? { target, within: open.filter(item => holds(item, target)).map(item => item.popup), x: event.clientX, y: event.clientY, travel: 0 } : null;
  }, true);
  // How far the pointer went, heard on its moves and its release (a scroll bar's drag moves nothing the page hears).
  const travel = (event: PointerEvent): void => {
    if (press) press.travel = Math.max(press.travel, Math.hypot(event.clientX - press.x, event.clientY - press.y));
  };
  addEventListener("pointermove", travel, true);
  addEventListener("pointercancel", () => { press = null; }, true);
  addEventListener("pointerup", event => {
    travel(event);
    const pressed = press;
    press = null;
    // A drag moved the pointer as far as a card's carry starts (board.ts): not a click.
    if (pressed && pressed.travel < token("--control") / 4) dismiss(pressed.target, pressed.within);
  }, true);
  // Esc closes the top popup only, before the frame or dialog around it hears it.
  addEventListener("keydown", event => {
    const top = opened().at(-1);
    if (event.key !== "Escape" || !top) return;
    event.preventDefault();
    event.stopPropagation();
    hidePopup(top.popup);
  }, true);
}
