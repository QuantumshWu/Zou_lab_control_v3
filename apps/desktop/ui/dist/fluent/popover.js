import { token } from "./dom.js";
const shown = [];
/** A press while a popup is open, until its release: where, and how far it has gone since. */
let press = null;
let watching = false;
/** A box hidden or taken away closes its popup (its size goes to nothing). */
const boxes = new ResizeObserver(entries => {
    for (const entry of entries) {
        const box = entry.target;
        if (box.isConnected && box.checkVisibility())
            continue;
        for (const open of opened())
            if (open.box === box)
                hidePopup(open.popup);
    }
});
const grabWheel = (event) => {
    const open = opened();
    if (open.length && !open.some(({ popup }) => popup.contains(event.target)))
        event.preventDefault();
};
/** The popups open now, bottom first (one taken out of the page with its host closed silently). */
function opened() {
    for (let at = shown.length - 1; at >= 0; at--)
        if (!isShown(shown[at].popup))
            forget(shown[at].popup);
    return [...shown];
}
/** Open `popup` beside `box` (null: where its owner places it, a context menu at the pointer). */
export function showPopup(popup, box) {
    if (isShown(popup))
        return;
    watch();
    for (const open of opened().reverse())
        if (!(box && open.popup.contains(box)))
            hidePopup(open.popup);
    popup.popover = "manual";
    // A modal dialog's controls are the only live ones: a popup of one stands in it.
    const host = box?.closest("dialog") ?? document.body;
    if (popup.parentElement !== host)
        host.append(popup);
    if (!popup.dataset.popup) {
        popup.dataset.popup = "";
        popup.addEventListener("beforetoggle", event => { if (event.newState === "closed")
            forget(popup); });
    }
    if (!shown.length)
        addEventListener("wheel", grabWheel, { capture: true, passive: false });
    shown.push({ popup, box });
    if (box)
        boxes.observe(box);
    if (box instanceof HTMLElement)
        popup.showPopover({ source: box });
    else
        popup.showPopover();
}
/** Close `popup`, and first the popups that stand in it. */
export function hidePopup(popup) {
    if (!isShown(popup))
        return;
    for (const open of opened().reverse())
        if (open.box && popup.contains(open.box))
            hidePopup(open.popup);
    popup.hidePopover();
}
export function isShown(popup) {
    return popup.matches(":popover-open");
}
/** The popup open beside `box`, if one is. */
export function popupOf(box) {
    return opened().find(open => open.box === box)?.popup;
}
function forget(popup) {
    const at = shown.findIndex(open => open.popup === popup);
    if (at < 0)
        return;
    const box = shown.splice(at, 1)[0].box;
    if (box && !shown.some(open => open.box === box))
        boxes.unobserve(box);
    if (!shown.length)
        removeEventListener("wheel", grabWheel, { capture: true });
}
/** Whether `target` is inside an open popup or its box: the popups from the top down to that one close. */
function dismiss(target) {
    for (const open of opened().reverse()) {
        if (target && (open.popup.contains(target) || open.box?.contains(target)))
            return;
        hidePopup(open.popup);
    }
}
function watch() {
    if (watching)
        return;
    watching = true;
    // Capture on the window: heard before any control under the pointer takes the press.
    addEventListener("pointerdown", event => {
        press = opened().length ? { target: event.target, x: event.clientX, y: event.clientY, travel: 0 } : null;
    }, true);
    // How far the pointer went, heard on its moves and its release (a scroll bar's drag moves nothing the page hears).
    const travel = (event) => {
        if (press)
            press.travel = Math.max(press.travel, Math.hypot(event.clientX - press.x, event.clientY - press.y));
    };
    addEventListener("pointermove", travel, true);
    addEventListener("pointercancel", () => { press = null; }, true);
    addEventListener("pointerup", event => {
        travel(event);
        const pressed = press;
        press = null;
        // A drag moved the pointer as far as a card's carry starts (board.ts): not a click.
        if (pressed && pressed.travel < token("--control") / 4)
            dismiss(pressed.target);
    }, true);
    // Esc closes the top popup only, before the frame or dialog around it hears it.
    addEventListener("keydown", event => {
        const top = opened().at(-1);
        if (event.key !== "Escape" || !top)
            return;
        event.preventDefault();
        event.stopPropagation();
        hidePopup(top.popup);
    }, true);
}
