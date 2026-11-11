/** How long lines wait before they are sent, and how many at most wait. */
const SEND_EVERY_MS = 1000;
const WAITING_LIMIT = 200;
const WORDS_LIMIT = 80;
/** The keys that are commands, said with their modifiers; any key held with Ctrl or Meta is one too. */
const COMMAND_KEYS = new Set(["Enter", "Escape", "Tab", "Delete", "Backspace", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight",
    "Home", "End", "PageUp", "PageDown", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12"]);
const words = (text) => {
    const said = (text ?? "").replace(/\s+/g, " ").trim();
    return said ? said.slice(0, WORDS_LIMIT) : undefined;
};
/** What a control is, for someone who was not looking: its kind, its own words, and the item, panel or
 * card it stands in. Never its value. */
export function describe(target) {
    if (!(target instanceof Element))
        return {};
    const control = target.closest("button, select, input, textarea, [role], [data-item], a, label") ?? target;
    const field = control instanceof HTMLInputElement || control instanceof HTMLTextAreaElement || control instanceof HTMLSelectElement;
    return {
        tag: control.tagName.toLowerCase(),
        role: control.getAttribute("role") ?? undefined,
        id: control.id || undefined,
        class: words(control.getAttribute("class")),
        // A field's words are its label and its name; a button's are what it says.
        words: words(control.getAttribute("aria-label")) ?? words(control.getAttribute("title")) ?? (field ? undefined : words(control.textContent)),
        name: field ? control.name || undefined : undefined,
        item: target.closest("[data-item]")?.dataset.item,
        panel: target.closest("[data-panel]")?.dataset.panel,
        popup: target.closest("[popover]") ? true : undefined,
    };
}
/** Whether a key press is a command (kept, with its modifiers) rather than typing (never kept). */
export function isCommandKey(event) {
    return COMMAND_KEYS.has(event.key) || ((event.ctrlKey || event.metaKey || event.altKey) && event.key.length === 1);
}
/** Keep this window's journal: `post` sends a batch to the core. Returns `note`, for a line of the page's own. */
export function startJournal(post, page) {
    let waiting = [];
    let dropped = 0;
    const now = () => Math.round((performance.timeOrigin + performance.now()) * 1000);
    const note = (ev, facts = {}) => {
        if (waiting.length >= WAITING_LIMIT) {
            dropped++;
            return;
        }
        waiting.push({ us: now(), ev, facts: { page, ...facts } });
    };
    const send = () => {
        if (dropped) {
            waiting.push({ us: now(), ev: "dropped", facts: { page, lines: dropped } });
            dropped = 0;
        }
        if (!waiting.length)
            return;
        const lines = waiting;
        waiting = [];
        try {
            post({ action: "journal", lines });
        }
        catch { /* the journal never fails the page */ }
    };
    setInterval(send, SEND_EVERY_MS);
    addEventListener("pagehide", send);
    // Heard before any control: a handler that stops the event does not hide it from the journal.
    addEventListener("click", event => note("click", describe(event.target)), true);
    addEventListener("dblclick", event => note("double_click", describe(event.target)), true);
    addEventListener("contextmenu", event => note("context_menu", describe(event.target)), true);
    addEventListener("keydown", event => {
        if (event.repeat || !isCommandKey(event))
            return;
        const held = [event.ctrlKey && "Ctrl", event.altKey && "Alt", event.shiftKey && "Shift", event.metaKey && "Meta"].filter(Boolean);
        note("key", { key: [...held, event.key].join("+"), ...describe(event.target) });
    }, true);
    addEventListener("change", event => note("change", describe(event.target)), true);
    // Every popup of the interface is a popover (fluent/popover.ts): its toggle says it opened or closed.
    addEventListener("toggle", event => {
        const state = event.newState;
        if (event.target instanceof HTMLElement && event.target.hasAttribute("popover"))
            note(state === "open" ? "popup_opened" : "popup_closed", { class: words(event.target.className) });
    }, true);
    document.addEventListener("visibilitychange", () => note(document.hidden ? "hidden" : "shown"));
    addEventListener("focus", () => note("focused"));
    addEventListener("blur", () => note("blurred"));
    addEventListener("error", event => note("error", { message: words(event.message), source: event.filename, line: event.lineno, column: event.colno }));
    addEventListener("unhandledrejection", event => note("error", { message: words(String(event.reason?.message ?? event.reason)), rejection: true }));
    note("opened", { url: location.pathname.split("/").pop(), dpr: devicePixelRatio });
    return note;
}
