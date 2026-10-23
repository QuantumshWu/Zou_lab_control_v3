// Entry: this window's one transport, its board of plot panels (the Task
// Console's live board, the Figure Viewer's opened Figure, the Pulse
// Editor's preview pixels) and its one view.
import { confirm } from "./fluent/index.js";
import { Shell } from "./shell.js";
import { Transport } from "./transport.js";
import { PanelBoard } from "./views/board.js";
import { FIRST_REQUEST, WINDOWS, windowKey } from "./views/index.js";
const key = windowKey();
const review = `review-${new URLSearchParams(location.search).get("node") ?? ""}`;
const transport = new Transport(FIRST_REQUEST);
const board = new PanelBoard(transport, key === "figure-viewer" ? "figure-panels" : "panels");
const shell = new Shell(transport, WINDOWS[key](board));
board.changed = () => shell.status();
transport.attach({
    state: state => {
        // The Figure Viewer's document comes as its own section (state.viewer).
        // A message carries only the sections that changed: an answer with none still settles its request.
        if (key === "figure-viewer") {
            if (state.viewer || state.request !== undefined)
                board.accept({ ...state.viewer, type: "state", request: state.request, error: state.error });
        }
        // A point review's window draws its own review panel alone.
        else if (key === "review")
            board.accept(state.panels ? { ...state, panels: state.panels.filter(panel => panel.id === review) } : state);
        else
            board.accept(state);
        shell.accept(state);
    },
    frame: frame => board.frame(frame),
    image: (frame, image) => board.image(frame, image),
    shared: event => board.shared(event),
    releaseMappings: mapping => board.releaseMappings(mapping),
    changed: () => shell.status(),
    page: page => shell.view.showPage?.(page),
    caption: caption => shell.titleBar?.accept(caption),
    // v2 PulseGUI's close guard: each unsaved draft asks first; Cancel keeps the window as it was.
    closeRequested: async (unsaved) => {
        for (const what of unsaved)
            if (!await confirm(`Unsaved ${what}`, `Discard unsaved ${what} edits?`, { confirmText: "Discard" }))
                return;
        transport.send({ action: "close_pulse_editor" });
    },
});
// No browser menu (Back, Reload, Save as…) on the window; a text field keeps
// its edit menu, as a FluentLineEdit keeps its own.
document.addEventListener("contextmenu", event => {
    if (!(event.target instanceof HTMLInputElement))
        event.preventDefault();
});
// Words selected in a readout are that readout's: a press anywhere else drops
// them, as a FluentReadoutEdit deselects when it loses the focus -- no text
// stays painted selected while the operator works somewhere else.
document.addEventListener("pointerdown", event => {
    const selection = getSelection();
    if (!selection || selection.isCollapsed || !selection.rangeCount)
        return;
    const within = selection.getRangeAt(0).commonAncestorContainer;
    if (!(within instanceof Element ? within : within.parentElement)?.contains(event.target))
        selection.removeAllRanges();
}, true);
// A text field's selection goes with its focus, as a QLineEdit deselects on focus out:
// focused again by code (a step cell's press) it shows no old selection.  The window
// losing the activation keeps it (the field stays the document's active element).
document.addEventListener("focusout", event => {
    const field = event.target;
    if (!(field instanceof HTMLInputElement || field instanceof HTMLTextAreaElement) || document.activeElement === field)
        return;
    if (field.selectionStart !== null && field.selectionStart !== field.selectionEnd)
        field.setSelectionRange(field.selectionEnd, field.selectionEnd);
});
shell.status();
// Plots render at the device ratio v2's Qt canvas uses (its rounded ratio, set
// by scale.ts), so a front has v2's pixels; the canvas still shows it 1:1.
if (transport.connected && !transport.shared)
    transport.send({ action: "ready", dpr: Number(document.documentElement.dataset.qtDpr) || devicePixelRatio });
