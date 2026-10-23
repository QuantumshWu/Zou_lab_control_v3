// PulseEditor@Zou lab (v2 PulseEditorView): the header -- the board's dot,
// "PulseGUI · <pulse>", what the editor holds, Clear all -- over six pages.
// It renders the `pulse` state section and sends pulse_* commands; the
// preview's pixels arrive through the panel board's shared mappings.
import { TabWidget, command, element, frame, message, setElided, setState, setText, statusDot } from "../fluent/index.js";
import { queueStatus } from "../shell.js";
import { EditPage } from "./edit.js";
import { ConfigPage, PreviewPage, ScanPage, SubpulsePage, TargetPage } from "./pages.js";
import { confirmPulseDiscard } from "./widgets.js";
/** The board token -> the dot's word (PULSE_BOARD_STATES). */
const DOT = {
    "idle": "idle", "dirty-ready": "idle", "running-synced": "running", "running-stale": "warning", "unreachable": "error",
};
const PAGES = ["Edit", "Preview", "Scan", "Value Config", "Subpulse Config", "Target"];
export class PulseEditorView {
    ctx;
    key = "pulse-editor";
    title = "Pulse Editor";
    windowTitle = "PulseGUI@Zou lab";
    root = element("div", "window pulse-editor");
    dot = statusDot();
    heading = element("span", "heading pulse-heading", "PulseGUI · no pulse");
    summary = element("span", "summary pulse-summary elided");
    tabs = new TabWidget();
    edit;
    preview;
    scan;
    config;
    subpulse;
    target;
    shownMessage = 0;
    state = null;
    send;
    constructor(ctx, board) {
        this.ctx = ctx;
        this.send = (command, feedback) => ctx.transport.send(command, feedback);
        const say = (text) => { void message("Pulse", text.endsWith(".") ? text : `${text}.`); };
        this.edit = new EditPage(this.send, say, () => { if (this.tabs.currentKey !== "Edit") {
            this.tabs.setCurrent("Edit");
            this.send({ action: "pulse_page", page: "Edit" });
        } });
        this.preview = new PreviewPage(this.send);
        this.scan = new ScanPage(this.send);
        this.config = new ConfigPage(this.send);
        this.subpulse = new SubpulsePage(this.send, say, { before: () => this.edit.strip.beforeItem(), expectComponent: () => this.edit.expectComponent() });
        this.target = new TargetPage(this.send);
        board.external.set("pulse-preview", this.preview);
        const header = frame(false, "header pulse-header");
        const named = element("div", "identity pulse-identity");
        named.append(this.dot, this.heading);
        const clear = command("Clear all", () => void this.clearAll(), { role: "destructive" });
        header.append(named, this.summary, clear.root);
        const pages = {
            "Edit": this.edit.root, "Preview": this.preview.root, "Scan": this.scan.root, "Value Config": this.config.root,
            "Subpulse Config": this.subpulse.root, "Target": this.target.root,
        };
        for (const page of PAGES)
            this.tabs.addPermanent(page, page, pages[page]);
        this.tabs.onChange = key => { if (this.state && this.state.page !== key)
            this.send({ action: "pulse_page", page: key }); };
        this.root.append(header, this.tabs.root);
    }
    async clearAll() {
        const state = this.state;
        if (!state?.has_pulse)
            return;
        if (state.dirty && !await confirmPulseDiscard())
            return;
        this.send({ action: "pulse_clear", discard_changes: true }, "Clearing…");
    }
    update(state) {
        const pulse = state.pulse;
        if (!pulse)
            return;
        this.state = pulse;
        // v2: on the session's sequencer the editor is that device's window, "<role> Pulse Editor".
        const title = pulse.connection.locked ? `${pulse.connection.choices[0]?.label ?? ""} Pulse Editor` : this.windowTitle;
        if (document.title !== title)
            document.title = title;
        setState(this.dot, DOT[pulse.status]);
        setText(this.heading, pulse.title);
        setElided(this.summary, pulse.summary);
        this.edit.update(pulse);
        this.preview.update(pulse);
        this.scan.update(pulse);
        this.config.update(pulse);
        this.subpulse.update(pulse);
        this.target.update(pulse);
        if (this.tabs.currentKey !== pulse.page && PAGES.includes(pulse.page))
            this.tabs.setCurrent(pulse.page);
        if (pulse.message && pulse.message.id > this.shownMessage) {
            this.shownMessage = pulse.message.id;
            // v2 Sync: replacing unsaved work with the board's pulse asks first, then syncs again.
            if (pulse.message.kind === "sync_discard") {
                void confirmPulseDiscard().then(discard => { if (discard)
                    this.send({ action: "pulse_sync", discard_changes: true }, "Syncing…"); });
            }
            else
                void message("Pulse", pulse.message.text);
        }
    }
    status() {
        const [text, word] = queueStatus(this.ctx.transport, ["", "idle"]);
        if (word === "error" && this.state)
            setElided(this.summary, text);
    }
    shown() { if (this.state && this.state.page === "Preview")
        this.send({ action: "pulse_page", page: "Preview" }); }
    /** Shown with its pulse state: the pulse host sends it after the window's first state. */
    ready() { return this.state !== null; }
}
export const pulseEditor = (board) => (ctx) => new PulseEditorView(ctx, board);
