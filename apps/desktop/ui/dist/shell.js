// One page per window: v2 opened each of its windows on its own
// (Devices@, TaskConsole@, FigureViewer@, PulseGUI@Zou lab) and so does the
// host -- one native window per view, its page `index.html?window=<key>`
// showing that view alone. A view reaches another window through the host
// (show_window), as v2's windows opened one another.
import { element } from "./fluent/index.js";
import { TitleBar } from "./title_bar.js";
export class Shell {
    host = element("div", "shell-views");
    ctx;
    view;
    /** v2's FluentWindow title bar, on a window of the desktop's (a shared browser view has its own). */
    titleBar;
    latest = { type: "state" };
    constructor(transport, factory) {
        this.ctx = { transport, latest: this.latest, show: window => transport.send({ action: "show_window", window }) };
        this.titleBar = transport.connected && !transport.shared ? new TitleBar() : null;
        if (this.titleBar)
            document.body.append(this.titleBar.root);
        document.body.append(this.host);
        // A browser shown this window by Browser sharing watches only, and says so above the view.
        if (transport.shared)
            document.body.prepend(element("div", "note shared-mark", "Shared view · watching only"));
        this.view = factory(this.ctx);
        this.view.root.classList.add("window");
        // A window's view is first painted whole: it is laid out unseen until the host's state has filled it.
        if (transport.connected)
            this.host.style.visibility = "hidden";
        this.host.append(this.view.root);
        document.title = this.view.windowTitle;
        // The native window hides and shows the page: a hidden view holds no captured gesture.
        document.addEventListener("visibilitychange", () => { if (document.hidden)
            this.view.hidden?.();
        else
            this.view.shown?.(); });
        this.view.shown?.();
        this.view.status();
    }
    accept(state) {
        for (const [name, value] of Object.entries(state)) {
            if (name === "type" || name === "request" || name === "error" || name === "result")
                continue;
            if (value !== undefined)
                this.latest[name] = value;
        }
        this.view.update(state);
        if (this.host.style.visibility && (this.view.ready?.() ?? true))
            this.host.style.visibility = "";
    }
    status() { this.view.status(); }
}
/** Save image… (v2 save_screenshot): this window as it is on screen, to a PNG the operator picks. */
export function saveImage(transport) {
    transport.request({ action: "save_image" }, "Saving the window image…").then(result => {
        const path = result?.path;
        if (path)
            transport.note(`screenshot saved to ${path}`);
    }, () => { });
}
/** A view's status line from the one command queue: its error, else what is
 * under way, else the view's own idle words. */
export function queueStatus(transport, idle) {
    if (transport.lastError)
        return [transport.lastError, "error"];
    const busy = transport.busyText();
    if (busy)
        return [busy, "task"];
    return transport.notice ?? idle;
}
