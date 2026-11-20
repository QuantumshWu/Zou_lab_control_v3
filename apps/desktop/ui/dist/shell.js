// One page per window: v2 opened each of its windows on its own
// (Devices@, TaskConsole@, FigureViewer@, PulseGUI@Zou lab) and so does the
// host -- one native window per view, its page `index.html?window=<key>`
// showing that view alone. A view reaches another window through the host
// (show_window), as v2's windows opened one another.
import { element } from "./fluent/index.js";
import { TitleBar } from "./title_bar.js";
/** `then` once the page has laid out and drawn what it holds: on its next frame -- or, a page its window draws no
 * frames for while hidden, once this task is over (a measure then forces the layout it reads). */
function afterFrame(then) {
    if (document.visibilityState === "visible")
        requestAnimationFrame(() => then());
    else
        setTimeout(then);
}
export class Shell {
    host = element("div", "shell-views");
    ctx;
    view;
    /** v2's FluentWindow title bar, on a window of the desktop's (a shared browser view has its own). */
    titleBar;
    latest = { type: "state" };
    /** A window of the desktop's: made hidden, it comes on screen when its page says `window_ready`. */
    native;
    /** Whether the view has been shown: its first whole state laid out, or the page failed first. */
    shown = false;
    accepted = false;
    constructor(transport, factory) {
        this.ctx = { transport, latest: this.latest, show: window => transport.send({ action: "show_window", window }) };
        this.native = transport.connected && !transport.shared;
        // A page that fails before its view holds its first whole state is shown as it stands, its error said in
        // the status line and the journal as any other: no window stays hidden because of its page. (A page
        // that fails before it says `ready` at all is shown by the host when it has loaded.)
        if (this.native)
            for (const type of ["error", "unhandledrejection"])
                addEventListener(type, () => this.failed());
        this.titleBar = this.native ? new TitleBar() : null;
        if (this.titleBar)
            document.body.append(this.titleBar.root);
        document.body.append(this.host);
        // A browser shown this window by Browser sharing watches only, and says so above the view.
        if (transport.shared)
            document.body.prepend(element("div", "note shared-mark", "Shared view · watching only"));
        // A window's view is first painted whole: it is laid out unseen until the host's state has filled it.
        if (transport.connected)
            this.host.style.visibility = "hidden";
        this.view = factory(this.ctx);
        this.view.root.classList.add("window");
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
        this.accepted = true;
        try {
            this.view.update(state);
        }
        catch (error) {
            this.failed();
            throw error;
        }
        this.settle();
    }
    /** Also when a command of the view's is refused: its answer can complete the view's first state. */
    status() { this.view.status(); this.settle(); }
    /** Once the view holds its first whole state (the host's first state, and what `ready` waits for): it is
     * measured on the frame that lays it out -- after the fits the view asked that frame for -- its window
     * sized to it if the view sizes its window, and the window shown once that frame is drawn. */
    settle() {
        if (this.shown || !this.accepted || !(this.view.ready?.() ?? true))
            return;
        this.shown = true;
        this.host.style.visibility = "";
        if (this.native)
            afterFrame(() => {
                const size = this.view.measure?.();
                if (size)
                    this.ctx.transport.post({ action: "fit_window", ...size });
                afterFrame(() => this.ctx.transport.post({ action: "window_ready" }));
            });
    }
    failed() {
        if (this.shown)
            return;
        this.shown = true;
        this.host.style.visibility = "";
        if (this.native)
            this.ctx.transport.post({ action: "window_ready" });
    }
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
