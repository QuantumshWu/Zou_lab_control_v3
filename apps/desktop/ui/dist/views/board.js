// ConsoleBoardView + PanelCardView: the plot panels of a document, each a
// card with its finished picture, its title strip and its Setting frame.
// One board per plot window: the Task Console's live board, the Figure
// Viewer's opened Figure (each window's pixel mappings are its own). The
// board only copies finished pixels and forwards pointer and setting
// intents; every scientific decision is the host's.
import { Field, armable, button, comboFace, element, fieldKind, hintWidest, setFragments, setState, shareLabelColumn, token } from "../fluent/index.js";
import { CanvasPointer } from "./pointer.js";
/** A panel's Edit (`edit-<panel>`): drawn on its Edit page, never a card on the board. */
export const isEdit = (id) => id.startsWith("edit-");
/** A run's point review (`review-<node>`): its image on its own surface, never a board card. */
export const isReview = (id) => id.startsWith("review-");
/** An SLM Editor's pictures (`slm-<device>-<plot>`): drawn in its window, never board cards. */
export const isSlm = (id) => id.startsWith("slm-");
/** The cards that stand on another page, not on the board: Edits, point reviews, SLM Editor pictures. */
const offBoard = (id) => isEdit(id) || isReview(id) || isSlm(id);
const mapKey = (mapping, slot) => JSON.stringify([mapping, slot]);
export class PanelBoard {
    transport;
    /** #panels: cards in reading order, a card pad apart. */
    root = element("main");
    views = new Map();
    staged = new Map();
    /** Fronts the host delivers for a view of its own (the Pulse Editor's
     * preview), drawn by that view rather than a panel card. */
    external = new Map();
    signals = [];
    plotKinds = [];
    board = null;
    catalogKey = "";
    maps = new Map();
    gestures = true;
    /** Panels that take no pointer input whatever the Selectors say (an SLM Editor's wavefront). */
    quiet = new Set();
    /** Told when the set of panels or the presentation queue changes. */
    changed = () => { };
    /** Setting's Edit (v2 Panel Edit): the board's host opens the panel's Edit page. */
    onEdit = null;
    constructor(transport, id = "panels") {
        this.transport = transport;
        this.root.id = id;
        this.root.classList.add("panel-board");
        this.root.setAttribute("aria-label", "Figure panels");
        window.addEventListener("blur", () => { for (const view of this.views.values())
            view.cancelPointer(); });
        window.addEventListener("resize", () => { for (const view of this.views.values())
            view.placeSettings(); });
    }
    send(command, feedback) { return this.transport.send(command, feedback); }
    /** Whether the plots take pointer input: the hosting view's Selectors
     * (the Task Console's switch, v2 set_deriving; the Figure Viewer always). */
    get interactive() { return this.gestures; }
    set interactive(on) {
        if (on === this.gestures)
            return;
        this.gestures = on;
        if (!on)
            for (const view of this.views.values())
                view.cancelPointer();
    }
    /** The board's cards (an Edit is its page's, not the board's). */
    get size() { let count = 0; for (const id of this.views.keys())
        if (!offBoard(id))
            count++; return count; }
    /** Pictures still being presented (their state waits for the frame). */
    get presenting() { return this.staged.size > 0; }
    signalChoices(current) {
        // v2 names a node's signal "<node>/<output>" (no "@logic/" grammar in the words).
        const choices = this.signals.map(signal => [signal.id, signal.label.replace(/^@logic\//, "") + (!signal.available ? ` · ${signal.reason || "Waiting for data"}` : "")]);
        if (current && !this.signals.some(signal => signal.id === current))
            choices.push([current, `${current} · Unavailable`]);
        return choices;
    }
    /** The signal and plot-kind catalogue; true when it changed. */
    updateCatalog(state) {
        if (state.signals)
            this.signals = state.signals;
        if (state.plot_kinds)
            this.plotKinds = state.plot_kinds;
        const key = JSON.stringify([this.signals, this.plotKinds]);
        if (key === this.catalogKey)
            return false;
        this.catalogKey = key;
        return true;
    }
    /** The workbench board's entry for a panel (name, interval), when it runs. */
    boardPanel(id) { return this.board?.panels.find(entry => entry.panel === id); }
    frozen(id) { return this.board?.frozen[id] ?? ""; }
    accept(state) {
        if (state.request !== undefined && state.request === this.orderRequest)
            this.orderRequest = null;
        const boardChanged = Boolean(state.board) && JSON.stringify(state.board) !== JSON.stringify(this.board);
        if (state.board)
            this.board = state.board;
        const catalogChanged = this.updateCatalog(state) || boardChanged;
        for (const view of this.views.values()) {
            const panel = state.panels?.find(panel => panel.id === view.id);
            view.complete(state.error ? state.request : panel?.authoring_pending ? undefined : panel?.target_request ?? state.request, Boolean(state.error || panel?.error));
        }
        if (state.panels) {
            const alive = new Set(state.panels.map(panel => panel.id));
            for (const [id, view] of this.views)
                if (!alive.has(id)) {
                    view.cancelPointer();
                    view.root.remove();
                    this.views.delete(id);
                    this.staged.delete(id);
                }
            for (const panel of state.panels) {
                let view = this.views.get(panel.id);
                if (!view) {
                    view = new PanelView(this, panel.id);
                    this.views.set(panel.id, view);
                }
                const request = state.error ? state.request : panel.authoring_pending ? undefined : panel.target_request ?? state.request;
                if (panel.frame && panel.frame.sequence < view.drawnSequence)
                    continue;
                if (panel.frame && panel.frame.sequence > view.drawnSequence) {
                    this.staged.set(panel.id, { panel, request, failed: Boolean(state.error || panel.error) });
                    // A card with no picture yet has none its state could run ahead of.
                    if (view.drawnSequence < 0)
                        view.reserve(panel, request, Boolean(state.error || panel.error));
                }
                else
                    view.update(panel, request, Boolean(state.error || panel.error));
            }
            // Cards in the order the state lists them (board.panels order, v2's
            // order_committed), moving only the ones out of place -- unless the
            // operator just put them in another one and its answer is still due.
            this.order = state.panels.map(panel => panel.id);
            this.place();
        }
        if (catalogChanged)
            for (const view of this.views.values())
                view.refreshCatalog();
        this.root.classList.toggle("can-carry", Boolean(this.board) && this.size > 1);
        this.changed();
    }
    /** The cards in the order the state last listed them. */
    place() {
        let cursor = this.orderRequest !== null || this.carried ? null : this.root.firstElementChild;
        for (const id of this.order) {
            const view = this.views.get(id);
            // A panel's Edit card stands on its Edit page, a review's in its review.
            if (!view || offBoard(id))
                continue;
            const node = view.root;
            if (this.orderRequest !== null || this.carried) {
                if (!node.isConnected)
                    this.root.append(node);
                continue;
            }
            if (node !== cursor)
                this.root.insertBefore(node, cursor);
            else
                cursor = cursor.nextElementSibling;
        }
    }
    shared(event) {
        let buffer = null;
        try {
            const meta = typeof event.additionalData === "string" ? JSON.parse(event.additionalData) : event.additionalData;
            buffer = event.getBuffer();
            if (meta.type !== "shared-rgba-map" || !Number.isSafeInteger(meta.width) || !Number.isSafeInteger(meta.height) || meta.width < 1 || meta.height < 1
                || meta.stride !== meta.width * 4 || meta.byte_length !== meta.stride * meta.height || buffer.byteLength < meta.byte_length)
                throw new Error("The native image mapping has an invalid RGBA layout.");
            const key = mapKey(meta.mapping, meta.slot);
            if (this.maps.has(key))
                throw new Error("The native image mapping is already registered.");
            const image = new ImageData(new Uint8ClampedArray(buffer, 0, meta.byte_length), meta.width, meta.height, { colorSpace: "srgb" });
            this.maps.set(key, { meta, buffer, image });
            this.transport.post({ action: "map_ready", mapping: meta.mapping, slot: meta.slot });
        }
        catch (error) {
            if (buffer)
                this.transport.webview?.releaseBuffer(buffer);
            this.transport.report(error);
        }
    }
    /** A shared view's picture (no mapping to borrow or give back). */
    image(frame, image) {
        const view = this.views.get(frame.panel) ?? this.external.get(frame.panel);
        if (!view)
            return;
        try {
            view.draw(image, frame);
            this.changed();
        }
        catch (cause) {
            this.transport.report(cause);
        }
    }
    releaseMappings(mapping) {
        for (const [key, slot] of this.maps)
            if (slot.meta.mapping === mapping) {
                this.transport.webview?.releaseBuffer(slot.buffer);
                this.maps.delete(key);
            }
    }
    frame(frame) {
        let accepted = false;
        let error = "";
        try {
            const slot = this.maps.get(mapKey(frame.mapping, frame.slot));
            const view = this.views.get(frame.panel) ?? this.external.get(frame.panel);
            if (!slot || !view || slot.meta.panel !== frame.panel || slot.meta.width !== frame.width || slot.meta.height !== frame.height)
                throw new Error("The native frame does not match its panel and shared mapping.");
            accepted = view.draw(slot.image, frame);
            this.changed();
        }
        catch (cause) {
            error = String(cause);
            this.transport.report(cause);
        }
        finally {
            this.transport.post({ action: "frame_consumed", mapping: frame.mapping, slot: frame.slot, sequence: frame.sequence, accepted, error });
        }
    }
    // ------------------------------------------------------------ carrying
    /** A card carried by its title band: past a small threshold it follows the
     * pointer, the others already stand where letting go would leave them,
     * the slot is an ACCENT_TINT card with a dashed edge, and the board scrolls
     * under a pointer held near its top or bottom.  Letting go commits the
     * order (order_panels); Escape or a lost capture puts it back. */
    carried = null;
    orderRequest = null;
    order = [];
    carry(view, event) {
        if (!this.board || this.carried || this.size < 2 || offBoard(view.id))
            return;
        const grip = event.currentTarget;
        const start = { x: event.clientX, y: event.clientY };
        const threshold = token("--control") / 4;
        grip.setPointerCapture(event.pointerId);
        const move = (moved) => {
            if (!this.carried) {
                if (Math.hypot(moved.clientX - start.x, moved.clientY - start.y) < threshold)
                    return;
                this.lift(view, start);
            }
            this.follow(moved.clientX, moved.clientY);
        };
        const end = (commit) => {
            grip.removeEventListener("pointermove", move);
            grip.removeEventListener("pointerup", up);
            grip.removeEventListener("lostpointercapture", lost);
            window.removeEventListener("keydown", key, true);
            if (this.carried)
                this.drop(commit);
        };
        const up = () => end(true);
        const lost = () => end(false);
        const key = (pressed) => { if (pressed.key === "Escape") {
            pressed.preventDefault();
            pressed.stopPropagation();
            end(false);
        } };
        grip.addEventListener("pointermove", move);
        grip.addEventListener("pointerup", up);
        grip.addEventListener("lostpointercapture", lost);
        window.addEventListener("keydown", key, true);
    }
    lift(view, start) {
        const card = view.root, rect = card.getBoundingClientRect();
        const slot = element("div", "drop-slot");
        slot.style.width = `${rect.width}px`;
        slot.style.height = `${rect.height}px`;
        const before = [...this.views.keys()].filter(id => !offBoard(id)).sort((a, b) => this.index(a) - this.index(b));
        card.before(slot);
        card.classList.add("carried");
        document.body.classList.add("carrying");
        Object.assign(card.style, { left: `${rect.left}px`, top: `${rect.top}px`, width: `${rect.width}px` });
        this.carried = { view, slot, before, dx: start.x - rect.left, dy: start.y - rect.top, scroll: 0, pointer: start };
    }
    index(id) { return [...this.root.children].indexOf(this.views.get(id).root); }
    follow(x, y) {
        const carried = this.carried;
        carried.pointer = { x, y };
        const card = carried.view.root;
        card.style.left = `${x - carried.dx}px`;
        card.style.top = `${y - carried.dy}px`;
        // The slot goes before the card under the pointer's left half, after its right half.
        const others = [...this.root.children].filter((node) => node !== card && node !== carried.slot && node.classList.contains("panel"));
        const under = others.find(node => { const r = node.getBoundingClientRect(); return x >= r.left && x <= r.right && y >= r.top && y <= r.bottom; });
        if (under) {
            const r = under.getBoundingClientRect();
            const target = x < r.left + r.width / 2 ? under : under.nextElementSibling;
            if (target !== carried.slot && target !== carried.slot.nextElementSibling)
                this.root.insertBefore(carried.slot, target);
        }
        else if (others.length && y > Math.max(...others.map(node => node.getBoundingClientRect().bottom)))
            this.root.append(carried.slot);
        // Held near the board's top or bottom edge, the board scrolls under it.
        const view = this.root.getBoundingClientRect(), edge = token("--control") * 2;
        const speed = y < view.top + edge ? -1 : y > view.bottom - edge ? 1 : 0;
        window.clearInterval(carried.scroll);
        carried.scroll = speed ? window.setInterval(() => { this.root.scrollTop += speed * token("--row-gap") * 2; }, 20) : 0;
    }
    drop(commit) {
        const carried = this.carried;
        this.carried = null;
        window.clearInterval(carried.scroll);
        const card = carried.view.root;
        card.classList.remove("carried");
        document.body.classList.remove("carrying");
        card.style.left = card.style.top = card.style.width = "";
        if (commit)
            carried.slot.replaceWith(card);
        else
            carried.slot.remove();
        const order = [...this.root.children].map(node => node.dataset.panel).filter((id) => Boolean(id));
        if (commit && order.join() !== carried.before.join())
            this.orderRequest = this.send({ action: "order_panels", order }, "Reordering panels…");
        else if (!commit)
            for (const id of carried.before)
                this.root.append(this.views.get(id).root);
        carried.view.placeSettings();
    }
    /** The board leaves the screen with its view: no gesture stays captured. */
    hidden() { for (const view of this.views.values())
        view.cancelPointer(); }
    shown() { for (const view of this.views.values())
        view.placeSettings(); }
}
/** The Panel section holds the panel's own choices: its source and its kind. */
const PANEL_ROWS = new Set(["binding:signal", "semantic:kind", "semantic:cell_kind", "board:title", "board:overlay_signal", "board:size", "board:interval_ms"]);
/** v2's Panel section order: what it is, what it is called, what it shows, how often. */
const PANEL_ORDER = ["semantic:kind", "semantic:cell_kind", "board:title", "binding:signal", "board:overlay_signal", "board:size", "board:interval_ms"];
/** The board scheduler's closed refresh domain (v2 interval_form_field). */
const INTERVALS = [[100, "100"], [200, "200"], [400, "400"], [800, "800"]];
const SECTIONS = [["panel", "Panel"], ["semantic", "Semantic"], ["display", "Display"], ["fit", "Fit"]];
export class PanelView {
    board;
    id;
    root = element("article", "panel");
    canvas = element("canvas");
    context = this.canvas.getContext("2d", { alpha: false, colorSpace: "srgb" });
    name = element("span", "line");
    kind = element("span", "line");
    statusDot = element("span", "dot");
    settings = element("section", "settings");
    settingsStatus = element("p", "settings-status");
    form = element("div", "form");
    settingButton;
    fields = new Map();
    titles = new Map();
    model = element("select");
    modelFace = comboFace();
    expression = element("input");
    /** v2's "Two populations if ΔBIC >": a two-population model's evidence threshold. */
    bicGain = null;
    fitRows = [];
    fitError = element("p", "fit-error");
    modelChoices = "";
    fitDraft = false;
    fitVersion = 0;
    fitPending = null;
    settingsOpen = false;
    /** Shown in place on an Edit page rather than as a frame over the board. */
    inlined = false;
    closeButton;
    removeButton;
    cards = new Map();
    /** Moved by its title since it opened: placement keeps where it was put. */
    dragged = false;
    state = null;
    headerKey = "";
    caption = element("span", "row-label drag-handle");
    pointer;
    drawnSequence = -1;
    constructor(board, id) {
        this.board = board;
        this.id = id;
        this.root.dataset.panel = id;
        const heading = element("header", "panel-header");
        const title = element("div", "panel-name");
        this.name.textContent = id;
        title.append(this.name, this.kind);
        this.statusDot.hidden = true;
        this.settingButton = button("Setting", () => this.setSettingsOpen(!this.settingsOpen));
        this.settingButton.setAttribute("aria-expanded", "false");
        this.settingButton.setAttribute("aria-controls", `settings-${id}`);
        // Removing a plot panel takes two clicks (the header × and Setting's Remove).
        const close = armable(button("×", () => { }, "destructive glyph square close"), "remove this panel", () => this.board.send({ action: "close_panel", panel: id }, "Closing panel…"));
        close.setAttribute("aria-label", `Close ${id}`);
        const commands = element("div", "panel-commands");
        this.closeButton = close;
        commands.append(this.settingButton, close);
        heading.append(title, this.statusDot, commands);
        // The title band is the card's grip; its commands are clicked, not carried.
        heading.addEventListener("pointerdown", event => {
            if (event.button !== 0 || event.target.closest(".panel-commands"))
                return;
            this.board.carry(this, event);
        });
        const body = element("div", "panel-body");
        const surface = element("div", "surface");
        this.canvas.hidden = true;
        this.canvas.tabIndex = 0;
        this.canvas.setAttribute("aria-label", `${id} scientific plot`);
        // v2: no body placeholder -- an empty card is the blank frame its picture
        // will fill, and its strip's second line says why it holds none.
        surface.append(this.canvas);
        body.append(surface);
        this.buildSettings();
        this.root.append(heading, body, this.settings);
        this.pointer = new CanvasPointer(this.canvas, id, command => this.board.send(command), () => this.state?.axes || [], () => this.drawnSequence, () => this.board.interactive && !this.board.quiet.has(this.id));
    }
    buildSettings() {
        // FluentOverlayFrame: a frame inside the board, beside Setting, with its
        // title row (Setting · panel, ×) above the scrolling sections.
        this.settings.id = `settings-${this.id}`;
        this.settings.hidden = true;
        const header = element("div", "settings-header");
        const caption = this.caption;
        caption.textContent = `Setting · ${this.id}`;
        // The title carries the frame (the v2 drag handle), kept inside the board.
        caption.addEventListener("pointerdown", event => {
            if (event.button !== 0)
                return;
            event.preventDefault();
            caption.setPointerCapture(event.pointerId);
            // Where it stands as placed (offsetLeft/Top would round it to a whole CSS px).
            const start = { x: event.clientX, y: event.clientY, left: parseFloat(this.settings.style.left) || this.settings.offsetLeft, top: parseFloat(this.settings.style.top) || this.settings.offsetTop };
            const move = (moved) => {
                // In the card's padding box, which starts inside the card's edge.
                const card = this.root.getBoundingClientRect(), board = this.board.root.getBoundingClientRect();
                const edge = parseFloat(getComputedStyle(this.root).borderLeftWidth) || 0, inner = { left: card.left + edge, top: card.top + edge };
                const left = Math.max(board.left - inner.left, Math.min(start.left + moved.clientX - start.x, board.right - inner.left - this.settings.getBoundingClientRect().width));
                const top = Math.max(board.top - inner.top, start.top + moved.clientY - start.y);
                this.settings.style.left = `${left}px`;
                this.settings.style.top = `${top}px`;
                this.dragged = true;
            };
            const end = () => { caption.removeEventListener("pointermove", move); caption.removeEventListener("pointerup", end); caption.removeEventListener("lostpointercapture", end); };
            caption.addEventListener("pointermove", move);
            caption.addEventListener("pointerup", end);
            caption.addEventListener("lostpointercapture", end);
        });
        const close = button("×", () => this.setSettingsOpen(false), "quiet glyph square");
        close.title = "Close settings";
        header.append(caption, close);
        const body = element("div", "settings-body");
        this.settingsStatus.hidden = true;
        for (const [key, text] of SECTIONS) {
            const heading = element("div", "section-title");
            heading.append(element("span", "", text));
            heading.hidden = true;
            this.titles.set(key, heading);
        }
        const actions = element("div", "form-actions");
        const remove = armable(button("Remove", () => { }, "destructive"), "remove this panel", () => {
            this.setSettingsOpen(false);
            this.board.send({ action: "close_panel", panel: this.id }, "Closing panel…");
        });
        this.removeButton = remove;
        // v2's Setting row: Edit holds the panel still on its current data, in a page of its own
        // (where its Save figure group saves it); a double click returns the view (v2's gestures).
        const edit = button("Edit", () => { this.setSettingsOpen(false); this.board.onEdit?.(this.id); });
        edit.hidden = !this.board.onEdit || isEdit(this.id);
        actions.append(edit, element("span", "spacer"), remove);
        this.form.append(...SECTIONS.map(([key]) => this.titles.get(key)), actions);
        body.append(this.settingsStatus, this.form);
        this.settings.append(header, body);
        this.settings.addEventListener("keydown", event => {
            if (event.key === "Escape" && !this.inlined && !(event.target instanceof HTMLInputElement)) {
                event.stopPropagation();
                this.setSettingsOpen(false);
            }
        });
    }
    /** An Edit page shows the settings in place, always open (v2 PanelEditorView's
     * forms); the page, not the card, closes it. */
    inline(host) {
        this.inlined = true;
        this.settings.classList.add("inline");
        host.append(this.settings);
        // v2 PanelEditorView: each section is a card of its own (_section_card) under the page's one
        // label column; the card's own commands are not on the page.
        for (const [key] of SECTIONS) {
            const card = element("div", "card bordered panel-edit-card");
            const rows = element("div", "form");
            const title = this.titles.get(key);
            rows.append(title);
            card.append(rows);
            card.hidden = title.hidden;
            this.cards.set(key, card);
            host.append(card);
        }
        this.form.hidden = true;
        this.settingButton.hidden = this.closeButton.hidden = this.removeButton.hidden = true;
        this.setSettingsOpen(true);
    }
    /** A section shows while it has rows; on an Edit page its card goes with it. */
    showSection(key, shown) {
        this.titles.get(key).hidden = !shown;
        const card = this.cards.get(key);
        if (card)
            card.hidden = !shown;
    }
    setSettingsOpen(open) {
        if (this.inlined && !open)
            return;
        this.settingsOpen = open;
        this.settings.hidden = !open;
        this.settingButton.classList.toggle("active", open);
        this.settingButton.setAttribute("aria-expanded", String(open));
        if (open && this.state)
            this.updateSettings(this.state);
        if (open) {
            this.dragged = false;
            this.placeSettings();
        }
    }
    /** v2 show_fluent_popup_for_anchor, once per open (and again when the window,
     * the tab or the card moves it): beside its button -- the frame's right edge
     * is Setting's -- kept inside the board, as wide as its rows need, as tall as
     * they need but at least the popup minimum and at most what the board leaves
     * below it.  While it stands open only `widenSettings` changes it. */
    placeSettings() {
        if (!this.settingsOpen || this.inlined)
            return;
        const frame = this.settings, card = this.root.getBoundingClientRect(), anchor = this.settingButton.getBoundingClientRect();
        const board = this.board.root.getBoundingClientRect(), style = getComputedStyle(this.board.root);
        // The frame is placed in the card's padding box, which starts inside the card's edge.
        const edge = parseFloat(getComputedStyle(this.root).borderLeftWidth) || 0, inner = card.left + edge;
        const inset = parseFloat(style.paddingLeft) || 0;
        const gap = token("--popup-gap");
        const body = frame.querySelector(".settings-body"), scrolled = body.scrollTop;
        const width = Math.min(board.width - 2 * inset, this.settingsWidth());
        frame.style.width = `${width}px`;
        // The band's own height, not offsetHeight's whole CSS px (126 device px read as 125).
        const header = this.root.querySelector(".panel-header"), band = header.getBoundingClientRect().height;
        let top = frame.offsetTop;
        if (!this.dragged) {
            const left = Math.max(board.left + inset - inner, Math.min(anchor.right - inner - width, board.right - inset - inner - width));
            frame.style.left = `${left}px`;
            top = band + gap;
            frame.style.top = `${top}px`;
        }
        // v2's page is the board's viewport: the frame may reach its foot (the padding is the cards' margin, not a
        // wall), in whole device pixels so it never makes the board scroll.
        const foot = board.top + this.board.root.clientHeight - (card.top + edge + top);
        const room = Math.max(band * 4, Math.floor(foot * devicePixelRatio) / devicePixelRatio);
        frame.style.maxHeight = `${room}px`;
        frame.style.height = "";
        frame.style.height = `${Math.max(Math.min(token("--settings-min-h"), room), frame.getBoundingClientRect().height)}px`;
        body.scrollTop = scrolled;
    }
    /** v2 _sync_settings_body_size while the frame stands open: it neither moves
     * nor changes height (what comes or goes scrolls); rows that need more width
     * than it has widen its right edge, up to the board's, and it never narrows. */
    widenSettings() {
        const frame = this.settings, now = frame.getBoundingClientRect();
        const board = this.board.root.getBoundingClientRect(), inset = parseFloat(getComputedStyle(this.board.root).paddingLeft) || 0;
        const width = Math.min(this.settingsWidth(), board.right - inset - now.left);
        if (width > now.width)
            frame.style.width = `${width}px`;
    }
    /** What the frame's rows need (v2: the forms' widest row; a combo is its
     * widest choice, a message wraps): its max-content width in whole device
     * pixels up -- never a fraction short, which would cut a label. */
    settingsWidth() {
        const frame = this.settings, body = frame.querySelector(".settings-body"), scrolled = body.scrollTop, set = frame.style.width;
        frame.style.width = "max-content";
        const width = frame.getBoundingClientRect().width;
        frame.style.width = set;
        body.scrollTop = scrolled;
        return Math.ceil(width * devicePixelRatio - 1e-3) / devicePixelRatio;
    }
    update(panel, request, failed = false) {
        if (this.drawnSequence < 0 && !panel.frame && panel.display_size) {
            const scale = Number(document.documentElement.dataset.frameScale) || devicePixelRatio;
            this.fit({ width: Math.round(panel.display_size.width * scale), height: Math.round(panel.display_size.height * scale) });
            this.canvas.style.visibility = "hidden";
        }
        this.state = { ...panel, controls: panel.controls ?? this.state?.controls, fit: panel.fit ?? this.state?.fit,
            axes: panel.axes ?? this.state?.axes, structure: panel.structure ?? this.state?.structure,
            source_signal: panel.source_signal ?? this.state?.source_signal };
        this.updateHeader(this.state);
        // The card body reserves no status row: the dot rides the title strip and
        // the message travels as its tooltip, and in full in the Setting frame.
        const message = panel.error || panel.message || "";
        this.statusDot.hidden = !message;
        this.statusDot.title = message;
        setState(this.statusDot, panel.error ? "error" : "task");
        this.settingsStatus.textContent = message;
        this.settingsStatus.hidden = !message;
        this.settingsStatus.classList.toggle("error", Boolean(panel.error));
        if (this.settingsOpen || this.fields.size)
            this.updateSettings(this.state, request, failed);
    }
    refreshCatalog() {
        if (!this.state)
            return;
        this.updateHeader(this.state);
        if (this.settingsOpen || this.fields.size)
            this.updateSettings(this.state);
    }
    updateHeader(panel) {
        // What the card shows after its id (v2 _panel_caption): the operator's
        // name after a space, else "@" and the signal spelled with node names.
        const source = panel.source_signal || panel.binding?.signal || "";
        const entry = this.board.boardPanel(this.subject);
        // A saved Figure's dataset reads by the signal it was saved from (its label), spelled with node names.
        const label = this.board.signals.find(signal => signal.id === source)?.label;
        const saved = label && label !== source ? label.replace(/^@logic\//, "") : "";
        const signal = saved || (entry ? entry.name : label || source);
        // v2 opens an archive's dataset as a panel titled with its label: "figure-1 Site map".
        const title = entry?.title || (source.startsWith("@figure/") ? label : "");
        const words = title ? ` ${title}` : signal ? `@${signal}` : "";
        const shownId = panel.caption_id ?? this.id;
        const key = JSON.stringify([shownId, words, panel.structure, panel.kind, panel.focus]);
        if (key === this.headerKey)
            return;
        this.headerKey = key;
        this.caption.textContent = `Setting · ${shownId}`;
        // One line of numbers over one line of the axes they count, group i in
        // AXIS_GROUP_COLORS[i] on both lines; the panel's id in MUTED_INK. A line
        // too long for the card gives up text, never structure (v2 _band_fragments):
        // a typed name its tail, a signal's producer its middle (the output after
        // the slash stays), a group's contents their middle; ids, brackets and
        // separators stay whole.
        const name = [];
        const slash = words.lastIndexOf("/");
        if (!words)
            name.push({ text: shownId, elide: "right" });
        else {
            name.push({ text: shownId, className: "panel-id", elide: null });
            if (words.startsWith("@") && slash > 1)
                name.push({ text: words.slice(0, slash), elide: "middle" }, { text: words.slice(slash), elide: null });
            else
                name.push({ text: words, elide: "right" });
        }
        const kind = [];
        if (panel.structure) {
            name.push({ text: " ", elide: null });
            for (const [line, names] of [[name, false], [kind, true]]) {
                for (let domain = 0; domain < 3; domain++) {
                    const className = ["domain-repeat", "domain-point", "domain-cell"][domain];
                    if (domain)
                        line.push({ text: " × ", elide: null });
                    const items = panel.structure[domain] || [];
                    const text = items.map(item => String(item[names ? 0 : 1])).join(" × ") || (names ? "—" : "1");
                    line.push({ text: "(", className, elide: null }, { text, className, elide: "middle" }, { text: ")", className, elide: null });
                }
            }
            // v2 _vacancy_note: why the card holds no picture, as the operator's next step.
        }
        else
            kind.push({ text: `${this.board.plotKinds.find(item => item.id === panel.kind)?.label || panel.kind} · ${source ? "No data yet" : "Choose a signal in Setting"}`, className: "vacancy", elide: "right" });
        setFragments(this.name, name);
        setFragments(this.kind, kind);
        const whole = (line) => line.map(fragment => fragment.text).join("");
        this.name.title = `${whole(name)}\n${whole(kind)}\n${panel.kind}${panel.focus == null ? "" : ` · Cell ${panel.focus + 1}`}`;
        this.kind.title = this.name.title;
    }
    complete(request, failed = false) {
        for (const field of this.fields.values())
            field.complete(request, failed);
        if (this.fitPending && this.fitPending.id === request) {
            if (!failed && this.fitPending.version === this.fitVersion)
                this.fitDraft = false;
            this.fitPending = null;
        }
    }
    draw(image, frame) {
        const next = this.board.staged.get(this.id);
        if (frame.sequence < this.drawnSequence || (next?.panel.frame && frame.sequence < next.panel.frame.sequence))
            return false;
        if (frame.sequence > this.drawnSequence && next?.panel.frame?.sequence !== frame.sequence)
            throw new Error("A frame arrived without its matching accepted panel state.");
        this.fit(frame);
        this.context.putImageData(image, 0, 0);
        this.canvas.style.visibility = "";
        this.drawnSequence = frame.sequence;
        if (next?.panel.frame?.sequence === frame.sequence) {
            this.board.staged.delete(this.id);
            this.update(next.panel, next.request, next.failed);
        }
        return true;
    }
    /** A card that has drawn nothing yet: its state at once, in its first picture's
     * room (v2: an empty card is the blank frame its picture will fill), left
     * unpainted until that picture comes. */
    reserve(panel, request, failed = false) {
        this.fit(panel.frame);
        this.canvas.style.visibility = "hidden";
        this.update(panel, request, failed);
    }
    /** Whether the card holds its picture's room (drawn, or reserved for the picture on its way). */
    get placed() { return !this.canvas.hidden; }
    fit(frame) {
        if (this.canvas.width !== frame.width)
            this.canvas.width = frame.width;
        if (this.canvas.height !== frame.height)
            this.canvas.height = frame.height;
        // A shared view shows the operator's pixels at the operator's size.
        this.canvas.style.width = `${frame.width / (Number(document.documentElement.dataset.frameScale) || devicePixelRatio)}px`;
        this.canvas.hidden = false;
    }
    cancelPointer() { this.pointer.cancel(); }
    /** The board panel this card is of: itself, or the panel an Edit holds still. */
    get subject() { return isEdit(this.id) ? this.id.slice("edit-".length) : this.id; }
    sectionOf(spec) {
        // v2: the grid's Cell fit value is display state shown beside the fit it annotates.
        if (spec.section === "display" && spec.name === "facet_fit_parameter")
            return "fit";
        // v2: the scientific Window is the last of the display rows.
        if (spec.section === "binding" && spec.name === "window")
            return "display";
        return PANEL_ROWS.has(`${spec.section}:${spec.name}`) || spec.section === "binding" || spec.section === "board" ? "panel" : spec.section;
    }
    /** Put `rows` right after `title`, in order, moving only what is out of place. */
    place(title, rows) {
        const parent = title.parentElement ?? this.form;
        let cursor = title.nextElementSibling;
        for (const row of rows) {
            if (row !== cursor)
                parent.insertBefore(row, cursor);
            cursor = row.nextElementSibling;
        }
    }
    updateSettings(panel, request, failed = false) {
        // v2 _form_spec: no Plot kind row (a panel's kind is chosen when it is added), and a
        // Window only where v2's kinds declare one (curve, histogram, rolling).
        const windowed = ["curve", "histogram", "rolling"].includes(panel.kind);
        const window = panel.binding && windowed
            ? { section: "binding", name: "window", label: "Window", kind: "integer", value: panel.binding.window, minimum: 1, required: true } : null;
        // The binding's Window stands where the plot's spec holds it (a `bound` display row), else last.
        const held = (panel.controls ?? []).some(control => control.bound);
        const controls = [
            ...(panel.binding ? [
                { section: "binding", name: "signal", label: "Signal", kind: "choice", value: panel.binding.signal,
                    // The live board's picker is v2's tree (its producers' signals); a viewer's lists its data.
                    ...(this.board.boardPanel(this.subject)?.signal_tree ? { tree: this.board.boardPanel(this.subject).signal_tree }
                        : { choices: this.board.signalChoices(String(this.fields.get("binding:signal")?.value || panel.binding.signal)) }) },
            ] : []),
            // v2 PanelEditorView's Panel card has no Cell kind (the Setting frame's has).
            ...(panel.controls || []).filter(control => control.section !== "semantic" || (control.name !== "kind" && !(this.inlined && control.name === "cell_kind"))).flatMap(control => {
                if (control.bound)
                    return window ? [window] : [];
                if (control.section !== "semantic" || control.name !== "cell_kind")
                    return [control];
                // v2 Cell kind: "automatic (from data)" first, then the kinds by their own names.
                const kinds = (control.choices ?? []).filter(([value]) => value !== "").map(([value]) => [value, String(value)]);
                return [{ ...control, choices: [["", "automatic (from data)"], ...kinds] }];
            }),
            ...(window && !held ? [window] : []),
        ];
        // The board's own rows of this panel (workbench board.panels): its name and refresh.
        const entry = this.board.boardPanel(this.subject);
        // v2 size_form_field: the panel's size preset, a choice of the plotting owner's sizes.
        if (panel.size_choices?.length)
            controls.push({ section: "board", name: "size", label: "Panel size", kind: "choice", value: panel.size ?? null,
                choices: panel.size_choices.map(size => [size, size]) });
        // v2's Overlay (paints_images): a panel whose surfaces are images -- an Image,
        // or a grid of image cells -- is drawn with the site rings of a judgement of
        // its measurement; its display rows then hold the rings' Point labels.
        if (entry && (panel.controls ?? []).some(control => control.name === "show_point_labels")) {
            // No overlay is the box's none row, "Off" in MUTED_INK, as v2's.
            controls.push({ section: "board", name: "overlay_signal", label: "Overlay", kind: "choice", value: entry.overlay_signal || null,
                allow_none: true, placeholder: "Off", tree: entry.overlay_tree });
        }
        if (entry)
            controls.push(
            // v2's Panel name field holds the name the card shows (its title, else its default name).
            { section: "board", name: "title", label: "Panel name", kind: "text", value: entry.title || entry.name, allow_none: true, placeholder: entry.name }, { section: "board", name: "interval_ms", label: "Update interval", kind: "choice", segmented: true, unit: "ms", value: entry.interval_ms, choices: INTERVALS });
        // While a task runs, a panel showing its protected signals keeps its
        // source and semantics; display parameters stay editable (frozen).
        const frozen = this.board.frozen(this.subject);
        if (frozen)
            for (const [index, spec] of controls.entries()) {
                if (spec.section === "binding" || spec.section === "semantic" || spec.name === "overlay_signal")
                    controls[index] = { ...spec, unavailable_reason: spec.unavailable_reason || frozen };
            }
        const rank = (spec) => { const at = PANEL_ORDER.indexOf(`${spec.section}:${spec.name}`); return at < 0 ? PANEL_ORDER.length : at; };
        const ordered = [...controls.filter(spec => this.sectionOf(spec) === "panel").sort((a, b) => rank(a) - rank(b)), ...controls.filter(spec => this.sectionOf(spec) !== "panel")];
        controls.splice(0, controls.length, ...ordered);
        const alive = new Set();
        for (const spec of controls) {
            const key = `${spec.section}:${spec.name}`;
            alive.add(key);
            let field = this.fields.get(key);
            if (field && field.kind !== fieldKind(spec)) {
                field.root.remove();
                this.fields.delete(key);
                field = undefined;
            }
            if (!field) {
                field = new Field(`${this.id}-${spec.section}`, spec, {
                    commit: value => this.configure(spec.section, spec, value),
                    scope: (steps, changed) => this.board.send({ action: "scope_step", panel: this.id, name: changed.name, steps }, `Updating ${changed.label}…`),
                }, !PANEL_ROWS.has(key) || key === "binding:signal");
                this.fields.set(key, field);
            }
            else
                field.update(spec, request, failed);
        }
        for (const [key, field] of this.fields)
            if (!alive.has(key)) {
                field.root.remove();
                this.fields.delete(key);
            }
        for (const [section] of SECTIONS) {
            if (section === "fit")
                continue;
            const rows = controls.filter(spec => this.sectionOf(spec) === section).map(spec => this.fields.get(`${spec.section}:${spec.name}`).root);
            this.showSection(section, rows.length > 0);
            this.place(this.titles.get(section), rows);
        }
        this.updateFit(panel, request, failed);
        if (this.settingsOpen && !this.inlined) {
            // v2 share_label_column: one label column, as wide as the longest label's whole words.
            shareLabelColumn(this.form);
            this.widenSettings();
        }
    }
    /** One edited setting as a configure command; a binding edit carries the
     * other binding field's newest draft so the two never race. */
    configure(section, spec, value) {
        if (section === "board" && spec.name === "size")
            return this.board.send({ action: "configure", panel: this.id, size: value }, "Resizing the panel…");
        if (section === "board" && spec.name === "overlay_signal")
            return this.board.send({ action: "set_panel_overlay", panel: this.subject, signal: value || null }, "Changing the overlay…");
        if (section === "board") {
            return spec.name === "title"
                ? this.board.send({ action: "rename_panel", panel: this.subject, title: value === "" ? null : value }, "Renaming the panel…")
                : this.board.send({ action: "set_panel_interval", panel: this.subject, interval_ms: value }, "Updating the interval…");
        }
        const binding = this.state?.binding;
        if (section === "binding" && binding) {
            const draft = {
                signal: this.fields.get("binding:signal")?.value ?? binding.signal,
                window: this.fields.get("binding:window")?.value ?? binding.window,
                [spec.name]: value,
            };
            return this.board.send({ action: "configure", panel: this.subject, binding: draft }, `Updating ${spec.label}…`);
        }
        const group = section === "display" ? "parameters" : section;
        return this.board.send({ action: "configure", panel: this.id, [group]: { [spec.name]: value } }, `Updating ${spec.label}…`);
    }
    updateFit(panel, request, failed = false) {
        const fitState = panel.fit;
        const title = this.titles.get("fit");
        this.showSection("fit", Boolean(fitState));
        if (!fitState) {
            for (const row of this.fitRows)
                row.remove();
            this.fitRows = [];
            return;
        }
        if (!this.fitRows.length) {
            const row = (label, control) => {
                const node = element("div", "field");
                const cell = element("div", "field-control");
                label.classList.add("field-label");
                cell.append(control);
                node.append(label, cell);
                return node;
            };
            const modelLabel = element("label", "", "Fit model");
            this.model.id = `fit-model-${this.id}`;
            modelLabel.htmlFor = this.model.id;
            const expressionLabel = element("label", "", "Parameters");
            this.expression.id = `fit-expression-${this.id}`;
            this.expression.type = "text";
            this.expression.spellcheck = false;
            expressionLabel.htmlFor = this.expression.id;
            this.expression.addEventListener("input", () => { this.fitDraft = true; this.fitVersion++; });
            this.expression.addEventListener("change", () => this.configureFit());
            this.expression.addEventListener("keydown", event => {
                if (event.key === "Enter") {
                    this.configureFit();
                    this.expression.blur();
                }
                if (event.key === "Escape") {
                    this.fitDraft = false;
                    this.expression.value = this.state?.fit?.expression || "";
                    this.expression.blur();
                    event.stopPropagation();
                }
            });
            this.model.addEventListener("change", () => { this.fitDraft = true; this.fitVersion++; this.configureFit(); });
            this.fitRows = [row(modelLabel, this.model), row(expressionLabel, this.expression), this.fitError];
            this.fitError.hidden = true;
        }
        const signature = JSON.stringify(fitState.models);
        if (signature !== this.modelChoices) {
            this.modelChoices = signature;
            const selected = this.model.value;
            this.model.replaceChildren(this.modelFace, new Option("(none)", ""), ...fitState.models.map(model => new Option(model.label, model.key)));
            hintWidest(this.model, ["(none)", ...fitState.models.map(model => model.label)]);
            if (this.fitDraft || this.fitPending)
                this.model.value = selected;
        }
        this.complete(request, failed);
        if (!this.fitDraft && !this.fitPending) {
            this.model.value = fitState.selected || "";
            if (this.expression.value !== fitState.expression)
                this.expression.value = fitState.expression;
        }
        // v2 panel_state's fit fields: Fit model while there are models; Parameters (described by
        // the model's own symbols), a two-population model's ΔBIC and the grid's Cell fit value
        // once one is chosen.
        const [modelRow, expressionRow] = this.fitRows;
        const chosen = this.model.value !== "";
        const symbols = (fitState.models.find(model => model.key === this.model.value)?.parameters ?? []).map(([, symbol]) => symbol);
        const syntax = symbols.length ? `${symbols[0]}=1 fixes it, ${symbols[0]}=guess(1) seeds it` : "symbol=value fixes it, symbol=guess(value) seeds it";
        const described = symbols.length ? `${symbols.join(", ")}  --  ${syntax}` : syntax;
        // v2 qt_form: a text box's placeholder is its description cut at 48; the whole line is its tooltip.
        if (this.expression.placeholder !== described.slice(0, 48)) {
            this.expression.placeholder = described.slice(0, 48);
            this.expression.title = described;
        }
        const bic = this.bicRow(fitState.configuration);
        const facet = this.fields.get("display:facet_fit_parameter")?.root;
        const rows = [
            ...(fitState.models.length || chosen ? [modelRow] : []),
            ...(chosen ? [expressionRow, ...(bic ? [bic] : []), ...(facet ? [facet] : [])] : []),
            this.fitError,
        ];
        for (const row of [modelRow, expressionRow, this.bicGain?.root, facet])
            if (row && !rows.includes(row))
                row.remove();
        this.showSection("fit", rows.length > 1);
        this.place(title, rows);
        // v2 fit_expression_refusal: what the operator typed that the model would not take.
        const refused = fitState.error ? `Parameter expression ignored; automatic fit is active: ${fitState.error}` : "";
        this.fitError.textContent = refused;
        this.fitError.hidden = !refused;
    }
    /** v2's "Two populations if ΔBIC >": shown while the accepted fit asks the two-population question. */
    bicRow(configuration) {
        if (!("min_bic_gain" in configuration))
            return null;
        // A plot number keeps its own spelling in a plain box (v2 _NumberHandler: 10.0, never a spin box).
        const gain = configuration.min_bic_gain;
        const spec = { section: "display", name: "min_bic_gain", label: "Two populations if ΔBIC >", kind: "number", value: gain, allow_none: false, step: 1,
            lossless: true, text: typeof gain === "number" && Number.isInteger(gain) ? gain.toFixed(1) : String(gain),
            description: "the evidence two populations must exceed over one; short of it the fit reports one population" };
        if (!this.bicGain)
            this.bicGain = new Field(`${this.id}-fit`, spec, {
                commit: value => this.board.send({ action: "configure", panel: this.id, fit: { ...this.state?.fit?.configuration, min_bic_gain: value } }, "Updating fit…"),
            });
        else
            this.bicGain.update(spec);
        return this.bicGain.root;
    }
    configureFit() {
        if (!this.fitDraft || this.fitPending?.version === this.fitVersion)
            return;
        const fit = { ...this.state?.fit?.configuration, model: this.model.value || null, expression: this.expression.value };
        const id = this.board.send({ action: "configure", panel: this.id, fit }, "Updating fit…");
        this.fitPending = { id, version: this.fitVersion };
    }
}
