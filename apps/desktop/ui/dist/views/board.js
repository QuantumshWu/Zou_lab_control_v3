// ConsoleBoardView + PanelCardView: the plot panels of a document, each a
// card with its finished picture, its title strip and its Setting frame.
// One board per plot window: the Task Console's live board, the Figure
// Viewer's opened Figure (each window's pixel mappings are its own). The
// board only copies finished pixels and forwards pointer and setting
// intents; every scientific decision is the host's.
import { Field, armable, button, comboFace, element, fieldKind, hintWidest, setFragments, setState, shareLabelColumn, token } from "../fluent/index.js";
import { BoardMetrics, GeomProxy, drop, firstFreeSlot, minBoardWidth, pack, slotUnder } from "./board_layout.js";
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
/** v2 ConsoleBoardView: one rule set places every card.  Every card has a place, a card let go
 * onto another changes places with it, a card let go on empty space goes there, cards that
 * overlap make way, and everything falls north-west (board_layout).  A drop records where it
 * left EVERY card for the board's column count (how many of the narrowest card fit side by side),
 * and nothing else writes those places; a card no drop has placed takes the first free slot, in
 * board order, at the width the board has now.  A column count no drop was made at shows the
 * nearest wider count's arrangement packed narrower (none wider: the nearest narrower), so
 * narrowing packs without destroying the arrangement and widening restores it.  The places live
 * as long as the window; the workbench keeps the order (order_panels). */
export class PanelBoard {
    transport;
    /** #panels: the cards, absolutely placed in whole device px a card pad apart (the padding's left
     * and top are BoardMetrics' margins), over the extent that makes the board scroll. */
    root = element("main");
    /** The board's scroll extent (v2 setMinimumSize): as tall as the lowest card, the padding after it. */
    extent = element("div", "board-extent");
    /** v2 _arrangements: where the last drop at each column count left every card, by panel id.
     * Written by a drop alone: packing reads it and never writes it back (that would ratchet). */
    arrangements = new Map();
    /** Where each card was last put (device px at a pixel ratio): only a moved card is written. */
    drawn = new Map();
    /** A card's or the board's size changed: settle again before the frame paints. */
    observer = new ResizeObserver(() => this.resized());
    frameRequest = 0;
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
        this.root.append(this.extent);
        this.observer.observe(this.root);
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
        const answered = state.request !== undefined && state.request === this.orderRequest;
        if (answered)
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
            const cards = this.boardIds();
            // A card that leaves takes its gesture with it (v2 set_cards ends the drag).
            if (this.gesture && !alive.has(this.gesture.view.id))
                this.gesture.end(false);
            for (const [id, view] of this.views)
                if (!alive.has(id)) {
                    view.cancelPointer();
                    this.observer.unobserve(view.root);
                    view.root.remove();
                    this.views.delete(id);
                    this.staged.delete(id);
                    this.forget(id);
                }
            for (const panel of state.panels) {
                let view = this.views.get(panel.id);
                if (!view) {
                    view = new PanelView(this, panel.id);
                    this.views.set(panel.id, view);
                    // A board card's size is its own (its picture's): the board settles again when it changes.
                    if (!offBoard(panel.id))
                        this.observer.observe(view.root, { box: "border-box" });
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
            this.reorder(state.panels.map(panel => panel.id));
            this.place();
            // Cards came or went, or the order changed: v2 set_cards settles the board (a carried card's frame again).
            if (this.boardIds().join() !== cards.join()) {
                if (this.carried)
                    this.reframe();
                else
                    this.schedule();
            }
        }
        else if (answered)
            this.place();
        if (catalogChanged)
            for (const view of this.views.values())
                view.refreshCatalog();
        this.root.classList.toggle("can-carry", Boolean(this.board) && this.size > 1);
        this.changed();
    }
    /** The order the workbench holds (state.panels, v2's order_committed) is the persisted fact: one
     * that changes from elsewhere (a layout loaded, MCP) lays the board out afresh, every card in a
     * first free slot in that order -- unless it is a drop's own order coming back.  While a drop's
     * order_panels answer is due, a state's order is older than it: only its arrivals and departures count. */
    reorder(incoming) {
        const known = new Set(this.order), arriving = new Set(incoming);
        if (this.orderRequest !== null) {
            this.order = [...this.order.filter(id => arriving.has(id)), ...incoming.filter(id => !known.has(id))];
            return;
        }
        const kept = (ids, other) => ids.filter(id => other.has(id) && !offBoard(id)).join();
        if (kept(this.order, arriving) !== kept(incoming, known))
            this.arrangements.clear();
        this.order = incoming;
    }
    /** A card that left: no place, no arrangement keeps it (v2 set_cards). */
    forget(id) {
        this.drawn.delete(id);
        for (const [columns, places] of this.arrangements) {
            places.delete(id);
            if (!places.size)
                this.arrangements.delete(columns);
        }
    }
    /** The board's cards in board order (an Edit, a review and an SLM Editor picture stand elsewhere). */
    boardIds() { return this.order.filter(id => !offBoard(id) && this.views.has(id)); }
    /** The cards' nodes in board order -- reading order, for the keyboard -- moving only the ones out
     * of place, never while one is carried or a drop's order is still due. */
    place() {
        let cursor = this.orderRequest !== null || this.carried ? null : this.extent.nextElementSibling;
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
    // ------------------------------------------------------------- placing
    /** v2 _pack_current, now: every card where the board's places settle at its width, the
     * board's extent under the lowest.  Null while the board is not laid out (a page not shown)
     * or a card is carried (its frame stands still until it is let go). */
    layout() {
        if (this.frameRequest) {
            cancelAnimationFrame(this.frameRequest);
            this.frameRequest = 0;
        }
        if (this.carried)
            return null;
        const frame = this.settle();
        if (!frame)
            return null;
        this.reach(this.put(frame.board), frame.metrics);
        return frame.board;
    }
    /** Settle once, in the next frame, however many states asked. */
    schedule() {
        if (!this.frameRequest)
            this.frameRequest = requestAnimationFrame(() => { this.frameRequest = 0; this.layout(); });
    }
    /** After layout, before paint: a card's picture or the board's width changed. */
    resized() {
        const carried = this.carried;
        if (!carried) {
            this.layout();
            return;
        }
        // v2 resizeEvent: a carried card's frame was settled at the old width.
        if (Math.round(this.extent.getBoundingClientRect().width * devicePixelRatio) !== carried.frame.room)
            this.reframe();
    }
    /** v2 _settle at the board's width, every card measured first (reads only): a card the
     * arrangement places starts there and falls; every other card -- one no drop has placed --
     * then takes the first free slot, in board order.  Whole device px; the margins (the gap) are
     * the board's padding, a card's right edge may reach its content edge. */
    settle() {
        const ids = this.boardIds(), dpr = devicePixelRatio;
        const metrics = new BoardMetrics(Math.round((parseFloat(getComputedStyle(this.root).paddingLeft) || 0) * dpr));
        const room = Math.round(this.extent.getBoundingClientRect().width * dpr);
        const sizes = ids.map(id => this.views.get(id).root.getBoundingClientRect()).map(rect => [Math.round(rect.width * dpr), Math.round(rect.height * dpr)]);
        if (ids.length && (room <= 0 || sizes.some(([width, height]) => width <= 0 || height <= 0)))
            return null;
        const board = new Map(ids.map((id, index) => [id, new GeomProxy(...sizes[index])]));
        const boardW = Math.max(room + 2 * metrics.gap, minBoardWidth(board.values(), metrics));
        const places = this.places([...board.values()], boardW, metrics);
        const placed = [];
        for (const [id, proxy] of board) {
            const at = places?.get(id);
            if (at) {
                [proxy.col, proxy.row] = at;
                placed.push(proxy);
            }
        }
        pack(placed, metrics, boardW);
        for (const [id, proxy] of board)
            if (!places?.has(id)) {
                [proxy.col, proxy.row] = firstFreeSlot(proxy, placed, boardW, metrics);
                placed.push(proxy);
            }
        return { board, ids, boardW, room, metrics };
    }
    /** v2 _columns: how many of the narrowest card fit side by side on a board this wide. */
    columns(cards, boardW, metrics) {
        return Math.max(1, Math.floor((boardW - metrics.gap) / (Math.min(...cards.map(card => card.width)) + metrics.gap)));
    }
    /** v2 _places: the arrangement the last drop at this column count left, else the nearest wider
     * one's (a narrower board packs it), else the nearest narrower one's. */
    places(cards, boardW, metrics) {
        if (!this.arrangements.size || !cards.length)
            return undefined;
        const columns = this.columns(cards, boardW, metrics), counts = [...this.arrangements.keys()], wider = counts.filter(count => count > columns);
        return this.arrangements.get(columns) ?? this.arrangements.get(wider.length ? Math.min(...wider) : Math.max(...counts));
    }
    /** v2 _place: every card but the carried one to its place (writes only, a moved card's alone);
     * the lowest bottom edge among them all. */
    put(board, carried) {
        const dpr = devicePixelRatio, moved = [];
        let bottom = 0;
        for (const [id, proxy] of board) {
            bottom = Math.max(bottom, proxy.row + proxy.height);
            const view = this.views.get(id), at = `${proxy.col},${proxy.row},${dpr}`;
            if (id === carried || this.drawn.get(id) === at)
                continue;
            view.root.style.left = `${proxy.col / dpr}px`;
            view.root.style.top = `${proxy.row / dpr}px`;
            this.drawn.set(id, at);
            moved.push(view);
        }
        // An open Setting frame rides its card, kept inside the board where the card now stands.
        for (const view of moved)
            view.placeSettings();
        return bottom;
    }
    /** The board scrolls to `bottom` and its own bottom padding (v2 setMinimumSize(..., bottom + gap)). */
    reach(bottom, metrics) {
        this.extent.style.height = `${Math.max(0, bottom - metrics.gap) / devicePixelRatio}px`;
    }
    // ------------------------------------------------------------ carrying
    /** v2 ConsoleBoardView's drag.  A card carried by its title band follows the pointer past a
     * small threshold.  The slot is chosen in the board as it stood at pick-up (the frame), which
     * stands still, so the choice cannot chase its own consequence: under the pointer's x on the
     * line of the card's strip, nearest where the card is drawn (slotUnder).  While it is carried
     * the other cards already stand where letting go will leave them -- preview and drop are one
     * computation (drop) -- with the slot under it an ACCENT_TINT card with a dashed edge, and held
     * near the board's top or bottom edge the board scrolls under it.  Let go, the places are
     * recorded for this column count and the cards' reading order committed (order_panels); Escape
     * or a lost capture puts every card back. */
    carried = null;
    /** A press on a title band until it is let go, and how to end it from outside (its card left, the board hid). */
    gesture = null;
    orderRequest = null;
    order = [];
    carry(view, event) {
        if (!this.board || this.gesture || this.size < 2 || offBoard(view.id))
            return;
        const grip = event.currentTarget;
        const start = { x: event.clientX, y: event.clientY };
        const threshold = token("--control") / 4;
        grip.setPointerCapture(event.pointerId);
        const move = (moved) => {
            if (!this.carried && (Math.hypot(moved.clientX - start.x, moved.clientY - start.y) < threshold || !this.lift(view, start)))
                return;
            this.follow(moved.clientX, moved.clientY);
        };
        const end = (commit) => {
            grip.removeEventListener("pointermove", move);
            grip.removeEventListener("pointerup", up);
            grip.removeEventListener("lostpointercapture", lost);
            window.removeEventListener("keydown", key, true);
            this.gesture = null;
            if (this.carried)
                this.letGo(commit);
        };
        // Let go where the pointer is released: the slot under it.
        const up = (released) => { if (this.carried)
            this.carried.pointer = { x: released.clientX, y: released.clientY }; end(true); };
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
        this.gesture = { view, end };
    }
    /** v2 _card_drag_started: the frame is the board as it stands; the card leaves it to follow the
     * pointer (fixed, so it never stretches the board it is carried over), held where it was pressed. */
    lift(view, start) {
        const frame = this.settle();
        if (!frame)
            return false;
        const card = view.root, rect = card.getBoundingClientRect(), strip = card.querySelector(".panel-header").getBoundingClientRect(), dpr = devicePixelRatio;
        const plate = element("div", "drop-slot");
        this.carried = { view, frame, slot: null, plate, hold: { x: start.x - rect.left, y: start.y - rect.top },
            strip: Math.round((strip.top + strip.height / 2 - rect.top) * dpr), band: Math.max(1, Math.round(strip.height * dpr)), pointer: start, scroll: 0 };
        // Under every card: the cards stand after it.
        this.extent.after(plate);
        this.drawn.delete(view.id);
        card.classList.add("carried");
        document.body.classList.add("carrying");
        Object.assign(card.style, { left: `${rect.left}px`, top: `${rect.top}px`, width: `${rect.width}px` });
        return true;
    }
    follow(x, y) {
        const carried = this.carried, box = this.root.getBoundingClientRect();
        carried.pointer = { x, y };
        this.aim(box);
        Object.assign(carried.view.root.style, { left: `${x - carried.hold.x}px`, top: `${y - carried.hold.y}px` });
        this.steer(box);
    }
    /** v2 _show_drop: the board as letting go here would leave it, and the slot drawn where the card will rest. */
    aim(box = this.root.getBoundingClientRect()) {
        const carried = this.carried, { board, boardW, metrics } = carried.frame, dpr = devicePixelRatio;
        // Board px of a client point: from the padding box's corner, scrolled.
        const left = box.left + this.root.clientLeft - this.root.scrollLeft, top = box.top + this.root.clientTop - this.root.scrollTop;
        const seen = [Math.round((carried.pointer.x - carried.hold.x - left) * dpr), Math.round((carried.pointer.y - carried.hold.y - top) * dpr)];
        const others = [...board].filter(([id]) => id !== carried.view.id).map(([, proxy]) => proxy);
        const slot = slotUnder([Math.round((carried.pointer.x - left) * dpr), seen[1] + carried.strip], board.get(carried.view.id), others, metrics, boardW, seen);
        if (carried.slot && carried.slot[0] === slot[0] && carried.slot[1] === slot[1])
            return;
        carried.slot = slot;
        const preview = this.droppedAt(slot), rest = preview.get(carried.view.id);
        const bottom = this.put(preview, carried.view.id);
        Object.assign(carried.plate.style, { left: `${rest.col / dpr}px`, top: `${rest.row / dpr}px`, width: `${rest.width / dpr}px`, height: `${rest.height / dpr}px` });
        // Room for one more row while a card is carried: the slot after the last card must be reachable.
        const lowest = Math.max(0, ...others.map(proxy => proxy.row + proxy.height));
        this.reach(Math.max(bottom, lowest + metrics.gap + rest.height), metrics);
    }
    /** v2 _dropped_at: every card at rest once the carried one is let go at `slot`, on copies of the frame. */
    droppedAt(slot) {
        const carried = this.carried, { board, ids, boardW, metrics } = carried.frame;
        const copies = new Map([...board].map(([id, proxy]) => [id, new GeomProxy(proxy.width, proxy.height, proxy.col, proxy.row)]));
        drop(ids.map(id => copies.get(id)), copies.get(carried.view.id), slot, metrics, boardW);
        return copies;
    }
    /** v2 _frame_drag again: the board's width or its set of cards changed under a carried card. */
    reframe() {
        const carried = this.carried, frame = this.settle();
        if (!frame || !frame.board.has(carried.view.id)) {
            this.gesture?.end(false);
            return;
        }
        carried.frame = frame;
        carried.slot = null;
        this.aim();
    }
    /** v2 _card_dropped: the drop records where it left EVERY card at this column count, the board
     * settles (to those places) and the cards' reading order is committed.  Not committed (Escape,
     * a lost capture), every card goes back where it stood. */
    letGo(commit) {
        const carried = this.carried;
        if (commit)
            this.aim();
        const preview = commit && carried.slot ? this.droppedAt(carried.slot) : null;
        this.carried = null;
        window.clearInterval(carried.scroll);
        carried.plate.remove();
        carried.view.root.classList.remove("carried");
        carried.view.root.style.width = "";
        document.body.classList.remove("carrying");
        if (preview) {
            const { boardW, metrics } = carried.frame;
            this.arrangements.set(this.columns([...preview.values()], boardW, metrics), new Map([...preview].map(([id, proxy]) => [id, [proxy.col, proxy.row]])));
        }
        const packed = this.layout() ?? preview;
        if (preview && packed)
            this.commit(packed);
    }
    /** v2 order_committed: the board's cards in the reading order of where they rest, sent when it changed. */
    commit(packed) {
        const ids = this.boardIds().filter(id => packed.has(id));
        const order = [...ids].sort((a, b) => packed.get(a).row - packed.get(b).row || packed.get(a).col - packed.get(b).col);
        if (order.join() === ids.join())
            return;
        const on = new Set(order);
        let next = 0;
        this.order = this.order.map(id => on.has(id) ? order[next++] : id);
        this.orderRequest = this.send({ action: "order_panels", order }, "Reordering panels…");
    }
    /** v2 _scroll_speed: device px per tick -- none unless the pointer is within one title strip of
     * the board's top or bottom edge (or past it), faster the deeper in, up to a quarter strip. */
    speed(box = this.root.getBoundingClientRect()) {
        const carried = this.carried;
        if (!carried)
            return 0;
        const dpr = devicePixelRatio, band = carried.band;
        const y = Math.round((carried.pointer.y - box.top - this.root.clientTop) * dpr), height = Math.round(this.root.clientHeight * dpr);
        if (y < band)
            return -Math.max(1, Math.floor(Math.min(band - y, band) / 4));
        if (y > height - band)
            return Math.max(1, Math.floor(Math.min(y - (height - band), band) / 4));
        return 0;
    }
    /** v2 _steer_scroll: a tick every 20 ms while the pointer holds the card at an edge. */
    steer(box) {
        const carried = this.carried;
        if (!this.speed(box)) {
            window.clearInterval(carried.scroll);
            carried.scroll = 0;
        }
        else if (!carried.scroll)
            carried.scroll = window.setInterval(() => this.step(), 20);
    }
    /** v2 _scroll_step: the board moved under a pointer that did not -- the card keeps its hold
     * and is aimed at the slot now under it.  At the board's end it stops. */
    step() {
        const carried = this.carried, speed = this.speed();
        if (!carried)
            return;
        const before = this.root.scrollTop;
        // At least a whole CSS px, so a fractional step is never rounded away.
        if (speed)
            this.root.scrollTop = before + Math.sign(speed) * Math.max(1, Math.abs(speed) / devicePixelRatio);
        if (!speed || this.root.scrollTop === before) {
            window.clearInterval(carried.scroll);
            carried.scroll = 0;
            return;
        }
        this.aim();
    }
    /** The board leaves the screen with its view: no gesture stays captured, no card carried. */
    hidden() { this.gesture?.end(false); for (const view of this.views.values())
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
    modelFace = comboFace(this.model);
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
            // An open list's Esc closes the list only (v2 FluentComboBox.keyPressEvent).
            if (event.key === "Escape" && !this.inlined && !(event.target instanceof HTMLInputElement) && !event.target.closest("select:open")) {
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
        // What the panel has pinned (v2 data_scope): each Scope row's axis and the coordinate it shows.
        const pinned = (panel.controls ?? []).flatMap(control => control.section === "semantic" && control.cycle && control.cycle.index !== null
            ? [[control.label, control.cycle.label]] : []);
        const key = JSON.stringify([shownId, words, panel.structure, panel.kind, panel.focus, pinned]);
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
        const domains = ["domain-repeat", "domain-point", "domain-cell"];
        if (panel.structure) {
            name.push({ text: " ", elide: null });
            for (const [line, names] of [[name, false], [kind, true]]) {
                for (let domain = 0; domain < 3; domain++) {
                    const className = domains[domain];
                    if (domain)
                        line.push({ text: " × ", elide: null });
                    const items = panel.structure[domain] || [];
                    const text = items.map(item => String(item[names ? 0 : 1])).join(" × ") || (names ? "—" : "1");
                    line.push({ text: "(", className, elide: null }, { text, className, elide: "middle" }, { text: ")", className, elide: null });
                }
            }
        }
        // The pinned coordinates close the axes' line, each in its axis's group colour: the part
        // that changes under the operator's hands (v2 _band_fragments).
        for (const [axis, coordinate] of pinned) {
            const domain = panel.structure?.findIndex(items => items.some(item => String(item[0]) === axis)) ?? -1;
            if (kind.length)
                kind.push({ text: ", ", elide: null });
            kind.push({ text: `${axis}=${coordinate}`, className: domains[domain], elide: "middle" });
        }
        // v2 _vacancy_note: why the card holds no picture, as the operator's next step.
        if (!kind.length)
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
