// v2 PointReviewView for a run's point-selection request, in its own window
// (review:<node>, titled with the request's title, opened as v2's review
// dialog was: a screen fit, the other windows waiting until it closes).  The
// review image on its own surface (review-<node>, the host's Image panel on
// the read the review opened with, at its own resolution, every candidate
// drawn as the review marks it), the candidates listed beside it -- each a
// check box "label · (x, y)", ticked = kept -- under a filter, the counts in a
// status line, and the commands: Exclude selected, Restore selected, Reset,
// Stop, and the request's own confirm.  On the image a click toggles the
// candidate under it, a drag selects the ones it encloses, a click on none
// clears the selection (v2 _ImagePointReviewInteraction); the list and the
// image show one state, which is the window's until the confirm answers the
// excluded ids.  Closing the window is Stop.
import { StatusStrip, checkBox, command, element, setChecked, wordsWidth } from "../fluent/index.js";
/** v2: a press that moves this far (CSS px) is a box; a click takes a point this near. */
const DRAG = 6, NEAR = 10;
class PointReview {
    host;
    root = element("div", "point-review");
    panel;
    points;
    initial;
    excluded;
    selected = new Set();
    boxes = new Map();
    surface = element("div", "point-review-surface");
    band = element("div", "point-review-band");
    strip = new StatusStrip();
    exclude;
    restore;
    confirm;
    press = null;
    drawn;
    /** `answered`: the confirm's excluded ids, or null for Stop. */
    constructor(host, row, request, answered) {
        this.host = host;
        const payload = request.payload;
        this.points = payload.points ?? (payload.point_ids ?? []).map(id => [id, id, NaN, NaN]);
        this.initial = new Set(payload.excluded_point_ids ?? []);
        this.excluded = new Set(this.initial);
        this.panel = `review-${row.node_id}`;
        const filter = element("input");
        filter.type = "text";
        filter.placeholder = "Filter sites";
        filter.spellcheck = false;
        const list = element("div", "point-list");
        let widest = 0;
        for (const [id, label, x, y] of this.points) {
            const text = Number.isFinite(x) ? `${label}  ·  (${x.toFixed(2)}, ${y.toFixed(2)})` : label;
            widest = Math.max(widest, wordsWidth(text));
            const box = checkBox(text, kept => this.commit(kept ? [...this.excluded].filter(item => item !== id) : [...this.excluded, id], [...this.selected]));
            box.title = id;
            this.boxes.set(id, box);
            list.append(box);
        }
        filter.addEventListener("input", () => {
            const wanted = filter.value.trim().toLowerCase();
            for (const [id, box] of this.boxes)
                box.hidden = Boolean(wanted) && !`${box.textContent} ${id}`.toLowerCase().includes(wanted);
        });
        const side = element("div", "card bordered point-review-side");
        // v2's side wants its widest row (QCheckBox.sizeHint: the box, its gap, the
        // words' QFontMetrics.size()), every row counted once as built -- the scroll
        // area keeps that hint, so a filter never narrows it; the filter's own hint
        // is under the side's minimum at every scale.
        side.style.setProperty("--review-side-rows", `calc(${widest} * var(--q) + var(--check-size) + var(--check-gap))`);
        side.append(filter, list);
        this.surface.append(this.band);
        this.band.hidden = true;
        const plot = element("div", "card bordered point-review-plot");
        plot.append(this.surface);
        const body = element("div", "point-review-body");
        body.append(plot, side);
        this.strip.root.classList.add("flush");
        const message = element("p", "wrap point-review-message", request.message ?? "");
        message.hidden = !request.message;
        // Red like every command that takes something away; Restore brings points back.
        this.exclude = command("Exclude selected", () => this.retain(false), { role: "destructive" }).root;
        this.restore = command("Restore selected", () => this.retain(true)).root;
        const reset = command("Reset", () => this.commit([...this.initial], []), { role: "destructive",
            title: "Discard your exclusions and go back to the points this review started with" });
        const stop = command(`Stop ${row.name}`, () => answered(null), { role: "destructive" });
        const confirm = command(payload.confirm_label || "Continue", () => answered(this.points.map(([id]) => id).filter(id => this.excluded.has(id))), { role: "primary" });
        this.confirm = confirm.root;
        const actions = element("div", "point-review-actions");
        actions.append(this.exclude, this.restore, reset.root, element("span", "spacer"), stop.root, this.confirm);
        this.root.append(message, body, this.strip.root, actions);
        this.surface.addEventListener("pointerdown", event => this.down(event), { capture: true });
        this.surface.addEventListener("pointermove", event => this.move(event), { capture: true });
        this.surface.addEventListener("pointerup", event => this.up(event), { capture: true });
        this.surface.addEventListener("lostpointercapture", () => { this.press = null; this.band.hidden = true; });
        // A request that names no points (an older plugin) is reviewed by its list alone.
        this.drawn = Boolean(payload.points);
        this.root.classList.toggle("listed", !this.drawn);
        this.show();
    }
    /** Each accepted state: the surface takes the review's picture once the host has drawn it. */
    update() {
        const view = this.host.board.views.get(this.panel);
        if (view && view.root.parentElement !== this.surface)
            this.surface.prepend(view.root);
    }
    /** The candidates where the host says they stand on the drawn picture, in canvas CSS px. */
    placed() {
        const canvas = this.host.board.views.get(this.panel)?.canvas;
        const review = this.host.latest().panels?.find(panel => panel.id === this.panel)?.review;
        if (!canvas || !review)
            return [];
        const box = canvas.getBoundingClientRect();
        return review.points.filter(point => point[4] !== null && point[5] !== null)
            .map(([id, , , , nx, ny]) => ({ id, x: box.left + nx * box.width, y: box.top + ny * box.height }));
    }
    down(event) {
        if (event.button !== 0 || !event.target.closest(".surface"))
            return;
        event.stopPropagation();
        event.preventDefault();
        this.surface.setPointerCapture(event.pointerId);
        this.press = { x: event.clientX, y: event.clientY };
    }
    move(event) {
        if (!this.press)
            return;
        event.stopPropagation();
        const origin = this.surface.getBoundingClientRect();
        const [left, right] = [this.press.x, event.clientX].sort((a, b) => a - b);
        const [top, bottom] = [this.press.y, event.clientY].sort((a, b) => a - b);
        const shown = right - left > DRAG || bottom - top > DRAG;
        this.band.hidden = !shown;
        if (shown)
            Object.assign(this.band.style, { left: `${left - origin.left}px`, top: `${top - origin.top}px`, width: `${right - left}px`, height: `${bottom - top}px` });
    }
    up(event) {
        const press = this.press;
        if (!press)
            return;
        event.stopPropagation();
        this.press = null;
        this.band.hidden = true;
        const [left, right] = [press.x, event.clientX].sort((a, b) => a - b);
        const [top, bottom] = [press.y, event.clientY].sort((a, b) => a - b);
        const placed = this.placed();
        if (right - left > DRAG || bottom - top > DRAG) {
            this.commit([...this.excluded], placed.filter(point => point.x >= left && point.x <= right && point.y >= top && point.y <= bottom).map(point => point.id));
            return;
        }
        let nearest = null;
        for (const point of placed) {
            const distance = Math.hypot(point.x - press.x, point.y - press.y);
            if (distance <= NEAR && (!nearest || distance < nearest.distance))
                nearest = { id: point.id, distance };
        }
        if (!nearest) {
            this.commit([...this.excluded], []);
            return;
        }
        const id = nearest.id;
        this.commit(this.excluded.has(id) ? [...this.excluded].filter(item => item !== id) : [...this.excluded, id], []);
    }
    /** Exclude selected / Restore selected: the selection's points out of the fit or back in. */
    retain(kept) {
        const excluded = new Set(this.excluded);
        for (const id of this.selected) {
            if (kept)
                excluded.delete(id);
            else
                excluded.add(id);
        }
        this.commit([...excluded], []);
    }
    commit(excluded, selected) {
        this.excluded = new Set(excluded);
        this.selected = new Set(selected);
        this.show();
        if (this.drawn)
            this.host.send({ action: "point_review_state", panel: this.panel, excluded: [...this.excluded], selected: [...this.selected] });
    }
    show() {
        for (const [id, box] of this.boxes)
            setChecked(box, !this.excluded.has(id));
        const detected = this.points.length, width = String(detected).length;
        // Each count in the detected count's width, in figure spaces (v2 figure_padded): the words after it stay put.
        const pad = (count) => String(count).padStart(width, "\u2007");
        const suffix = this.selected.size ? ` · Selected ${pad(this.selected.size)}` : "";
        this.strip.show(`Detected ${detected} · Excluded ${pad(this.excluded.size)} · Final ${pad(detected - this.excluded.size)}${suffix}`, this.selected.size ? "task" : "idle");
        this.exclude.disabled = this.restore.disabled = !this.selected.size;
        this.confirm.disabled = this.excluded.size >= detected;
    }
}
/** The review's window (`?window=review&node=<node>`): the run's current
 * point-selection request, reviewed until it is answered or gone; then the
 * window asks to be closed. */
class ReviewWindow {
    ctx;
    board;
    node;
    key = "review";
    title = "Point review";
    windowTitle = "Point review";
    root = element("div", "window review-window");
    review = null;
    closing = false;
    waiting = element("p", "note", "Waiting for the run's review…");
    constructor(ctx, board, node) {
        this.ctx = ctx;
        this.board = board;
        this.node = node;
        this.root.append(this.waiting);
    }
    update(_state) {
        const row = this.ctx.latest.logic?.find(item => item.node_id === this.node);
        const request = row?.run?.request ?? null;
        if (!row || request?.kind !== "point-selection") {
            // Answered here, over MCP, or the run is over: the window goes.
            if (this.review && !this.closing)
                this.close();
            return;
        }
        if (this.review?.request !== request.request) {
            const view = new PointReview({ board: this.board, latest: () => this.ctx.latest, send: (command, feedback) => this.ctx.transport.send(command, feedback) }, row, request, excluded => this.answer(row, request, excluded));
            this.review = { request: request.request, view };
            this.root.replaceChildren(view.root);
            // v2's review window is titled with the request's own words.
            document.title = request.title || "Point review";
        }
        this.review.view.update();
    }
    /** The window's words are its counts; the command queue has none of its own here. */
    status() { }
    answer(row, request, excluded) {
        this.ctx.transport.send({ action: "answer_request", node: row.node_id, request: request.request,
            response: excluded ? { excluded_point_ids: excluded } : null }, excluded ? "Continuing the run…" : "Stopping the run…");
        this.close();
    }
    close() {
        this.closing = true;
        this.ctx.transport.send({ action: "close_point_review", node: this.node });
    }
}
export const reviewWindow = (board, node) => (ctx) => new ReviewWindow(ctx, board, node);
