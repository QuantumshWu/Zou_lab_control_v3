const buttonOf = (event) => event.button === 1 ? "middle" : event.button === 2 ? "right" : "left";
export class CanvasPointer {
    canvas;
    panel;
    send;
    axes;
    sequence;
    enabled;
    activePointer = null;
    activeButton = null;
    pressAxes = null;
    move = null;
    moveFrame = 0;
    lastPoint = { x: 0.5, y: 0.5 };
    constructor(canvas, panel, send, axes, sequence, enabled = () => true) {
        this.canvas = canvas;
        this.panel = panel;
        this.send = send;
        this.axes = axes;
        this.sequence = sequence;
        this.enabled = enabled;
        this.bind();
    }
    /** Give up a captured gesture (blur, the canvas going away, Selectors off). */
    cancel() {
        this.move = null;
        if (this.moveFrame)
            cancelAnimationFrame(this.moveFrame);
        this.moveFrame = 0;
        if (this.activePointer === null)
            return;
        const pointer = this.activePointer;
        this.activePointer = null;
        this.activeButton = null;
        this.pressAxes = null;
        if (this.canvas.hasPointerCapture(pointer))
            this.canvas.releasePointerCapture(pointer);
        this.send({ action: "pointer", panel: this.panel, event: "cancel", axes: null, displayed_sequence: this.sequence(), ...this.lastPoint });
    }
    axesAt(x, y) {
        const axes = this.axes();
        for (let index = axes.length - 1; index >= 0; index--) {
            const axis = axes[index];
            const [left, top, right, bottom] = axis.bounds;
            if (x >= left && x <= right && y >= top && y <= bottom)
                return axis;
        }
        return null;
    }
    bind() {
        const canvas = this.canvas;
        const modifiers = (event) => [event.shiftKey ? "shift" : "", event.ctrlKey ? "control" : "", event.altKey ? "alt" : "", event.metaKey ? "meta" : ""].filter(Boolean);
        // A canvas not laid out (its page hidden) has no place to point at: nothing is sent.
        const placed = () => { const rect = canvas.getBoundingClientRect(); return rect.width > 0 && rect.height > 0; };
        const message = (event, kind) => {
            const rect = canvas.getBoundingClientRect();
            this.lastPoint = { x: (event.clientX - rect.left) / rect.width, y: (event.clientY - rect.top) / rect.height };
            if (kind === "press")
                this.pressAxes = this.axesAt(this.lastPoint.x, this.lastPoint.y);
            const axes = kind === "leave" ? null
                : (kind === "move" || kind === "release") && this.activePointer !== null ? this.pressAxes
                    : this.axesAt(this.lastPoint.x, this.lastPoint.y);
            return { action: "pointer", panel: this.panel, event: kind, ...this.lastPoint, axes,
                button: kind === "move" ? this.activeButton : buttonOf(event),
                displayed_sequence: this.sequence(),
                buttons: [event.buttons & 1 ? "left" : "", event.buttons & 4 ? "middle" : "", event.buttons & 2 ? "right" : ""].filter(Boolean), modifiers: modifiers(event) };
        };
        const flushMove = () => {
            if (this.moveFrame)
                cancelAnimationFrame(this.moveFrame);
            this.moveFrame = 0;
            if (this.move) {
                this.send(this.move);
                this.move = null;
            }
        };
        canvas.addEventListener("pointerdown", event => {
            if (this.activePointer !== null) {
                this.cancel();
                return;
            }
            if (!this.enabled() || !placed())
                return;
            event.preventDefault();
            canvas.focus({ preventScroll: true });
            this.activePointer = event.pointerId;
            this.activeButton = buttonOf(event);
            canvas.setPointerCapture(event.pointerId);
            this.send(message(event, "press"));
        });
        canvas.addEventListener("pointermove", event => {
            if (this.activePointer !== null && this.activePointer !== event.pointerId)
                return;
            if ((this.activePointer === null && !this.enabled()) || !placed())
                return;
            this.move = message(event, "move");
            if (!this.moveFrame)
                this.moveFrame = requestAnimationFrame(flushMove);
        });
        canvas.addEventListener("pointerup", event => {
            if (this.activePointer !== event.pointerId)
                return;
            if (!placed()) {
                this.cancel();
                return;
            }
            event.preventDefault();
            flushMove();
            this.send(message(event, "release"));
            this.activePointer = null;
            this.activeButton = null;
            this.pressAxes = null;
            if (canvas.hasPointerCapture(event.pointerId))
                canvas.releasePointerCapture(event.pointerId);
        });
        canvas.addEventListener("dblclick", event => { if (!this.enabled() || !placed())
            return; event.preventDefault(); flushMove(); this.send(message(event, "double_click")); });
        canvas.addEventListener("pointerleave", event => {
            if (this.activePointer !== null || !this.enabled() || !placed())
                return;
            flushMove();
            this.send(message(event, "leave"));
        });
        canvas.addEventListener("pointercancel", () => this.cancel());
        canvas.addEventListener("lostpointercapture", () => this.cancel());
        canvas.addEventListener("contextmenu", event => event.preventDefault());
        canvas.addEventListener("wheel", event => {
            if (!this.enabled() || !placed())
                return;
            event.preventDefault();
            flushMove();
            const unit = event.deltaMode === WheelEvent.DOM_DELTA_LINE ? 16 : event.deltaMode === WheelEvent.DOM_DELTA_PAGE ? canvas.clientHeight : 1;
            this.send({ ...message(event, "scroll"), steps: -event.deltaY * unit / 100 });
        }, { passive: false });
        canvas.addEventListener("keydown", event => {
            if (event.key !== "Escape" || !this.enabled())
                return;
            event.preventDefault();
            flushMove();
            this.send({ action: "pointer", panel: this.panel, event: "key", key: "Escape", axes: null, displayed_sequence: this.sequence(), ...this.lastPoint });
        });
    }
}
