// One canvas's pointer input, as the host reads it: every press, drag,
// release, wheel, double-click and Escape becomes a `pointer` command with
// the normalized position, the axes under it (captured at press for the whole
// drag), the button and the frame on screen. Moves coalesce once per
// animation frame. Wheel travel on the same axes and in the same direction
// also accumulates per frame, without losing fractional steps. The host
// decides what a gesture does; this only says where it happened.
//
// `enabled` is the Selectors switch of whoever shows the canvas (v2
// set_interaction_enabled / set_deriving): off, the plot takes no input --
// no drag, hover, zoom or pan -- and the wheel scrolls the page.
import type { Axis, Command } from "../protocol.js";

type Button = "left" | "middle" | "right";
const buttonOf = (event: MouseEvent): Button => event.button === 1 ? "middle" : event.button === 2 ? "right" : "left";

export class CanvasPointer {
  private activePointer: number | null = null;
  private activeButton: Button | null = null;
  private pressAxes: Axis | null = null;
  private move: Command | null = null;
  private moveFrame = 0;
  private wheel: Command | null = null;
  private wheelFrame = 0;
  private lastPoint = { x: 0.5, y: 0.5 };

  constructor(private canvas: HTMLCanvasElement, private panel: string, private send: (command: Command) => void,
              private axes: () => readonly Axis[], private sequence: () => number, private enabled: () => boolean = () => true) {
    this.bind();
  }

  /** Give up a captured gesture (blur, the canvas going away, Selectors off). */
  cancel(): void {
    this.wheel = null;
    if (this.wheelFrame) cancelAnimationFrame(this.wheelFrame);
    this.wheelFrame = 0;
    this.move = null;
    if (this.moveFrame) cancelAnimationFrame(this.moveFrame);
    this.moveFrame = 0;
    if (this.activePointer === null) return;
    const pointer = this.activePointer;
    this.activePointer = null;
    this.activeButton = null;
    this.pressAxes = null;
    if (this.canvas.hasPointerCapture(pointer)) this.canvas.releasePointerCapture(pointer);
    this.send({ action: "pointer", panel: this.panel, event: "cancel", axes: null, displayed_sequence: this.sequence(), ...this.lastPoint });
  }

  private axesAt(x: number, y: number): Axis | null {
    const axes = this.axes();
    for (let index = axes.length - 1; index >= 0; index--) {
      const axis = axes[index]!;
      const [left, top, right, bottom] = axis.bounds;
      if (x >= left && x <= right && y >= top && y <= bottom) return axis;
    }
    return null;
  }

  private flushWheel(): void {
    if (this.wheelFrame) cancelAnimationFrame(this.wheelFrame);
    this.wheelFrame = 0;
    const command = this.wheel;
    this.wheel = null;
    if (command) this.send(command);
  }

  private bind(): void {
    const canvas = this.canvas;
    const modifiers = (event: MouseEvent): string[] => [event.shiftKey ? "shift" : "", event.ctrlKey ? "control" : "", event.altKey ? "alt" : "", event.metaKey ? "meta" : ""].filter(Boolean);
    // A canvas not laid out (its page hidden) has no place to point at: nothing is sent.
    const placed = (): boolean => { const rect = canvas.getBoundingClientRect(); return rect.width > 0 && rect.height > 0; };
    const message = (event: MouseEvent, kind: string): Command => {
      const rect = canvas.getBoundingClientRect();
      this.lastPoint = { x: (event.clientX - rect.left) / rect.width, y: (event.clientY - rect.top) / rect.height };
      if (kind === "press") this.pressAxes = this.axesAt(this.lastPoint.x, this.lastPoint.y);
      const axes = kind === "leave" ? null
        : (kind === "move" || kind === "release") && this.activePointer !== null ? this.pressAxes
        : this.axesAt(this.lastPoint.x, this.lastPoint.y);
      return { action: "pointer", panel: this.panel, event: kind, ...this.lastPoint, axes,
        button: kind === "move" ? this.activeButton : buttonOf(event),
        displayed_sequence: this.sequence(), time: event.timeStamp / 1000,
        buttons: [event.buttons & 1 ? "left" : "", event.buttons & 4 ? "middle" : "", event.buttons & 2 ? "right" : ""].filter(Boolean), modifiers: modifiers(event) };
    };
    const flushMove = (): void => {
      if (this.moveFrame) cancelAnimationFrame(this.moveFrame);
      this.moveFrame = 0;
      if (this.move) { this.send(this.move); this.move = null; }
    };
    canvas.addEventListener("pointerdown", event => {
      if (this.activePointer !== null) { this.cancel(); return; }
      if (!this.enabled() || !placed()) return;
      event.preventDefault();
      this.flushWheel();
      flushMove();
      canvas.focus({ preventScroll: true });
      this.activePointer = event.pointerId;
      this.activeButton = buttonOf(event);
      canvas.setPointerCapture(event.pointerId);
      this.send(message(event, "press"));
    });
    canvas.addEventListener("pointermove", event => {
      if (this.activePointer !== null && this.activePointer !== event.pointerId) return;
      if ((this.activePointer === null && !this.enabled()) || !placed()) return;
      this.flushWheel();
      this.move = message(event, "move");
      if (!this.moveFrame) this.moveFrame = requestAnimationFrame(flushMove);
    });
    canvas.addEventListener("pointerup", event => {
      if (this.activePointer !== event.pointerId) return;
      if (!placed()) { this.cancel(); return; }
      event.preventDefault();
      this.flushWheel();
      flushMove();
      this.send(message(event, "release"));
      this.activePointer = null;
      this.activeButton = null;
      this.pressAxes = null;
      if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
    });
    canvas.addEventListener("dblclick", event => { if (!this.enabled() || !placed()) return; event.preventDefault(); this.flushWheel(); flushMove(); this.send(message(event, "double_click")); });
    canvas.addEventListener("pointerleave", event => {
      if (this.activePointer !== null || !this.enabled() || !placed()) return;
      this.flushWheel();
      flushMove();
      this.send(message(event, "leave"));
    });
    canvas.addEventListener("pointercancel", () => this.cancel());
    canvas.addEventListener("lostpointercapture", () => this.cancel());
    canvas.addEventListener("contextmenu", event => event.preventDefault());
    canvas.addEventListener("wheel", event => {
      if (!this.enabled() || !placed()) return;
      event.preventDefault();
      flushMove();
      const unit = event.deltaMode === WheelEvent.DOM_DELTA_LINE ? 16 : event.deltaMode === WheelEvent.DOM_DELTA_PAGE ? canvas.clientHeight : 1;
      const steps = -event.deltaY * unit / 100;
      if (!Number.isFinite(steps) || steps === 0) return;
      // These notifications have no individual reply waiter. The native lane
      // may also combine adjacent travel while an earlier frame is rendering.
      const command = { ...message(event, "scroll"), steps, _coalesce_scroll: true };
      const prior = this.wheel;
      const identity = (value: Command): string => {
        const axis = value.axes as Axis | null;
        return JSON.stringify([axis?.role, axis?.cell_index, value.modifiers]);
      };
      if (prior && identity(prior) === identity(command) && Math.sign(Number(prior.steps)) === Math.sign(steps)) {
        command.steps += Number(prior.steps);
      } else { this.flushWheel(); }
      this.wheel = command;
      if (!this.wheelFrame) this.wheelFrame = requestAnimationFrame(() => this.flushWheel());
    }, { passive: false });
    canvas.addEventListener("keydown", event => {
      if (event.key !== "Escape" || !this.enabled()) return;
      event.preventDefault();
      this.flushWheel();
      flushMove();
      this.send({ action: "pointer", panel: this.panel, event: "key", key: "Escape", axes: null, displayed_sequence: this.sequence(), ...this.lastPoint });
    });
  }
}
