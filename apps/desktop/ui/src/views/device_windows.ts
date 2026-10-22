// A device's own windows, as v2 opened them from Devices@: its control
// (v2 open_device_control, "<role> control": half the standard width and its
// content's height, opened once the device's fields are read) and its log
// ("<role> log@Zou lab"). Each shows one device row of the workbench's
// `devices`; the host closes both when the device closes.
import { element } from "../fluent/index.js";
import type { State } from "../protocol.js";
import type { View, ViewContext } from "../shell.js";
import type { DeviceRow } from "../workbench.js";
import { DeviceControlView, DeviceLogView } from "./device_control.js";

const row = (ctx: ViewContext, id: string): DeviceRow | undefined => (ctx.latest.devices as DeviceRow[] | undefined)?.find(device => device.instance_id === id);

class DeviceControlWindow implements View {
  readonly key = "control";
  readonly title = "Control";
  readonly windowTitle: string;
  readonly root = element("div", "device-window");
  private view: DeviceControlView;
  private named = "";
  private shownOnce = false;

  constructor(private ctx: ViewContext, private device: string) {
    this.windowTitle = `${device} control`;
    this.view = new DeviceControlView(device, ctx);
    this.root.append(this.view.root);
    ctx.transport.send({ action: "read_device", device }, "Reading the device…");
  }

  update(state: State): void {
    const device = row(this.ctx, this.device);
    if (!device) return;
    this.view.update(device, state.request, Boolean(state.error));
    const title = `${device.role} control`;
    if (title !== this.named) { this.named = title; document.title = title; }
    // v2: the window is built from the fields the device declares, so it opens once they are read, as tall as they are.
    if (!this.shownOnce && device.fields) {
      this.shownOnce = true;
      requestAnimationFrame(() => {
        this.root.classList.add("measuring");
        // An overflowing box's scroll width leaves out its end padding.
        const height = this.root.getBoundingClientRect().height;
        const width = Math.ceil(this.view.root.scrollWidth + parseFloat(getComputedStyle(this.view.root).paddingRight));
        this.root.classList.remove("measuring");
        this.ctx.transport.post({ action: "fit_window", width, height });
        this.ctx.transport.post({ action: "show_window", window: `control:${this.device}` });
      });
    }
  }
  status(): void {}
}

class DeviceLogWindow implements View {
  readonly key = "log";
  readonly title = "Log";
  readonly windowTitle: string;
  readonly root = element("div", "device-window");
  private named = "";

  constructor(private ctx: ViewContext, private device: string) {
    this.windowTitle = `${device} log@Zou lab`;
    this.root.append(new DeviceLogView(device, ctx).root);
  }

  update(_state: State): void {
    const device = row(this.ctx, this.device);
    const title = `${device?.role ?? this.device} log@Zou lab`;
    if (title !== this.named) { this.named = title; document.title = title; }
  }
  status(): void {}
}

export const deviceControl = (device: string) => (ctx: ViewContext): View => new DeviceControlWindow(ctx, device);
export const deviceLog = (device: string) => (ctx: ViewContext): View => new DeviceLogWindow(ctx, device);
