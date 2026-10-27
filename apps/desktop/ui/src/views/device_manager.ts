// DeviceManager@Zou lab: the Config page authors the apparatus draft (one
// card per device family, each device a ruled row of its family's card:
// Role, type, Remove, its parameters), lists discovered hardware and the
// loaded session (Control, Log, Close), and ends in the file commands and
// the one lifecycle command.  Control and Log open the device's own windows
// (v2's "<role> control" and "<role> log@Zou lab"; a sequencer's control is
// the Pulse Editor's window).  Every edit is a §4.2 command.
import {
  ComboBox, Field, GroupBox, Menu, StatusStrip, command, element, elided, fieldKind, form, nameWords, note, ruledList,
  TabWidget, setChecked, setState, shareLabelColumn, statusDot, switchButton, type Command as Button, type PathAsk, type Value,
} from "../fluent/index.js";
import type { State } from "../protocol.js";
import { queueStatus, type View, type ViewContext } from "../shell.js";
import type { Apparatus, DeviceRow, Share, WorkbenchControl } from "../workbench.js";

/** v2's Loaded session: the devices the session holds (a closing one until it is gone). */
const LOADED = new Set(["open", "closing"]);
const LIVE = new Set(["opening", "open", "closing", "failed"]);
/** The lifecycle command by what apply_devices would do now (v2 texts). */
const LIFECYCLE: Record<NonNullable<Apparatus["action"]>, string> = { init: "Init devices", apply: "Apply device changes", shut_down: "Shut down devices" };

/** A form of fields that keeps its rows across projections. */
class Fields {
  readonly root = form();
  private fields = new Map<string, Field>();
  constructor(private id: string, private commit: (name: string, value: Value, label: string) => number, private browse: (ask: PathAsk) => Promise<string | null>) {}
  update(controls: readonly WorkbenchControl[], request?: number, failed = false): void {
    const alive = new Set(controls.map(control => control.name));
    for (const [name, field] of this.fields) if (!alive.has(name)) { field.root.remove(); this.fields.delete(name); }
    let cursor = this.root.firstElementChild;
    for (const shown of controls) {
      // A device's numbers are spin boxes, a blank-allowed one too.
      const control = { ...shown, spin: true };
      let field = this.fields.get(control.name);
      if (field && field.kind !== fieldKind(control)) {
        if (field.root === cursor) cursor = cursor.nextElementSibling;
        field.root.remove(); this.fields.delete(control.name); field = undefined;
      }
      if (!field) field = new Field(this.id, control, { commit: (value, spec) => this.commit(spec.name, value, spec.label), browse: this.browse });
      else field.update(control, request, failed);
      this.fields.set(control.name, field);
      if (field.root !== cursor) this.root.insertBefore(field.root, cursor); else cursor = cursor.nextElementSibling;
    }
  }
}

/** One draft device: its header row on the form's columns (Role, its box,
 * the type, Remove and the disclosure chevron) and its parameters. */
class DeviceCard {
  readonly root = element("div", "device-card");
  private label = element("label", "field-label", "Role");
  private role = element("input");
  private types: ComboBox;
  private disclosure: HTMLButtonElement;
  private params: Fields;
  private collapsed = false;
  private roleDraft = false;

  constructor(readonly id: string, private ctx: ViewContext) {
    const header = form("device-header");
    const label = this.label;
    this.role.type = "text";
    this.role.spellcheck = false;
    this.role.placeholder = "role";
    this.role.id = `device-${id}-role`;
    label.htmlFor = this.role.id;
    this.role.addEventListener("input", () => { this.roleDraft = true; });
    this.role.addEventListener("change", () => {
      this.ctx.transport.send({ action: "edit_device", device: id, role: this.role.value.trim() }, "Renaming the device…");
    });
    this.role.addEventListener("keydown", event => { if (event.key === "Enter") this.role.blur(); });
    this.role.addEventListener("blur", () => { this.roleDraft = false; });
    this.types = new ComboBox(value => { if (typeof value === "string") this.ctx.transport.send({ action: "edit_device", device: id, type_id: value }, "Changing the device type…"); });
    const remove = command("Remove", () => this.ctx.transport.send({ action: "remove_device", device: id }, "Removing the device…"), { role: "destructive" });
    this.disclosure = element("button", "quiet square disclosure");
    this.disclosure.type = "button";
    this.disclosure.addEventListener("click", () => { this.collapsed = !this.collapsed; this.showCollapsed(); });
    const cell = element("div", "field-control");
    cell.append(this.role, this.types.root, remove.root, this.disclosure);
    const row = element("div", "field");
    row.append(label, cell);
    header.append(row);
    this.params = new Fields(`device-${id}`, (name, value, words) =>
      this.ctx.transport.send({ action: "edit_device", device: id, parameters: { [name]: value } }, `Updating ${words}…`), ask => this.ctx.transport.choosePath(ask));
    this.root.append(header, this.params.root);
    this.showCollapsed();
  }

  private showCollapsed(): void {
    this.disclosure.classList.toggle("open", !this.collapsed);
    this.disclosure.title = this.collapsed ? "Show parameters" : "Hide parameters";
    this.params.root.hidden = this.collapsed;
  }

  update(device: DeviceRow, apparatus: Apparatus, request?: number, failed = false): void {
    // The name the operator is still typing is theirs until they leave the box.
    if (!this.roleDraft && this.role.value !== device.role) this.role.value = device.role;
    const family = apparatus.types.filter(type => type.family === device.family && (type.addable || type.type_id === device.type_id));
    // v2 set_device_choices: a type is named by its id, the types in id order.
    const choices = family.map(type => ({ value: type.type_id, label: type.type_id }))
      .sort((a, b) => a.value.localeCompare(b.value));
    // A row of another family's type (a device another computer publishes sits with its kind).
    if (!choices.some(choice => choice.value === device.type_id)) choices.unshift({ value: device.type_id, label: device.type_id });
    this.types.setChoices(choices);
    this.types.value = device.type_id;
    // v2: the Role label says the internal id; why a device failed is the status line's (v2 _outcome).
    this.label.title = `Internal ID: ${device.instance_id}`;
    this.params.update(device.controls, request, failed);
  }
}

/** One family's card: its title and Add device, then its devices as a ruled list. */
class FamilyCard {
  readonly box: GroupBox;
  readonly list = ruledList();
  private add: Button;
  constructor(readonly family: string, label: string, ctx: ViewContext, firstType: () => string | null) {
    this.add = command("Add device", () => {
      const type = firstType();
      if (type) ctx.transport.send({ action: "add_device", type_id: type }, "Adding a device…");
    });
    this.box = new GroupBox(label, [this.add.root]);
    this.box.body.append(this.list);
  }
  set addable(on: boolean) { this.add.disabled = !on; }
}

export class DeviceManagerView implements View {
  readonly key = "device-manager";
  readonly title = "Device Manager";
  readonly windowTitle = "Devices@Zou lab";
  readonly root = element("div", "window device-manager");
  /** v2 DeviceManagerView: a FluentTabWidget whose one permanent page, Config, is the window's white page. */
  private tabs = new TabWidget();
  private page = element("div", "config-page");
  private dot = statusDot();
  private documentName = element("span", "document-name elided");
  private left = element("div", "config-column left");
  private right = element("div", "config-column right");
  private families = new Map<string, FamilyCard>();
  private cards = new Map<string, DeviceCard>();
  private unavailable = note("", true);
  private discovered = new GroupBox("Discovered hardware");
  private discoveredList = ruledList();
  private discoveredEmpty = note("No discovered hardware", true);
  private loaded = new GroupBox("Loaded session");
  private loadedList = ruledList();
  private loadedEmpty = note("No active installation", true);
  private sharing: GroupBox;
  private shareButton: Button;
  private shareAddress = element("input");
  private shareCopy: Button;
  private shareAddressRow = element("div", "list-row");
  private shareStatus = note("", true);
  private shareClients = ruledList();
  private scan: Button;
  private newButton: Button;
  private save: Button;
  private files: Button[];
  private lifecycle: Button;
  private strip = new StatusStrip();

  constructor(private ctx: ViewContext) {
    const header = element("div", "title-row config-header");
    header.append(element("span", "heading", "Devices"), element("span", "spacer"), this.dot, this.documentName);
    this.left.append(this.unavailable);
    // v2 Scan hardware: every family at once; the status line says what was found and what could not be asked.
    this.scan = command("Scan hardware", () => ctx.transport.send({ action: "discover_devices" }, "scanning hardware"));
    this.discovered = new GroupBox("Discovered hardware", [this.scan.root]);
    this.discovered.body.append(this.discoveredList, this.discoveredEmpty);
    this.loaded.body.append(this.loadedList, this.loadedEmpty);
    // v2's Browser sharing card: Start/Stop, the address to open, and every browser to Allow or Reject.
    this.shareButton = command("Start sharing", () => {
      const active = (this.ctx.latest.share as Share | undefined)?.active;
      this.ctx.transport.send({ action: active ? "share_stop" : "share_start" }, active ? "Stopping sharing…" : "Starting sharing…");
    }, { words: ["Start sharing", "Stop sharing"] });
    this.sharing = new GroupBox("Browser sharing", [this.shareButton.root]);
    this.shareAddress.readOnly = true;
    this.shareAddress.className = "readout";
    this.shareCopy = command("Copy address", () => {
      this.shareAddress.select();
      const copied = document.execCommand("copy");
      this.ctx.transport.note(copied ? "sharing address copied" : this.shareAddress.value, "task");
    });
    this.shareAddressRow.append(this.shareAddress, this.shareCopy.root);
    this.sharing.body.append(this.shareAddressRow, this.shareStatus, this.shareClients, note("Trusted LAN only. Approve each browser here.", true));
    // v2's column, in v2's order.
    this.right.append(this.discovered.root, this.loaded.root,
      note("Control opens each loaded device in its own window; this window authors the device graph.", true), this.sharing.root);
    const columns = element("div", "config-columns");
    columns.append(this.left, this.right);
    const actions = element("div", "config-actions");
    this.newButton = command("New…", () => this.showTemplates());
    const open = command("Open…", () => ctx.transport.send({ action: "load_apparatus" }, "Opening an apparatus…"));
    this.save = command("Save", () => {
      // v2: a draft not read in full from its file (unread) goes through Save as, which asks before replacing.
      const apparatus = this.apparatus();
      const path = apparatus?.unread ? "" : apparatus?.path;
      ctx.transport.send({ action: "save_apparatus", ...(path ? { path } : {}) }, "Saving the apparatus…");
    });
    const saveAs = command("Save as…", () => ctx.transport.send({ action: "save_apparatus" }, "Saving the apparatus…"));
    // As wide as its words of the moment (v2 set_lifecycle sets the text; the bar's stretch takes the rest).
    this.lifecycle = command(LIFECYCLE.init, () => this.runLifecycle(), { role: "primary" });
    this.files = [this.newButton, open, this.save, saveAs];
    actions.append(this.newButton.root, open.root, this.save.root, saveAs.root, element("span", "spacer"), this.lifecycle.root);
    this.page.append(header, columns, actions, this.strip.root);
    this.tabs.addPermanent("config", "Config", this.page);
    this.root.append(this.tabs.root);
  }

  private apparatus(): Apparatus | undefined { return this.ctx.latest.apparatus as Apparatus | undefined; }
  private devices(): DeviceRow[] { return (this.ctx.latest.devices as DeviceRow[] | undefined) ?? []; }

  private runLifecycle(): void {
    const action = this.apparatus()?.action;
    // v2's words while the device worker runs.
    if (action === "shut_down") this.ctx.transport.send({ action: "shut_down_devices" }, "shutting down devices");
    else if (action) this.ctx.transport.send({ action: "apply_devices" }, action === "init" ? "initializing devices" : "applying device changes");
  }

  private showTemplates(): void {
    const templates = this.apparatus()?.templates ?? [];
    new Menu(templates.map(template => ({ label: nameWords(template), action: () => this.ctx.transport.send({ action: "use_template", template }, `New ${template} apparatus…`) })))
      .openBelow(this.newButton.root);
  }

  update(state: State): void {
    const apparatus = this.apparatus();
    const devices = this.devices();
    const busy = Boolean(apparatus?.busy);
    // The document: its name and the unsaved star; the dot is the session.
    const path = apparatus?.path ?? "";
    const name = (path.split(/[\\/]/).pop() || "untitled") + (apparatus?.dirty ? "*" : "");
    if (this.documentName.textContent !== name) this.documentName.textContent = name;
    this.documentName.title = `${path || "untitled"} (${apparatus?.dirty ? "unsaved draft" : "saved"})`;
    const live = devices.some(device => LIVE.has(device.state));
    const changed = apparatus?.action === "apply";
    setState(this.dot, changed ? "warning" : live ? "running" : "idle");
    this.dot.title = changed ? "Configuration differs from the active installation" : live ? "Active installation" : "No active installation";
    // The Config page is the draft (v2 set_devices): a loaded device it no longer holds is the
    // Loaded session's, and Apply closes it.
    const draft = devices.filter(device => device.pending !== "remove");
    this.updateFamilies(apparatus, draft, state);
    this.updateDiscovered(apparatus, draft);
    this.updateLoaded(devices, busy);
    this.updateSharing(this.ctx.latest.share as Share | undefined);
    // One primary at a time: Save while the draft is unsaved, else the lifecycle.
    const action = apparatus?.action ?? null;
    this.lifecycle.text = action ? LIFECYCLE[action] : LIFECYCLE.init;
    this.lifecycle.setRole(action === "shut_down" ? "destructive" : apparatus?.dirty ? "standard" : "primary");
    this.lifecycle.disabled = busy || !action;
    this.save.setDirty(Boolean(apparatus?.dirty));
    for (const button of this.files) button.disabled = busy || !apparatus;
    this.newButton.disabled = busy || !apparatus?.templates.length;
    this.scan.disabled = busy || !apparatus;
    // One label column for every card of the column, measured while on screen.
    if (!this.root.hidden) requestAnimationFrame(() => shareLabelColumn(this.left));
    this.status();
  }

  private updateFamilies(apparatus: Apparatus | undefined, devices: DeviceRow[], state: State): void {
    const types = apparatus?.types ?? [];
    // v2: one card per family of the addable types, in type-id order.
    const byId = [...types].sort((a, b) => a.type_id.localeCompare(b.type_id));
    const order = [...new Set([...byId.filter(type => type.addable).map(type => type.family), ...devices.map(device => device.family)])];
    for (const [family, card] of this.families) if (!order.includes(family)) { card.box.root.remove(); this.families.delete(family); }
    let cursor: Element | null = this.left.firstElementChild;
    for (const family of order) {
      let card = this.families.get(family);
      const label = types.find(type => type.family === family)?.family_label ?? family;
      if (!card) {
        card = new FamilyCard(family, label, this.ctx, () => types.find(type => type.family === family && type.addable)?.type_id ?? null);
        this.families.set(family, card);
      }
      card.box.title = label;
      card.addable = !apparatus?.busy && types.some(type => type.family === family && type.addable);
      if (card.box.root !== cursor) this.left.insertBefore(card.box.root, cursor); else cursor = cursor.nextElementSibling;
    }
    const alive = new Set(devices.map(device => device.instance_id));
    for (const [id, card] of this.cards) if (!alive.has(id)) { card.root.remove(); this.cards.delete(id); }
    const lists = new Map<string, Element[]>();
    for (const device of devices) {
      let card = this.cards.get(device.instance_id);
      if (!card) { card = new DeviceCard(device.instance_id, this.ctx); this.cards.set(device.instance_id, card); }
      card.update(device, apparatus ?? { path: "", dirty: false, types: [], unavailable: [], templates: [], discovered: [], action: null, busy: null }, state.request, Boolean(state.error));
      lists.set(device.family, [...(lists.get(device.family) ?? []), card.root]);
    }
    for (const [family, card] of this.families) {
      const nodes = lists.get(family) ?? [];
      let at = card.list.firstElementChild;
      for (const node of nodes) { if (node !== at) card.list.insertBefore(node, at); else at = at.nextElementSibling; }
    }
    // What this machine cannot offer today, and why (a missing vendor runtime), by family as v2
    // (UnavailableDeviceTypes.family: the folder under devices/).
    const family = (module: string): string => {
      const parts = module.split(".");
      const at = parts.indexOf("devices");
      return at >= 0 && at + 1 < parts.length ? parts[at + 1]! : parts.length > 1 ? parts[parts.length - 2]! : module;
    };
    // One line per reason, naming every family that gave it (no plugin environment: every family, once).
    const reasons = new Map<string, string[]>();
    for (const item of apparatus?.unavailable ?? []) {
      const families = reasons.get(item.reason) ?? [];
      if (!families.includes(family(item.module))) families.push(family(item.module));
      reasons.set(item.reason, families);
    }
    const words = [...reasons].map(([reason, families]) => `${families.join(", ")}: ${reason}`).join("\n");
    this.unavailable.textContent = words;
    this.unavailable.hidden = !words;
  }

  private updateDiscovered(apparatus: Apparatus | undefined, devices: DeviceRow[]): void {
    const found = apparatus?.discovered ?? [];
    // Only a machine with a type that can scan offers the scan (v2 set_discovery_enabled).
    const scannable = (apparatus?.types ?? []).some(type => type.discoverable);
    this.scan.disabled = !scannable || Boolean(apparatus?.busy);
    this.scan.root.title = scannable ? "" : "No installed device type declares discovery";
    const configured = new Set(devices.map(device => device.instance_id));
    const signature = JSON.stringify([found, [...configured], apparatus?.busy]);
    if (this.discoveredList.dataset.signature === signature) return;
    this.discoveredList.dataset.signature = signature;
    this.discoveredList.replaceChildren(...found.map(item => {
      const line = element("div", "list-row");
      const add = command("Add", () => this.ctx.transport.send({ action: "add_discovered", instance_id: item.instance_id }, "Adding the discovered device…"));
      add.disabled = configured.has(item.instance_id) || Boolean(apparatus?.busy);
      line.append(elided(item.role, "list-title"), element("span", "note list-detail", `${item.type_id} · ${item.instance_id}`), add.root);
      return line;
    }));
    this.discoveredEmpty.hidden = found.length > 0;
  }

  /** v2 set_loaded_devices: a row per device the session holds, its role and type; Control, Log, Close
   * and the Remote switch, all of them disabled while a device change runs (v2 _refresh_controls). */
  private updateLoaded(devices: DeviceRow[], busy: boolean): void {
    const loaded = devices.filter(device => LOADED.has(device.state) || device.close_failed);
    const signature = JSON.stringify([loaded.map(device => [device.instance_id, device.role, device.type_id, device.state, device.close_failed, device.error, device.published, device.publishable]), busy]);
    if (this.loadedList.dataset.signature === signature) return;
    this.loadedList.dataset.signature = signature;
    this.loadedList.replaceChildren(...loaded.map(device => {
      const line = element("div", "list-row");
      const identity = element("div", "list-identity");
      identity.append(elided(device.role, "list-title"), element("span", "note list-detail", `${device.type_id}${device.close_failed ? " · Close failed" : ""}`));
      line.title = `Internal ID: ${device.instance_id}${device.error ? `\n${device.error}` : ""}`;
      const control = command("Control", () => this.openControl(device.instance_id));
      const log = command("Log", () => this.openLog(device.instance_id));
      // Close ends hardware: the destructive face, never the light blue of Control.
      const close = command("Close", () => this.ctx.transport.send({ action: "close_device", device: device.instance_id }, `closing ${device.role}`), { role: "destructive" });
      // Remote: a switch, a state; it shows the bench's answer, never the click (v2 _remote_clicked).
      const remote = switchButton(on => this.ctx.transport.send({ action: "publish_device", device: device.instance_id, published: on },
        `${device.role}: ${on ? "publishing on" : "withdrawing from"} the bench fabric`), "Remote");
      setChecked(remote, Boolean(device.published));
      for (const button of [control, log, close]) button.disabled = busy;
      control.disabled = busy || device.state !== "open";
      // A device another computer publishes is published there, not here (publish_device refuses it).
      remote.disabled = busy || device.state !== "open" || (!device.publishable && !device.published);
      if (device.remote) remote.title = `Published by ${device.remote.publisher}`;
      line.append(identity, control.root, log.root, close.root, remote);
      return line;
    }));
    this.loadedEmpty.hidden = loaded.length > 0;
  }

  /** v2 set_sharing: the button says what it would do, the status counts, a row per browser. */
  private updateSharing(share: Share | undefined): void {
    this.sharing.root.hidden = !share;
    if (!share) return;
    const signature = JSON.stringify(share);
    if (this.sharing.root.dataset.signature === signature) return;
    this.sharing.root.dataset.signature = signature;
    this.shareButton.text = share.active ? "Stop sharing" : "Start sharing";
    this.shareButton.setRole(share.active ? "destructive" : "standard");
    this.shareAddress.value = share.urls[0] ?? "";
    this.shareAddress.title = share.urls.join("\n");
    this.shareAddressRow.hidden = !share.urls.length;
    const pending = share.clients.filter(client => client.status === "pending").length;
    const connected = share.clients.length - pending;
    this.shareStatus.textContent = share.error || (pending ? `${pending} awaiting approval · ${connected} connected`
      : connected ? `${connected} connected` : share.active ? "Waiting for browser requests." : "Sharing is off.");
    this.shareStatus.classList.toggle("error", Boolean(share.error));
    this.shareStatus.classList.toggle("warning", !share.error && pending > 0);
    const browserName = (agent: string): string =>
      ([["Edg", "Edge"], ["OPR/", "Opera"], ["Firefox/", "Firefox"], ["FxiOS/", "Firefox"], ["Chrome/", "Chrome"], ["CriOS/", "Chrome"], ["Safari/", "Safari"]] as const)
        .find(([marker]) => agent.includes(marker))?.[1] ?? "Browser";
    this.shareClients.replaceChildren(...share.clients.map(client => {
      const line = element("div", "list-row");
      const identity = element("div", "list-identity");
      const detail = element("span", `note list-detail${client.status === "pending" ? " warning" : ""}`,
        `${browserName(client.browser)} · ${client.status === "pending" ? "Pending" : "Connected"}`);
      detail.title = client.browser;
      identity.append(elided(client.address, "list-title"), detail);
      line.append(identity);
      if (client.status === "pending") line.append(command("Allow", () => this.ctx.transport.send({ action: "share_decide", client: client.id, allow: true }, "Allowing the browser…")).root);
      line.append(command(client.status === "pending" ? "Reject" : "Disconnect",
        () => this.ctx.transport.send({ action: "share_decide", client: client.id, allow: false }, "Disconnecting the browser…"), { role: "destructive" }).root);
      return line;
    }));
  }

  private openControl(id: string): void {
    const device = this.devices().find(item => item.instance_id === id);
    if (!device) return;
    // A published device is the other computer's: the workbench refuses its read in v2's words, no window opens.
    if (device.published) { this.ctx.transport.send({ action: "read_device", device: id }, `Reading ${device.role}…`); return; }
    // A sequencer's control is the Pulse Editor's window (v2 control_factory: PulseGUI over the session's sequencer).
    if (device.family === "sequencer") { this.ctx.show("pulse-editor"); return; }
    // A device with an editor of its own opens it (v2 control_factory: an SLM's Editor).
    if (device.control_window) { this.ctx.show(`${device.control_window}:${id}`); return; }
    // v2: the control is read first, then its window opens (its page shows it once the fields are in). The window reads
    // the device itself, opening and brought forward: one reader, never two readings superseding each other.
    this.ctx.show(`control:${id}`);
  }

  private openLog(id: string): void { this.ctx.show(`log:${id}`); }

  status(): void {
    const apparatus = this.apparatus();
    // v2: the strip says the last thing that happened (2 devices from apparatus.json, … initialized: …).
    const report = apparatus?.report;
    const [text, word] = queueStatus(this.ctx.transport, apparatus?.busy ? [apparatus.busy, "task"]
      : report?.text ? [report.text, report.severity] : apparatus ? ["Ready", "idle"] : ["The device workbench is not running in this build.", "warning"]);
    // v2's Device Manager refuses a command as a warning (orange); red is for what failed once started.
    this.strip.show(text, word === "error" && text === this.ctx.transport.lastError ? "warning" : word);
  }

  shown(): void { requestAnimationFrame(() => shareLabelColumn(this.left)); }
}

export const deviceManager = (ctx: ViewContext): View => new DeviceManagerView(ctx);
