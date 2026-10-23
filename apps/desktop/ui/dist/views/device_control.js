// Device Control (v2 DeviceControlView) and the device log, as pages of the
// Device Manager.  The page shows the instrument's fields as the workbench
// reads them (read_device) -- its reading, the desired value, the device's
// own limits -- and tunes one field per Apply (tune_device).  Who may tune
// what (claims, protected fields, accepted risk) is the workbench's answer,
// shown as each row's reason; the page never decides it.
import { Field, StatusStrip, alignUnitPickers, command, element, readoutText, setChecked, setElided, setState, statusDot, switchButton, titleRow, } from "../fluent/index.js";
import { queueStatus } from "../shell.js";
/** How long a live field waits after an edit before it tunes (v2: 75 ms). */
const LIVE_DELAY_MS = 75;
export class DeviceControlView {
    id;
    ctx;
    root = element("section", "device-control");
    owner;
    refresh = command("Refresh", () => this.read());
    risk;
    reason = element("p", "note wrap");
    table = element("div", "control-table");
    rows = new Map();
    strip = new StatusStrip();
    device = null;
    localRequest = -1;
    /** The strip's last word of its own (v2 show_status): "applied <field>" until the next read. */
    note = null;
    constructor(id, ctx) {
        this.id = id;
        this.ctx = ctx;
        this.risk = switchButton(on => ctx.transport.send({ action: "accept_device_risk", device: id, accepted: on }, on ? "Accepting the risk…" : "Withdrawing the risk…"), "Accept risk");
        this.risk.classList.add("row-switch");
        const title = titleRow("Owner: none", [this.refresh.root, this.risk]);
        this.owner = title.querySelector(".heading");
        const headings = ["Field", "Current", "Desired", "Device limits", "Live", "Apply", "Status"]
            .map((words, index) => element("span", `column-heading${index === 1 ? " right" : index === 2 ? " field" : index === 4 || index === 5 ? " centre" : ""}`, words));
        this.table.append(...headings);
        this.root.append(title, this.reason, this.table, this.strip.root);
    }
    read() { this.note = null; this.ctx.transport.send({ action: "read_device", device: this.id }, "refreshing device settings"); }
    update(device, request, failed = false) {
        this.device = device;
        // The owners by the names the console calls them; the policy's reason under them.
        const control = device.control;
        this.owner.textContent = `Owner: ${control?.owners.length ? control.owners.join(", ") : "none"}`;
        this.reason.textContent = control?.reason ?? "Reading the device…";
        setChecked(this.risk, Boolean(device.risk_accepted));
        this.risk.disabled = !control?.risk_enabled;
        const fields = device.fields ?? [];
        const alive = new Set(fields.map(field => field.name));
        for (const [name, row] of this.rows)
            if (!alive.has(name)) {
                this.dropRow(row);
                this.rows.delete(name);
            }
        for (const spec of fields) {
            let row = this.rows.get(spec.name);
            if (!row) {
                row = this.makeRow(spec);
                this.rows.set(spec.name, row);
            }
            // A field read in another spelling is a new number: the draft in the old one goes.
            if (row.spec.unit !== spec.unit) {
                row.draft = undefined;
                row.applied = "";
            }
            row.spec = spec;
            if (row.tuning !== null && row.tuning === request) {
                row.tuning = null;
                row.applied = failed ? "" : "Applied";
                row.error = failed ? this.ctx.transport.lastError : "";
                if (!failed) {
                    row.draft = undefined;
                    this.note = `applied ${spec.name}`;
                }
            }
            row.field.update(this.desiredSpec(row), request, failed);
            setElided(row.current, spec.current_text);
            row.limits.textContent = spec.limits_text;
            // Absent, not disabled: a field with no live write leaves its cell empty.
            row.live.hidden = !spec.live_write;
            setChecked(row.live, row.liveOn);
            row.live.disabled = !spec.editable;
            // Apply has something to do only while Desired differs from the reading.
            row.apply.disabled = !spec.editable || row.tuning !== null || row.draft === undefined || row.draft === spec.current;
            const [words, word] = row.tuning !== null ? ["Applying", "task"]
                : row.error ? [row.error, "error"]
                    : row.applied && spec.editable ? [row.applied, "idle"]
                        : [spec.status, spec.severity];
            setState(row.dot, word);
            setState(row.status, word);
            row.status.textContent = words;
            row.status.title = spec.reason || words;
            for (const cell of [row.live, row.apply])
                cell.title = spec.reason || "";
        }
        // One picker width, so every number box has the same right edge (v2 _align_headings).
        requestAnimationFrame(() => alignUnitPickers(this.table));
        this.status();
    }
    status() {
        // v2's words: ready (or that it has nothing to tune) once read, what the last Apply did.
        const fields = this.device?.fields;
        const [text, word] = queueStatus(this.ctx.transport, [!fields ? "refreshing device settings" : this.note ?? (fields.length ? "ready" : "No runtime controls"), "idle"]);
        this.strip.show(text, word);
    }
    desiredSpec(row) {
        const spec = row.spec;
        return { ...spec, value: row.draft !== undefined ? row.draft : spec.desired, unavailable_reason: spec.editable ? "" : spec.reason || "Read only", spin: true };
    }
    makeRow(spec) {
        const row = {};
        row.spec = spec;
        row.draft = undefined;
        row.liveOn = false;
        row.timer = 0;
        row.tuning = null;
        row.applied = "";
        row.error = "";
        row.field = new Field(`control-${this.id}`, { ...spec, value: spec.desired, spin: true }, {
            // Another spelling is the device's to read (an RF level depends on its
            // load): the row is re-read in it, reading, bounds and limits together.
            unit: unit => {
                window.clearTimeout(row.timer);
                return this.ctx.transport.send({ action: "read_device", device: this.id, units: { [row.spec.name]: unit } }, `Reading ${row.spec.label} in ${unit}…`);
            },
            commit: value => {
                row.draft = value;
                row.applied = "";
                row.error = "";
                if (row.liveOn) {
                    window.clearTimeout(row.timer);
                    row.timer = window.setTimeout(() => this.tune(row), LIVE_DELAY_MS);
                }
                // The draft is the page's until Apply: the field completes at once.
                const local = this.localRequest--;
                queueMicrotask(() => { if (this.device)
                    this.update(this.device, local); });
                return local;
            },
        });
        row.current = element("span", "control-current");
        row.limits = element("span", "note control-limits");
        row.limits.title = "The instrument's own range for this field; the editable window is the tighter of it and the bench policy";
        row.liveCell = element("span", "control-live");
        row.live = switchButton(on => { row.liveOn = on; if (!on)
            window.clearTimeout(row.timer); if (this.device)
            this.update(this.device); });
        row.liveCell.append(row.live);
        row.apply = command("Apply", () => this.tune(row)).root;
        const statusCell = element("span", "control-status");
        row.dot = statusDot();
        row.status = element("span", "status-text ink");
        statusCell.append(row.dot, row.status);
        // The row's cells in the table's columns: label, Current, the editor's
        // cell, Device limits, Live, Apply, Status; its problem spans the row.
        row.field.root.insertBefore(row.current, row.field.cell);
        row.field.root.insertBefore(row.limits, row.field.problem);
        row.field.root.insertBefore(row.liveCell, row.field.problem);
        row.field.root.insertBefore(row.apply, row.field.problem);
        row.field.root.insertBefore(statusCell, row.field.problem);
        this.table.append(row.field.root);
        return row;
    }
    tune(row) {
        window.clearTimeout(row.timer);
        const value = row.draft !== undefined ? row.draft : row.spec.desired;
        row.tuning = this.ctx.transport.send({ action: "tune_device", device: this.id, field: row.spec.name, value, ...(row.spec.unit ? { unit: row.spec.unit } : {}) }, `applying ${row.spec.name}`);
        row.error = "";
        if (this.device)
            this.update(this.device);
    }
    dropRow(row) { window.clearTimeout(row.timer); row.field.root.remove(); }
}
/** The device's own console (its stderr ring), read while the page is on
 * screen; a reader scrolled back into it stays where they were. */
export class DeviceLogView {
    id;
    ctx;
    root = element("section", "device-log");
    text = readoutText("No interactions recorded yet.");
    timer = 0;
    shown = "";
    constructor(id, ctx) {
        this.id = id;
        this.ctx = ctx;
        this.text.classList.add("log-text");
        this.root.append(this.text);
        this.timer = window.setInterval(() => { if (this.root.offsetParent)
            void this.poll(); }, 500);
        void this.poll();
    }
    async poll() {
        let result;
        try {
            result = await this.ctx.transport.request({ action: "get_device_log", device: this.id, lines: 2000 });
        }
        catch {
            return;
        }
        const lines = Array.isArray(result) ? result : Array.isArray(result?.lines) ? result.lines : typeof result === "string" ? result.split("\n") : [];
        const words = lines.length ? lines.map(String).join("\n") : "No interactions recorded yet.";
        if (words === this.shown)
            return;
        this.shown = words;
        const follow = this.text.scrollTop >= this.text.scrollHeight - this.text.clientHeight - 4;
        const position = this.text.scrollTop;
        this.text.textContent = words;
        this.text.scrollTop = follow ? this.text.scrollHeight : position;
    }
    stop() { window.clearInterval(this.timer); }
}
