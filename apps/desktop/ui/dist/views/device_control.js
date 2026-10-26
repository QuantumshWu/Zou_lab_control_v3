// Device Control (v2 DeviceControlView) and the device log, as pages of the
// Device Manager.  The page shows the instrument's fields as the workbench
// reads them (read_device) -- its reading, the desired value, the device's
// own limits -- and tunes one field per Apply (tune_device).  Who may tune
// what (claims, protected fields, accepted risk) is the workbench's answer,
// shown as each row's reason; the page never decides it.
import { Field, StatusStrip, alignUnitPickers, command, element, readoutText, setChecked, setElided, setState, setTitle, statusDot, switchButton, titleRow, } from "../fluent/index.js";
import { queueStatus } from "../shell.js";
/** How long a live field waits after an edit before it tunes (v2: 75 ms). */
const LIVE_DELAY_MS = 75;
/** Device limits' tooltip: what the column is. */
const LIMITS_TIP = "The instrument's own range for this field; the editable window is the tighter of it and the bench policy";
/** A draft the field's bounds admit (a whole number for an integer field). */
const within = (value, spec) => typeof value === "number"
    && (spec.minimum == null || value >= spec.minimum) && (spec.maximum == null || value <= spec.maximum) && (spec.kind !== "integer" || Number.isInteger(value));
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
    /** v2's tune worker: one write at a time, with only the latest waiting value per field. */
    active = null;
    queued = new Map();
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
    update(device, request, failed = false, result) {
        this.device = device;
        const completed = this.active?.request === request ? this.active : null;
        if (completed) {
            this.active = null;
            completed.row.tuning = null;
        }
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
            // A field read in another spelling: a displayed quantity is converted, a number the
            // operator wrote keeps its digits.  The reading, and a Desired that only mirrors it
            // (no draft, or the one just applied), are the device's re-read; a draft not yet
            // applied is now read in the new unit, checked against its bounds there, and waits
            // for the next edit or Apply -- a unit change never tunes by itself.
            const read = row.unitRequest !== null && row.unitRequest === request;
            if (read)
                row.unitRequest = null;
            if (row.spec.unit !== spec.unit || (read && !failed)) {
                const kept = row.draft !== undefined && row.draft !== row.spec.current && !row.applied;
                this.cancelLive(row);
                this.queued.delete(spec.name);
                row.revision++;
                row.applied = "";
                row.error = "";
                if (!kept)
                    row.draft = undefined;
                row.valid = !kept || within(row.draft, spec);
                row.outside = !row.valid;
                row.field.acceptEdit();
            }
            row.spec = spec;
            if (!spec.editable || device.state !== "open") {
                this.cancelLive(row);
                this.queued.delete(spec.name);
            }
            else if (!spec.live_write)
                this.cancelLive(row);
            if (completed?.row === row) {
                if (!failed)
                    this.note = `applied ${spec.name}`;
                if (row.revision === completed.revision && (spec.unit ?? "") === completed.unit && row.unitRequest === null) {
                    row.applied = failed ? "" : "Applied";
                    row.error = failed ? this.ctx.transport.lastError : "";
                    if (!failed) {
                        const reply = result;
                        row.draft = reply?.device === this.id && reply.readback !== undefined ? reply.readback : completed.value;
                        row.field.acceptEdit();
                    }
                }
            }
            row.field.update(this.desiredSpec(row), request, failed);
            if (row.outside && row.field.input instanceof HTMLInputElement)
                row.field.input.classList.add("invalid");
            // A reading shown rounded ("≈ -23.98 dBm") says every digit in its tooltip.
            setElided(row.current, spec.current_text, spec.current_exact ?? spec.current_text);
            row.limits.textContent = spec.limits_text;
            setTitle(row.limits, spec.limits_exact && spec.limits_exact !== spec.limits_text ? `${spec.limits_exact}\n${LIMITS_TIP}` : LIMITS_TIP);
            // Absent, not disabled: a field with no live write leaves its cell empty.
            row.live.hidden = !spec.live_write;
            setChecked(row.live, row.liveOn);
            row.live.disabled = !spec.editable || row.unitRequest !== null;
            // Apply has something to do only while Desired differs from the reading.
            row.apply.disabled = !spec.editable || !row.valid || row.unitRequest !== null || row.tuning !== null || row.draft === undefined || row.draft === spec.current;
            const waiting = this.queued.has(spec.name);
            const [words, word] = row.unitRequest !== null ? ["Reading unit", "task"]
                : row.tuning !== null ? [waiting ? "Applying; latest queued" : "Applying", "task"]
                    : waiting ? ["Queued latest", "task"]
                        : row.outside ? ["Desired is outside its bounds", "error"]
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
        this.drain();
    }
    status() {
        // v2's words: ready (or that it has nothing to tune) once read, what the last Apply did.
        const fields = this.device?.fields;
        const [text, word] = queueStatus(this.ctx.transport, [!fields ? "refreshing device settings" : this.note ?? (fields.length ? "ready" : "No runtime controls"), "idle"]);
        this.strip.show(text, word);
    }
    desiredSpec(row) {
        const spec = row.spec;
        return { ...spec, value: row.draft !== undefined ? row.draft : spec.desired, unavailable_reason: row.unitRequest !== null ? "Reading the new unit" : spec.editable ? "" : spec.reason || "Read only", spin: true, inexact: true };
    }
    makeRow(spec) {
        const row = {};
        row.spec = spec;
        row.draft = undefined;
        row.liveOn = false;
        row.timer = 0;
        row.tuning = null;
        row.revision = 0;
        row.valid = true;
        row.unitRequest = null;
        row.applied = "";
        row.error = "";
        row.outside = false;
        row.field = new Field(`control-${this.id}`, { ...spec, value: spec.desired, spin: true }, {
            // Another spelling is the device's to read (an RF level depends on its
            // load): the row is re-read in it, reading, bounds and limits together
            // (a draft keeps its digits: update).
            unit: unit => {
                this.cancelLive(row);
                this.queued.delete(row.spec.name);
                row.revision++;
                row.unitRequest = this.ctx.transport.send({ action: "read_device", device: this.id, units: { [row.spec.name]: unit } }, `Reading ${row.spec.label} in ${unit}…`);
                queueMicrotask(() => { if (this.device)
                    this.update(this.device); });
                return row.unitRequest;
            },
            edited: value => { this.edit(row, value); queueMicrotask(() => { if (this.device)
                this.update(this.device); }); },
            cancel: () => {
                this.cancelLive(row);
                this.queued.delete(row.spec.name);
                row.revision++;
                row.draft = undefined;
                row.valid = true;
                row.error = "";
                row.applied = "";
                row.outside = false;
                if (this.device)
                    this.update(this.device);
            },
            commit: value => {
                this.edit(row, value);
                // An explicit value/step completes the local Field, not the device write.
                const local = this.localRequest--;
                queueMicrotask(() => { if (this.device)
                    this.update(this.device, local); });
                return local;
            },
        });
        row.current = element("span", "control-current");
        row.limits = element("span", "note control-limits");
        row.liveCell = element("span", "control-live");
        row.live = switchButton(on => { row.liveOn = on; if (!on)
            this.cancelLive(row); if (this.device)
            this.update(this.device); });
        row.liveCell.append(row.live);
        row.apply = command("Apply", () => this.tune(row, false)).root;
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
    edit(row, value) {
        row.outside = false;
        const valid = value !== undefined;
        if (row.valid === valid && (!valid || Object.is(row.draft, value)))
            return;
        row.revision++;
        row.valid = valid;
        row.applied = "";
        row.error = "";
        if (valid)
            row.draft = value;
        this.cancelLive(row);
        if (valid && row.liveOn && row.spec.live_write && row.spec.editable && row.unitRequest === null) {
            row.timer = window.setTimeout(() => { row.timer = 0; this.tune(row, true); }, LIVE_DELAY_MS);
        }
    }
    cancelLive(row) {
        window.clearTimeout(row.timer);
        row.timer = 0;
        if (this.queued.get(row.spec.name)?.live)
            this.queued.delete(row.spec.name);
    }
    tune(row, live) {
        window.clearTimeout(row.timer);
        row.timer = 0;
        if (!row.valid || !row.spec.editable || row.unitRequest !== null || this.device?.state !== "open"
            || (live && (!row.liveOn || !row.spec.live_write)))
            return;
        const value = row.draft !== undefined ? row.draft : row.spec.desired;
        this.queued.set(row.spec.name, { row, value, unit: row.spec.unit ?? "", revision: row.revision, live });
        row.error = "";
        if (this.device)
            this.update(this.device);
    }
    drain() {
        if (this.active)
            return;
        for (const [name, next] of this.queued) {
            this.queued.delete(name);
            const row = this.rows.get(name);
            if (row !== next.row || !row.spec.editable || row.unitRequest !== null || (row.spec.unit ?? "") !== next.unit
                || this.device?.state !== "open" || (next.live && (!row.liveOn || !row.spec.live_write || !row.valid || row.revision !== next.revision)))
                continue;
            const request = this.ctx.transport.send({ action: "tune_device", device: this.id, field: name, value: next.value,
                ...(next.unit ? { unit: next.unit } : {}) }, `applying ${name}`);
            this.active = { ...next, request };
            row.tuning = request;
            if (this.device)
                this.update(this.device);
            return;
        }
    }
    dropRow(row) { this.cancelLive(row); this.queued.delete(row.spec.name); row.field.root.remove(); }
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
