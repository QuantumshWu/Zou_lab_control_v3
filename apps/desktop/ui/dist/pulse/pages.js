// The Pulse Editor's other pages: Preview (the native pulse timeline),
// Scan, Value Config, Subpulse Config and Target (v2 preview_view,
// scan_view, config_view, component_view, target_view).
import { ComboBox, EditableCombo, Menu, SpinBox, command, element, setChecked, setMiddle, setState, setText, setTitle, switchButton, token } from "../fluent/index.js";
import { Strip } from "./edit.js";
import { CanvasPointer } from "../views/pointer.js";
import { Drawn, RepeatsBox, TextBox, editCommand } from "./widgets.js";
// ------------------------------------------------------------------ Preview
export class PreviewPage {
    send;
    root = element("div", "pulse-preview");
    offRows = switchButton(on => this.send({ action: "pulse_preview", include_off: on }, "Drawing…"), "Show off rows");
    selectors = switchButton(on => this.send({ action: "pulse_preview", selectors: on }), "Selectors");
    // v2 activated: picking the size already drawn pins it too.
    size = new ComboBox(value => this.send({ action: "pulse_preview", size: value }, "Drawing…"), "", true);
    status = element("span", "pulse-status-text ink");
    saveFigure;
    canvas = element("canvas");
    context = this.canvas.getContext("2d");
    placeholder = element("div", "surface-placeholder", "Open a pulse to see its timeline");
    drawn = 0;
    axes = [];
    gestures = false;
    pointer;
    constructor(send) {
        this.send = send;
        this.saveFigure = command("Save figure", () => send({ action: "pulse_preview", save: true }, "Saving the preview…"));
        const bar = element("div", "card bordered pulse-bar pulse-preview-bar");
        const size = element("span", "pulse-label", "Size");
        bar.append(this.offRows, this.selectors, size, this.size.root, this.status, element("span", "spacer"), this.saveFigure.root);
        const surface = element("div", "pulse-preview-surface");
        this.canvas.hidden = true;
        surface.append(this.canvas, this.placeholder);
        this.root.append(bar, surface);
        // The timeline takes zoom, pan and selector drags while Selectors is on (v2 set_interaction_enabled).
        this.canvas.tabIndex = 0;
        this.pointer = new CanvasPointer(this.canvas, "pulse-preview", command => send(command), () => this.axes, () => this.drawn, () => this.gestures);
        // v2 PulsePreviewView.eventFilter: a wheel over the drawing is the drawing's, Selectors on or off; it never scrolls the page.
        this.canvas.addEventListener("wheel", event => { if (!this.gestures)
            event.preventDefault(); }, { passive: false });
    }
    /** A front for the preview panel, copied from its shared mapping. */
    draw(image, frame) {
        if (frame.sequence < this.drawn)
            return false;
        this.drawn = frame.sequence;
        if (this.canvas.width !== frame.width)
            this.canvas.width = frame.width;
        if (this.canvas.height !== frame.height)
            this.canvas.height = frame.height;
        this.context.putImageData(image, 0, 0);
        this.canvas.style.width = `${frame.width / (Number(document.documentElement.dataset.frameScale) || devicePixelRatio)}px`;
        this.canvas.hidden = false;
        this.placeholder.hidden = true;
        return true;
    }
    inputs = new Drawn();
    update(state) {
        if (this.inputs.same(state.preview, state.has_pulse))
            return;
        const preview = state.preview;
        setChecked(this.offRows, preview.include_off);
        setChecked(this.selectors, preview.selectors);
        this.axes = preview.axes ?? [];
        if (this.gestures && !preview.selectors)
            this.pointer.cancel();
        this.gestures = preview.selectors;
        this.size.setChoices(preview.sizes.map(size => [size, size]));
        const words = preview.status.split(" · ");
        const drawnSize = words[words.length - 1] ?? null;
        this.size.value = preview.size ?? (preview.sizes.includes(drawnSize ?? "") ? drawnSize : null);
        setText(this.status, preview.status);
        setTitle(this.status, preview.status);
        this.status.dataset.state = preview.status_word;
        if (!state.has_pulse) {
            this.canvas.hidden = true;
            this.placeholder.hidden = false;
        }
    }
}
// ------------------------------------------------------------------ Scan
export class ScanPage {
    send;
    root = element("div", "pulse-scan");
    slots = element("p", "row-line pulse-slots");
    bindings = element("div", "pulse-bindings");
    repeats;
    hold;
    back;
    forward;
    source = element("textarea", "pulse-source");
    sourceDraft = false;
    table = element("pre", "pulse-table");
    tableNote = element("p", "note wrap");
    progress = element("span", "status-text ink");
    run;
    constructor(send) {
        this.send = send;
        const bound = element("section", "card bordered");
        bound.append(element("div", "heading", "Bound fields"), this.slots, this.bindings);
        this.repeats = new RepeatsBox(repeats => send({ action: "pulse_scan", op: "repeats", repeats }, "Changing Scan repeats…"));
        this.hold = command("Hold current point", () => send({ action: "pulse_hold" }, "Holding..."));
        this.back = command("◀ Step", () => send({ action: "pulse_step", delta: -1 }, "Holding..."));
        this.forward = command("Step ▶", () => send({ action: "pulse_step", delta: 1 }, "Holding..."));
        // Qt draws v2's ◀ ▶ from its own fallback (a text-sized triangle), not Segoe UI Symbol's large one.
        this.back.root.classList.add("step");
        this.forward.root.classList.add("step");
        const playback = element("section", "card bordered");
        const playRow = element("div", "pulse-inline playback");
        playRow.append(element("span", "pulse-label", "Scan repeats"), this.repeats.root, this.hold.root, this.back.root, this.forward.root, this.progress);
        playback.append(element("div", "heading", "Playback"), playRow);
        this.source.spellcheck = false;
        // v2 _on_code_changed: every change of the program is the editor's draft at once (its Run says so).
        this.source.addEventListener("input", () => { this.sourceDraft = true; this.commitSource(); });
        this.source.addEventListener("blur", () => { this.sourceDraft = false; });
        this.run = command("Run", () => send({ action: "pulse_scan", op: "run" }, "Running the scan program…"), { role: "primary" });
        const program = element("section", "card bordered pulse-program");
        const programTop = element("div", "pulse-inline pulse-equal");
        this.programCommands = [
            command("Open program…", () => send({ action: "pulse_scan", op: "load_program" }, "Opening…")),
            command("Template: columns", () => send({ action: "pulse_scan", op: "template", kind: "columns" })),
            command("Template: grid", () => send({ action: "pulse_scan", op: "template", kind: "grid" })),
            this.run,
            command("Open array…", () => send({ action: "pulse_scan", op: "load_array" }, "Opening…")),
            command("Save array", () => send({ action: "pulse_scan", op: "save_array" }, "Saving…")),
            command("Clear array", () => send({ action: "pulse_scan", op: "clear" }), { role: "destructive", title: "Remove the current scan table; keep the Python draft" }),
        ];
        const [openProgram, columns, grid, , openArray, saveArray, clearArray] = this.programCommands;
        programTop.append(openProgram.root, columns.root, grid.root);
        const programBottom = element("div", "pulse-inline pulse-equal");
        programBottom.append(this.run.root, openArray.root, saveArray.root, clearArray.root);
        program.append(element("div", "heading", "Generate the scan table (Python)"), programTop, this.source, programBottom);
        const table = element("section", "card bordered pulse-table-card");
        const tableBody = element("div", "pulse-table-body");
        tableBody.append(this.tableNote, this.table);
        table.append(element("div", "heading", "Scan table"), tableBody);
        const lower = element("div", "pulse-scan-lower");
        lower.append(program, table);
        this.root.append(bound, playback, lower);
    }
    programCommands;
    commitSource() { this.send({ action: "pulse_scan", op: "edit_source", source: this.source.value }); }
    inputs = new Drawn();
    update(state) {
        if (this.inputs.same(state.scan, state.control, state.has_pulse))
            return;
        const scan = state.scan;
        setText(this.slots, scan.slots_text);
        const groups = new Map();
        for (const binding of scan.bindings) {
            const group = groups.get(binding.group_id) ?? { label: binding.group_label, members: [] };
            group.members.push(binding);
            groups.set(binding.group_id, group);
        }
        const titled = [...groups.keys()].some(Boolean);
        this.bindings.replaceChildren(...[...groups.entries()].sort(([a], [b]) => Number(Boolean(a)) - Number(Boolean(b))).flatMap(([, group]) => [
            ...(titled ? [element("div", "section-title", group.label || "Pulse")] : []),
            ...group.members.map(binding => {
                const words = [binding.scan ? "Scan" : "", binding.source === "api" ? "API" : binding.source === "config" ? "Config" : ""].filter(Boolean).join(" + ");
                // A mark that plays nothing today reads as a kept one: the whole line muted.
                return element("p", `row-line elided${binding.note ? " note" : ""}`, `${binding.label} · ${words}${binding.note ? ` — ${binding.note}` : ""}`);
            }),
        ]));
        this.repeats.show(scan.repeats, state.has_pulse);
        this.hold.disabled = !state.control.can_hold;
        this.back.disabled = this.forward.disabled = !state.control.can_step;
        if (!this.sourceDraft && document.activeElement !== this.source && this.source.value !== scan.source_text)
            this.source.value = scan.source_text;
        // v2 set_run_dirty / set_workspace_busy: Run* while the program is newer than its table; the program's commands wait while busy.
        this.run.setDirty(scan.source_dirty);
        for (const button of this.programCommands)
            button.disabled = scan.busy;
        setText(this.table, scan.table_text);
        this.table.hidden = !scan.table_text;
        setText(this.tableNote, scan.table_note);
        this.tableNote.hidden = !scan.table_note;
        setText(this.progress, scan.progress_text);
        setState(this.progress, scan.progress_word);
    }
}
/** v2 PulseConfigView: the value file (its commands, path and status, the
 * Name / Value / Unit table) and the Pulse's bindings (each field's Config
 * name, chosen or typed). */
export class ConfigPage {
    send;
    root = element("div", "pulse-config");
    mode = element("p", "row-line ink");
    bindings = element("div", "pulse-config-bindings");
    placeholder = element("p", "row-line", "No value bindings yet. Choose Config from a field's source control in Edit.");
    path = element("p", "row-line elided");
    active = element("p", "row-line elided");
    table = element("div", "pulse-config-table card bordered");
    rows = element("div", "pulse-config-rows");
    entries = [];
    picked = new Set();
    anchor = -1;
    current = null;
    editing = null;
    fileCommands;
    save;
    refresh;
    unload;
    add;
    removeRows;
    busy = false;
    bindingRows = new Map();
    groupTitles = new Map();
    header = element("div", "pulse-config-binding head");
    constructor(send) {
        this.send = send;
        const bindings = element("section", "card bordered");
        for (const word of ["Pulse field", "Config name", "Default", "Config value", "Status"])
            this.header.append(element("span", "", word));
        bindings.append(element("div", "heading", "Pulse bindings"), this.mode, element("p", "row-line elided", "Choose a Config name for each field. Save the Pulse to keep these bindings."), this.bindings);
        this.refresh = command("Refresh", () => send({ action: "pulse_config", op: "refresh" }, "Refreshing…"));
        this.save = command("Save", () => send({ action: "pulse_config", op: "save" }, "Saving the value file…"));
        this.unload = command("Unload", () => send({ action: "pulse_config", op: "unload" }, "Unloading…"), { role: "destructive" });
        this.fileCommands = [command("New file", () => send({ action: "pulse_config", op: "new" })), command("Open…", () => send({ action: "pulse_config", op: "load" }, "Opening…")),
            this.refresh, this.save, command("Save as…", () => send({ action: "pulse_config", op: "save", save_as: true }, "Saving…")), this.unload];
        const commands = element("div", "pulse-inline");
        commands.append(...this.fileCommands.map(button => button.root));
        const head = element("div", "pulse-config-row head");
        head.append(element("span", "", "Name"), element("span", "", "Value"), element("span", "", "Unit"));
        this.table.append(head, this.rows);
        this.table.tabIndex = 0;
        this.table.addEventListener("keydown", event => this.key(event));
        this.table.addEventListener("copy", event => { const text = this.copied(); if (!this.editing && text) {
            event.preventDefault();
            event.clipboardData?.setData("text/plain", text);
        } });
        this.table.addEventListener("paste", event => { if (this.editing || this.busy)
            return; event.preventDefault(); void this.paste(event.clipboardData?.getData("text/plain") ?? ""); });
        this.table.addEventListener("contextmenu", event => {
            event.preventDefault();
            if (this.busy)
                return;
            new Menu([{ label: "Copy", action: () => this.copy() }, { label: "Paste", action: () => void this.paste() }, { label: "Clear", action: () => this.clear() }]).openAt(event.clientX, event.clientY);
        });
        const edits = element("div", "pulse-inline");
        // v2 _add_entry: a new row, its Name being typed.
        this.add = command("Add parameter", () => { this.commit([...this.entries, ["", "", ""]]); this.pendingEdit = this.entries.length; });
        this.removeRows = command("Remove selected", () => { if (this.picked.size)
            this.commit(this.entries.filter((_, row) => !this.picked.has(row))); }, { role: "destructive" });
        edits.append(this.add.root, this.removeRows.root);
        const file = element("section", "card bordered");
        file.append(element("div", "heading", "Value file"), commands, this.path, this.active, this.table, edits);
        this.root.append(bindings, file);
    }
    /** The row whose Name starts being typed once the table holds it (Add parameter). */
    pendingEdit = -1;
    commit(entries) { this.send({ action: "pulse_config", op: "edit", rows: entries }, "Editing the value file…"); }
    // ---------------------------------------------------------- the value table (FluentTableView)
    select(row, event) {
        if (event.shiftKey && this.anchor >= 0) {
            if (!event.ctrlKey)
                this.picked.clear();
            for (let at = Math.min(this.anchor, row); at <= Math.max(this.anchor, row); at++)
                this.picked.add(at);
        }
        else if (event.ctrlKey) {
            if (this.picked.has(row))
                this.picked.delete(row);
            else
                this.picked.add(row);
            this.anchor = row;
        }
        else {
            this.picked = new Set([row]);
            this.anchor = row;
        }
        this.paintRows();
    }
    paintRows() { [...this.rows.children].forEach((node, row) => node.classList.toggle("selected", this.picked.has(row))); }
    drawRows() {
        this.rows.replaceChildren(...this.entries.map((entry, row) => {
            const node = element("div", "pulse-config-row");
            entry.forEach((value, column) => {
                const cell = element("span", "elided", value);
                cell.dataset.column = String(column);
                node.append(cell);
            });
            // A click picks the row (Ctrl adds, Shift spans); a click on the picked cell, or a double click, types into it.
            node.addEventListener("pointerdown", event => {
                if (this.busy || event.button !== 0 || event.target.closest("input"))
                    return;
                const column = Number(event.target.closest("[data-column]")?.dataset.column ?? 0);
                const again = this.picked.size === 1 && this.picked.has(row) && this.current?.[0] === row && this.current[1] === column;
                this.current = [row, column];
                this.select(row, event);
                this.table.focus({ preventScroll: true });
                if (again && !event.shiftKey && !event.ctrlKey)
                    setTimeout(() => this.edit(row, column, null));
            });
            node.addEventListener("dblclick", event => {
                if (this.busy)
                    return;
                this.edit(row, Number(event.target.closest("[data-column]")?.dataset.column ?? 0), null);
            });
            return node;
        }));
        this.paintRows();
    }
    /** One cell typed into: Enter or leaving commits, Escape cancels, Tab and the arrows move on (v2 delegate + FluentTableView.eventFilter). */
    edit(row, column, typed) {
        const cell = this.rows.children[row]?.children[column];
        if (!cell || this.busy)
            return;
        this.current = [row, column];
        if (!this.picked.has(row)) {
            this.picked = new Set([row]);
            this.anchor = row;
            this.paintRows();
        }
        const input = element("input");
        input.type = "text";
        input.spellcheck = false;
        input.value = typed ?? this.entries[row][column];
        cell.replaceChildren(input);
        this.editing = { row, column, input };
        let done = false;
        const finish = (keep, next) => {
            if (done)
                return;
            done = true;
            this.editing = null;
            const value = input.value;
            cell.textContent = keep ? value : this.entries[row][column];
            if (keep && value !== this.entries[row][column]) {
                const entries = this.entries.map(entry => [...entry]);
                entries[row][column] = value;
                this.commit(entries);
            }
            this.table.focus({ preventScroll: true });
            if (next)
                setTimeout(() => this.edit(next[0], next[1], null));
        };
        input.addEventListener("keydown", event => {
            event.stopPropagation();
            if (event.key === "Enter") {
                event.preventDefault();
                finish(true, null);
            }
            else if (event.key === "Escape") {
                event.preventDefault();
                finish(false, null);
            }
            else if (event.key === "Tab") {
                event.preventDefault();
                const step = event.shiftKey ? -1 : 1;
                const flat = row * 3 + column + step;
                finish(true, flat >= 0 && flat < this.entries.length * 3 ? [Math.floor(flat / 3), flat % 3] : null);
            }
            else if ((event.key === "ArrowUp" || event.key === "ArrowDown") && !event.shiftKey && !event.ctrlKey && !event.altKey) {
                const next = row + (event.key === "ArrowUp" ? -1 : 1);
                if (next >= 0 && next < this.entries.length) {
                    event.preventDefault();
                    finish(true, [next, column]);
                }
            }
        });
        input.addEventListener("blur", () => finish(true, null));
        input.focus();
        if (typed === null)
            input.select();
    }
    key(event) {
        if (this.editing || this.busy || event.target !== this.table)
            return;
        const control = event.ctrlKey || event.metaKey;
        if (event.key === "Delete" || event.key === "Backspace") {
            event.preventDefault();
            this.clear();
            return;
        }
        const current = this.current;
        if (!current)
            return;
        const [row, column] = current;
        const moves = { ArrowUp: [row - 1, column], ArrowDown: [row + 1, column], ArrowLeft: [row, column - 1], ArrowRight: [row, column + 1] };
        const moved = moves[event.key];
        if (moved) {
            event.preventDefault();
            const next = [Math.max(0, Math.min(this.entries.length - 1, moved[0])), Math.max(0, Math.min(2, moved[1]))];
            this.current = next;
            this.select(next[0], event);
        }
        else if (event.key === "F2") {
            event.preventDefault();
            this.edit(row, column, null);
        }
        else if (event.key.length === 1 && !control && !event.altKey) {
            event.preventDefault();
            this.edit(row, column, event.key);
        }
    }
    /** v2 copy_selection: the picked rows, tab-separated (Ctrl+C or the menu's Copy). */
    copied() {
        const rows = [...this.picked].sort((a, b) => a - b);
        return rows.map(row => this.entries[row].join("\t")).join("\n");
    }
    copy() { const text = this.copied(); if (text)
        void navigator.clipboard.writeText(text); }
    /** v2 paste_clipboard: a tab-separated block from the current cell, inside the table (Ctrl+V or the menu's Paste). */
    async paste(given) {
        const start = this.current ?? (this.picked.size ? [Math.min(...this.picked), 0] : null);
        if (!start)
            return;
        const text = given ?? await navigator.clipboard.readText();
        // str.splitlines: no empty line after a closing newline.
        const lines = text.replace(/\r\n?/g, "\n").replace(/\n$/, "").split("\n").map(line => line.split("\t"));
        const entries = this.entries.map(entry => [...entry]);
        lines.forEach((fields, dr) => fields.forEach((value, dc) => {
            const row = entries[start[0] + dr], column = start[1] + dc;
            if (row && column < 3)
                row[column] = value;
        }));
        this.commit(entries);
    }
    /** v2 clear_selection: the picked rows' cells, blank. */
    clear() {
        if (!this.picked.size)
            return;
        this.commit(this.entries.map((entry, row) => this.picked.has(row) ? ["", "", ""] : entry));
    }
    // ---------------------------------------------------------- the page
    inputs = new Drawn();
    update(state) {
        if (this.inputs.same(state.config))
            return;
        const config = state.config;
        this.busy = config.busy;
        // v2 set_config_mode: a note, or why Config cannot apply in the error ink.
        setText(this.mode, config.error || (config.enabled ? "Config is on. Saved sources are applied." : "Config is off. Pulse defaults are used."));
        setState(this.mode, config.error ? "error" : "idle");
        this.drawBindings(config);
        setText(this.path, config.file_path || "No value file selected");
        setTitle(this.path, config.file_path);
        this.path.classList.toggle("note", !config.file_path);
        const loaded = config.active_path ? `Loaded file: ${config.active_path.split(/[\\/]/).pop()}` : "No value file selected";
        setText(this.active, config.dirty ? `${loaded} · unsaved edits do not participate in Fire` : loaded);
        setTitle(this.active, config.active_path);
        for (const button of this.fileCommands)
            button.disabled = config.busy;
        this.refresh.disabled = !config.file_path || config.busy;
        this.save.setDirty(config.dirty);
        this.unload.disabled = !(config.active_path || config.file_path) || config.busy;
        this.add.disabled = this.removeRows.disabled = config.busy;
        this.table.classList.toggle("disabled", config.busy);
        // The table follows the file's entries; a cell being typed into is left alone until it is done.
        const entries = config.entries.map(row => [...row]);
        if (JSON.stringify(entries) !== JSON.stringify(this.entries) && !this.editing) {
            // Rows taken away or added: the picked and current numbers name other rows now, so
            // nothing stays picked or current -- no unseen row for a paste or a keystroke to land on.
            if (entries.length !== this.entries.length) {
                this.picked.clear();
                this.anchor = -1;
                this.current = null;
            }
            this.entries = entries;
            this.drawRows();
        }
        if (this.pendingEdit >= 0 && this.pendingEdit < this.entries.length) {
            const row = this.pendingEdit;
            this.pendingEdit = -1;
            this.picked = new Set([row]);
            this.anchor = row;
            this.edit(row, 0, null);
        }
    }
    /** v2 set_page's binding grid: a section title per group (when there are groups), one row per field. */
    drawBindings(config) {
        // No bindings: the grid says so under its header, across its columns.
        const groups = new Map(config.binding_groups.map(([fieldId, groupId, label]) => [fieldId, [groupId, label]]));
        const grouped = new Map();
        for (const binding of config.bindings) {
            const [groupId, label] = groups.get(binding[0]) ?? ["", "Pulse"];
            const group = grouped.get(groupId) ?? { label, rows: [] };
            group.rows.push(binding);
            grouped.set(groupId, group);
        }
        const order = [...grouped.keys()].sort((a, b) => Number(Boolean(a)) - Number(Boolean(b)));
        const titled = order.some(Boolean);
        const nodes = [this.header, ...(config.bindings.length ? [] : [this.placeholder])];
        for (const groupId of order) {
            const group = grouped.get(groupId);
            if (titled) {
                let title = this.groupTitles.get(groupId);
                if (!title) {
                    title = element("div", "section-title pulse-config-group");
                    this.groupTitles.set(groupId, title);
                }
                setText(title, group.label || "Pulse");
                nodes.push(title);
            }
            for (const [fieldId, label, key, defaultText, value, status] of group.rows) {
                let row = this.bindingRows.get(fieldId);
                if (!row) {
                    const name = new EditableCombo(text => this.send({ action: "pulse_config", op: "binding", field_id: fieldId, key: text }, "Naming the Config value…"), "Config name");
                    const field = element("span", "elided"), status = element("span", "elided");
                    const values = [element("span", "pulse-config-value"), element("span", "pulse-config-value")];
                    const node = element("div", "pulse-config-binding");
                    node.append(field, name.root, ...values, status);
                    row = { row: node, field, name, values, status };
                    this.bindingRows.set(fieldId, row);
                }
                for (const [node, text] of [[row.field, label], [row.status, status]]) {
                    setText(node, text);
                    setTitle(node, text);
                }
                // The two values are readouts (v2 FluentReadoutEdit): a long one gives up its middle.
                setMiddle(row.values[0], defaultText);
                setMiddle(row.values[1], value);
                // A field that kept its Config name under Default or API reads muted: the name is there to switch back to.
                row.row.classList.toggle("parked", config.parked.includes(fieldId));
                row.name.fill(config.available_names, key);
                row.name.disabled = config.busy;
                nodes.push(row.row);
            }
        }
        for (const fieldId of [...this.bindingRows.keys()])
            if (!config.bindings.some(([id]) => id === fieldId))
                this.bindingRows.delete(fieldId);
        for (const groupId of [...this.groupTitles.keys()])
            if (!titled || !grouped.has(groupId))
                this.groupTitles.delete(groupId);
        if (nodes.length !== this.bindings.children.length || nodes.some((node, at) => this.bindings.children[at] !== node))
            this.bindings.replaceChildren(...nodes);
    }
}
// ------------------------------------------------------------------ Subpulse Config
export class SubpulsePage {
    send;
    root = element("div", "pulse-subpulse");
    choose = new ComboBox(value => this.send({ action: "pulse_subpulse", op: "context", component_id: value ?? "" }, "Opening the Subpulse…"));
    commands;
    fileRow = element("div", "pulse-inline pulse-subpulse-row");
    sourcePath;
    browse;
    pathLine = element("p", "row-line elided");
    scope = element("p", "row-line elided");
    status = element("span", "status-text ink");
    hint = element("div", "pulse-inline");
    toolbar = element("div", "pulse-inline pulse-subpulse-toolbar");
    hidden = new ComboBox(() => { }, "All ports shown");
    addPort;
    strip;
    bindings = element("section", "card bordered pulse-subpulse-bindings");
    bindingRows = element("div", "pulse-subpulse-binding-grid");
    placeholder = element("div", "pulse-placeholder", "Choose a Subpulse above, or create or open a Subpulse file.");
    body = element("div", "pulse-subpulse-body");
    page = null;
    bindingRowsShown = new Map();
    constructor(send, say, edit) {
        this.send = send;
        const op = (name, extra = {}, feedback) => () => send({ action: "pulse_subpulse", op: name, component_id: this.page?.selected_id ?? "", ...extra }, feedback);
        this.commands = {
            new: command("New file", op("new")), open: command("Open…", op("open", {}, "Opening…")),
            save: command("Save", op("save", {}, "Saving…")),
            // An instance in the Pulse is exported to a file; an independent file is saved as another (v2 component_view).
            save_as: command("Save as…", () => send({ action: "pulse_subpulse", op: this.page?.context_id ? "export" : "save_as", component_id: this.page?.context_id ?? "" }, "Saving…")),
            // v2 component_action("insert"): where the Edit page's selection says, then the new block's name ready to type.
            insert: command("Insert into Pulse", () => { edit.expectComponent(); op("insert", { before_item: edit.before() }, "Inserting…")(); }),
        };
        const head = element("div", "card bordered pulse-subpulse-head");
        const top = element("div", "pulse-inline pulse-subpulse-row");
        const label = (text) => {
            const node = element("span", "pulse-label");
            node.append(element("span", "", text), element("span", "sizer", text === "Subpulse" ? "Subpulse file" : "Subpulse"));
            return node;
        };
        top.append(label("Subpulse"), this.choose.root, ...Object.values(this.commands).map(button => button.root));
        this.sourcePath = new TextBox(text => {
            const record = this.record();
            if (record && text.trim() !== record.path)
                send({ action: "pulse_subpulse", op: "source", component_id: record.component_id, source: record.source, path: text.trim() }, "Choosing the Subpulse file…");
        });
        this.sourcePath.root.placeholder = "Choose a Subpulse file";
        this.browse = command("Browse…", () => {
            const record = this.record();
            if (record)
                send({ action: "pulse_subpulse", op: "source", component_id: record.component_id, source: record.source, browse: true }, "Choosing the Subpulse file…");
        });
        this.fileRow.append(label("Subpulse file"), this.sourcePath.root, this.browse.root);
        this.hint.append(this.scope, this.status);
        head.append(top, this.fileRow, this.pathLine, this.hint);
        // The schedule's own commands stand above its scrolling body (embedded PulseScheduleView.toolbar).
        this.strip = new Strip(send, "subpulse", say);
        this.addPort = command("Add", () => {
            const ports = this.page?.document?.ports ?? [];
            const key = this.hidden.value;
            if (typeof key === "string")
                this.send(editCommand("subpulse", "visible_ports", { ports: [...ports.filter(port => port.visible).map(port => port.key), key] }), "Showing the output…");
        });
        this.toolbar.append(command("Add period", () => this.strip.insert("period")).root, command("Add spacer", () => this.strip.insert("spacer")).root, command("Add bracket", () => this.strip.addBracket()).root, command("Remove", () => this.strip.remove(), { role: "destructive" }).root, element("span", "spacer"), this.hidden.root, this.addPort.root, command("Hide off", () => this.strip.hideOff(), { title: "Hide the ports no period drives" }).root, command("Show all", () => this.strip.showAll()).root);
        this.bindings.append(element("div", "heading", "Subpulse bindings"), this.bindingRows);
        this.body.append(this.strip.root, this.bindings);
        this.root.append(head, this.toolbar, this.body, this.placeholder);
    }
    /** The selected Subpulse's source row (id, source, path, status). */
    record() {
        const page = this.page;
        return page?.sources.find(row => row.component_id === page.selected_id);
    }
    inputs = new Drawn();
    update(state) {
        if (this.inputs.same(state.subpulse))
            return;
        const page = this.page = state.subpulse;
        this.choose.setChoices([["", "Independent file"], ...page.contexts.filter(([id]) => id).map(([id, name]) => [id, name])]);
        this.choose.value = page.selected_id;
        const schedule = page.document;
        const instance = Boolean(page.context_id);
        this.commands.save.text = instance ? "Save Pulse" : "Save";
        this.commands.save_as.text = instance ? "Export file..." : "Save as...";
        this.commands.insert.text = instance ? "Duplicate in Pulse" : "Insert into Pulse";
        for (const [name, button] of Object.entries(this.commands))
            button.disabled = page.busy || (!schedule && name !== "new" && name !== "open");
        this.commands.save.setDirty(page.dirty);
        const record = this.record();
        this.fileRow.hidden = !record;
        if (record) {
            if (document.activeElement !== this.sourcePath.root)
                this.sourcePath.show(record.path, !page.busy);
            setTitle(this.sourcePath.root, record.path || "Choose a Subpulse file");
            this.browse.disabled = page.busy;
            setText(this.status, record.status);
            setTitle(this.status, record.status);
            setState(this.status, record.word);
        }
        this.status.hidden = !(record && record.status && !(page.context_id && record.source !== "config"));
        setText(this.pathLine, page.path || (instance ? "Embedded in the current Pulse" : "Unsaved Subpulse"));
        setTitle(this.pathLine, page.path);
        this.pathLine.hidden = !schedule || Boolean(page.selected_id);
        setText(this.scope, instance ? "Default in Pulse — Save Pulse keeps these edits." : page.selected_id ? "Config source — Save writes this Subpulse file." : "Independent file — Save writes this file.");
        this.hint.hidden = !schedule;
        this.toolbar.hidden = this.body.hidden = !schedule;
        this.root.classList.toggle("open", Boolean(schedule));
        this.placeholder.hidden = Boolean(schedule);
        if (schedule) {
            this.strip.update(state, schedule, schedule.item_order ?? [], !page.busy);
            const hidden = schedule.ports.filter(port => !port.visible);
            this.hidden.setChoices(hidden.map(port => [port.key, port.label]));
            if (!hidden.length)
                this.hidden.value = null;
            else if (this.hidden.value === null || !hidden.some(port => port.key === this.hidden.value))
                this.hidden.value = hidden[0].key;
            this.hidden.disabled = !hidden.length;
            this.addPort.disabled = !hidden.length;
        }
        // The bindings: a field's label beside its Config name, chosen or typed (v2 Subpulse bindings).
        this.bindings.hidden = !page.bindings.length;
        const nodes = page.bindings.flatMap(([fieldId, label, key]) => {
            let row = this.bindingRowsShown.get(fieldId);
            if (!row) {
                const name = new EditableCombo(text => this.send(editCommand("subpulse", "config_binding", { field_id: fieldId, key: text }), "Naming the Config value…"), "Config name");
                row = [element("span", "pulse-label elided", label), name];
                this.bindingRowsShown.set(fieldId, row);
            }
            setText(row[0], label);
            row[1].fill([...new Set(page.config_names)], key);
            row[1].disabled = page.busy;
            return [row[0], row[1].root];
        });
        for (const fieldId of [...this.bindingRowsShown.keys()])
            if (!page.bindings.some(([id]) => id === fieldId))
                this.bindingRowsShown.delete(fieldId);
        if (nodes.length !== this.bindingRows.children.length || nodes.some((node, at) => this.bindingRows.children[at] !== node))
            this.bindingRows.replaceChildren(...nodes);
    }
}
// ------------------------------------------------------------------ Target
/** v2 PulseTargetView: the digital and DAC rows of the wiring, its names
 * always the operator's, its topology only offline; one line says what the
 * page shows or what the last Apply or check found. */
export class TargetPage {
    send;
    root = element("div", "pulse-target");
    status = element("span", "pulse-status-text ink");
    digital = element("section", "card bordered pulse-target-card");
    dac = element("section", "card bordered pulse-target-card");
    lists = element("div", "pulse-target-lists");
    addDigital;
    addDac;
    apply;
    records = [];
    /** Each DAC row's Width box (v2 keeps it beside the endpoints it describes). */
    widths = new Map();
    source = "";
    editable = false;
    dirty = false;
    reserved = new Set();
    /** A refusal of the page's own (a width that does not match its endpoints), until the page is next shown. */
    local = null;
    /** A DAC's bus width rule (v2 TargetWidthRule: minimum, default, maximum). */
    dacRule = [1, 2, null];
    constructor(send) {
        this.send = send;
        this.addDigital = command("Add digital", () => this.add("digital"));
        this.addDac = command("Add DAC", () => this.add("dac"));
        this.apply = command("Apply target", () => { const records = this.current(); if (records)
            send({ action: "pulse_target", records }, "Applying the target…"); }, { role: "primary" });
        const bar = element("div", "card bordered pulse-bar pulse-target-bar");
        bar.append(this.status, element("span", "spacer"), this.addDigital.root, this.addDac.root, this.apply.root);
        this.lists.append(this.digital, this.dac);
        this.root.append(bar, this.lists);
    }
    width(record) { return record.kind === "dac" ? this.widths.get(record.key) ?? Math.max(this.dacRule[0], record.endpoints.length) : 1; }
    /** v2 _current_records: the rows as typed, refused where a width does not match its endpoints. */
    current() {
        for (const record of this.records) {
            const width = this.width(record);
            const listed = record.endpoints.filter(Boolean).length;
            if (listed !== width) {
                this.say(`${record.signal.trim() || record.key}: width is ${width} but ${listed} endpoints are listed`);
                return null;
            }
        }
        return this.records.map(record => ({ ...record, signal: record.signal.trim(), endpoints: record.endpoints.filter(Boolean),
            clock_endpoint: record.kind === "dac" ? (record.clock_endpoint ?? "").trim() : null }));
    }
    say(text) { this.local = text; this.showStatus(); }
    /** v2 _allocate_key: the first free `<stem>_<n>`, past every row's key, its clock and the target's own. */
    allocate(stem) {
        const used = new Set([...this.records.flatMap(record => [record.key, record.clock_key ?? ""]), ...this.reserved]);
        let index = 1;
        while (used.has(`${stem}_${index}`))
            index++;
        return `${stem}_${index}`;
    }
    /** v2 _add_digital / _add_dac: a named, wired row (a digital before the DACs), shown at once. */
    add(kind) {
        if (!this.current())
            return;
        const key = this.allocate(kind);
        if (kind === "digital") {
            const split = this.records.findIndex(record => record.kind === "dac");
            const record = { key, kind, signal: key, endpoints: [`endpoint:${key}`], clock_key: null, clock_endpoint: null };
            this.records.splice(split < 0 ? this.records.length : split, 0, record);
        }
        else {
            const clock = this.allocate(`${key}_clock`);
            const width = this.dacRule[1];
            this.records.push({ key, kind, signal: key, endpoints: Array.from({ length: width }, (_, bit) => `endpoint:${key}[${bit}]`), clock_key: clock, clock_endpoint: `endpoint:${clock}` });
            this.widths.set(key, width);
        }
        this.touch();
        // QScrollArea.ensureWidgetVisible: the row in view with its 50 px margin.
        const row = this.lists.querySelector(`[data-key="${CSS.escape(key)}"]`);
        if (row) {
            const margin = 50 * token("--edge"), view = this.lists.getBoundingClientRect(), box = row.getBoundingClientRect();
            if (box.bottom + margin > view.bottom)
                this.lists.scrollTop += box.bottom + margin - view.bottom;
            else if (box.top - margin < view.top)
                this.lists.scrollTop -= view.top - (box.top - margin);
        }
    }
    touch() { this.dirty = true; this.local = null; this.render(); this.showStatus(); }
    render() {
        const digital = this.records.filter(record => record.kind === "digital");
        const dac = this.records.filter(record => record.kind === "dac");
        const header = (title, columns, kind = "") => {
            const row = element("div", `pulse-target-row head ${kind}`.trim());
            for (const column of columns)
                row.append(element("span", "", column));
            return [element("div", "heading", title), row];
        };
        // The row remover every list has (v2 FluentButton "×", destructive): the ghost ×, red under the pointer.
        const remove = (record) => {
            const cross = command("×", () => { if (!this.current())
                return; this.records = this.records.filter(item => item !== record); this.touch(); }, { role: "destructive", glyph: true, square: true, title: "Remove this port row" });
            cross.disabled = !this.editable;
            return cross.root;
        };
        const edit = (value, changed) => {
            const box = new TextBox(text => { changed(text); this.dirty = true; });
            box.show(value, true);
            return box.root;
        };
        this.digital.replaceChildren(...header(`Digital outputs · ${digital.length}`, ["Signal name", "Endpoint"]), ...digital.map(record => {
            const row = element("div", "pulse-target-row");
            row.dataset.key = record.key;
            const endpoint = edit(record.endpoints.join(", "), text => { record.endpoints = text.split(",").map(part => part.trim()).filter(Boolean); });
            endpoint.disabled = !this.editable;
            row.append(edit(record.signal, text => { record.signal = text; }), endpoint, remove(record));
            return row;
        }));
        const [lowest, , highest] = this.dacRule;
        this.dac.replaceChildren(...header(`DAC outputs · ${dac.length}`, ["Signal name", "Width", "Data endpoints (bit order)", "Latch endpoint"], "dac"), ...dac.map(record => {
            const row = element("div", "pulse-target-row dac");
            row.dataset.key = record.key;
            const endpoints = edit(record.endpoints.join(", "), text => { record.endpoints = text.split(",").map(part => part.trim()).filter(Boolean); });
            const clock = edit(record.clock_endpoint ?? "", text => { record.clock_endpoint = text; });
            endpoints.disabled = clock.disabled = !this.editable;
            // v2 _resize_dac_endpoints: the Width box grows the list with placeholder endpoints or cuts it.
            const resize = (value) => {
                const width = Math.max(lowest, Math.min(highest ?? value, Math.round(value)));
                if (!Number.isFinite(width))
                    return;
                this.widths.set(record.key, width);
                const listed = record.endpoints;
                record.endpoints = listed.length < width ? [...listed, ...Array.from({ length: width - listed.length }, (_, bit) => `endpoint:${record.key}[${listed.length + bit}]`)] : listed.slice(0, width);
                endpoints.value = record.endpoints.join(", ");
                this.dirty = true;
                count.show(String(width), this.editable);
                spin.refresh();
            };
            const width = element("div", "pulse-dac-width");
            const count = new TextBox(text => resize(Number(text)));
            count.numeric = { kind: "int", lo: lowest, hi: highest, resolution: 0, allowAny: true };
            count.root.inputMode = "numeric";
            width.append(count.root);
            const spin = new SpinBox(width, count.root, { step: () => 1, bounds: () => [lowest, highest], commit: resize });
            count.show(String(this.width(record)), this.editable);
            spin.refresh();
            row.append(edit(record.signal, text => { record.signal = text; }), width, endpoints, clock, remove(record));
            return row;
        }));
    }
    target = null;
    /** v2's one status line: the page's caption, until the last Apply (or the page's own check) says something. */
    showStatus() {
        const target = this.target;
        const [text, word] = this.local !== null ? [this.local, "error"] : target?.feedback ? [target.feedback, target.feedback_word] : [target?.caption ?? "", "idle"];
        setText(this.status, text);
        setTitle(this.status, text);
        this.status.dataset.state = word;
    }
    inputs = new Drawn();
    update(state) {
        if (this.inputs.same(state.target, state.has_pulse))
            return;
        const target = this.target = state.target;
        this.local = null;
        this.editable = target.editable;
        this.reserved = new Set(target.reserved);
        const rule = target.width_rules?.dac;
        this.dacRule = rule ? [rule[0] ?? 1, rule[1] ?? 2, rule[2] ?? null] : [1, 2, null];
        this.addDigital.disabled = this.addDac.disabled = !target.editable;
        const source = JSON.stringify(target.records);
        if (source !== this.source && (!this.dirty || target.feedback_word === "task")) {
            this.source = source;
            this.records = target.records.map(record => ({ ...record, endpoints: [...record.endpoints] }));
            // v2 _reconcile_row: the Width box says how many endpoints the row has.
            this.widths = new Map(this.records.filter(record => record.kind === "dac").map(record => [record.key, Math.max(1, record.endpoints.length)]));
            this.dirty = false;
            if (!this.root.contains(document.activeElement))
                this.render();
        }
        else if (!this.root.contains(document.activeElement))
            this.render();
        this.showStatus();
    }
}
