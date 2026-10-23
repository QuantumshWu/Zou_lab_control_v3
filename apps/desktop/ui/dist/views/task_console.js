// TaskConsole@Zou lab: the header (identity, the one chooser and its Add,
// Selectors, Pause, Save…, Open…, Clear), the status strip with Stop task,
// and the Monitor / Logic / Edit pages.  The board, the rows and the pages
// draw the workbench's state; every button is one §4 command.
import { ComboBox, Dialog, Field, StatusStrip, TabWidget, command, confirm, element, note, setChecked, setElided, setState, statusDot, switchButton, } from "../fluent/index.js";
import { queueStatus, saveImage } from "../shell.js";
import { LogicEditorView, LogicRowView, PublisherEditorView, projectRow } from "./logic.js";
import { PanelEditView } from "./panel_edit.js";
const ADD_WORDS = { plot: "Add panel", logic: "Add node" };
/** A node type in the chooser: its layer, then its name (v2 "Processor: occupancy"). */
const LAYER_WORDS = { measurement: "Measurement", processor: "Processor", task: "Task", panel_publisher: "Panel publisher" };
/** TASK_CONSOLE_PANEL_KINDS: what the console's chooser offers, in v2's order. */
const CONSOLE_PLOT_KINDS = ["image", "curve", "rolling", "histogram", "facet_grid"];
const SEVERITY = { idle: 0, task: 0, running: 1, warning: 2, error: 3 };
/** v2 _running_logic: a node still at work, started or waiting to start (a follower waiting for its source is not). */
const working = (row) => row.phase !== "idle" && row.phase !== "following";
const counted = (count, noun) => `${count} ${noun}${count === 1 ? "" : "s"}`;
export class TaskConsoleView {
    ctx;
    board;
    key = "task-console";
    title = "Task Console";
    windowTitle = "TaskConsole@Zou lab";
    root = element("div", "window task-console");
    dot = statusDot();
    summary = element("span", "summary");
    kinds = new ComboBox(() => { this.kindPicked = true; this.updateAdd(); });
    /** The operator chose a kind: a refill keeps it; until then the first (v2 "Plot: image") is shown. */
    kindPicked = false;
    add;
    selectors;
    onScreen = false;
    pause;
    fileCommands;
    strip;
    tabs = new TabWidget();
    monitor = element("div", "monitor-page");
    monitorHint = note("No panels yet. Choose a Plot in the header and click Add panel, or turn on Plot on a Logic row before starting it.", true);
    logicPage = element("div", "logic-page");
    logicList = element("div", "logic-list");
    logicHint = note("No logic nodes yet. Choose a measurement, processor or task in the header and click Add node; it starts stopped. Open its Edit page to set parameters, then press Start.", true);
    rows = new Map();
    editors = new Map();
    publisherEditors = new Map();
    /** Each panel's Edit page, by its Edit's id (`edit-<panel>`). */
    panelEdits = new Map();
    /** The Edit the operator asked for, brought forward when it arrives. */
    editFocus = null;
    addRequest = null;
    requests = new Map();
    /** The point reviews already asked of the host: one window each. */
    reviewsAsked = new Set();
    kindsSignature = "";
    constructor(ctx, board) {
        this.ctx = ctx;
        this.board = board;
        const header = element("header", "card header");
        const identity = element("div", "identity");
        // v2: the dot and the word a dot gap apart (the strip's), the summary a row gap after them.
        identity.append(this.dot, element("span", "name", "task"));
        this.kinds.root.classList.add("kind-combo");
        this.add = command(ADD_WORDS.plot, () => this.addSelection(), { words: Object.values(ADD_WORDS) });
        this.selectors = switchButton(on => this.send({ action: "set_board", selectors: on }, on ? "Turning selectors on…" : "Turning selectors off…"), "Selectors");
        this.selectors.classList.add("row-switch");
        this.pause = command("Pause", () => this.send({ action: "set_board", paused: !this.boardState()?.paused }, "Updating the board…"), { words: ["Pause", "Resume"] });
        const image = command("Save image…", () => saveImage(this.ctx.transport), { title: "Save one screenshot of TaskConsole" });
        const save = command("Save…", () => this.send({ action: "save_layout" }, "Saving the layout…"), { title: "Save the layout and signal wiring" });
        const open = command("Open…", () => void this.openLayout(), { title: "Open a saved layout and signal wiring" });
        const clear = command("Clear", () => void this.clearBoard(), { role: "destructive", title: "Remove every panel and logic node from the board" });
        this.fileCommands = [save, open, clear];
        header.append(identity, this.summary, this.kinds.root, this.add.root, this.selectors, this.pause.root, image.root, save.root, open.root, clear.root);
        this.strip = new StatusStrip({ action: ["Stop task", () => { const task = this.boardState()?.active_task; if (task)
                    this.send({ action: "stop_node", node: task }, "Stopping the task…"); }] });
        this.strip.root.id = "status";
        this.strip.text.id = "status-text";
        this.monitor.append(this.monitorHint);
        this.logicPage.append(this.logicHint, this.logicList);
        this.tabs.root.classList.add("console-tabs");
        this.tabs.addPermanent("monitor", "Monitor", this.monitor);
        this.tabs.addPermanent("logic", "Logic", this.logicPage);
        this.tabs.setCurrent("monitor");
        this.root.append(header, this.strip.root, this.tabs.root);
        this.updateAdd();
        // Setting's Edit (v2 Panel Edit) opens the panel's Edit page here.
        board.onEdit = panel => this.editPanel(panel);
    }
    /** v2 view.monitor / view.logic: a remote client shows the Monitor or Logic page. */
    showPage(page) { if (page === "monitor" || page === "logic")
        this.tabs.setCurrent(page); }
    editPanel(panel) {
        const edit = `edit-${panel}`;
        if (this.tabs.has(`edit:${edit}`)) {
            this.tabs.setCurrent(`edit:${edit}`);
            return;
        }
        this.editFocus = edit;
        this.send({ action: "edit_panel", panel }, "Holding the panel still…");
    }
    /** The Edit pages follow the Edits the state carries: one opens with its Edit, closes with it. */
    updatePanelEdits(board, rows) {
        const panels = (this.ctx.latest.panels ?? []);
        const edits = panels.filter(row => row.edit);
        for (const [edit] of this.panelEdits)
            if (!edits.some(row => row.id === edit)) {
                this.tabs.remove(`edit:${edit}`);
                this.panelEdits.delete(edit);
            }
        for (const row of edits) {
            const entry = board?.panels.find(item => item.panel === row.edit.panel);
            const name = entry?.title ?? entry?.name ?? row.edit.panel;
            let page = this.panelEdits.get(row.id);
            if (!page) {
                page = new PanelEditView(row.id, row.edit.panel, this.board, (command, feedback) => this.send(command, feedback), node => this.edit(node), this.ctx.transport);
                this.panelEdits.set(row.id, page);
                this.tabs.addClosable(`edit:${row.id}`, `Edit panel · ${name}`, page.root, () => this.closePanelEdit(row.id), this.editFocus === row.id);
                if (this.editFocus === row.id)
                    this.editFocus = null;
            }
            page.update(row, name, rows, String(this.ctx.latest.workspace?.today ?? ""));
            this.tabs.setTitle(`edit:${row.id}`, `Edit panel · ${name}`);
        }
    }
    closePanelEdit(edit) {
        this.panelEdits.delete(edit);
        this.tabs.remove(`edit:${edit}`);
        this.send({ action: "close_panel", panel: edit }, "Closing the Edit…");
    }
    send(command, feedback) { return this.ctx.transport.send(command, feedback); }
    boardState() { return this.ctx.latest.board; }
    logic() { return this.ctx.latest.logic; }
    nodeRows() { return this.logic() ?? []; }
    shown() {
        this.onScreen = true;
        this.monitor.append(this.board.root);
        this.board.interactive = this.boardState()?.selectors ?? true;
        this.board.shown();
    }
    hidden() { this.onScreen = false; this.board.hidden(); }
    update(state) {
        const latest = this.ctx.latest;
        const board = this.boardState();
        const rows = this.nodeRows();
        const takeover = board?.active_task ?? null;
        this.updateKinds(latest);
        // Header: a board-wide command needs the workbench's board.
        const workbench = Boolean(board);
        for (const control of [this.pause, ...this.fileCommands])
            control.disabled = !workbench;
        this.selectors.disabled = !workbench;
        this.fileCommands[1].disabled = !workbench || Boolean(takeover);
        this.fileCommands[2].disabled = !workbench || Boolean(takeover);
        setChecked(this.selectors, Boolean(board?.selectors));
        if (this.onScreen)
            this.board.interactive = board?.selectors ?? true;
        this.pause.text = board?.paused ? "Resume" : "Pause";
        this.pause.setActive(Boolean(board?.paused));
        // One rule for the summary (v2): panels, live/paused, nodes running.
        const running = rows.filter(row => row.phase === "running").length;
        const words = [counted(this.board.size, "panel"), board?.paused ? "paused" : "live", ...(rows.length ? [`${running}/${counted(rows.length, "node")} running`] : [])].join(" · ");
        setElided(this.summary, words);
        this.monitorHint.hidden = this.board.size > 0;
        // Rows, in board order, then each panel's published outputs (v2 panel publishers).
        const publishers = this.publishers(board);
        const alive = new Set([...rows.map(row => row.node_id), ...publishers.map(item => item.projection.id)]);
        for (const [id, view] of this.rows)
            if (!alive.has(id)) {
                view.root.remove();
                this.rows.delete(id);
            }
        for (const [id] of this.editors)
            if (!alive.has(id))
                this.closeEditor(id);
        for (const [id] of this.publisherEditors)
            if (!alive.has(id))
                this.closeEditor(id);
        let worst = "idle";
        let cursor = this.logicList.firstElementChild;
        for (const row of rows) {
            let view = this.rows.get(row.node_id);
            if (!view) {
                view = new LogicRowView(row.node_id, this.intents);
                this.rows.set(row.node_id, view);
            }
            const projection = this.project(row);
            const gate = { active: Boolean(takeover), owns: takeover === row.node_id };
            view.update(projection, gate);
            if (SEVERITY[view.state] > SEVERITY[worst])
                worst = view.state;
            if (view.root !== cursor)
                this.logicList.insertBefore(view.root, cursor);
            else
                cursor = cursor.nextElementSibling;
            const editor = this.editors.get(row.node_id);
            if (editor) {
                editor.update(row, projection, gate, state.request, Boolean(state.error));
                this.tabs.setTitle(`edit:${row.node_id}`, `Edit · ${row.name}`);
            }
            this.showRequest(row);
        }
        for (const { projection, outputs, locked } of publishers) {
            let view = this.rows.get(projection.id);
            if (!view) {
                view = new LogicRowView(projection.id, this.intents);
                this.rows.set(projection.id, view);
            }
            view.update(projection, { active: Boolean(takeover), owns: false });
            if (view.root !== cursor)
                this.logicList.insertBefore(view.root, cursor);
            else
                cursor = cursor.nextElementSibling;
            const editor = this.publisherEditors.get(projection.id);
            if (editor) {
                editor.update(projection.name, outputs, locked, state.request, Boolean(state.error));
                this.tabs.setTitle(`edit:${projection.id}`, `Edit · ${projection.name} outputs`);
            }
        }
        this.logicHint.hidden = rows.length + publishers.length > 0;
        this.updatePanelEdits(board, rows);
        setState(this.dot, worst === "task" ? "idle" : worst);
        this.strip.setActionVisible(Boolean(takeover));
        // A node just added opens its Edit page (v2: Add node focuses its editor).
        if (this.addRequest && state.request === this.addRequest.id) {
            const fresh = rows.find(row => !this.addRequest.before.has(row.node_id));
            this.addRequest = null;
            if (fresh && !state.error)
                this.edit(fresh.node_id);
        }
        this.status();
    }
    /** Each board panel as its publisher row: the outputs it can publish. */
    publishers(board) {
        return (board?.panels ?? []).map(entry => {
            const outputs = entry.outputs;
            const name = entry.title ?? entry.name;
            return {
                projection: { id: `panel:${entry.panel}`, name, details: [entry.panel], kind: "panel_publisher", state: "idle", status: "", canStart: false, canStop: false,
                    pending: false, edited: false, previewOffered: false, autoPreview: false, publisher: true,
                    // v2: the signals the panel publishes now, by their output names, in signal order.
                    outputs: outputs.filter(output => output.live).sort((a, b) => (a.signal < b.signal ? -1 : a.signal > b.signal ? 1 : 0))
                        .map(output => [output.name, output.shape ?? "", `published · ${output.signal}`]) },
                outputs, locked: board?.frozen[entry.panel] ?? "",
            };
        });
    }
    project(row) { return projectRow(row); }
    status() {
        // v2: the strip keeps the console's newest line (a command's outcome, a note, the running
        // Task's status), said once by the workbench; a picture being presented says nothing.
        const report = this.boardState()?.report;
        const [text, word] = queueStatus(this.ctx.transport, report?.text ? [report.text, report.severity] : ["Ready", "idle"]);
        this.strip.show(text, word);
    }
    // -------------------------------------------------------------- header
    updateKinds(latest) {
        // v2 TaskConsoleView: the console's five plot kinds by their own names, the first the
        // default; then each node type by its api name, measurements, processors, tasks.
        const offered = new Set((latest.plot_kinds ?? []).map(kind => kind.id));
        const plots = CONSOLE_PLOT_KINDS.filter(id => offered.has(id)).map(id => ({ value: `plot:${id}`, label: `Plot: ${id}` }));
        const layer = (kind) => ({ measurement: 0, processor: 1, task: 2 }[kind] ?? 3);
        const nodes = [...(latest.node_types ?? [])]
            .sort((a, b) => layer(a.node_kind) - layer(b.node_kind) || a.type_id.localeCompare(b.type_id))
            .map(type => ({
            value: `logic:${type.type_id}`, label: `${LAYER_WORDS[type.node_kind] ?? type.node_kind}: ${type.type_id}`,
            title: type.publishes ? `Publishes: ${type.publishes}` : "",
        }));
        const signature = JSON.stringify([plots, nodes]);
        if (signature === this.kindsSignature)
            return;
        this.kindsSignature = signature;
        const current = this.kinds.value;
        this.kinds.setChoices([...plots, ...nodes]);
        const all = [...plots, ...nodes].map(choice => choice.value);
        this.kinds.value = this.kindPicked && typeof current === "string" && all.includes(current) ? current : all[0] ?? null;
        this.kinds.disabled = all.length === 0;
        this.updateAdd();
    }
    updateAdd() {
        const value = String(this.kinds.value ?? "");
        const family = value.startsWith("logic:") ? "logic" : "plot";
        this.add.text = ADD_WORDS[family];
        this.add.disabled = !value || (family === "logic" && Boolean(this.boardState()?.active_task));
    }
    addSelection() {
        const value = String(this.kinds.value ?? "");
        const [family, key] = [value.slice(0, value.indexOf(":")), value.slice(value.indexOf(":") + 1)];
        if (family === "plot") {
            // v2: a blank panel of the chosen kind; its signal is chosen in its Setting.
            this.send({ action: "add_panel", kind: key }, "Adding panel…");
        }
        else if (family === "logic") {
            const id = this.send({ action: "add_node", type_id: key }, "Adding node…");
            this.addRequest = { id, before: new Set(this.nodeRows().map(row => row.node_id)) };
        }
    }
    /** v2 _confirm_board_replaced: what goes, and what is stopped first. */
    replaceText(verb) {
        const rows = this.nodeRows();
        const parts = [this.board.size ? counted(this.board.size, "panel") : "", rows.length ? counted(rows.length, "logic node") : ""].filter(Boolean);
        const running = rows.filter(working).map(row => row.name);
        return `${verb} removes ${parts.join(" and ")} from this board.${running.length ? ` It first stops ${running.join(", ")}.` : ""} A board that was not saved cannot be brought back.`;
    }
    async clearBoard() {
        const rows = this.nodeRows();
        // An empty board asks nothing; the workbench says "the board is already empty" (a line
        // shown here alone would be gone at the next state push).
        if (!this.board.size && !rows.length) {
            this.send({ action: "clear_board" });
            return;
        }
        if (!await confirm("Replace the board?", this.replaceText("Clearing"), { confirmText: "Clear" }))
            return;
        this.send({ action: "clear_board", stop_running: rows.some(working) }, "Clearing the board…");
    }
    async openLayout() {
        const running = this.nodeRows().some(working);
        if (running && !await confirm("Replace the board?", this.replaceText("Opening this layout"), { confirmText: "Stop and open" }))
            return;
        this.send({ action: "load_layout", ...(running ? { stop_running: true } : {}) }, "Opening a layout…");
    }
    // ---------------------------------------------------------------- rows
    intents = {
        start: id => {
            const row = this.logic()?.find(item => item.node_id === id);
            this.send({ action: "start_node", node: id, ...(row ? { expected_draft_revision: row.draft_revision } : {}) }, `Starting ${row?.name ?? id}…`);
        },
        stop: id => {
            this.send({ action: "stop_node", node: id }, `Stopping ${this.nameOf(id)}…`);
        },
        edit: id => id.startsWith("panel:") ? this.editPublisher(id) : this.edit(id),
        remove: id => this.send({ action: "remove_node", node: id }, `Removing ${this.nameOf(id)}…`),
        preview: (id, on) => this.send({ action: "edit_node", node: id, auto_preview: on }, "Updating Plot…"),
        rename: (id, name) => this.send({ action: "rename_node", node: id, name }, "Renaming…"),
        editDraft: (id, change, label) => {
            return this.send({ action: "edit_node", node: id, ...change }, `Updating ${label}…`);
        },
        revert: id => this.send({ action: "revert_node", node: id }, "Reverting…"),
        check: id => this.send({ action: "check_node", node: id }, "Checking files…"),
        choose: ask => this.ctx.transport.choosePath(ask),
    };
    nameOf(id) { return this.nodeRows().find(row => row.node_id === id)?.name ?? id; }
    edit(id) {
        const key = `edit:${id}`;
        if (this.tabs.has(key)) {
            this.tabs.setCurrent(key);
            return;
        }
        const row = this.nodeRows().find(item => item.node_id === id);
        if (!row)
            return;
        const editor = new LogicEditorView(id, this.intents);
        this.editors.set(id, editor);
        const takeover = this.boardState()?.active_task ?? null;
        editor.update(row, this.project(row), { active: Boolean(takeover), owns: takeover === id });
        // v2 handle._logic_editor_tab / _panel_editor_tab: "Edit · <name>", "Edit panel · <name>".
        this.tabs.addClosable(key, `Edit · ${row.name}`, editor.root, () => this.closeEditor(id));
    }
    editPublisher(id) {
        const key = `edit:${id}`;
        if (this.tabs.has(key)) {
            this.tabs.setCurrent(key);
            return;
        }
        const item = this.publishers(this.boardState()).find(entry => entry.projection.id === id);
        if (!item)
            return;
        const panel = id.slice("panel:".length);
        const editor = new PublisherEditorView(id, outputs => this.send({ action: "publish_panel_outputs", panel, outputs }, "Updating published outputs…"));
        this.publisherEditors.set(id, editor);
        editor.update(item.projection.name, item.outputs, item.locked);
        // v2 _logic_editor_tab: "Edit · <api name>", a publisher's being "<panel name> outputs".
        this.tabs.addClosable(key, `Edit · ${item.projection.name} outputs`, editor.root, () => this.closeEditor(id));
    }
    closeEditor(id) {
        this.editors.delete(id);
        this.publisherEditors.delete(id);
        this.tabs.remove(`edit:${id}`);
    }
    /** An operator request of a run (answer_request): a card with the request's
     * own words and fields; Submit answers, Cancel answers null (= Stop). */
    showRequest(row) {
        const request = row.run?.request ?? null;
        for (const [key, dialog] of this.requests)
            if (key.startsWith(`${row.node_id}#`) && key !== request?.request) {
                dialog.close("gone");
                this.requests.delete(key);
            }
        if (!request || this.requests.has(request.request))
            return;
        // A point review is a window of its own (v2 opened its review window), not a card here.
        if (request.kind === "point-selection") {
            this.pointReview(row, request);
            return;
        }
        const dialog = new Dialog(request.title || "The run asks");
        this.requests.set(request.request, dialog);
        if (request.message)
            dialog.content.append(element("div", "readout-text dialog-text", request.message));
        if (request.kind === "manual-axis") {
            this.manualAxis(row, request, dialog);
            return;
        }
        const fields = Array.isArray(request.payload?.fields) ? request.payload.fields : [];
        const answers = {};
        if (fields.length) {
            const grid = element("div", "form");
            for (const spec of fields) {
                answers[spec.name] = spec.value;
                const field = new Field(`request-${row.node_id}`, spec, { commit: value => { answers[spec.name] = value; queueMicrotask(() => field.update({ ...spec, value })); return 0; },
                    browse: ask => this.ctx.transport.choosePath(ask) });
                grid.append(field.root);
            }
            dialog.content.append(grid);
        }
        dialog.addAction("Cancel", "cancel", "destructive");
        dialog.addAction("Submit", "submit", "primary");
        void dialog.open().then(answer => {
            this.requests.delete(request.request);
            if (answer === "gone")
                return;
            this.send({ action: "answer_request", node: row.node_id, request: request.request, response: answer === "submit" ? answers : null }, answer === "submit" ? "Answering the run…" : "Stopping the run…");
        });
    }
    /** v2 manual_axis_setting for a manual-axis request: the run stands aside
     * for a hand moving a knob nothing here can advance ("Set <axis> to <value>,
     * then continue."). Continue answers {}; Stop answers null, which stops the
     * run rather than retrying against a hand that has moved on. */
    manualAxis(row, request, dialog) {
        dialog.root.classList.add("manual-axis");
        dialog.addAction("Stop", "stop", "destructive");
        dialog.addAction("Continue", "continue", "primary");
        void dialog.open().then(answer => {
            this.requests.delete(request.request);
            if (answer === "gone")
                return;
            const proceed = answer === "continue";
            this.send({ action: "answer_request", node: row.node_id, request: request.request, response: proceed ? {} : null }, proceed ? "Continuing the run…" : "Stopping the run…");
        });
    }
    /** v2 PointReviewView for a point-selection request: its own window (point_review.ts), asked of the host once. */
    pointReview(row, request) {
        if (this.reviewsAsked.has(request.request))
            return;
        this.reviewsAsked.add(request.request);
        this.send({ action: "open_point_review", node: row.node_id, request: request.request }, "Opening the review…");
    }
}
export const taskConsole = (board) => (ctx) => new TaskConsoleView(ctx, board);
