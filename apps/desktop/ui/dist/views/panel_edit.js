// v2 PanelEditorView: one board panel's Edit page in the Task Console. Its
// picture is the panel held still on one read (the Edit, `edit-<panel>`, a
// panel of the live document the host keeps on that read); its settings are
// the panel's own, shown in place (Panel / Semantic / Display / Fit) and
// shared with the board's card both ways; Refresh snapshot takes the panel's
// newest data; Direct producer opens the node that makes the signal; Save
// figure writes the held picture and its archive (v2's inline group).
import { checked, command, element, Field, form, note, PathEdit, Readout, row as settingRow, sectionTitle, Segmented, setChecked, shareLabelColumn, switchButton, wholeText } from "../fluent/index.js";
/** v2 _IMAGE_FORMATS. */
const FORMATS = ["png", "pdf", "svg"];
/** The moment a file is written, as v2 stamps it (%Y%m%d_%H%M%S). */
function stampNow() {
    const at = new Date(), two = (value) => String(value).padStart(2, "0");
    return `${at.getFullYear()}${two(at.getMonth() + 1)}${two(at.getDate())}_${two(at.getHours())}${two(at.getMinutes())}${two(at.getSeconds())}`;
}
/** v2 stamped_file_name: the operator's name (what a Windows name cannot hold as _) and the moment. */
export function stampedFileName(stem, extension, stamp) {
    const text = [...stem.trim()].map(char => '<>:"/\\|?*'.includes(char) || char.charCodeAt(0) < 32 ? "_" : char).join("") || "untitled";
    return `${text}_${stamp}.${extension}`;
}
const baseName = (path) => path.split(/[\\/]/).pop() ?? path;
export class PanelEditView {
    edit;
    panel;
    board;
    transport;
    root = element("section", "logic-editor panel-edit");
    title = new Readout("", true, true);
    kind = element("span", "note");
    forms = element("div", "panel-edit-forms");
    holder = element("div", "panel-edit-surface");
    held = element("p", "note panel-edit-held");
    refresh;
    producerText = element("p", "note panel-edit-held");
    /** The producer's draft fields a selector of this panel writes, read only (v2 producer_form). */
    producerForm = form();
    producerFields = new Map();
    openProducer;
    producerNode = "";
    /** The page's labels as last measured into its one column. */
    labels = "";
    adopted = null;
    words = "";
    // Save figure (v2 PanelEditorView's group): where, called what, which format, the file it makes.
    directory;
    autoName;
    name = element("input");
    format;
    extension = FORMATS[0];
    file = new Readout("");
    saveFigure;
    directorySet = false;
    nameTyped = false;
    stamp = stampNow();
    panelName = "";
    canSave = false;
    body;
    constructor(edit, panel, board, send, openNode, transport) {
        this.edit = edit;
        this.panel = panel;
        this.board = board;
        this.transport = transport;
        const header = element("div", "logic-editor-header");
        const titleRow = element("div", "logic-editor-title");
        this.title.root.classList.add("heading");
        this.kind.title = "The plot kind is fixed when the panel is added";
        titleRow.append(this.title.root, this.kind);
        header.append(titleRow);
        this.refresh = command("Refresh snapshot", () => send({ action: "refresh_panel_edit", panel }, "Refreshing the snapshot…"));
        this.openProducer = command("Open producer", () => { if (this.producerNode)
            openNode(this.producerNode); }, { title: "Open this node's existing Logic Edit tab" });
        const snapshot = element("div", "card bordered panel-edit-card");
        const refreshRow = element("div", "form-actions");
        refreshRow.append(this.refresh.root);
        snapshot.append(sectionTitle("Frozen snapshot"), this.holder, this.held, refreshRow);
        const producer = element("div", "card bordered panel-edit-card");
        const producerRow = element("div", "form-actions");
        producerRow.append(element("span", "spacer"), this.openProducer.root);
        producer.append(sectionTitle("Direct producer"), this.producerText, this.producerForm, producerRow);
        this.directory = new PathEdit(() => this.updateSave(), () => void this.chooseDirectory());
        this.autoName = switchButton(on => { setChecked(this.autoName, on); this.updateSave(); }, "Auto-name");
        setChecked(this.autoName, true);
        this.name.type = "text";
        this.name.placeholder = "File name";
        this.name.spellcheck = false;
        this.name.addEventListener("input", () => { this.nameTyped = true; this.updateSave(); });
        this.format = new Segmented(value => { this.extension = String(value); this.format.setCurrent(this.extension); this.updateSave(); });
        this.format.setChoices(FORMATS.map(extension => [extension, extension.toUpperCase()]));
        this.format.setCurrent(this.extension);
        this.saveFigure = command("Save figure", () => this.save(), { role: "primary" });
        const save = element("div", "card bordered panel-edit-card");
        const rows = form();
        rows.append(settingRow("Directory", this.directory.root), settingRow("Name", this.autoName, this.name), settingRow("Format", this.format.root), settingRow("File", this.file.root));
        const saveRow = element("div", "form-actions");
        saveRow.append(element("span", "spacer"), this.saveFigure.root);
        save.append(sectionTitle("Save figure"), rows, saveRow);
        // v2: the page's heading scrolls with its cards.
        const body = element("div", "logic-editor-body");
        body.append(header, this.forms, snapshot, producer, save);
        this.body = body;
        this.holder.append(note("No frozen plot surface", true));
        this.held.textContent = "No frozen snapshot";
        this.root.append(body);
        this.updateSave();
    }
    /** v2 _save_path: the directory and the auto (stamped) or typed name, in the picked format; null until both are there. */
    savePath() {
        const directory = this.directory.value.trim().replace(/[\\/]+$/, "");
        if (!directory)
            return null;
        let name;
        if (checked(this.autoName))
            name = stampedFileName(this.panelName, this.extension, this.stamp);
        else {
            // A name's own dots stay ("scan_1.5us"); only a picture format's suffix is replaced by the picked one.
            const typed = this.name.value.trim().replace(/[\\/]/g, "_").replace(/\.(png|pdf|svg)$/i, "");
            if (!typed)
                return null;
            name = `${typed}.${this.extension}`;
        }
        return `${directory.replace(/\//g, "\\")}\\${name}`;
    }
    updateSave() {
        this.name.disabled = checked(this.autoName);
        const path = this.savePath();
        this.file.text = path ?? "";
        this.saveFigure.disabled = !(this.canSave && path);
    }
    async chooseDirectory() {
        const chosen = await this.transport.choosePath({ title: "Choose where to save", start: this.directory.value, dir: true });
        if (chosen) {
            this.directory.value = chosen;
            this.updateSave();
        }
    }
    /** v2 _save_figure: the held picture as an image and its archive; the next auto name is stamped anew. */
    save() {
        const path = this.savePath();
        if (!path || this.saveFigure.disabled)
            return;
        const name = this.panelName;
        this.transport.request({ action: "save", panel: this.edit, path }, `saving ${name}`, (_state, sections) => sections.saved).then(saved => {
            const files = Array.isArray(saved) ? saved.map(String) : [];
            if (files.length === 2)
                this.transport.note(`panel saved to ${baseName(files[0])} and ${baseName(files[1])}`);
        }, () => { });
        this.stamp = stampNow();
        this.updateSave();
    }
    /** Each accepted state: the Edit's card (adopted once), what it holds, its producer. */
    update(row, name, logic, today) {
        const view = this.board.views.get(this.edit);
        if (view && view !== this.adopted) {
            this.adopted = view;
            this.holder.replaceChildren(view.root);
            view.inline(this.forms);
        }
        // Saved where the day's data goes, called after the panel, until the operator says otherwise.
        if (!this.directorySet && today) {
            this.directorySet = true;
            this.directory.value = today;
        }
        if (!this.nameTyped && this.name.value !== name)
            this.name.value = name;
        this.panelName = name;
        this.canSave = Boolean(row.edit && row.frame);
        this.updateSave();
        this.title.text = name;
        // v2: the kind, and a grid's cell kind when one is named.
        const cell = row.controls?.find(control => control.section === "semantic" && control.name === "cell_kind")?.value;
        const kind = `(${(row.kind || "automatic").replace("_", " ")}${typeof cell === "string" && cell ? ` · ${cell.replace("_", " ")} cells` : ""})`;
        if (this.kind.textContent !== kind)
            this.kind.textContent = kind;
        // v2 _snapshot_text: Signal, revision, shape, and what Save keeps.
        const edit = row.edit;
        const signal = row.binding?.signal ?? "";
        const signalText = (this.board.signals.find(item => item.id === signal)?.label ?? signal).replace(/^@logic\//, "") || "(unknown)";
        const shape = edit ? (edit.shape.length === 1 ? `(${edit.shape[0]},)` : `(${edit.shape.join(", ")})`) : "";
        const words = edit ? `Signal: ${signalText} · revision ${edit.revision} · shape ${shape} · ${edit.status}` : "No frozen snapshot is displayed.";
        if (words !== this.words) {
            this.words = words;
            this.held.textContent = words;
        }
        // The live data has moved on from the held read: said in the warning colour, so Save is not mistaken for the newest.
        this.held.classList.toggle("warning", Boolean(edit?.advanced));
        // The node that makes the signal (v2 Direct producer): @logic/<node>/<output>.
        const node = /^@logic\/([^/]+)\//.exec(signal)?.[1] ?? "";
        const producer = logic.find(item => item.node_id === node);
        this.producerNode = producer ? node : "";
        const text = producer ? `Logic node: ${producer.name}` : "This signal has no editable direct producer.";
        if (this.producerText.textContent !== text)
            this.producerText.textContent = text;
        this.openProducer.disabled = !producer;
        // Its selector-linked draft fields, as the draft holds them, never edited here.
        const linked = new Set(producer?.selector_fields ?? []);
        const shown = (producer?.controls ?? []).filter(control => linked.has(control.name));
        for (const [name, field] of this.producerFields)
            if (!shown.some(control => control.name === name)) {
                field.root.remove();
                this.producerFields.delete(name);
            }
        for (const control of shown) {
            const spec = { ...control, enabled: false };
            let field = this.producerFields.get(control.name);
            if (!field) {
                field = new Field(`producer-${this.edit}`, spec, { commit: () => 0 });
                this.producerFields.set(control.name, field);
            }
            else
                field.update(spec);
            if (field.root.parentElement !== this.producerForm)
                this.producerForm.append(field.root);
        }
        this.producerForm.hidden = shown.length === 0;
        // One label column down the page (v2 _share_label_column), measured when its words change.
        const labels = [...this.body.querySelectorAll(".form .field-label")].map(wholeText).join("|");
        if (labels !== this.labels) {
            this.labels = labels;
            requestAnimationFrame(() => shareLabelColumn(this.body));
        }
    }
}
