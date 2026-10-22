// The control gallery (v2 examples/gallery.py): every control of the
// library with fake data, for looking at and for screenshots.  Reached only
// by index.html#gallery; it sends no command.
import {
  ComboBox, Field, GroupBox, InfoPane, Legend, Menu, Readout, Segmented, SelectionGrid, StatusLine, StatusStrip, TabWidget, TreeCombo,
  checkBox, choiceTree, command, confirm, element, form, message, note, prompt, ruledList, sectionTitle, setChecked, spinBox, statusDot, switchButton,
  type FieldSpec, type Value,
} from "../fluent/index.js";
import type { View, ViewContext } from "../shell.js";

function labelled(name: string, ...controls: HTMLElement[]): HTMLElement {
  const cell = element("div", "gallery-item");
  cell.append(element("span", "note gallery-name", name), ...controls);
  return cell;
}

export const gallery = (_ctx: ViewContext): View => {
  const root = element("div", "window gallery");
  const page = element("div", "gallery-page");
  root.append(page);
  const section = (title: string, ...items: HTMLElement[]): void => {
    const box = new GroupBox(title);
    const row = element("div", "gallery-row");
    row.append(...items);
    box.body.append(row);
    page.append(box.root);
  };

  // Labels and status.
  const line = new StatusLine("running · 3 frames", "running");
  section("Typography / labels / status", labelled("StatusDot", statusDot("running")), labelled("SectionLabel", element("span", "heading", "Section label")),
    labelled("ElidedLabel", element("span", "elided gallery-elide", "An eliding label with a longer tooltip that does not fit")),
    labelled("StatusLine", line.root), labelled("Note", note("Muted information that still carries meaning.")));

  // Commands.
  const dirty = command("Save", () => {});
  dirty.setDirty(true);
  const active = command("Pause", () => {}, { words: ["Pause", "Resume"] });
  active.setActive(true);
  active.text = "Resume";
  const disabled = command("Add panel", () => {});
  disabled.disabled = true;
  section("Buttons", labelled("primary", command("Start", () => {}, { role: "primary" }).root), labelled("standard", command("Refresh", () => {}).root),
    labelled("destructive", command("Remove", () => {}, { role: "destructive" }).root), labelled("dirty", dirty.root), labelled("active", active.root),
    labelled("disabled", disabled.root), labelled("quiet ×", command("×", () => {}, { role: "quiet", glyph: true, square: true }).root),
    labelled("remove × (two clicks)", command("×", () => {}, { role: "destructive", glyph: true, square: true, confirm: "remove this row" }).root),
    labelled("destructive-primary", command("Delete", () => {}, { role: "destructive-primary" }).root));

  // Choices.
  const onSwitch = switchButton(on => setChecked(onSwitch, on), "Selectors");
  setChecked(onSwitch, true);
  const offSwitch = switchButton(on => setChecked(offSwitch, on), "Plot");
  const check = checkBox("Fake option", () => {});
  setChecked(check, true);
  const segments = new Segmented(value => segments.setCurrent(value));
  segments.setChoices([["preview", "Preview"], ["run", "Run"], ["scan", "Scan"]]);
  segments.setCurrent("run");
  const combo = new ComboBox(value => { combo.value = value; });
  combo.setChoices([{ value: "image", label: "Plot: Image" }, { value: "curve", label: "Plot: Curve" }, { value: "occupancy", label: "Processor: occupancy", group: "Logic" }]);
  combo.value = "image";
  // v2 gallery's FluentTreeComboBox: choices under their sources, one with none.
  const tree = new TreeCombo(key => { tree.value = key; });
  tree.setTree(choiceTree([{ key: "temperature", group: "sensor", label: "Temperature · float · ready", text: "sensor · Temperature" },
    { key: "count", group: "counter", label: "Count · int · ready", text: "counter · Count" }, { key: "waiting", label: "waiting · unassigned" }], "temperature", "(no source)"), "temperature", "(none)");
  const edit = element("input");
  edit.type = "text";
  edit.value = "editable text";
  const spin = spinBox({ value: 3, step: () => 1, bounds: () => [0, null], commit: () => {} });
  section("Choices and text", labelled("Switch", onSwitch), labelled("Switch · off", offSwitch), labelled("CheckBox", check), labelled("Segmented", segments.root),
    labelled("ComboBox", combo.root), labelled("TreeComboBox", tree.root), labelled("LineEdit", edit), labelled("SpinBox", spin.root), labelled("Readout", new Readout("C:/Users/lab/workspace/data/2026_10_20/calibration-2/final/calibration.npz").root));

  // A parameter form.
  const specs: FieldSpec[] = [
    { name: "label", label: "Label", kind: "text", value: "fake panel" },
    { name: "count", label: "Count", kind: "integer", value: 3, minimum: 0, step: 1 },
    { name: "mode", label: "Mode", kind: "choice", value: "preview", choices: [["preview", "Preview"], ["run", "Run"]] },
    { name: "relim", label: "Limits", kind: "choice", value: "auto", segmented: true, choices: [["auto", "Auto"], ["keep", "Keep"], ["manual", "Manual"]] },
    { name: "title", label: "Title", kind: "text", value: null, allow_none: true, automatic: true, automatic_value: "frames" },
    { name: "frequency", label: "Frequency", kind: "number", value: 120, step: 0.5, unit: "MHz",
      units: [{ value: "Hz", label: "Hz", group: "Hz" }, { value: "kHz", label: "kHz", group: "Hz" }, { value: "MHz", label: "MHz", group: "Hz" }] },
    { name: "enabled", label: "Enabled", kind: "boolean", value: true },
    { name: "seed", label: "Seed", kind: "integer", value: null, allow_none: true, placeholder: "random", problem: "Seed must be below 2^32" },
  ];
  const grid = form();
  grid.append(sectionTitle("Parameters", command("Fit", () => {}).root));
  const fields: Field[] = [];
  for (const spec of specs) {
    let current: FieldSpec = spec;
    const field: Field = new Field("gallery", spec, { commit: (value: Value) => { current = { ...current, value }; queueMicrotask(() => field.update(current, 0)); return 0; } });
    fields.push(field);
    grid.append(field.root);
  }
  section("ParameterForm", grid);

  // Cards, ruled list, legend.
  const card = new GroupBox("Camera", [command("Add device", () => {}).root]);
  const list = ruledList();
  for (const role of ["mot_camera", "dcam_0"]) {
    const row = element("div", "list-row");
    row.append(element("span", "list-title", role), element("span", "note list-detail", "camera.virtual_mot · Open"), command("Control", () => {}).root, command("Close", () => {}, { role: "destructive" }).root);
    list.append(row);
  }
  card.body.append(list);
  const legend = new Legend();
  legend.setRows([["temperature", "1", "kelvin"], ["smoothed", "1", ""], ["frames", "96×128", "camera frames"]]);
  const emptyLegend = new Legend();
  emptyLegend.setRows([]);
  section("GroupBox · ruled list · legend", card.root, labelled("PublishedItemsLegend", legend.root, emptyLegend.root));

  // Tabs.
  const tabs = new TabWidget();
  tabs.root.classList.add("gallery-tabs");
  tabs.addPermanent("monitor", "Monitor", note("Monitor page"));
  tabs.addPermanent("logic", "Logic", note("Logic page"));
  tabs.addClosable("edit", "Edit: occupancy", note("Edit page"), () => tabs.remove("edit"), false);
  section("TabWidget", tabs.root);

  // Selection grid.
  const selection = new SelectionGrid({ rowActions: [{ value: "remove", glyph: "×", title: "Remove this row" }] });
  selection.setGrid({ rows: ["shot 1", "shot 2", "shot 3"], columns: ["rest", "load", "image", "", "check"], narrowColumns: [3],
    cells: [[0, 0, "rest"], [0, 1, "load"], [0, 2, "image"], [0, 4, "check"], [1, 0, "rest"], [1, 1, "load"], [1, 2, "image"], [1, 4, "check"], [2, 0, "rest"], [2, 2, "image"], [2, 4, "check"]] });
  selection.setSelected([[0, 0], [1, 1], [2, 2]]);
  selection.setStates({ "0,2": "derived", "1,0": "error", "1,4": "kept", "2,4": "missing" });
  section("SelectionGrid", selection.root);

  // Info pane and strip.
  const info = new InfoPane("Artifact", { commit: () => {}, browse: () => {}, refresh: () => {} });
  info.root.classList.add("gallery-info");
  info.setPath("C:/lab/data/frames.npz");
  info.setTabs([["Summary", [["Name", "Fake figure"], ["Value", 42], ["shape", [1, 6, 96, 128]], ["axis", Array.from({ length: 96 }, (_, i) => i)],
    ["run", { started: "2026-10-20T13:02", devices: { camera: "mot_camera", sequencer: "pulse" } }], ["Open", { text: "Open folder", action: "open" }]]],
    ["State", [["state", "stopped"]]], ["Raw", [["format", "zlc.figure"]]]]);
  info.setStatus("Fake data only.");
  const strip = new StatusStrip({ action: ["Stop task", () => {}] });
  strip.show("seamless_scan: point 3 of 20", "task");
  strip.setActionVisible(true);
  section("InfoPane · StatusStrip", info.root, labelled("StatusStrip", strip.root));

  // Menu and dialogs.
  const menuButton = command("New…", () => new Menu([{ label: "virtual", action: () => {} }, { label: "hardware", action: () => {} }, "separator", { label: "withheld", action: () => {}, disabled: true }]).openBelow(menuButton.root));
  section("Menu · dialogs", menuButton.root,
    command("Message…", () => void message("Saved", "C:/lab/workspace/apparatus.json was written.")).root,
    command("Confirm…", () => void confirm("Replace the board?", "Clearing removes 3 panels and 2 logic nodes from this board. A board that was not saved cannot be brought back.", { confirmText: "Clear" })).root,
    command("Prompt…", () => void prompt("Rename", "Name", "Occupancy")).root);

  return { key: "gallery", title: "Gallery", windowTitle: "zlc_ui control gallery — fake data only", root, update() {}, status() {} };
};
