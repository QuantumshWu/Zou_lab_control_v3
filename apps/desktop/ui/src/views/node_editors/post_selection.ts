// Occupancy's post-selection (v2 PostSelectionEditor), under its
// calibrations: the title with its switch; while the switch is on, the Mode
// strip, then one "Keep where" row in the same control column -- frame A,
// the relation strip, frame B in Relation mode; each frame with its
// Any | 1 | 0 strip in Values mode.  The model (occupancy/editor.py) offers
// the frames the source carries, named and ordered as the calibrations'
// columns, reads the condition in the title row, and says what is refused
// (red) or what the source lacks (amber).  The switch and the rows follow the
// model, so they change together; the other mode's values stay in the draft
// as they are.
import { ComboBox, Segmented, element, form, row as formRow, setChecked, setText, switchButton, type Value } from "../../fluent/index.js";
import type { EditorFacts, LogicRow } from "../../workbench.js";
import type { DraftEdit, NodeEditor } from "./index.js";

type Strip = { label: string; choices: [Value, string][]; current: Value };
type Frame = { label: string; choices: [string, string][]; current: string };
type Facts = EditorFacts & {
  title: string; on: boolean; summary: string; relation_mode: boolean;
  strips: Record<"mode" | "relation" | "value_a" | "value_b", Strip>;
  frames: Record<"frame_a" | "frame_b", Frame>;
  problem: string; note: string;
};

export class PostSelectionEditor implements NodeEditor {
  readonly root = element("div", "node-editor post-selection");
  private title = element("span", "heading");
  private toggle: HTMLButtonElement;
  private summary = element("span", "summary-text");
  private strips: Record<keyof Facts["strips"], Segmented>;
  private frames: Record<keyof Facts["frames"], ComboBox>;
  private rows = form();
  private modeRow: HTMLDivElement;
  private problem = element("p", "field-error");
  private note = element("p", "field-error warning");
  private facts: Facts | null = null;

  constructor(private edit: DraftEdit) {
    this.toggle = switchButton(on => this.edit({ post_selection: on }, this.facts?.title ?? "Post-selection"));
    const header = element("div", "title-row editor-title");
    header.append(this.title, this.toggle, this.summary);
    const strip = (name: keyof Facts["strips"]): Segmented => new Segmented(value => this.edit({ [name]: value }, this.facts?.strips[name].label ?? name));
    this.strips = { mode: strip("mode"), relation: strip("relation"), value_a: strip("value_a"), value_b: strip("value_b") };
    const combo = (name: keyof Facts["frames"]): ComboBox => new ComboBox(value => this.edit({ [name]: value }, this.facts?.frames[name].label ?? name));
    this.frames = { frame_a: combo("frame_a"), frame_b: combo("frame_b") };
    // A frame and its value stand closer than the two frames do.
    const slot = (frame: ComboBox, value: Segmented): HTMLDivElement => {
      const holder = element("div", "post-slot");
      holder.append(frame.root, value.root);
      return holder;
    };
    const condition = element("div", "post-condition");
    condition.append(slot(this.frames.frame_a, this.strips.value_a), this.strips.relation.root, slot(this.frames.frame_b, this.strips.value_b));
    this.modeRow = formRow("Mode", this.strips.mode.root);
    this.rows.append(this.modeRow, formRow("Keep where", condition));
    this.root.append(header, this.rows, this.problem, this.note);
  }

  update(_row: LogicRow, raw: EditorFacts, locked: string): void {
    const facts = raw as Facts;
    this.facts = facts;
    setText(this.title, facts.title);
    setChecked(this.toggle, facts.on);
    this.toggle.disabled = Boolean(locked);
    setText(this.summary, facts.summary);
    this.rows.hidden = !facts.on;
    setText(this.modeRow.firstElementChild!, facts.strips.mode.label);
    for (const [name, strip] of Object.entries(this.strips) as [keyof Facts["strips"], Segmented][]) {
      strip.setChoices(facts.strips[name].choices);
      strip.setCurrent(facts.strips[name].current);
      strip.disabled = Boolean(locked);
    }
    this.strips.relation.root.hidden = !facts.relation_mode;
    this.strips.value_a.root.hidden = this.strips.value_b.root.hidden = facts.relation_mode;
    for (const [name, combo] of Object.entries(this.frames) as [keyof Facts["frames"], ComboBox][]) {
      combo.setChoices(facts.frames[name].choices);
      combo.value = facts.frames[name].current;
      combo.title = facts.frames[name].label;
      combo.disabled = Boolean(locked);
    }
    for (const [line, text] of [[this.problem, facts.problem], [this.note, facts.note]] as const) { setText(line, text); line.hidden = !text; }
  }
}
