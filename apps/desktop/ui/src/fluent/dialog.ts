import { command, type Role } from "./button.js";
import { element } from "./dom.js";

/** FluentCardDialog: THE chrome every modal wears -- a white rounded card
 * with the DIVIDER edge, no platform frame, centred on the window and kept
 * on it, dragged by its own body.  The page behind is not dimmed (Qt's
 * modal dims nothing); it only stops taking clicks. */
export class Dialog {
  readonly root = element("dialog", "dialog");
  readonly content = element("div", "dialog-content");
  readonly actions = element("div", "dialog-actions");
  private answer: (value: string) => void = () => {};

  constructor(title: string) {
    const heading = element("div", "dialog-title", title);
    this.root.append(heading, this.content, this.actions);
    this.root.addEventListener("cancel", event => { event.preventDefault(); this.close("cancel"); });
    this.root.addEventListener("pointerdown", event => this.drag(event));
  }

  /** A command at the row's end; its click closes the dialog with `value`. */
  addAction(text: string, value: string, role: Role = "standard"): HTMLButtonElement {
    const action = command(text, () => this.close(value), { role });
    this.actions.append(action.root);
    return action.root;
  }

  open(): Promise<string> {
    document.body.append(this.root);
    this.root.showModal();
    return new Promise(resolve => { this.answer = resolve; });
  }

  close(value: string): void {
    if (!this.root.open) return;
    this.root.close();
    this.root.remove();
    this.answer(value);
  }

  /** The card is its own title bar: a press nothing inside wanted drags it. */
  private drag(event: PointerEvent): void {
    if (event.button !== 0 || (event.target as Element).closest("button, input, select, textarea, .readout-text")) return;
    const rect = this.root.getBoundingClientRect();
    const origin = { x: event.clientX - rect.left, y: event.clientY - rect.top };
    this.root.setPointerCapture(event.pointerId);
    const move = (moved: PointerEvent): void => {
      const left = Math.max(0, Math.min(innerWidth - rect.width, moved.clientX - origin.x));
      const top = Math.max(0, Math.min(innerHeight - rect.height, moved.clientY - origin.y));
      this.root.style.margin = "0";
      this.root.style.left = `${left}px`;
      this.root.style.top = `${top}px`;
    };
    // The capture lost without a release (the window losing the mouse) ends the drag too.
    const end = (): void => { this.root.removeEventListener("pointermove", move); this.root.removeEventListener("pointerup", end); this.root.removeEventListener("lostpointercapture", end); };
    this.root.addEventListener("pointermove", move);
    this.root.addEventListener("pointerup", end);
    this.root.addEventListener("lostpointercapture", end);
  }
}

function body(text: string): HTMLDivElement {
  // A wrap-anywhere selectable block: a path or a traceback breaks inside a
  // token instead of running past the card; past its cap it scrolls.
  return element("div", "readout-text dialog-text", text);
}

/** fluent_message: a title, the message and one OK (acknowledging destroys nothing). */
export async function message(title: string, text: string): Promise<void> {
  const dialog = new Dialog(title);
  dialog.content.append(body(text));
  dialog.addAction("OK", "ok", "primary");
  await dialog.open();
}

/** fluent_confirm: one modal yes/no question.  A warning confirms something
 * that cannot be taken back, so its confirm is the one white-on-red fill. */
export async function confirm(title: string, text: string, options: { confirmText?: string; cancelText?: string; kind?: "warning" | "info"; cancelRole?: Role } = {}): Promise<boolean> {
  const dialog = new Dialog(title);
  dialog.content.append(body(text));
  dialog.addAction(options.cancelText ?? "Cancel", "cancel", options.cancelRole ?? "standard");
  dialog.addAction(options.confirmText ?? "Confirm", "confirm", (options.kind ?? "warning") === "warning" ? "destructive-primary" : "primary");
  return await dialog.open() === "confirm";
}

/** A save's files already there (full paths, in one folder), as the host refused them: one name a
 * line and their folder once, then Cancel or the red Overwrite. Cancel is told to `status`: nothing
 * was saved and those files are kept. */
export async function confirmOverwrite(files: readonly string[], status: { note(text: string): void }): Promise<boolean> {
  const names = files.map(file => file.split(/[\\/]/).pop() ?? file), one = files.length === 1;
  const folder = files[0]!.slice(0, -names[0]!.length - 1);
  if (await confirm(one ? "File already exists" : "Files already exist", `${names.join("\n")}\n${one ? "is" : "are"} already in ${folder}\n\nOverwrite ${one ? "it" : "them"}?`,
                    { confirmText: "Overwrite" })) return true;
  status.note(`nothing was saved; ${names.join(" and ")} kept`);
  return false;
}

/** FluentInputDialog: one line of text, OK commits it, Cancel or Escape gives null. */
export async function prompt(title: string, label: string, value = ""): Promise<string | null> {
  const dialog = new Dialog(title);
  const edit = element("input");
  edit.type = "text";
  edit.spellcheck = false;
  edit.value = value;
  const words = element("label", "field-label", label);
  const line = element("div", "dialog-field");
  line.append(words, edit);
  dialog.content.append(line);
  dialog.addAction("Cancel", "cancel");
  const ok = dialog.addAction("OK", "ok", "primary");
  edit.addEventListener("keydown", event => { if (event.key === "Enter") ok.click(); });
  const answer = dialog.open();
  edit.focus();
  edit.select();
  return await answer === "ok" ? edit.value : null;
}
