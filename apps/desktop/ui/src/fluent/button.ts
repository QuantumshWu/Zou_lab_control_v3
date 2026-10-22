import { assign, element } from "./dom.js";
import { fitWords, setGlyph } from "./ink.js";

/** What a command DOES, which is all its face may say (zlc_ui BUTTON_ROLES):
 * `primary` the surface's ONE run or commit command (ACCENT_STRONG, the only
 * dark blue face); `standard` every other command (BUTTON_TINT); `quiet` a
 * one-glyph command that closes a view (a ghost, never red); `destructive`
 * anything that takes away, stops or shuts down (RED_TINT, RED_DARK ink);
 * `destructive-primary` the confirm of a destructive question (white on
 * RED_DARK).  Active and armed are states, not roles. */
export type Role = "standard" | "primary" | "quiet" | "destructive" | "destructive-primary";
const ROLE_CLASSES: readonly Role[] = ["primary", "quiet", "destructive", "destructive-primary"];

/** How long a two-click remover waits for its second click (arm_interval_ms:
 * the system double-click interval, never under a quarter second). */
export const ARM_INTERVAL_MS = 500;

/** FluentButton(confirm_twice=...): the first click arms the white-on-red
 * face, a second within the interval acts; left alone it returns to rest.
 * Only removing a plot panel or a logic node is armed. */
export function armable(node: HTMLButtonElement, phrase: string, confirmed: () => void): HTMLButtonElement {
  let timer = 0;
  const rest = (): void => { window.clearTimeout(timer); timer = 0; node.classList.remove("armed"); node.title = `Click twice to ${phrase}`; };
  node.title = `Click twice to ${phrase}`;
  node.addEventListener("click", () => {
    if (node.classList.contains("armed")) { rest(); confirmed(); return; }
    node.classList.add("armed");
    node.title = `Click again to ${phrase}`;
    timer = window.setTimeout(rest, ARM_INTERVAL_MS);
  });
  // Hidden, disabled or left: back to rest (v2 disarms on hide/disable).
  node.addEventListener("blur", rest);
  return node;
}

export type CommandOptions = {
  role?: Role;
  /** A one-symbol label (×): larger, centred on its ink, little padding. */
  glyph?: boolean;
  /** As wide as it is tall (a glyph command). */
  square?: boolean;
  title?: string;
  /** Every label this command can say: its slot keeps the widest, so what
   * stands beside it never moves (fit_button_words). */
  words?: readonly string[];
  /** Two-click remover: the phrase its tooltip words ("remove this node"). */
  confirm?: string;
};

/** FluentButton: a command drawn by its role.  The label is never elided;
 * a toggle command shows that it is on with `active`, set by its owner from
 * accepted state, never by its own click. */
export class Command {
  readonly root: HTMLButtonElement;
  private label: string;
  private role: Role;
  private dirty = false;
  private readonly words: HTMLSpanElement | null = null;

  constructor(text: string, action: () => void, options: CommandOptions = {}) {
    this.root = element("button");
    this.root.type = "button";
    this.label = text;
    this.role = options.role ?? "standard";
    if (options.glyph) this.root.classList.add("glyph");
    if (options.square) this.root.classList.add("square");
    if (options.title) this.root.title = options.title;
    if (options.words?.length) {
      // Every word in one grid cell: the slot is the widest of them, only
      // the current one is visible (v2 measures sizeHint for each word).
      this.words = element("span", "words");
      this.root.append(this.words);
      for (const word of options.words) this.words.append(element("span", "", word));
    }
    if (options.confirm) armable(this.root, options.confirm, action);
    else this.root.addEventListener("click", action);
    this.show();
  }

  get text(): string { return this.label; }
  set text(text: string) { if (text !== this.label) { this.label = text; this.show(); } }
  set disabled(disabled: boolean) { assign(this.root, "disabled", disabled); if (disabled && this.root.classList.contains("armed")) this.root.classList.remove("armed"); }
  get disabled(): boolean { return this.root.disabled; }
  set hidden(hidden: boolean) { this.root.hidden = hidden; }
  set title(text: string) { if (this.root.title !== text) this.root.title = text; }

  setRole(role: Role): void { if (role !== this.role) { this.role = role; this.show(); } }
  /** set_active: the SELECTED face a toggle command keeps while it is on. */
  setActive(active: boolean): void { this.root.classList.toggle("active", active); }
  /** set_dirty: pressing this would save or apply something new -- drawn
   * primary with a trailing `*`, the one thing on its surface to press. */
  setDirty(dirty: boolean): void { if (dirty !== this.dirty) { this.dirty = dirty; this.show(); } }

  private show(): void {
    const shown = this.dirty && !this.label.endsWith("*") ? `${this.label}*` : this.label;
    const role = this.dirty ? "primary" : this.role;
    for (const name of ROLE_CLASSES) this.root.classList.toggle(name, name === role);
    if (this.root.classList.contains("glyph")) { setGlyph(this.root, shown, "glyph"); return; }
    if (!this.words) { if (this.root.textContent !== shown) this.root.textContent = shown; fitWords(this.root, [shown]); return; }
    let current = [...this.words.children].find(node => node.textContent === shown);
    if (!current) { current = element("span", "", shown); this.words.append(current); }
    for (const node of this.words.children) node.classList.toggle("current", node === current);
    fitWords(this.root, [...this.words.children].map(node => node.textContent ?? ""));
  }
}

/** A command button: `command("Save…", save, { role: "primary" }).root`. */
export function command(text: string, action: () => void, options: CommandOptions = {}): Command {
  return new Command(text, action, options);
}
