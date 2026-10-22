import { command, type Command } from "./button.js";
import { element, setText, setTitle } from "./dom.js";

/** The one status vocabulary (zlc_ui STATUS_CUES): `idle` nothing under way
 * and nothing wrong, `running` something lasting that is live, `task` an
 * operation under way or just done, `warning` needs attention, `error`
 * failed.  Dot colour and ink both come from this word. */
export type StatusWord = "idle" | "running" | "task" | "warning" | "error";

/** One status word -> its dot and ink. */
export function setState(node: HTMLElement, word: StatusWord): void {
  if (node.dataset.state !== word) node.dataset.state = word;
}

/** FluentStatusDot: a round severity or state mark. */
export function statusDot(word: StatusWord = "idle", title = ""): HTMLSpanElement {
  const dot = element("span", "dot");
  dot.setAttribute("aria-hidden", "true");
  setState(dot, word);
  if (title) dot.title = title;
  return dot;
}

/** A dot and its words, a status gap apart, the words in the word's ink. */
export class StatusLine {
  readonly root = element("div", "status-line");
  readonly dot = statusDot();
  readonly text = element("span", "status-text ink");
  constructor(text = "", word: StatusWord = "idle") {
    this.root.append(this.dot, this.text);
    this.show(text, word);
  }
  show(text: string, word: StatusWord): void {
    setState(this.dot, word);
    setState(this.text, word);
    setText(this.text, text);
    setTitle(this.text, text);
  }
}

/** FluentStatusStrip: the persistent one-line status surface, mounted once
 * and fed forever -- a dot, one eliding line (all of it in the tooltip) and
 * an optional destructive action at its end (the console's Stop task).  The
 * newest message is the true one; an empty one falls back to the last idle
 * line rather than bringing an older error back (StatusStrip.show_status). */
export class StatusStrip {
  readonly root = element("div", "card strip");
  readonly dot = statusDot();
  readonly text = element("span", "strip-text");
  readonly trailing = element("div", "strip-trailing");
  readonly action: Command | null = null;
  private idleText = "";

  constructor(options: { action?: [string, () => void] } = {}) {
    this.root.role = "status";
    this.root.setAttribute("aria-live", "polite");
    this.root.append(this.dot, this.text, this.trailing);
    if (options.action) {
      this.action = command(options.action[0], options.action[1], { role: "destructive" });
      this.action.hidden = true;
      this.trailing.append(this.action.root);
    }
    this.show("", "idle");
  }

  show(text: string, word: StatusWord): void {
    if (!text) { word = "idle"; text = this.idleText; }
    else if (word === "idle") this.idleText = text;
    setState(this.root, word);
    setState(this.dot, word);
    // Something under way, or failed: said by class too (the live-chain
    // acceptance waits on #status.busy clearing).
    this.root.classList.toggle("busy", word === "task");
    this.root.classList.toggle("error", word === "error");
    // One eliding line, the whole message in its tooltip.
    setText(this.text, text.replace(/\s*\n\s*/g, " · "));
    setTitle(this.root, text);
  }

  setActionVisible(visible: boolean): void { if (this.action) this.action.hidden = !visible; }
}
