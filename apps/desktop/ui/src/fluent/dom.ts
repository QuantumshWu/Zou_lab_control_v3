// DOM helpers every control is built from.  No framework: a control is a
// class or function that owns its elements and is told its state; it never
// decides anything a command or the accepted state should decide.
import { fitWords, setGlyph } from "./ink.js";

export type Value = string | number | boolean | null;

export function element<K extends keyof HTMLElementTagNameMap>(tag: K, className = "", text = ""): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

/** A plain command button (FluentButton, standard role unless `className`
 * says otherwise; a `glyph` one is a one-symbol label centred on its ink).
 * See button.ts for roles, the dirty star and two-click removers. */
export function button(text: string, action: () => void, className = ""): HTMLButtonElement {
  const node = element("button", className);
  if (node.classList.contains("glyph")) setGlyph(node, text, "glyph");
  else { node.textContent = text; if (text) fitWords(node, [text]); }
  node.type = "button";
  node.addEventListener("click", action);
  return node;
}

/** Set text only when it differs: a state arrives many times a second, and
 * rewriting unchanged text restarts selections and tooltips. */
export function setText(node: Element, text: string): void {
  if (node.textContent !== text) node.textContent = text;
}

export function setTitle(node: HTMLElement, text: string): void {
  if (node.title !== text) node.title = text;
}

/** ElidedLabel: one line that ends in an ellipsis when it does not fit, the
 * whole text in its tooltip. */
export function elided(text = "", className = ""): HTMLSpanElement {
  const node = element("span", `elided ${className}`.trim(), text);
  node.title = text;
  return node;
}

export function setElided(node: HTMLElement, text: string, tooltip = text): void {
  setText(node, text);
  setTitle(node, tooltip);
}

const ACRONYMS = new Set(["api", "dac", "daq", "id", "psf", "rf", "roi", "slm", "ttl"]);
/** zlc_ui.status.name_words: a snake_case name read as words, sentence case, the lab's acronyms in capitals. */
export function nameWords(name: string): string {
  const words = name.replace(/_/g, " ").split(/\s+/).filter(Boolean).map(word => ACRONYMS.has(word.toLowerCase()) ? word.toUpperCase() : word);
  if (words.length) words[0] = words[0]!.slice(0, 1).toUpperCase() + words[0]!.slice(1);
  return words.join(" ");
}

/** A muted one-line note (muted_note_label); `wrap` lets it run to several lines. */
export function note(text = "", wrap = false): HTMLParagraphElement {
  return element("p", wrap ? "note wrap" : "note", text);
}

let probe: HTMLDivElement | null = null;
/** A CSS length token in CSS px (tokens.css).  Read through a probe: a custom
 * property reads back as its unresolved calc() text, not as a length. */
export function token(name: string): number {
  if (!probe) {
    probe = document.createElement("div");
    probe.style.cssText = "position:absolute;visibility:hidden;height:0;left:0;top:0;";
    probe.setAttribute("aria-hidden", "true");
  }
  if (!probe.isConnected) document.body.append(probe);
  probe.style.width = `var(${name})`;
  return probe.getBoundingClientRect().width;
}

/** Replace `host`'s children with `nodes` only where they differ, so a node
 * keeps its focus, scroll position and hover across reconciliations. */
export function reconcileChildren(host: Element, nodes: readonly Element[]): void {
  let cursor = host.firstElementChild;
  for (const node of nodes) {
    if (node !== cursor) host.insertBefore(node, cursor);
    else cursor = cursor.nextElementSibling;
  }
  while (cursor) { const next = cursor.nextElementSibling; cursor.remove(); cursor = next; }
}

/** Keyed views: one view per key, created once, dropped when its key goes. */
export class Keyed<T extends { root: Element }> {
  readonly views = new Map<string, T>();
  constructor(private make: (key: string) => T, private drop: (view: T) => void = view => view.root.remove()) {}
  sync(keys: readonly string[]): T[] {
    const alive = new Set(keys);
    for (const [key, view] of this.views) if (!alive.has(key)) { this.drop(view); this.views.delete(key); }
    return keys.map(key => {
      let view = this.views.get(key);
      if (!view) { view = this.make(key); this.views.set(key, view); }
      return view;
    });
  }
  get(key: string): T | undefined { return this.views.get(key); }
}

/** v2 setting_label_width, the one rule every label column is: the widest words' advance in whole Qt px
 * (QFontMetrics; never less than the words, which are cut where they do not fit), the label trail after
 * them, never below `minimum` -- in whole device px up. */
export function labelColumnWidth(advances: readonly number[], minimum = token("--label-min")): number {
  const q = token("--q");
  const widest = Math.max(minimum, ...advances.map(advance => Math.max(Math.round(advance / q) * q, advance) + token("--label-trail")));
  return Math.ceil(widest * devicePixelRatio - 1e-3) / devicePixelRatio;
}
