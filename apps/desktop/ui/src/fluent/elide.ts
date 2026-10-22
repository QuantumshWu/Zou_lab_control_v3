// A line too long for its box ends in an ellipsis, as zlc_ui's labels do
// (QFontMetrics.elidedText); it never runs over its neighbours or is cut
// without one.  The right side is CSS (`.elided` and the other one-line
// classes of fluent.css: text-overflow: ellipsis), so nothing is measured.
// The middle -- only where v2 elides in the middle: a value or a path, a
// node's name, a form label, an info tree name, a panel's producer and axes --
// is measured here with the box's own font, each text once per font, and
// refitted only when its box is resized or its text changes: all the boxes
// a state changed are read together and then written together, so a state
// costs one layout however many it touches.
import { setText } from "./dom.js";

/** Where a text gives up room: its tail (Qt ElideRight) or its middle (ElideMiddle). */
export type Elide = "right" | "middle";
/** v2's tooltips: ElidedLabel and the readouts always say the whole text,
 * FluentCheckBox and a form label only while it is cut. */
export type Tip = "always" | "cut" | "none";

const ELLIPSIS = "…";
let pen: CanvasRenderingContext2D | null | undefined;
const widths = new Map<string, Map<string, number>>();

/** The advance of a text in a CSS font, each text measured once per font. */
export function measurer(font: string): (text: string) => number {
  if (pen === undefined) pen = document.createElement("canvas").getContext("2d");
  let known = widths.get(font);
  if (!known) widths.set(font, known = new Map());
  const cache = known;
  return text => {
    let width = cache.get(text);
    if (width === undefined) {
      if (!pen) return 0;
      if (pen.font !== font) pen.font = font;
      width = pen.measureText(text).width;
      cache.set(text, width);
    }
    return width;
  };
}

/** A node's font as the canvas takes it (the `font` shorthand is empty when a font-variant has no shorthand spelling). */
export function fontOf(style: CSSStyleDeclaration): string {
  return `${style.fontStyle} ${style.fontWeight} ${style.fontSize} ${style.fontFamily}`;
}

const graphemes = (text: string): string[] => [...new Intl.Segmenter().segment(text)].map(part => part.segment);

/** Qt's ElideMiddle (QTextEngine::elidedText): a text wider than `width` keeps
 * its two ends, a grapheme more of each while both fit beside the ellipsis. */
export function elideMiddle(text: string, width: number, measure: (text: string) => number): string {
  if (measure(text) <= width) return text;
  const parts = graphemes(text);
  const available = width - measure(ELLIPSIS);
  let [left, right, nextLeft, nextRight, leftWidth, rightWidth] = [0, parts.length, 0, parts.length, 0, 0];
  do {
    [left, right] = [nextLeft, nextRight];
    nextLeft += 1;
    nextRight -= 1;
    leftWidth += measure(parts[nextLeft - 1]!);
    rightWidth += measure(parts[nextRight]!);
  } while (nextLeft < parts.length && nextRight > 0 && leftWidth + rightWidth < available);
  return `${parts.slice(0, left).join("")}${ELLIPSIS}${parts.slice(right).join("")}`;
}

/** Qt's ElideRight: the head that fits beside the ellipsis. */
export function elideRight(text: string, width: number, measure: (text: string) => number): string {
  if (measure(text) <= width) return text;
  const parts = graphemes(text);
  const available = width - measure(ELLIPSIS);
  let kept = 0, used = 0;
  while (kept < parts.length && used + measure(parts[kept]!) <= available) used += measure(parts[kept++]!);
  return `${parts.slice(0, kept).join("")}${ELLIPSIS}`;
}

/** What a node says in full: its text before any eliding. */
export function wholeText(node: HTMLElement): string {
  const entry = fitted.get(node);
  return entry?.kind === "middle" ? entry.text : entry?.kind === "fragments" ? entry.fragments.map(fragment => fragment.text).join("") : node.textContent ?? "";
}

/** One coloured piece of a line: `elide` is how it gives up room, null for
 * structure that never does (a bracket, a separator, an id). */
export type Fragment = { text: string; className?: string; elide: Elide | null };

type Fitted = { kind: "middle"; text: string; tip: Tip } | { kind: "fragments"; fragments: readonly Fragment[]; signature: string };
const fitted = new WeakMap<HTMLElement, Fitted>();
/** What a fragment line last drew (its shown texts and classes). */
const drawn = new WeakMap<HTMLElement, string>();
const pending = new Set<HTMLElement>();
const resized = new ResizeObserver(entries => fitAll(entries.map(entry => entry.target as HTMLElement)));

function watch(node: HTMLElement, entry: Fitted): void {
  if (!fitted.has(node)) resized.observe(node);
  fitted.set(node, entry);
  // After the change that asked for it, with every other one of that change.
  if (!pending.size) queueMicrotask(() => { const nodes = [...pending]; pending.clear(); fitAll(nodes); });
  pending.add(node);
}

/** Read every node's room (one layout), then write every text. */
function fitAll(nodes: readonly HTMLElement[]): void {
  const writes = nodes.map(node => {
    const entry = fitted.get(node);
    if (!entry || !node.isConnected) return null;
    const style = getComputedStyle(node);
    // The content box: a node's width comes from its layout (a column, flex: 1), never from its text;
    // read to the fraction (clientWidth's whole CSS px would cut a text that fits by a fraction).
    const inset = [style.paddingLeft, style.paddingRight, style.borderLeftWidth, style.borderRightWidth].reduce((sum, side) => sum + (parseFloat(side) || 0), 0);
    const width = node.clientWidth ? node.getBoundingClientRect().width - inset : 0;
    return { node, entry, width, inset, measure: measurer(fontOf(style)) };
  });
  for (const write of writes) {
    if (!write) continue;
    const { node, entry, width, inset, measure } = write;
    if (entry.kind === "middle") {
      const shown = width > 0 ? elideMiddle(entry.text, width, measure) : entry.text;
      setText(node, shown);
      // A box sized by its words takes the whole text's room (--whole-width, QLabel's sizeHint), never
      // what it shows: sized by an elided text it would cut more on every refit and never grow back.
      const whole = `${Math.ceil(measure(entry.text) * devicePixelRatio) / devicePixelRatio + inset}px`;
      if (node.style.getPropertyValue("--whole-width") !== whole) node.style.setProperty("--whole-width", whole);
      if (entry.tip !== "none") {
        const title = entry.tip === "always" || shown !== entry.text ? entry.text : "";
        if (node.title !== title) node.title = title;
      }
    } else showFragments(node, entry.fragments, width, measure);
  }
}

/** `node` says `text`, its middle given up while it is wider than the node
 * (v2 ElidedLabel(mode=ElideMiddle), set_display_elide(ElideMiddle)). */
export function setMiddle(node: HTMLElement, text: string, tip: Tip = "always"): void {
  const entry = fitted.get(node);
  if (entry?.kind === "middle" && entry.text === text && entry.tip === tip) return;
  // Whole at once (what reads it in this turn reads it all); cut before it is painted.
  setText(node, text);
  watch(node, { kind: "middle", text, tip });
}

/** v2 elide_fragments: a line of fragments shortened to its node's width with
 * its structure kept -- the fixed pieces whole, the room left shared between
 * the others by equal claim (a piece that fits in its share hands the rest
 * back), none below an ellipsis. */
export function setFragments(node: HTMLElement, fragments: readonly Fragment[]): void {
  const signature = JSON.stringify(fragments);
  const entry = fitted.get(node);
  if (entry?.kind === "fragments" && entry.signature === signature) return;
  watch(node, { kind: "fragments", fragments: fragments.map(fragment => ({ ...fragment })), signature });
}

function showFragments(node: HTMLElement, fragments: readonly Fragment[], width: number, measure: (text: string) => number): void {
  const natural = fragments.map(fragment => measure(fragment.text));
  const shown = fragments.map(fragment => fragment.text);
  if (width > 0 && natural.reduce((sum, value) => sum + value, 0) > width) {
    const fixed = fragments.reduce((sum, fragment, at) => sum + (fragment.elide ? 0 : natural[at]!), 0);
    let claimants = fragments.flatMap((fragment, at) => fragment.elide && natural[at] ? [at] : []);
    const floor = measure(ELLIPSIS);
    let remaining = Math.max(0, width - fixed);
    while (claimants.length) {
      const share = remaining / claimants.length;
      const satisfied = claimants.filter(at => natural[at]! <= share);
      if (!satisfied.length) {
        for (const at of claimants) {
          const fragment = fragments[at]!;
          const allotted = Math.max(Math.floor(share), floor);
          shown[at] = fragment.elide === "middle" ? elideMiddle(fragment.text, allotted, measure) : elideRight(fragment.text, allotted, measure);
        }
        break;
      }
      for (const at of satisfied) remaining -= natural[at]!;
      claimants = claimants.filter(at => !satisfied.includes(at));
    }
  }
  const signature = JSON.stringify([shown, fragments.map(fragment => fragment.className ?? "")]);
  if (drawn.get(node) === signature) return;
  drawn.set(node, signature);
  node.replaceChildren(...fragments.map((fragment, at) => {
    if (!fragment.className) return document.createTextNode(shown[at]!);
    const span = document.createElement("span");
    span.className = fragment.className;
    span.textContent = shown[at]!;
    return span;
  }));
}

// A tooltip only while the text is cut (FluentCheckBox; `data-tip-cut`):
// decided when the pointer comes, so nothing is measured until somebody points
// at it.  The words that clip are the node's `.words`, else the node itself; a
// tooltip its owner set is kept.
const tipped = new WeakMap<HTMLElement, string>();
document.addEventListener("pointerover", event => {
  const node = (event.target as Element | null)?.closest?.<HTMLElement>("[data-tip-cut]");
  if (!node) return;
  if (node.title !== "" && node.title !== (tipped.get(node) ?? "")) return;
  const words = node.querySelector<HTMLElement>(".words") ?? node;
  const tip = words.scrollWidth > words.clientWidth ? words.textContent ?? "" : "";
  tipped.set(node, tip);
  if (node.title !== tip) node.title = tip;
}, { passive: true });
