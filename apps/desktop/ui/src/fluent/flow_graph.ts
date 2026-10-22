import { element } from "./dom.js";
import { elideMiddle, elideRight, measurer } from "./elide.js";
import { pointPx } from "./ink.js";

/** One node of a graph the owner projects (v2 InfoPane.set_graph): a Logic
 * run or a Device, its words, and the info row it stands for (a click on the
 * card opens that row, so the picture is a map of the tabs). */
export type GraphNode = { id: string; kind: "logic" | "device"; title: string; subtitle: string; root: boolean; tooltip: string; row: [string, string] | null };
export type GraphEdge = { source: string; target: string; kind: "causal" | "device"; label: string };
export type Graph = { nodes: GraphNode[]; edges: GraphEdge[] };

const SVG = "http://www.w3.org/2000/svg";
const svg = <K extends keyof SVGElementTagNameMap>(tag: K, attributes: Record<string, string | number> = {}): SVGElementTagNameMap[K] => {
  const node = document.createElementNS(SVG, tag);
  for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, String(value));
  return node;
};

/** FlowView: v2's layered Flow of an archive's saved lineage. Devices stand
 * on the top layer; each Logic run one layer below its deepest parent; a
 * layer is ordered by its neighbours' barycentres (two sweeps), centred, and
 * cards shrink to fit the pane down to a minimum. Causal edges are GRAPHITE
 * curves, device edges dashed ORANGE; an edge that skips a layer runs in a
 * lane right of the cards. */
export class FlowGraph {
  readonly root = element("div", "flow-graph");
  rowRequested: (tab: string, label: string) => void = () => {};
  private graph: Graph = { nodes: [], edges: [] };
  private drawnWidth = 0;

  constructor() {
    new ResizeObserver(() => { if (this.root.clientWidth && this.root.clientWidth !== this.drawnWidth) this.draw(); }).observe(this.root);
  }

  set(graph: Graph): void { this.graph = graph; this.draw(); }

  private draw(): void {
    const style = getComputedStyle(document.documentElement);
    const qt = parseFloat(style.getPropertyValue("--qt")) || 1, fluent = parseFloat(style.getPropertyValue("--fluent")) || 1;
    const px = (value: number, minimum = 1): number => Math.max(minimum, Math.round(value * fluent)) * qt;
    const fontPt = Math.max(8, Math.round(12 * fluent));
    // QFont.setBold(True) is weight 700.
    const font = (pt: number, bold = false): string => `${bold ? "700 " : ""}${pointPx(pt) * qt}px "Segoe UI", sans-serif`;
    const color = (name: string): string => style.getPropertyValue(name).trim();
    this.drawnWidth = this.root.clientWidth;
    const { nodes, edges } = this.graph;
    if (!nodes.length) {
      this.root.replaceChildren(element("div", "flow-empty", "No saved Logic flow"));
      return;
    }
    // Ranks: devices 0, Logic by the longest causal path from a source (1 at a source).
    const logic = nodes.filter(node => node.kind === "logic").map(node => node.id);
    const causal = edges.filter(edge => edge.kind === "causal");
    const rank = new Map<string, number>();
    const pending = new Map(logic.map(id => [id, causal.filter(edge => edge.target === id).length]));
    const ready = logic.filter(id => pending.get(id) === 0);
    for (const id of ready) rank.set(id, 1);
    while (ready.length) {
      const id = ready.shift()!;
      for (const edge of causal.filter(edge => edge.source === id)) {
        rank.set(edge.target, Math.max(rank.get(edge.target) ?? 1, rank.get(id)! + 1));
        pending.set(edge.target, pending.get(edge.target)! - 1);
        if (pending.get(edge.target) === 0) ready.push(edge.target);
      }
    }
    for (const node of nodes) if (node.kind === "device") rank.set(node.id, 0);
    const layers = new Map<number, string[]>();
    for (const node of nodes) { const at = rank.get(node.id) ?? 1; layers.set(at, [...(layers.get(at) ?? []), node.id]); }
    const ranks = [...layers.keys()].sort((a, b) => a - b);
    const positions = (): Map<string, number> => new Map([...layers.values()].flatMap(layer => layer.map((id, index) => [id, (index + 0.5) / layer.length] as [string, number])));
    const reorder = (at: number, fromPredecessors: boolean): void => {
      const layer = layers.get(at)!, place = positions(), original = new Map(layer.map((id, index) => [id, index]));
      const score = (id: string): number => {
        const neighbours = edges.filter(edge => (fromPredecessors ? edge.target : edge.source) === id)
          .map(edge => fromPredecessors ? edge.source : edge.target).filter(other => rank.get(other) !== at);
        return neighbours.length ? neighbours.reduce((sum, other) => sum + place.get(other)!, 0) / neighbours.length : place.get(id)!;
      };
      layer.sort((a, b) => score(a) - score(b) || original.get(a)! - original.get(b)!);
    };
    for (let sweep = 0; sweep < 2; sweep++) {
      for (const at of ranks.slice(1)) reorder(at, true);
      for (const at of ranks.slice(0, -1).reverse()) reorder(at, false);
    }
    const cardHeight = px(62, 50), gapX = px(16, 12), gapY = px(58, 44), margin = px(18, 14);
    // v2 settles the width once, as the pane gives it less a small inset and the scroll bar's room.
    const available = Math.max(px(220, 190), this.root.clientWidth - px(8, 6) - px(12, 10) - 2 * margin);
    const geometry = new Map(ranks.map(at => {
      const count = layers.get(at)!.length;
      const width = Math.max(px(105, 90), Math.min(px(210, 176), (available - (count - 1) * gapX) / count));
      return [at, { card: width, layer: count * width + Math.max(0, count - 1) * gapX }] as const;
    }));
    const widest = Math.max(available, ...[...geometry.values()].map(entry => entry.layer));
    const rects = new Map<string, { x: number; y: number; w: number; h: number }>();
    let y = margin;
    for (const at of ranks) {
      const { card, layer } = geometry.get(at)!;
      let x = margin + (widest - layer) / 2;
      for (const id of layers.get(at)!) { rects.set(id, { x, y, w: card, h: cardHeight }); x += card + gapX; }
      y += cardHeight + gapY;
    }
    const long = edges.filter(edge => (rank.get(edge.target) ?? 0) - (rank.get(edge.source) ?? 0) > 1);
    const laneBase = margin + widest + px(22, 16), laneStep = px(12, 9);
    const lanes = new Map(long.map((edge, index) => [edge, laneBase + index * laneStep]));
    const outgoing = new Map<string, GraphEdge[]>(), incoming = new Map<string, GraphEdge[]>();
    for (const edge of edges) { outgoing.set(edge.source, [...(outgoing.get(edge.source) ?? []), edge]); incoming.set(edge.target, [...(incoming.get(edge.target) ?? []), edge]); }
    const centre = (id: string): number => rects.get(id)!.x + rects.get(id)!.w / 2;
    for (const list of outgoing.values()) list.sort((a, b) => centre(a.target) - centre(b.target));
    for (const list of incoming.values()) list.sort((a, b) => centre(a.source) - centre(b.source));
    const width = Math.max(widest + 2 * margin, ...[...lanes.values()].map(lane => lane + margin));
    const height = Math.max(...[...rects.values()].map(rect => rect.y + rect.h)) + margin;
    const picture = svg("svg", { width, height, viewBox: `0 0 ${width} ${height}` });
    const stroke = px(1.4, 1), arrow = px(6, 5), shoulder = px(16, 12);
    for (const edge of edges) {
      const from = rects.get(edge.source), to = rects.get(edge.target);
      if (!from || !to) continue;
      const out = outgoing.get(edge.source)!, into = incoming.get(edge.target)!;
      const start = { x: from.x + (out.indexOf(edge) + 1) / (out.length + 1) * from.w, y: from.y + from.h };
      const end = { x: to.x + (into.indexOf(edge) + 1) / (into.length + 1) * to.w, y: to.y };
      const lane = lanes.get(edge);
      const middle = (start.y + end.y) / 2;
      const d = lane !== undefined
        ? `M${start.x} ${start.y}L${start.x} ${start.y + shoulder}L${lane} ${start.y + shoulder}L${lane} ${end.y - shoulder}L${end.x} ${end.y - shoulder}L${end.x} ${end.y}`
        : `M${start.x} ${start.y}C${start.x} ${middle} ${end.x} ${middle} ${end.x} ${end.y}`;
      const ink = color(edge.kind === "device" ? "--orange" : "--graphite");
      const path = svg("path", { d, fill: "none", stroke: ink, "stroke-width": stroke, ...(edge.kind === "device" ? { "stroke-dasharray": `${4 * stroke} ${2 * stroke}` } : {}) });
      path.append(Object.assign(svg("title"), { textContent: edge.label || (edge.kind === "device" ? "device use" : "causal input") }));
      const head = svg("polygon", { points: `${end.x},${end.y} ${end.x - arrow},${end.y - arrow * 1.35} ${end.x + arrow},${end.y - arrow * 1.35}`, fill: ink });
      picture.append(path, head);
    }
    const fit = (text: string, fontSpec: string, room: number, middle: boolean): string =>
      (middle ? elideMiddle : elideRight)(text, room, measurer(fontSpec));
    const inset = px(10, 8);
    for (const node of nodes) {
      const rect = rects.get(node.id)!, device = node.kind === "device";
      const card = svg("g", { class: `flow-card${node.row ? " opens" : ""}` });
      card.append(Object.assign(svg("title"), { textContent: node.tooltip }));
      card.append(svg("rect", { x: rect.x, y: rect.y, width: rect.w, height: rect.h, rx: px(5, 4), fill: color(device ? "--orange-tint" : "--surface"),
        stroke: color(device ? "--orange" : "--divider"), "stroke-width": px(1) }));
      const words = (text: string, at: number, size: number, ink: string, bold = false, middle = false): SVGTextElement => {
        const spec = font(size, bold);
        // A QGraphicsSimpleTextItem's position is the top of its line (the font's ascent above the baseline).
        const line = svg("text", { x: rect.x + inset, y: rect.y + at, fill: ink, "dominant-baseline": "text-before-edge", style: `font: ${spec}` });
        line.textContent = fit(text, spec, rect.w - 2 * inset, middle);
        return line;
      };
      card.append(words(`${device ? "Device" : "Logic"}${node.root ? " · this figure" : ""}`, px(5, 4), Math.max(6, fontPt - 4), color(device ? "--orange-dark" : "--muted-ink")),
        words(node.title, px(19, 15), Math.max(8, fontPt - 1), color("--text"), true, true),
        words(node.subtitle, px(40, 32), Math.max(7, fontPt - 3), color("--muted-ink")));
      if (node.row) { const [tab, label] = node.row; card.addEventListener("click", () => this.rowRequested(tab, label)); }
      picture.append(card);
    }
    this.root.replaceChildren(picture);
  }
}
