// The wire: commands the window sends on the bus and the accepted state it
// is told.  Every UI action is one of these commands; the window keeps no
// state of its own beyond drafts, focus and which page it shows.
import type { FieldSpec } from "./fluent/index.js";
export type { Value } from "./fluent/index.js";

/** Backend transform: the UI reads only its normalized hit-test rectangle. */
export type Axis = Readonly<{ bounds: readonly [number, number, number, number]; [key: string]: unknown }>;
/** A panel setting: a form field and the configure group it belongs to. */
/** `bound`: where a live panel's Window (its binding's row) stands among the display rows. */
export type Control = FieldSpec & { section: "binding" | "semantic" | "display" | "board"; bound?: boolean };
export type Signal = { id: string; label: string; available: boolean; reason?: string };
export type PlotKind = { id: string; label: string };
export type Panel = {
  id: string;
  title: string;
  kind: string;
  focus: number | null;
  binding?: { signal: string; window: number };
  source_signal?: string;
  /** What the card calls the panel when it is not its id (a viewer's own panel-<k>). */
  caption_id?: string;
  structure?: [string, number][][];
  axes?: Axis[];
  controls?: Control[];
  fit?: { models: { key: string; label: string; parameters?: [string, string][] }[]; selected: string | null; expression: string; configuration: Record<string, unknown>; error?: string };
  error?: string;
  message?: string;
  authoring_pending?: boolean;
  target_request?: number | null;
  frame?: { width: number; height: number; sequence: number };
  /** Authored logical dimensions for an empty panel, before a raster exists. */
  display_size?: { width: number; height: number };
  /** The size preset the panel draws at, and the presets it offers (zlc_plot PANEL_PRESETS). */
  size?: string | null;
  size_choices?: string[];
  /** A point review's candidates: id, label, x, y on the image's axes, and where each
   * stands on the picture (canvas fractions, null before it is drawn). */
  review?: { node: string; points: [string, string, number, number, number | null, number | null][] };
};
export type State = {
  type: "state";
  request?: number;
  document?: { path: string; title: string };
  panels?: Panel[];
  signals?: Signal[];
  plot_kinds?: PlotKind[];
  acquisition?: { state: "idle" | "starting" | "running" | "stopping" | "stopped" | "error"; produced?: number; error?: string };
  /** The Figure Viewer's own document: the opened archive, its panels and the Data it offers. */
  viewer?: { document: { path: string; title: string; info?: unknown; flow?: unknown }; panels: Panel[]; signals: Signal[]; plot_kinds?: PlotKind[]; error?: string | null };
  error?: string;
  busy?: boolean;
  message?: string;
  [section: string]: unknown;
};
export type Frame = { type: "frame"; mapping: string | number; slot: number; panel: string; sequence: number; width: number; height: number };
export type MapInfo = { type: "shared-rgba-map"; mapping: string | number; slot: number; panel: string; width: number; height: number; stride: number; byte_length: number };
export type Command = { action: string; panel?: string; [key: string]: unknown };
export type Send = (command: Command, feedback?: string) => number;
export type { FieldSpec };

type SharedEvent = { additionalData: MapInfo | string; getBuffer(): ArrayBuffer };
type WebView = {
  addEventListener(type: "message", callback: (event: { data: unknown }) => void): void;
  addEventListener(type: "sharedbufferreceived", callback: (event: SharedEvent) => void): void;
  releaseBuffer(buffer: ArrayBuffer): void;
};
declare global {
  interface Window { ipc?: { postMessage(message: string): void }; chrome?: { webview?: WebView } }
}

