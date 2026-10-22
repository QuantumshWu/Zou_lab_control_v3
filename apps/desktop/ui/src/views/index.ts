// The windows the host opens, one page each (`index.html?window=<view>&…`),
// with v2's names; a line adding a window changes its one entry here.
import type { ViewFactory } from "../shell.js";
import type { PanelBoard } from "./board.js";
import { deviceManager } from "./device_manager.js";
import { deviceControl, deviceLog } from "./device_windows.js";
import { figureViewer } from "./figure_viewer.js";
import { reviewWindow } from "./point_review.js";
import { slmEditor } from "./slm_editor.js";
import { gallery } from "./gallery.js";
import { installation } from "./installation.js";
import { pulseEditor } from "../pulse/view.js";
import { taskConsole } from "./task_console.js";

/** The page's query: which window, of what. */
const query = new URLSearchParams(location.search);

export const WINDOWS: Record<string, (board: PanelBoard) => ViewFactory> = {
  "devices": () => deviceManager,
  "task-console": board => taskConsole(board),
  "figure-viewer": board => figureViewer(board),
  "pulse-editor": board => pulseEditor(board),
  // A device's own windows (v2 open_device_control and its log).
  "control": () => deviceControl(query.get("device") ?? ""),
  "log": () => deviceLog(query.get("device") ?? ""),
  // A run's point review (v2 PointReviewView's own window).
  "review": board => reviewWindow(board, query.get("node") ?? ""),
  // An SLM's Control: its SLM Editor (v2 open_slm_control).
  "slm-editor": board => slmEditor(board, query.get("device") ?? ""),
  // installation.exe's one window.
  "installation": () => installation,
  // Development only: every control with fake data.
  "gallery": () => gallery,
};

/** Where this window numbers its commands from (windows.rs gives each window its own base), so answers never cross windows. */
export const FIRST_REQUEST = Number(query.get("base")) || 1;

/** Which window this page is: the host's `?window=`, else a development hash (`#gallery`). */
export function windowKey(): string {
  const named = query.get("window") ?? location.hash.slice(1);
  return named in WINDOWS ? named : "task-console";
}
