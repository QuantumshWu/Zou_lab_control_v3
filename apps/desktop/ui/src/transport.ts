// The window's one connection to the native host: commands out on
// window.ipc, accepted states and frames in on chrome.webview.  The command
// queue is the host's; the window only remembers what it is waiting for, so
// every view can say "Updating…" or the last error in its status strip.
// Each window numbers its commands from its own base, and answers to
// another window's commands are not this window's to show.
import type { Command, Frame, MapInfo, State } from "./protocol.js";
import type { PathAsk } from "./fluent/index.js";
import type { CaptionState } from "./title_bar.js";

export type SharedBuffer = { additionalData: MapInfo | string; getBuffer(): ArrayBuffer };

export type Handlers = {
  state(state: State): void;
  frame(frame: Frame): void;
  shared(event: SharedBuffer): void;
  releaseMappings(mapping: string | number): void;
  /** A shared view's picture: its pixels, fetched from the instrument. */
  image(frame: Frame, image: ImageData): void;
  /** Something the queue or the status line should show again. */
  changed(): void;
  /** The host shows one of the view's pages (show_window {page}). */
  page?(page: string): void;
  /** The title bar's buttons as the host's caption window sees the pointer (frame.rs). */
  caption?(caption: CaptionState): void;
  /** A standalone Pulse Editor's close found unsaved drafts ("Pulse", "Config", "Subpulse"). */
  closeRequested?(unsaved: string[]): void;
};

/** Words a shared view says when something would change the instrument. */
const READ_ONLY = "This is a shared view: only the experiment computer changes anything.";

/** One change to the state this page holds (delta.rs): set, delete a key, or splice an array. */
type Change = [path: (string | number)[], value?: unknown, start?: number, removed?: number];
type Container = Record<string | number, unknown>;
const ENVELOPE = new Set(["type", "request", "error", "result", "changes"]);

/** Apply `changes` to `sections` copy-on-write: what changed is new, everything
 * else keeps its identity (a view can skip what is the same object as last
 * time). Returns the new sections and the names of the sections touched. */
export function applyChanges(sections: Container, changes: readonly Change[]): [Container, Set<string>] {
  const root: Container = { ...sections };
  const copied = new Set<unknown>([root]);
  const own = (parent: Container, key: string | number): Container => {
    let child = parent[key] as Container;
    if (!copied.has(child)) {
      child = (Array.isArray(child) ? child.slice() : { ...child }) as Container;
      parent[key] = child;
      copied.add(child);
    }
    return child;
  };
  const touched = new Set<string>();
  for (const [path, value, start, removed] of changes) {
    touched.add(String(path[0]));
    let parent = root;
    for (let at = 0; at < path.length - 1; at++) parent = own(parent, path[at]!);
    const last = path[path.length - 1]!;
    if (value === undefined && start === undefined) delete parent[last];
    else if (start === undefined) parent[last] = value;
    else (own(parent, last) as unknown as unknown[]).splice(start, removed!, ...(value as unknown[]));
  }
  return [root, touched];
}

export class Transport {
  readonly webview = window.chrome?.webview;
  /** Served to a browser by Browser sharing (share.rs): it watches, it never commands. */
  readonly shared = document.documentElement.dataset.share === "1";
  readonly connected = this.shared || Boolean(window.chrome?.webview && window.ipc);
  /** The shared view's ticket for pictures, once the operator allowed it. */
  private token = "";
  /** Per panel: the newest picture told of, and whether one is on its way. */
  private wanted = new Map<string, Frame>();
  private fetching = new Set<string>();
  /** Commands sent and not yet answered, with the words the status shows. */
  readonly pending = new Map<number, string>();
  lastError = "";
  /** What a finished command left to say until the next one (v2 _report: "screenshot saved to …"). */
  notice: [string, "idle" | "task" | "warning"] | null = null;
  private nextRequest: number;
  private answers = new Map<number, { resolve(result: unknown): void; reject(error: Error): void; pick(state: State, sections: Readonly<Container>): unknown }>();
  private handlers: Handlers | null = null;
  /** The state sections this page holds; each state message says what changed in them. */
  private sections: Container = {};

  constructor(private readonly firstRequest = 1) {
    this.nextRequest = firstRequest;
    if (!this.connected) this.lastError = "Open this interface in the ZLC desktop application.";
  }

  attach(handlers: Handlers): void {
    this.handlers = handlers;
    if (this.shared) { this.watch(); return; }
    this.webview?.addEventListener("sharedbufferreceived", event => handlers.shared(event as SharedBuffer));
    this.webview?.addEventListener("message", event => {
      try { this.receive(typeof event.data === "string" ? JSON.parse(event.data) : event.data); }
      catch (error) { this.report(error); }
    });
  }

  /** Send one command; `feedback` is what the status line says until it is answered. */
  send(command: Command, feedback?: string): number {
    const id = this.nextRequest++;
    if (this.shared) {
      // A pointer's own motion is no intent; anything else is said once, in the status line.
      if (command.action !== "pointer" && command.action !== "frontend_error") this.note(READ_ONLY, "warning");
      return id;
    }
    if (!window.ipc) { this.lastError = "The native desktop connection is unavailable."; this.handlers?.changed(); return id; }
    if (feedback) { this.pending.set(id, feedback); this.lastError = ""; this.notice = null; this.handlers?.changed(); }
    window.ipc.postMessage(JSON.stringify({ id, ...command }));
    return id;
  }

  /** Send a command whose answer the caller needs (a query): resolves with the
   * accepted state's `result` (or what `pick` takes of it, or of the sections
   * this page holds once it is applied) for this request,
   * rejects with its error. */
  request(command: Command, feedback?: string, pick: (state: State, sections: Readonly<Container>) => unknown = state => state.result): Promise<unknown> {
    if (this.shared) { this.send(command, feedback); return Promise.reject(Error(READ_ONLY)); }
    const id = this.send(command, feedback);
    return new Promise((resolve, reject) => this.answers.set(id, { resolve, reject, pick }));
  }

  /** A path's Browse… (v2 FluentPathEdit): the core shows this window's native dialog; the chosen
   * path, or null (cancelled, or refused: the status line says why). */
  choosePath({ dir, ...ask }: PathAsk): Promise<string | null> {
    return this.request({ action: dir ? "choose_folder" : "choose_file", ...ask })
      .then(chosen => (chosen as { path?: string } | null)?.path ?? null, () => null);
  }

  /** Say what a finished command did, in the status line, until the next command. */
  note(text: string, word: "idle" | "task" | "warning" = "task"): void { this.notice = [text, word]; this.handlers?.changed(); }

  /** Internal acknowledgements (map_ready, frame_consumed) are never awaited. */
  post(command: Command): void { this.send(command); }

  report(error: unknown): void {
    this.lastError = error instanceof Error ? error.message : String(error);
    this.handlers?.changed();
    this.send({ action: "frontend_error", error: this.lastError });
  }

  /** Whether `request` is one this window sent. */
  private owns(request: unknown): boolean { return typeof request === "number" && request >= this.firstRequest && request < this.firstRequest + 1e12; }

  /** A state message as the views read it: its envelope and every section it
   * changed, whole (a shared view's stream sends changed sections whole). */
  private expand(data: { [key: string]: unknown }): State {
    const state: Record<string, unknown> = { type: "state" };
    for (const key of ["request", "error", "result"]) if (key in data) state[key] = data[key];
    if (Array.isArray(data.changes)) {
      const [sections, touched] = applyChanges(this.sections, data.changes as Change[]);
      this.sections = sections;
      for (const key of touched) state[key] = sections[key];
    } else {
      for (const [key, value] of Object.entries(data)) if (!ENVELOPE.has(key)) state[key] = this.sections[key] = value;
    }
    return state as State;
  }

  private receive(data: { type?: string; [key: string]: unknown }): void {
    const handlers = this.handlers!;
    if (data.type === "state") {
      const state = this.expand(data);
      // Another window's answer: its sections are news, its request and error are not ours.
      if (state.request !== undefined && state.request !== 0 && !this.owns(state.request)) { handlers.state({ ...state, request: undefined, error: undefined }); handlers.changed(); return; }
      if (state.request !== undefined) {
        this.pending.delete(state.request);
        const answer = this.answers.get(state.request);
        if (answer) {
          this.answers.delete(state.request);
          if (state.error) answer.reject(Error(state.error)); else answer.resolve(answer.pick(state, this.sections));
        }
      }
      if (state.error) this.lastError = state.error;
      else if (state.request !== undefined) this.lastError = "";
      handlers.state(state);
    } else if (data.type === "frame") handlers.frame(data as unknown as Frame);
    else if (data.type === "page") handlers.page?.(String(data.page));
    else if (data.type === "caption") handlers.caption?.(data as unknown as CaptionState);
    else if (data.type === "close_requested") handlers.closeRequested?.((data.unsaved as string[] | undefined) ?? []);
    else if (data.type === "busy") {
      const request = Number(data.request);
      if (this.owns(request) && !this.pending.has(request)) this.pending.set(request, "Updating…");
    } else if (data.type === "release_mappings") {
      handlers.releaseMappings(data.mapping as string | number);
      this.post({ action: "mappings_released", mapping: data.mapping });
    } else if (data.type === "error") {
      if (data.request !== undefined && data.request !== 0 && !this.owns(data.request)) { handlers.changed(); return; }
      if (data.request !== undefined) {
        const request = Number(data.request);
        this.pending.delete(request);
        this.answers.get(request)?.reject(Error(String(data.error)));
        this.answers.delete(request);
      }
      this.lastError = String(data.error);
    }
    handlers.changed();
  }

  /** The status line's words while something is under way, or "". */
  busyText(): string { return [...this.pending.values()].at(-1) ?? ""; }

  // ------------------------------------------------------------ a shared view
  /** The instrument's event stream: pending until the operator allows this
   * browser, then the window's state (changed sections) and its pictures. */
  private watch(): void {
    const stream = new EventSource("events");
    stream.onmessage = event => {
      const message = JSON.parse(event.data) as { type: string; [key: string]: unknown };
      if (message.type === "pending") this.notice = [String(message.message), "warning"];
      else if (message.type === "hello") {
        this.token = String(message.token);
        // The pictures are drawn at the operator's pixel ratio; they show at its size here.
        document.documentElement.dataset.frameScale = String(Number(message.dpr) || 1);
        this.notice = ["Shared view of the experiment computer: watching only", "idle"];
      } else if (message.type === "closed") {
        stream.close();
        this.token = "";
        this.lastError = String(message.message);
      } else if (message.type === "frame") {
        this.wanted.set(String(message.panel), message as unknown as Frame);
        void this.fetchPicture(String(message.panel));
      } else if (message.type === "state") this.handlers?.state(message as unknown as State);
      this.handlers?.changed();
    };
    stream.onerror = () => {
      if (stream.readyState === EventSource.CLOSED && !this.lastError) { this.lastError = "The experiment computer stopped sharing."; this.handlers?.changed(); }
    };
  }

  /** The newest picture of a panel, one fetch at a time; a newer notice while it travels fetches again. */
  private async fetchPicture(panel: string): Promise<void> {
    if (this.fetching.has(panel)) return;
    this.fetching.add(panel);
    try {
      for (;;) {
        const frame = this.wanted.get(panel);
        if (!frame || !this.token) return;
        this.wanted.delete(panel);
        const response = await fetch(`frame/${encodeURIComponent(panel)}?token=${this.token}&sequence=${frame.sequence}`, { cache: "no-store" });
        // 410: a newer picture replaced it (its notice is already here or on its way).
        if (!response.ok) continue;
        const bitmap = await createImageBitmap(await response.blob(), { colorSpaceConversion: "none", premultiplyAlpha: "none" });
        const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
        const context = canvas.getContext("2d", { colorSpace: "srgb" })!;
        context.drawImage(bitmap, 0, 0);
        bitmap.close();
        this.handlers?.image({ ...frame, type: "frame", mapping: "share", slot: 0 }, context.getImageData(0, 0, canvas.width, canvas.height, { colorSpace: "srgb" }));
      }
    } catch (error) { this.lastError = `A shared picture could not be shown: ${error instanceof Error ? error.message : String(error)}`; this.handlers?.changed(); }
    finally { this.fetching.delete(panel); }
  }
}
