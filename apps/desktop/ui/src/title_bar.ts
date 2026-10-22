// v2's window title bar: zlc_ui's FluentWindow on qframelesswindow's
// StandardTitleBar (32 Qt px): the product mark on the body's left column, the
// window title in the UI font, and TitleBarButton's minimize, maximize and
// close (46 x 32 Qt px, glyphs at v2's own coordinates). The host's caption
// window answers the band's hit tests (frame.rs) and tells the page which
// button the pointer is over or pressing, and whether the window is maximized.
import { element } from "./fluent/index.js";

export type CaptionState = { hover: string | null; pressed: string | null; maximized: boolean; width?: number };

export class TitleBar {
  readonly root = element("div", "title-bar");
  private title = element("span", "title-bar-title");
  private buttons = new Map<string, HTMLElement>();

  constructor() {
    const mark = element("img", "title-bar-mark");
    mark.src = "zlc.ico";
    mark.alt = "";
    mark.draggable = false;
    this.root.append(mark, this.title);
    for (const name of ["minimize", "maximize", "close"]) {
      const button = element("div", `title-bar-button ${name}`);
      const glyph = element("span", "glyph");
      // The restore glyph's window behind ends in a short stroke left of its corner.
      if (name === "maximize") glyph.append(element("span", "stub"));
      button.append(glyph);
      this.buttons.set(name, button);
      this.root.append(button);
    }
    // FluentWindow's title label follows windowTitleChanged.
    const show = (): void => { if (this.title.textContent !== document.title) this.title.textContent = document.title; };
    const head = document.querySelector("title");
    if (head) new MutationObserver(show).observe(head, { childList: true, characterData: true, subtree: true });
    show();
  }

  accept(caption: CaptionState): void {
    for (const [name, button] of this.buttons) {
      button.classList.toggle("hover", caption.hover === name && caption.pressed !== name);
      button.classList.toggle("pressed", caption.pressed === name);
    }
    this.root.classList.toggle("maximized", caption.maximized);
    // The window's own width: the page can be a device pixel or two wider, past its right edge.
    if (caption.width) this.root.style.setProperty("--frame-w", `${caption.width / devicePixelRatio}px`);
  }
}
