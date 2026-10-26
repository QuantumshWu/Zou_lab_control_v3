// Runtime inputs of tokens.css, written before the body is parsed (a classic
// script in <head>, so the first paint already has v2's metrics).
//
// zlc_ui sizes everything in Qt logical px.  Qt 5 with AA_EnableHighDpiScaling
// paints one logical px as round-half-up(Windows scale) device px, and the
// Fluent scale is resolve_fluent_auto_scale() of the Qt-logical available
// screen.  Everything else in tokens.css derives from the two numbers set here,
// plus the few lengths that only the font itself can answer.
(() => {
  const root = document.documentElement;
  const AUTO_SCALE_BASIS = [1280, 790], AUTO_SCALE_MARGIN = [48, 88];  // style.py
  const FLUENT_SCALE_MIN = 0.72, FLUENT_SCALE_MAX = 1.25;
  const NUMBER_SAMPLE = "-1234567890.12345678";                       // qt_form._NUMBER_SAMPLE
  const SPIN_HINT_SAMPLE = "-123.456e-12";                              // FluentDoubleSpinBox.sizeHint
  const COUNT_SAMPLE = "10000";                                       // pulse/_layout.py COUNT_SAMPLE

  const mask = (svg: string): string => `url("data:image/svg+xml,${encodeURIComponent(svg)}")`;

  // paint_chevron: (cx - h, cy - s*h/2) -> (cx, cy + s*h/2) -> (cx + h, cy - s*h/2), round cap and join,
  // in a `width` x `height` box at (cx, cy) of it.
  function chevron(width: number, height: number, cx: number, cy: number, half: number, stroke: number, up: boolean): string {
    const sign = up ? -1 : 1;
    const points = [[cx - half, cy - sign * half / 2], [cx, cy + sign * half / 2], [cx + half, cy - sign * half / 2]];
    return mask(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}"><polyline points="${points.map(p => p.join(",")).join(" ")}" fill="none" stroke="#000" stroke-width="${stroke}" stroke-linecap="round" stroke-linejoin="round"/></svg>`);
  }

  function tick(side: number): string {
    // _paint_tick: 0.26..0.75 across, 0.34..0.68 down, that span centred on the box, pen max(1.5, side / 8).
    const dx = side / 2 - (0.26 + 0.75) / 2 * side, dy = side / 2 - (0.34 + 0.68) / 2 * side;
    const points = [[0.26, 0.52], [0.43, 0.68], [0.75, 0.34]].map(([x, y]) => `${dx + x! * side},${dy + y! * side}`).join(" ");
    return mask(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${side} ${side}"><polyline points="${points}" fill="none" stroke="#000" stroke-width="${Math.max(1.5, side / 8)}" stroke-linecap="round" stroke-linejoin="round"/></svg>`);
  }

  // Python's round(): half to even (v2 asks it whether a stroke is an odd number of device px).
  const pyRound = (value: number): number => { const floor = Math.floor(value); return value - floor === 0.5 ? floor + (floor & 1) : Math.round(value); };

  function apply(): void {
    const dpr = window.devicePixelRatio || 1;
    // Qt::HighDpiScaleFactorRoundingPolicy::Round (Qt 5 default): .5 rounds up.
    const qtDpr = Math.max(1, Math.floor(dpr + 0.5));
    const qt = qtDpr / dpr;
    const availW = screen.availWidth * dpr / qtDpr, availH = screen.availHeight * dpr / qtDpr;
    const auto = Math.min(1, Math.max(0.1, (availW - AUTO_SCALE_MARGIN[0]!) / AUTO_SCALE_BASIS[0]!),
      Math.max(0.1, (availH - AUTO_SCALE_MARGIN[1]!) / AUTO_SCALE_BASIS[1]!));
    const fluent = Math.max(FLUENT_SCALE_MIN, Math.min(FLUENT_SCALE_MAX, auto));
    const scaled = (value: number, minimum = 1): number => Math.max(minimum, Math.round(value * fluent));
    root.style.setProperty("--qt", String(qt));
    root.style.setProperty("--fluent", String(fluent));
    // One device pixel in CSS px: lines and marks placed by calc() snap to it.
    root.style.setProperty("--dpx", `${1 / dpr}px`);

    // The font answers the rest: fluent_font_size() pt, in the whole px Qt asks for (--font-size).
    const fontQt = Math.round(Math.max(8, Math.round(12 * fluent)) * 4 / 3), fontPx = fontQt * qt;
    const context = document.createElement("canvas").getContext("2d");
    if (context) {
      // centred_line_top: v2 puts a control's line on a whole Qt px, its
      // capitals on the control's middle -- floor(h / 2 + capHeight / 2 -
      // ascent + 0.5) -- and its baseline the engine's whole-px ascent below.
      // The font's outline metrics, read at 16x the size for their fractions:
      context.font = `${16 * fontQt}px "Segoe UI"`;
      const outline = context.measureText("H");
      const ascentQt = outline.fontBoundingBoxAscent / 16, capQt = outline.actualBoundingBoxAscent / 16;
      const lineAt = (height: number): number => (Math.floor(height / 2 + capQt / 2 - ascentQt + 0.5) + Math.round(ascentQt)) * qt;
      // This page's line box is Blink's whole CSS-px ascent + descent, centred
      // in its box less the bottom padding --cap-lift: the padding that puts
      // its baseline where v2's is in a control-high box.
      context.font = `${fontPx}px "Segoe UI"`;
      const capital = context.measureText("H");
      const ascent = capital.fontBoundingBoxAscent, line = ascent + capital.fontBoundingBoxDescent;
      const control = Math.max(22, Math.round(30 * fluent));
      const capLift = Math.max(0, control * qt - line - 2 * (lineAt(control) - ascent));
      root.style.setProperty("--cap-lift", `${capLift}px`);
      // _tab_cap_centre: a closable tab's disk on the middle of its label's
      // capitals, where .tab puts them (its line box centred on the band
      // above the underline, as QTabBar centres it), on a device px.
      const labelH = Math.max(26, Math.round(32 * fluent)) * qt, disk = (Math.max(14, Math.round(18 * fluent)) + 1 & ~1) * qtDpr;
      const baseline = Math.round(((labelH - line) / 2 + ascent) * dpr);
      root.style.setProperty("--tab-close-top", `${Math.floor(baseline - capital.actualBoundingBoxAscent * dpr / 2 - disk / 2 + 0.5) / dpr}px`);
      // A line box, rounded up to a whole Qt px as a QTextDocument line is.
      root.style.setProperty("--font-line", `${Math.ceil(line / qt - 1e-6) * qt}px`);
      // v2 sizes by QFontMetrics.horizontalAdvance: a whole Qt px.
      const advance = (text: string): string => `${Math.round(context.measureText(text).width / qt) * qt}px`;
      root.style.setProperty("--number-text", advance(NUMBER_SAMPLE));
      root.style.setProperty("--spin-hint-text", advance(SPIN_HINT_SAMPLE));
      root.style.setProperty("--count-text", advance(COUNT_SAMPLE));
      // QLineEdit.sizeHint, which a window opened at its content reads: 17 x's, each a whole px.
      root.style.setProperty("--edit-hint-text", `${17 * Math.round(context.measureText("x").width / qt) * qt}px`);
      // DeviceControlView's Status column: its widest word whole.
      root.style.setProperty("--control-status-text", advance("Applying; latest queued"));
      // Its Device limits column gives way to the window down to its heading, whole.
      root.style.setProperty("--control-limits-text", advance("Device limits"));
      // _STEP_CHIP_SAMPLE, the chip every number width reserves room for: its
      // ink in the chip's font (0.8 of the field's, in the whole px Qt asks
      // for), up to a whole Qt px (_step_chip_metrics).  fluent/ink.ts draws the chips.
      context.font = `${Math.round(Math.max(8, Math.round(12 * fluent)) * 0.8 * 4 / 3) * qt}px "Segoe UI"`;
      const chip = context.measureText("±0.0001");
      root.style.setProperty("--chip-sample-ink", `${Math.ceil((chip.actualBoundingBoxLeft + chip.actualBoundingBoxRight) / qt - 1e-6) * qt}px`);
    }
    // Each chevron is a mask as large as the part it sits in, drawn in device
    // px at v2's centre (paint_chevron): x on a device px, or on a pixel's
    // middle when the stroke is an odd number of device px; y on the
    // half-pixel grid.  The part's edges lie on device px, so the chevron does.
    const half = scaled(8) / 2 * qtDpr, stroke = Math.max(1, scaled(3) / 2) * qtDpr, odd = pyRound(stroke) & 1 ? 0.5 : 0;
    const control = Math.max(22, Math.round(30 * fluent)) * qtDpr, edge = qtDpr, inner = control - 2 * edge, column = scaled(16) * qtDpr;
    const columnX = Math.floor(column / 2 + 0.5) + odd;
    // A field's drop column (stepper_column) on its inner rect: every combo's one chevron.
    root.style.setProperty("--chevron-drop", chevron(column, inner, columnX, inner / 2, half, stroke, false));
    // A spin box's cells: their chevrons a quarter of the inner height from its middle (_paint_spin_stepper).
    const rise = Math.floor(inner / 2 + 0.5) / 2;
    root.style.setProperty("--chevron-step-up", chevron(column, inner, columnX, rise, half, stroke, true));
    root.style.setProperty("--chevron-step-down", chevron(column, inner, columnX, inner - rise, half, stroke, false));
    // A disclosure (a control-square quiet button): on its centre, inside its edge.
    const centreX = Math.floor(control / 2 + 0.5) - edge + odd, centreY = control / 2 - edge;
    root.style.setProperty("--chevron-disclosure-down", chevron(inner, inner, centreX, centreY, half, stroke, false));
    root.style.setProperty("--chevron-disclosure-up", chevron(inner, inner, centreX, centreY, half, stroke, true));
    // A tree combo row's branch: the chevron alone.
    root.style.setProperty("--chevron-down", chevron(2 * half + stroke, half + stroke, half + stroke / 2, (half + stroke) / 2, half, stroke, false));
    // InfoTree.drawBranches: half scaled_px(4, 3), pen scaled_px(1.5, 1), on a
    // device px of its branch column -- the box an even number of device px
    // across, so its centre is one (rotated a quarter turn while closed).
    const infoHalf = Math.max(3, Math.round(4 * fluent)) * qtDpr, infoPen = Math.max(1, Math.round(1.5 * fluent)) * qtDpr;
    const infoBox = 2 * Math.ceil(infoHalf + infoPen / 2 + 1);
    root.style.setProperty("--info-chevron-box", `${infoBox / dpr}px`);
    root.style.setProperty("--info-chevron-mask", chevron(infoBox, infoBox, infoBox / 2, infoBox / 2, infoHalf, infoPen, false));
    // The tick spans the indicator's whole rect: the sheet's even width and its edge each side.
    const side = scaled(16, 2);
    root.style.setProperty("--tick", tick(side + (side & 1) + 2));
    root.dataset.qtDpr = String(qtDpr);
    resolution?.removeEventListener("change", apply);
    resolution = matchMedia(`(resolution: ${dpr}dppx)`);
    resolution.addEventListener("change", apply);
  }
  let resolution: MediaQueryList | null = null;
  apply();
})();
