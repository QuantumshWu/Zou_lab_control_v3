// paint_centred_ink: the one way a small symbol or label is centred in its
// face -- a command's ×, a tab's ✕, the tab overflow ⋯, a step chip, a
// binding badge's letter.  A line box centred puts a × 2.5 px and a ⋯ 1.2 px
// below the middle of its face, and a digit a pixel low; v2 lays the text out
// and moves it so the centre of its own INK lands on the face's, by a
// translation snapped to the window's device pixels.  Here the face holds an
// SVG whose user unit is one device pixel, the text placed at that snapped
// origin; its ink is read back once per font and text.
const SVG = "http://www.w3.org/2000/svg";
/** Qt asks the system for whole pixels: a font of `pt` points is qRound(pt * 4 / 3) px. */
export const pointPx = (pt) => Math.round(pt * 4 / 3);
/** This window's metrics as scale.ts wrote them: device px per Qt px, and
 * scaled_px / scaled_even_px of the Fluent scale. */
export function metrics() {
    const root = document.documentElement;
    const qtDpr = Number(root.dataset.qtDpr) || Math.max(1, Math.floor(devicePixelRatio + 0.5));
    const fluent = parseFloat(root.style.getPropertyValue("--fluent")) || 1;
    const scaled = (value, minimum = 1) => Math.max(minimum, Math.round(value * fluent));
    const even = (value, minimum = 2) => { const size = scaled(value, minimum); return size + (size & 1); };
    return { qtDpr, fontPt: scaled(12, 8), scaled, even };
}
/** A font of `px` Qt px in this window's device px: Segoe UI, and for a
 * symbol it lacks (⋯, ✕) the face Qt falls back to on Windows. */
export function glyphFont(px, weight = 400) {
    return `${weight} ${px * metrics().qtDpr}px "Segoe UI", "MS UI Gothic"`;
}
const advances = new Map();
let ruler = null;
/** QFontMetrics.size(text).width() of the body font -- the words' advance up
 * to a whole Qt px -- in Qt px: what a FluentButton's words take. */
export function wordsWidth(text) {
    const m = metrics(), qt = m.qtDpr / devicePixelRatio;
    const key = `${qt}|${m.fontPt}|${text}`;
    let width = advances.get(key);
    if (width === undefined) {
        ruler ??= document.createElement("canvas").getContext("2d");
        if (!ruler)
            return 0;
        ruler.font = `400 ${pointPx(m.fontPt) * qt}px "Segoe UI", system-ui, sans-serif`;
        width = Math.ceil(ruler.measureText(text).width / qt - 1e-3);
        advances.set(key, width);
    }
    return width;
}
/** A FluentButton's sizeHint: its words (the widest it can say) and its padding and edges. */
export function fitWords(button, words) {
    const widest = Math.max(0, ...words.map(wordsWidth));
    const width = widest ? `calc(${widest} * var(--q) + 2 * var(--button-pad-h) + 2 * var(--edge))` : "";
    if (button.style.minWidth !== width)
        button.style.minWidth = width;
}
const inks = new Map();
let pen = null;
/** The ink box of `text` in `font` (device px), from its baseline origin
 * (QPainterPath.boundingRect): drawn once at the size it is shown and read
 * back, each edge to a fraction of a pixel by its coverage. */
export function inkBox(text, font) {
    const key = `${font}|${text}`;
    let ink = inks.get(key);
    if (ink)
        return ink;
    ink = { left: 0, top: 0, right: 0, bottom: 0 };
    pen ??= document.createElement("canvas").getContext("2d", { willReadFrequently: true });
    if (pen) {
        pen.font = font;
        const size = parseFloat(/([\d.]+)px/.exec(font)?.[1] ?? "16");
        const width = Math.ceil(pen.measureText(text).width + 2 * size), height = Math.ceil(3 * size);
        const [originX, originY] = [Math.round(size), Math.round(2 * size)];
        pen.canvas.width = width;
        pen.canvas.height = height;
        pen.font = font;
        pen.fillText(text, originX, originY);
        const alpha = pen.getImageData(0, 0, width, height).data;
        const columns = new Float32Array(width), rows = new Float32Array(height);
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const cover = alpha[(y * width + x) * 4 + 3] / 255;
                if (cover > columns[x])
                    columns[x] = cover;
                if (cover > rows[y])
                    rows[y] = cover;
            }
        }
        let [left, right, top, bottom] = [0, width - 1, 0, height - 1];
        while (left < right && !columns[left])
            left++;
        while (right > left && !columns[right])
            right--;
        while (top < bottom && !rows[top])
            top++;
        while (bottom > top && !rows[bottom])
            bottom--;
        // An edge pixel the outline covers in part is that part of a pixel.
        ink = { left: left + 1 - columns[left] - originX, right: right + columns[right] - originX,
            top: top + 1 - rows[top] - originY, bottom: bottom + rows[bottom] - originY };
    }
    inks.set(key, ink);
    return ink;
}
/** The baseline origin that puts `ink`'s centre on (cx, cy), by a whole
 * device px translation (device_snap); an ink that cannot be centred
 * exactly keeps its odd pixel below it, as v2's letters do. */
export function centredOrigin(ink, cx, cy) {
    return [Math.floor(cx - (ink.left + ink.right) / 2 + 0.5), Math.ceil(cy - (ink.top + ink.bottom) / 2 - 0.5)];
}
/** One text of an SVG whose user unit is a device px, its ink centred on (cx, cy). */
export function inkText(text, font, cx, cy) {
    const [x, y] = centredOrigin(inkBox(text, font), cx, cy);
    const node = document.createElementNS(SVG, "text");
    node.setAttribute("x", String(x));
    node.setAttribute("y", String(y));
    node.setAttribute("style", `font: ${font}`);
    node.textContent = text;
    return node;
}
export function glyphFace(kind, text) {
    const m = metrics(), d = m.qtDpr;
    // A control's face inside its edge: control_height() less FIELD_BORDER_PX each side.
    const inner = (m.scaled(30, 22) - 2) * d;
    if (kind === "tab-close") {
        // _FluentTabCloseButton: the hover disk scaled_even_px(18, 14); bold max(9, int(size * 0.62)) px.
        const side = m.even(18, 14);
        return { width: side * d, height: side * d, font: glyphFont(Math.max(9, Math.floor(side * 0.62)), 700), ink: 0 };
    }
    if (kind === "tab-overflow") {
        // The corner ⋯: scaled_px(26, 20) square (inside its edge here); bold max(11, int(size * 0.72)) px.
        const size = m.scaled(26, 20);
        return { width: (size - 2) * d, height: (size - 2) * d, font: glyphFont(Math.max(11, Math.floor(size * 0.72)), 700), ink: 0 };
    }
    if (kind === "chip") {
        // _StepChip: 0.8 of the field's font; the text's ink plus scaled_px(6, 4) each side,
        // and the pill its ink plus scaled_px(3, 2) above and below, of the inner rect's parity.
        const font = glyphFont(pointPx(m.fontPt * 0.8));
        const box = inkBox(text, font);
        const ink = Math.ceil((box.right - box.left) / d - 1e-6) * d;
        let tall = Math.min(inner, Math.ceil(((box.bottom - box.top) / d + 2 * m.scaled(3, 2)) * d));
        tall += (inner - tall) & 1;
        return { width: ink + 2 * m.scaled(6, 4) * d, height: tall, font, ink };
    }
    // FluentButton glyph: the face; round(font pt * 1.35) pt.
    return { width: inner, height: inner, font: glyphFont(pointPx(Math.round(m.fontPt * 1.35))), ink: 0 };
}
/** `face` shows `text` as a glyph of `kind`, its ink centred: an SVG filling
 * the face (fluent.css svg.ink), in the face's ink (currentColor).  A chip's
 * face takes the pill's size. */
export function setGlyph(face, text, kind) {
    const signature = `${kind}|${text}|${devicePixelRatio}`;
    if (face.dataset.ink === signature)
        return;
    face.dataset.ink = signature;
    const { width, height, font } = glyphFace(kind, text);
    if (kind === "chip") {
        face.style.width = `${width / devicePixelRatio}px`;
        face.style.height = `${height / devicePixelRatio}px`;
    }
    const svg = document.createElementNS(SVG, "svg");
    svg.setAttribute("class", "ink");
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.append(inkText(text, font, width / 2, height / 2));
    face.replaceChildren(svg);
    watch();
}
// A window moved to a screen of another scale draws its glyphs again (after
// scale.ts, whose listener was added first, has written the new metrics).
let screen = null;
function watch() {
    if (screen?.matches)
        return;
    screen?.removeEventListener("change", redraw);
    screen = matchMedia(`(resolution: ${devicePixelRatio}dppx)`);
    screen.addEventListener("change", redraw);
}
function redraw() {
    inks.clear();
    for (const face of document.querySelectorAll("[data-ink]")) {
        const [kind, text] = face.dataset.ink.split("|");
        setGlyph(face, text, kind);
    }
}
