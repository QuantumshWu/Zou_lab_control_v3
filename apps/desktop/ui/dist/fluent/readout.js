import { element } from "./dom.js";
import { setMiddle } from "./elide.js";
/** FluentReadoutEdit: a read-only but selectable value -- plain TEXT ink on
 * the surface, no box, its text starting where a field's text starts.  A
 * value too long for its row is elided in the MIDDLE; the tooltip has it
 * all.  `flush` starts it at its own edge (a value that continues a line). */
export class Readout {
    root = element("span", "readout-edit");
    full = "";
    /** `plain`: a label, not a value -- eliding in the middle but neither
     * selectable nor inset (a node name in its column). */
    constructor(text = "", flush = false, plain = false) {
        this.root.classList.toggle("flush", flush || plain);
        this.root.classList.toggle("plain", plain);
        this.text = text;
    }
    get text() { return this.full; }
    set text(text) {
        if (text === this.full)
            return;
        this.full = text;
        // The whole text in the tooltip, unless the owner gives it another.
        this.root.title = text;
        setMiddle(this.root, text, "none");
    }
}
/** A multi-line readout: wraps anywhere (a path, a traceback), selectable. */
export function readoutText(text = "") { return element("div", "readout-text", text); }
/** PublishedItemsLegend: `(name, shape, description)` rows as small neutral
 * chips that wrap -- the name in TEXT, its shape in MUTED_INK, the
 * descriptions in the tooltip.  With none it says so where a chip's name
 * would start, one chip high. */
export class Legend {
    root = element("div", "legend");
    signature = "";
    setRows(rows) {
        const signature = JSON.stringify(rows);
        if (signature === this.signature)
            return;
        this.signature = signature;
        this.root.title = rows.filter(([, , description]) => description).map(([name, shape, description]) => `${name} ${shape} — ${description}`).join("\n");
        if (!rows.length) {
            this.root.replaceChildren(element("span", "legend-empty", "no outputs"));
            return;
        }
        this.root.replaceChildren(...rows.map(([name, shape]) => {
            const chip = element("span", "chip");
            chip.append(element("span", "chip-name", name));
            if (shape)
                chip.append(element("span", "chip-shape", shape));
            return chip;
        }));
    }
    get text() {
        const chips = [...this.root.querySelectorAll(".chip")].map(chip => chip.textContent?.trim()).filter(Boolean);
        return chips.length ? `outputs: ${chips.join(", ")}` : "no outputs";
    }
}
