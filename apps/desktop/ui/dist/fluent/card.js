import { element, elided } from "./dom.js";
/** FluentFrame: a white rounded region.  `bordered` gives it the DIVIDER
 * edge of a delineated card; the default is edgeless (a header or a strip
 * that must not read as a boxed card). */
export function frame(bordered = false, className = "") {
    return element("div", `card${bordered ? " bordered" : ""} ${className}`.trim());
}
/** card_title_row: the semibold title (as tall as a control) on the left,
 * `lead` right after it, a muted eliding `summary` taking the free width,
 * and the `trailing` commands at the row's end.  A card's command sits on its
 * title line, never alone on a row under it. */
export function titleRow(title, trailing = [], options = {}) {
    const row = element("div", "title-row");
    const heading = typeof title === "string" ? element("span", "heading", title) : title;
    heading.classList.add("heading");
    row.append(heading, ...(options.lead ?? []));
    row.append(options.summary ?? element("span", "spacer"));
    row.append(...trailing);
    return row;
}
/** A muted one-line summary for a title row, right-aligned against the
 * commands after it, eliding rather than widening its card. */
export function summary(text = "") {
    return elided(text, "summary-text");
}
/** FluentGroupBox drawing its own title row: a bordered white card, its
 * title row first, then its body, one card pad all round and a row gap
 * between rows.  `selected` is a fill (ACCENT_TINT), never a frame. */
export class GroupBox {
    root = element("section", "card bordered group");
    header;
    heading;
    body = element("div", "group-body");
    constructor(title, trailing = [], options = {}) {
        this.header = titleRow(title, trailing, options);
        this.heading = this.header.querySelector(".heading");
        this.root.append(this.header, this.body);
    }
    set title(text) { if (this.heading.textContent !== text)
        this.heading.textContent = text; }
    setSelected(selected) { this.root.classList.toggle("selected", selected); }
}
/** A ruled list: items inside a card, each but the last ruled off from the
 * next by a DIVIDER hairline in the middle of a half-section break (a rule
 * separates items; it never closes the list -- the card's edge does). */
export function ruledList(className = "") {
    return element("div", `ruled ${className}`.trim());
}
