import { element, setText, token } from "./dom.js";
import { setGlyph, wordsWidth } from "./ink.js";
import { Menu } from "./menu.js";
/** FluentTabWidget: a white rounded card, pivot tabs on its top (MUTED_INK,
 * TEXT and the ACCENT underline when current), the page straight below.
 * Permanent tabs always exist (Monitor, Logic, Config); a closable tab
 * (an Edit page) carries a quiet × that asks its owner to close it -- the
 * widget never removes a page behind its owner's back.  The first tab's
 * underline starts on the pages' content edge (`--tab-inset`).  One lone
 * permanent page shows no bar, unless `loneBar`.  When the tabs do not fit,
 * the widest are capped to one shared width and elide, and the corner ⋯
 * lists every full title. */
export class TabWidget {
    options;
    root = element("div", "tabs");
    bar = element("div", "tab-bar");
    strip = element("div", "tab-strip");
    stack = element("div", "tab-pages");
    overflow = element("button", "quiet tab-overflow");
    tabs = new Map();
    current = null;
    onChange = () => { };
    constructor(options = {}) {
        this.options = options;
        this.strip.role = "tablist";
        setGlyph(this.overflow, "⋯", "tab-overflow");
        this.overflow.type = "button";
        this.overflow.title = "All tabs";
        this.overflow.tabIndex = -1;
        this.overflow.hidden = true;
        this.overflow.addEventListener("click", () => {
            new Menu([...this.tabs.values()].map(tab => ({ label: tab.label.textContent || "", action: () => this.setCurrent(tab.key) }))).openBelow(this.overflow);
        });
        this.bar.append(this.strip, this.overflow);
        this.root.append(this.bar, this.stack);
        new ResizeObserver(() => this.fit()).observe(this.bar);
    }
    get currentKey() { return this.current; }
    has(key) { return this.tabs.has(key); }
    page(key) { return this.tabs.get(key)?.page; }
    addPermanent(key, title, page) { this.add(key, title, page, null, false); }
    addClosable(key, title, page, close, focus = true) {
        this.add(key, title, page, close, focus);
    }
    add(key, title, page, close, focus) {
        if (this.tabs.has(key))
            throw new Error(`Tab ${key} already exists.`);
        const tab = element("div", "tab");
        tab.role = "tab";
        tab.tabIndex = -1;
        const label = element("span", "tab-label", title);
        // QTabBar.tabSizeHint: the title's QFontMetrics.size(), a whole Qt px (a capped tab shrinks it).
        label.style.width = `calc(${wordsWidth(title)} * var(--q))`;
        tab.append(label);
        tab.title = title;
        tab.addEventListener("mousedown", event => { if (event.button === 0 && !event.target.closest(".tab-close"))
            this.setCurrent(key); });
        if (close) {
            // A quiet round × on the tab's label line: closing a view takes nothing away.
            const cross = element("button", "tab-close");
            setGlyph(cross, "✕", "tab-close");
            cross.type = "button";
            cross.tabIndex = -1;
            cross.title = "Close this tab";
            cross.addEventListener("click", event => { event.stopPropagation(); close(); });
            tab.append(cross);
        }
        page.classList.add("tab-page");
        page.hidden = true;
        page.role = "tabpanel";
        this.strip.append(tab);
        this.stack.append(page);
        this.tabs.set(key, { key, tab, label, page, closable: Boolean(close) });
        if (this.current === null || focus)
            this.setCurrent(key);
        this.showBar();
    }
    setTitle(key, title) {
        const tab = this.tabs.get(key);
        if (!tab || tab.tab.title === title)
            return;
        setText(tab.label, title);
        tab.label.style.width = `calc(${wordsWidth(title)} * var(--q))`;
        tab.tab.title = title;
        this.fit();
    }
    remove(key) {
        const tab = this.tabs.get(key);
        if (!tab)
            return;
        const keys = [...this.tabs.keys()];
        const index = keys.indexOf(key);
        tab.tab.remove();
        tab.page.remove();
        this.tabs.delete(key);
        if (this.current === key) {
            this.current = null;
            const next = keys[index + 1] ?? keys[index - 1];
            if (next !== undefined && next !== key)
                this.setCurrent(next);
        }
        this.showBar();
    }
    setCurrent(key) {
        const chosen = this.tabs.get(key);
        if (!chosen || this.current === key)
            return;
        this.current = key;
        // Told first: what the owner sends goes out before this page's layout below.
        this.onChange(key);
        for (const tab of this.tabs.values()) {
            const on = tab === chosen;
            tab.tab.setAttribute("aria-selected", String(on));
            tab.page.hidden = !on;
        }
        this.fit();
    }
    showBar() {
        const lone = !this.options.loneBar && this.tabs.size === 1 && ![...this.tabs.values()][0].closable;
        this.bar.hidden = lone;
        this.fit();
    }
    /** Water-filling: the largest cap such that capping every tab at it fits
     * the bar; tabs already narrower keep their natural width. */
    fit() {
        const tabs = [...this.tabs.values()];
        for (const tab of tabs)
            tab.tab.style.maxWidth = "";
        this.overflow.hidden = true;
        let room = this.strip.getBoundingClientRect().width;
        if (!room || !tabs.length)
            return;
        const widths = tabs.map(tab => tab.tab.getBoundingClientRect().width + parseFloat(getComputedStyle(tab.tab).marginLeft) + parseFloat(getComputedStyle(tab.tab).marginRight));
        const total = widths.reduce((sum, width) => sum + width, 0);
        if (total <= room + 0.5)
            return;
        // The corner ⋯ takes its room from the strip: the tabs share what it leaves.
        this.overflow.hidden = false;
        room = this.strip.getBoundingClientRect().width;
        const sorted = [...widths].sort((a, b) => a - b);
        const floor = token("--tab-floor");
        let cap = floor, prefix = 0;
        for (let k = 0; k < sorted.length; k++) {
            const share = (room - prefix) / (sorted.length - k);
            if (share <= sorted[k]) {
                cap = Math.max(share, floor);
                break;
            }
            prefix += sorted[k];
        }
        tabs.forEach((tab, index) => {
            if (widths[index] > cap) {
                const margins = widths[index] - tab.tab.getBoundingClientRect().width;
                tab.tab.style.maxWidth = `${Math.floor(cap - margins)}px`;
            }
        });
    }
}
