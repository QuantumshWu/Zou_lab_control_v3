// v2 zlc_ui/board/board_layout.py: pure north-west-gravity placement for a
// panel board.  It is GIVEN each card's rectangle and the clear gap between
// them, in whole pixels (the board's are device px), and knows nothing of the
// DOM: a card's size is what the card says, never a second answer here.
/** The one fact the packer cannot derive: the clear gap between cards. */
export class BoardMetrics {
    gap;
    constructor(gap) {
        this.gap = gap;
        if (!Number.isInteger(gap) || gap < 0)
            throw new RangeError("gap must be a non-negative whole number of pixels");
    }
}
/** A placement-only stand-in, so a trial pack never mutates a real card: the card's own rectangle. */
export class GeomProxy {
    col;
    row;
    width;
    height;
    constructor(width, height, col = 0, row = 0) {
        this.col = col;
        this.row = row;
        if (Math.trunc(width) <= 0 || Math.trunc(height) <= 0)
            throw new RangeError("a card rectangle must be positive");
        this.width = Math.trunc(width);
        this.height = Math.trunc(height);
    }
}
const ascending = (a, b) => a - b;
/** v2 _aabb: the card's AABB (x0, y0, x1, y1) -- top-left (col, row) plus its size. */
const aabb = (cfg) => [cfg.col, cfg.row, cfg.col + cfg.width, cfg.row + cfg.height];
/** v2 _overlaps_with_gap: `box`, EXPANDED by the gap on all sides, intersects a placed card --
 * a card left exactly a gap away is clear. */
export function overlapsWithGap(box, placed, metrics) {
    const gap = metrics.gap, [x0, y0, x1, y1] = box;
    for (const p of placed) {
        const [px0, py0, px1, py1] = aabb(p);
        if (x0 < px1 + gap && px0 < x1 + gap && y0 < py1 + gap && py0 < y1 + gap)
            return true;
    }
    return false;
}
/** v2 first_free_slot: the TOP-MOST then LEFT-MOST free place where `cfg` fits clear of every
 * `placed` card, inside `boardW` -- where a card with no place of its own GOES: Add tiles the top
 * row left to right, wraps, and never leaves a middle hole.  Candidates are the margin, each card's
 * far edges (+gap) and its near edges (to tuck under a wider one), swept by y then x. */
export function firstFreeSlot(cfg, placed, boardW, metrics) {
    const gap = metrics.gap, w = cfg.width, h = cfg.height;
    const xs = new Set([gap]), ys = new Set([gap]);
    for (const p of placed) {
        const [px0, py0, px1, py1] = aabb(p);
        xs.add(px1 + gap);
        ys.add(py1 + gap);
        xs.add(px0);
        ys.add(py0);
    }
    const maxX = Math.max(gap, boardW - gap - w);
    let candX = [...xs].filter(x => gap <= x && x <= maxX).sort(ascending);
    if (!candX.length)
        candX = [gap];
    for (const y of [...ys].filter(y => y >= gap).sort(ascending))
        for (const x of candX)
            if (!overlapsWithGap([x, y, x + w, y + h], placed, metrics))
                return [x, y];
    // No candidate fit (placing past the lowest card always clears).
    const bottom = Math.max(0, ...placed.map(p => p.row + p.height));
    return [gap, placed.length ? bottom + gap : gap];
}
/** v2 _rise_to: the margin, or one gap under the lowest card above (x, y) in its column band. */
function riseTo(x, y, width, boxes, gap) {
    let top = gap;
    for (const [px0, , px1, py1] of boxes)
        if (x < px1 + gap && px0 < x + width + gap && py1 + gap <= y)
            top = Math.max(top, py1 + gap);
    return top;
}
/** v2 _slide_to: the margin, or one gap right of the nearest card left of (x, y) in its row band. */
function slideTo(x, y, height, boxes, gap) {
    let left = gap;
    for (const [px0, py0, px1, py1] of boxes)
        if (y < py1 + gap && py0 < y + height + gap && px1 + gap <= x)
            left = Math.max(left, px1 + gap);
    return left;
}
/** v2 _below: the first row at or below `y` where a card at `x` is clear of every box --
 * straight DOWN its own column, never up and never sideways. */
function below(x, y, width, height, boxes, gap) {
    for (;;) {
        const hits = boxes.filter(([px0, py0, px1, py1]) => x < px1 + gap && px0 < x + width + gap && y < py1 + gap && py0 < y + height + gap).map(box => box[3]);
        if (!hits.length)
            return y;
        y = Math.max(...hits) + gap;
    }
}
/** v2 gravity_slot: where a card standing clear of every `placed` card comes to rest.  It rises
 * until a card above (or the top margin) stops it, slides left until a card beside it (or the left
 * margin) stops it, and repeats; it only moves up and left and stops at the FIRST thing in the
 * way.  Stopped by a card above, it RESTS UNDER it and slides no further left than the leftmost
 * such card's left edge: a card put below another stays below it. */
export function gravitySlot(cfg, placed, metrics) {
    const gap = metrics.gap, { width, height } = cfg, boxes = placed.map(aabb);
    let x = cfg.col, y = cfg.row;
    for (;;) {
        const top = riseTo(x, y, width, boxes, gap);
        let left = slideTo(x, top, height, boxes, gap);
        const under = boxes.filter(([px0, , px1, py1]) => py1 + gap === top && x < px1 + gap && px0 < x + width + gap).map(box => box[0]);
        if (under.length)
            left = Math.max(left, Math.min(x, ...under));
        if (left === x && top === y)
            return [x, y];
        x = left;
        y = top;
    }
}
/** v2 min_board_width: the NARROWEST a board packs to -- its widest card and both margins.  Never
 * the cards' current right extent, which would ratchet: narrowing could never reflow to one column. */
export function minBoardWidth(configs, metrics) {
    let widest = 0;
    for (const cfg of configs)
        widest = Math.max(widest, cfg.width);
    return widest + 2 * metrics.gap;
}
/** v2 _make_way: move `cfg` clear of the `settled` cards it overlaps.  RIGHT, just clear of them,
 * when its row has the room (every card it overlaps starts at or left of it, the place is on the
 * board and clear, and the way there passes no card); otherwise DOWN its own column to the first
 * clear row, and every card not settled yet below it in its column moves down as much (a column
 * makes way as one) -- those (`carriedDown`) later make way only further down. */
export function makeWay(cfg, settled, order, boardW, metrics, carriedDown) {
    const gap = metrics.gap, start = aabb(cfg);
    const hit = settled.filter(p => overlapsWithGap(start, [p], metrics));
    const x = Math.max(...hit.map(p => p.col + p.width)) + gap;
    const place = [x, cfg.row, x + cfg.width, cfg.row + cfg.height];
    const way = [cfg.col, cfg.row, x + cfg.width, cfg.row + cfg.height];
    const passed = order.some(p => p !== cfg && overlapsWithGap(way, [p], metrics) && !overlapsWithGap(start, [p], metrics) && !overlapsWithGap(place, [p], metrics));
    if (!carriedDown.has(cfg) && hit.every(p => p.col <= cfg.col) && x <= boardW - gap - cfg.width && !passed && !overlapsWithGap(place, settled, metrics)) {
        cfg.col = x;
        return;
    }
    const row = below(cfg.col, cfg.row, cfg.width, cfg.height, settled.map(aabb), gap);
    const down = row - cfg.row;
    for (const p of order) {
        if (p !== cfg && !settled.includes(p) && p.row >= cfg.row && p.col < cfg.col + cfg.width + gap && cfg.col < p.col + p.width + gap) {
            p.row += down;
            carriedDown.add(p);
        }
    }
    cfg.row = row;
}
/** v2 pack: settle every card of `order` under north-west gravity, from where it stands, on a
 * board `boardW` wide (clamped to fit the widest card).  1. Overlaps settle in priority -- the
 * cards of `first`, then the rest, each group in the reading order of where its cards stand; a
 * card overlapping one already settled makes way (makeWay).  2. Every card falls north-west
 * (gravitySlot), every other card an obstacle, until nothing moves. */
export function pack(order, metrics, boardW, first = []) {
    const cards = [...order], gap = metrics.gap;
    boardW = Math.max(boardW, minBoardWidth(cards, metrics));
    for (const cfg of cards) {
        cfg.col = Math.min(Math.max(Math.trunc(cfg.col), gap), boardW - gap - cfg.width);
        cfg.row = Math.max(Math.trunc(cfg.row), gap);
    }
    const rank = new Map(cards.map((cfg, index) => [cfg, index]));
    const ahead = new Set(first);
    const reading = (a, b) => a.row - b.row || a.col - b.col || rank.get(a) - rank.get(b);
    const settled = [];
    const carriedDown = new Set();
    for (const cfg of [...cards].sort((a, b) => Number(!ahead.has(a)) - Number(!ahead.has(b)) || reading(a, b))) {
        if (overlapsWithGap(aabb(cfg), settled, metrics))
            makeWay(cfg, settled, cards, boardW, metrics, carriedDown);
        settled.push(cfg);
    }
    let moved = true;
    while (moved) {
        moved = false;
        for (const cfg of [...cards].sort(reading)) {
            const [col, row] = gravitySlot(cfg, cards.filter(p => p !== cfg), metrics);
            if (col !== cfg.col || row !== cfg.row) {
                cfg.col = col;
                cfg.row = row;
                moved = true;
            }
        }
    }
}
/** v2 drop: the board once `carried` is let go at `slot`, in place; `order` is every card where it
 * stood at pick-up, settled.  The TARGET is the card whose place the slot's top-left is in: the
 * two swap (the target takes the carried card's place, the carried card the slot); on empty space
 * the carried card takes the slot.  pack settles the board with those two first.  Let go where it
 * was picked up, it finds every card where it stood. */
export function drop(order, carried, slot, metrics, boardW) {
    if (!order.includes(carried))
        throw new Error("the carried card must belong to the board");
    const [x, y] = slot;
    const target = order.find(cfg => cfg !== carried && cfg.col <= x && x < cfg.col + cfg.width && cfg.row <= y && y < cfg.row + cfg.height);
    if (target) {
        target.col = carried.col;
        target.row = carried.row;
    }
    carried.col = x;
    carried.row = y;
    pack(order, metrics, boardW, target ? [carried, target] : [carried]);
}
/** v2 _nearest: the anchor that keeps `at` inside a card `size` long, nearest `seen`; with none,
 * the anchor nearest `seen`.  Ties go to the smaller anchor. */
function nearest(anchors, at, seen, size) {
    const inside = anchors.filter(a => a <= at && at < a + size);
    return (inside.length ? inside : anchors).reduce((best, a) => Math.abs(a - seen) < Math.abs(best - seen) || (Math.abs(a - seen) === Math.abs(best - seen) && a < best) ? a : best);
}
/** v2 slot_under: the place a carried card is put at -- under the pointer, nearest where it is
 * seen.  Anchors are the margin and the other cards' edges (a gap past each far edge); on each
 * axis the slot is the anchor that keeps `point` inside the carried card nearest `seen` (its
 * top-left as drawn; default `point`).  The column comes first; the row is chosen only among the
 * cards in that column band.  An occupied place stays a candidate: let go there, the two swap
 * (drop).  `cfg` only says how big the carried card is. */
export function slotUnder(point, cfg, others, metrics, boardW, seen = point) {
    const gap = metrics.gap;
    boardW = Math.max(boardW, minBoardWidth([...others, cfg], metrics));
    const maxX = Math.max(gap, boardW - gap - cfg.width);
    const xs = new Set([gap]);
    for (const other of others) {
        const [left, , right] = aabb(other);
        xs.add(left);
        xs.add(right + gap);
    }
    let cols = [...xs].filter(x => gap <= x && x <= maxX).sort(ascending);
    if (!cols.length)
        cols = [gap];
    const x = nearest(cols, Math.trunc(point[0]), Math.trunc(seen[0]), cfg.width);
    const ys = new Set([gap]);
    for (const other of others) {
        const [left, top, right, bottom] = aabb(other);
        if (x < right + gap && left < x + cfg.width + gap) {
            ys.add(top);
            ys.add(bottom + gap);
        }
    }
    const rows = [...ys].filter(y => y >= gap).sort(ascending);
    return [x, nearest(rows, Math.trunc(point[1]), Math.trunc(seen[1]), cfg.height)];
}
