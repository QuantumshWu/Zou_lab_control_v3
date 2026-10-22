//! The flat timeline an operator edits: periods and bracket posts in the order
//! they are drawn (`zlc_ui.pulse.models.schedule_item_order`). Only the
//! brackets' inclusive endpoints are stored; this order is derived from them.
use std::collections::BTreeSet;

use zlc_pulse::model::{Bracket, Sequence};

pub const PERIOD: &str = "period";
pub const BRACKET: &str = "bracket";
pub const COMPONENT: &str = "component";

pub type Item = (String, String);

/// `"<bracket id>:start"` or `":end"`.
pub fn bracket_post_key(bracket_id: &str, side: &str) -> String { format!("{bracket_id}:{side}") }

/// The bracket id of a post key (`key.rpartition(":")[0]`).
pub fn post_bracket(key: &str) -> &str { key.rsplit_once(':').map_or("", |(id, _)| id) }

/// Half-open period gaps of one bracket's inclusive endpoints.
pub fn bracket_gap_bounds(period_ids: &[&str], start: Option<&str>, end: Option<&str>) -> (usize, usize) {
    let index = |id: &str| period_ids.iter().position(|item| *item == id).unwrap_or(period_ids.len());
    (start.map_or(period_ids.len(), index), end.map_or(0, |id| index(id) + 1))
}

/// Visual items from each bracket's endpoints (outermost first): at a gap the
/// ends closing there (innermost first), an empty bracket sitting in it, then
/// the starts opening there (outermost first).
pub fn schedule_item_order(period_ids: &[&str], brackets: &[Bracket]) -> Vec<Item> {
    let bounds: Vec<(usize, usize)> = brackets.iter()
        .map(|b| bracket_gap_bounds(period_ids, b.start_period_id.as_deref(), b.end_period_id.as_deref())).collect();
    let mut items = Vec::new();
    let post = |id: &str, side: &str| (BRACKET.to_owned(), bracket_post_key(id, side));
    for gap in 0..=period_ids.len() {
        for (bracket, (first, stop)) in brackets.iter().zip(&bounds).rev() {
            if *stop == gap && first < stop { items.push(post(&bracket.bracket_id, "end")); }
        }
        for (bracket, (first, stop)) in brackets.iter().zip(&bounds) {
            if *first == gap && *stop == gap {
                items.push(post(&bracket.bracket_id, "start"));
                items.push(post(&bracket.bracket_id, "end"));
            }
        }
        for (bracket, (first, stop)) in brackets.iter().zip(&bounds) {
            if *first == gap && first < stop { items.push(post(&bracket.bracket_id, "start")); }
        }
        if gap < period_ids.len() { items.push((PERIOD.to_owned(), period_ids[gap].to_owned())); }
    }
    items
}

/// `_sequence_item_order(sequence)`.
pub fn sequence_item_order(sequence: &Sequence) -> Vec<Item> { schedule_item_order(&sequence.period_ids(), &sequence.brackets) }

/// The first free `<stem><n>`, counting from one.
pub fn unique_id<'a>(existing: impl IntoIterator<Item = &'a str>, stem: &str) -> String {
    let taken: BTreeSet<&str> = existing.into_iter().collect();
    let mut ordinal = 1;
    while taken.contains(format!("{stem}{ordinal}").as_str()) { ordinal += 1; }
    format!("{stem}{ordinal}")
}

/// Python's `repr` of an item tuple: `('period', 'p1')`.
pub fn item_repr(item: &Item) -> String {
    format!("({}, {})", zlc_pulse::error::repr(&item.0), zlc_pulse::error::repr(&item.1))
}
